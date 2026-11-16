"""Pictures of a board from a camera that is known exactly: what a calibration is tested against.

Each pixel's ray is taken through the model's inverse (models.undistort, NumPy), met with the
board's plane and looked up in the board as printed. Nothing of OpenCV's camera code is in
that path, so a calibration that gets the camera back has been checked against something other
than itself. Several samples a pixel, a blur and sensor noise stand in for a lens and a sensor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import models
from .boards import BoardSpec

PAPER_MARGIN_MM = 12.0  # white paper around the pattern
TEXTURE_PX_PER_MM = 8.0


@dataclass
class SynthCamera:
    model: str
    K: np.ndarray
    dist: np.ndarray
    size: Tuple[int, int]  # width, height
    _rays: dict = field(default_factory=dict, repr=False, compare=False)

    def rays(self, ss: int) -> np.ndarray:
        """The ideal normalised coordinates of every sample: (H ss, W ss, 2)."""
        if ss not in self._rays:
            w, h = self.size
            u = (np.arange(w * ss) + 0.5) / ss - 0.5
            v = (np.arange(h * ss) + 0.5) / ss - 0.5
            out = np.empty((h * ss, w * ss, 2), dtype=np.float32)
            for r0 in range(0, h * ss, 256):  # in bands: the iteration's temporaries stay small
                vv = v[r0:r0 + 256]
                px = np.stack(np.meshgrid(u, vv), axis=-1)
                out[r0:r0 + len(vv)] = models.normalize(self.model, self.K, self.dist, px)
            self._rays[ss] = out
        return self._rays[ss]

    def to_dict(self) -> dict:
        return {"model": self.model, "image_size": list(self.size),
                "fx": self.K[0, 0], "fy": self.K[1, 1], "cx": self.K[0, 2], "cy": self.K[1, 2],
                "distortion": dict(zip(models.DIST_NAMES[self.model], [float(x) for x in self.dist]))}


def camera(model: str, size: Tuple[int, int], fx: float, fy: Optional[float] = None,
           cx: Optional[float] = None, cy: Optional[float] = None, dist: Sequence[float] = ()) -> SynthCamera:
    w, h = size
    K = np.array([[fx, 0, (w - 1) / 2.0 if cx is None else cx],
                  [0, fx if fy is None else fy, (h - 1) / 2.0 if cy is None else cy],
                  [0, 0, 1.0]])
    d = np.zeros(len(models.DIST_NAMES[models.check_model(model)]))
    d[: len(dist)] = dist
    return SynthCamera(model, K, d, (int(w), int(h)))


def _texture(spec: BoardSpec) -> np.ndarray:
    return spec.render(TEXTURE_PX_PER_MM, PAPER_MARGIN_MM)


def render(spec: BoardSpec, cam: SynthCamera, rvec, tvec, ss: int = 2, blur: float = 0.7,
           noise: float = 1.5, rng: Optional[np.random.Generator] = None,
           texture: Optional[np.ndarray] = None) -> np.ndarray:
    """The picture (grey, 8 bit) the camera takes of the board at this pose."""
    rng = rng or np.random.default_rng(0)
    tex = _texture(spec) if texture is None else texture
    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    rays = cam.rays(ss)
    x, y = rays[..., 0].astype(np.float64), rays[..., 1].astype(np.float64)
    # a ray lambda (x, y, 1) meets the board's plane (z_board = 0) where r3 . (lambda d - t) = 0
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
    den = r3[0] * x + r3[1] * y + r3[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (r3 @ t) / den
        px, py, pz = lam * x - t[0], lam * y - t[1], lam - t[2]
        bx = (r1[0] * px + r1[1] * py + r1[2] * pz) * 1000.0  # mm on the board
        by = (r2[0] * px + r2[1] * py + r2[2] * pz) * 1000.0
    hit = np.isfinite(lam) & (lam > 0)
    mx = np.where(hit, (bx + PAPER_MARGIN_MM) * TEXTURE_PX_PER_MM - 0.5, -10.0).astype(np.float32)
    my = np.where(hit, (by + PAPER_MARGIN_MM) * TEXTURE_PX_PER_MM - 0.5, -10.0).astype(np.float32)
    # paper is not white nor ink black; behind the board a wall
    paper = cv2.remap((tex.astype(np.float32) * (205.0 / 255.0) + 28.0), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
    wall = 105.0 + 12.0 * np.sin(np.linspace(0, 3.0, paper.shape[1], dtype=np.float32))[None, :]
    img = np.where(paper < 0, wall, paper)
    w, h = cam.size
    if ss > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    # light falling off to one side
    img = img * (0.86 + 0.14 * np.linspace(1.0, 0.0, w, dtype=np.float32))[None, :]
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _visible(spec: BoardSpec, cam: SynthCamera, rvec, tvec, border: float = 6.0) -> int:
    p = models.project(cam.model, cam.K, cam.dist, rvec, tvec, spec.corner_points())
    w, h = cam.size
    ok = np.all(np.isfinite(p), axis=1)
    ok &= (p[:, 0] > border) & (p[:, 0] < w - 1 - border) & (p[:, 1] > border) & (p[:, 1] < h - 1 - border)
    return int(ok.sum())


def random_poses(spec: BoardSpec, cam: SynthCamera, count: int, seed: int = 0, min_corners: int = 12,
                 max_tilt_deg: float = 45.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Views as a careful operator takes them: the board everywhere in the picture and out to
    its corners, near and far, tilted about both axes."""
    rng = np.random.default_rng(seed)
    w, h = cam.size
    bw, bh = (v / 1000.0 for v in spec.size_mm)
    centre = np.array([bw / 2, bh / 2, 0.0])
    whole = spec.type != "charuco"  # a chessboard has to be in the picture entirely
    need = spec.corner_count() if whole else min_corners
    cols = max(2, int(np.ceil(np.sqrt(count * w / h))))
    rows = max(2, int(np.ceil(count / cols)))
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    tries = 0
    while len(poses) < count and tries < count * 200:
        c, r = cells[tries % len(cells)]
        tries += 1
        lo, hi = (0.25, 0.75) if whole else (0.04, 0.96)
        u = (lo + (hi - lo) * (c + rng.uniform()) / cols) * (w - 1)
        v = (lo + (hi - lo) * (r + rng.uniform()) / rows) * (h - 1)
        ray = models.normalize(cam.model, cam.K, cam.dist, np.array([[u, v]]))[0]
        if not np.all(np.isfinite(ray)):
            continue
        frac = rng.uniform(0.3, 0.55) if whole else rng.uniform(0.35, 0.9)  # of the picture's height
        dist = cam.K[1, 1] * max(bw, bh) / (frac * h)
        tilt = np.radians(rng.uniform(0.0, max_tilt_deg)) * (0.6 if whole else 1.0)
        axis = rng.uniform(0, 2 * np.pi)
        Rt = cv2.Rodrigues(np.array([np.cos(axis), np.sin(axis), 0.0]) * tilt)[0]
        roll = np.radians(rng.uniform(-30, 30) + (90.0 if rng.uniform() < 0.3 else 0.0))
        Rz = cv2.Rodrigues(np.array([0, 0, roll]))[0]
        R = Rt @ Rz
        d = np.append(ray, 1.0)
        t = d / np.linalg.norm(d) * dist - R @ centre
        rvec = cv2.Rodrigues(R)[0].reshape(3)
        if _visible(spec, cam, rvec, t) >= need:
            poses.append((rvec, t))
    if len(poses) < count:
        raise ValueError("only %d of %d views of the board fit this camera" % (len(poses), count))
    return poses


def write_dataset(out_dir: str, spec: BoardSpec, cam: SynthCamera, count: int = 40, seed: int = 0,
                  noise: float = 1.5, blur: float = 0.7, ss: int = 2) -> List[str]:
    """Pictures, the board and the truth, laid out as `capture` leaves a session."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed + 1)
    tex = _texture(spec)
    paths, truth = [], []
    for i, (rvec, tvec) in enumerate(random_poses(spec, cam, count, seed)):
        name = "img_%04d.png" % (i + 1)
        cv2.imwrite(os.path.join(out_dir, name), render(spec, cam, rvec, tvec, ss, blur, noise, rng, tex))
        paths.append(os.path.join(out_dir, name))
        truth.append({"name": name, "rvec": [float(x) for x in rvec], "tvec": [float(x) for x in tvec]})
    with open(os.path.join(out_dir, "session.json"), "w") as f:
        json.dump({"board": spec.to_dict(), "source": {"kind": "synthetic", "seed": seed},
                   "image_size": list(cam.size)}, f, indent=2)
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump({"camera": cam.to_dict(), "views": truth}, f, indent=2)
    return paths
