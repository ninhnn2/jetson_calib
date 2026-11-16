"""From pictures of a board to a camera, and to how far that camera can be trusted.

What is done beyond one call to calibrateCamera:

  the model is chosen, not assumed   every candidate is fitted with some views held out, and
        judged by how well it predicts those. More coefficients always fit the views they were
        given better; held-out views say whether the lens needed them. Of the models within 5%
        of the best the simplest is taken.
  bad measurements are removed       a corner further from where the fitted camera puts it than
        4 times the median distance (and 0.3 px) is dropped, a view whose remaining corners
        are 3 times worse than the median view (and 0.5 px) is dropped, and the fit repeated.
  the answer comes with its error    one standard deviation per parameter (the solver's, or a
        bootstrap over the views), the coverage of the picture, the spread of tilts, and a
        grade against what SLAM and ADAS need.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import __version__, boards, models
from .boards import BoardSpec
from .export import Calibration
from .models import Fit, View

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")
PRESETS = {
    "auto": models.MODELS,
    # what Kalibr, VINS-Fusion, OpenVINS, basalt and ORB-SLAM3 can be given
    "slam": ("pinhole-radtan4", "fisheye"),
}
GRID_COLS = 16  # the picture is cut into GRID_COLS columns of square cells for the coverage
SELECT_TOLERANCE = 0.05
GRADES = {
    "A": "intrinsics good enough for SLAM, VIO and long-range (L4-class) geometry",
    "B": "good enough for ADAS L2 functions and 3D object detection",
    "C": "usable for undistortion and rough geometry; not for metric work",
    "D": "not to be used: take the views again",
}


def parse_models(text: str) -> Tuple[str, ...]:
    if text in PRESETS:
        return PRESETS[text]
    return tuple(models.check_model(m.strip()) for m in text.split(",") if m.strip())


def list_images(inputs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            out += [os.path.join(p, n) for n in sorted(os.listdir(p))
                    if n.lower().endswith(IMAGE_EXT) and not n.startswith(("preview", "."))]
        else:
            out.append(p)
    return out


def _spans_a_plane(obj: np.ndarray) -> bool:
    """Corners in at least three rows and three columns of the board. A line of corners, or a
    line and one corner beside it, leaves the board free to tip about that line: such a view
    says nothing a fit can rely on, and can come out behind the camera."""
    if len(obj) < 6:
        return False
    return min(len(np.unique(np.round(obj[:, 0], 6))), len(np.unique(np.round(obj[:, 1], 6)))) >= 3


def detect_files(spec: BoardSpec, paths: Sequence[str], min_corners: int = 8,
                 log: Callable[[str], None] = lambda s: None):
    """The board in every picture: (views, picture size, [(name, why not)])."""
    views: List[View] = []
    skipped: List[Tuple[str, str]] = []
    size = None
    for i, p in enumerate(paths):
        name = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append((name, "not a picture"))
            continue
        s = (img.shape[1], img.shape[0])
        if size is None:
            size = s
        elif s != size:
            skipped.append((name, "%dx%d, the others %dx%d" % (s + size)))
            continue
        det = boards.detect(spec, img, min_corners)
        if det is None:
            skipped.append((name, "board not found"))
        elif not _spans_a_plane(det.object_points(spec)):
            skipped.append((name, "corners in fewer than 3 rows or columns"))
        else:
            views.append(View(name, det.ids, det.object_points(spec), det.image))
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            log("  %d/%d pictures, board found in %d" % (i + 1, len(paths), len(views)))
    return views, size, skipped


def clean(model: str, views: Sequence[View], size, min_corners: int = 8, guess: Optional[Fit] = None):
    """The views without the corners and views that disagree with all the others.
    Returns (views, fit of them, corners dropped, [(view, why dropped)])."""
    views = list(views)
    dropped_views: List[Tuple[str, str]] = []
    dropped_points = 0
    for _ in range(6):
        f = models.fit(model, views, size, guess)
        norms = [np.linalg.norm(r, axis=1) for r in f.residuals(views)]
        norms = [np.where(np.isfinite(n), n, np.inf) for n in norms]  # a view the camera cannot explain
        thr = max(4.0 * float(np.median(np.concatenate(norms))), 0.3)
        kept = [n[n <= thr] for n in norms]
        vr = np.array([np.sqrt((k * k).mean()) if len(k) else np.inf for k in kept])
        vthr = max(3.0 * float(np.median(vr[np.isfinite(vr)])), 0.5)
        out, changed = [], False
        for v, n, r in zip(views, norms, vr):
            m = n <= thr
            if r > vthr:
                dropped_views.append((v.name, "its corners are off by %.2f px, the others' by %.2f"
                                      % (r, float(np.median(vr)))))
                changed = True
            elif m.all():
                out.append(v)
            elif m.sum() < min_corners or not _spans_a_plane(v.obj[m]):
                dropped_views.append((v.name, "too few good corners (%d of %d)" % (m.sum(), len(m))))
                changed = True
            else:
                dropped_points += int((~m).sum())
                out.append(v.keep(m))
                changed = True
        if len(out) < 3:
            raise ValueError("fewer than 3 views agree with each other")
        views = out
        if not changed:
            return views, f, dropped_points, dropped_views
    return views, models.fit(model, views, size, guess), dropped_points, dropped_views


def cross_validate(model: str, views: Sequence[View], size, folds: int = 5, seed: int = 0,
                   guess: Optional[Fit] = None) -> Optional[Tuple[float, np.ndarray]]:
    """The error on views the fit has not seen: each fold is left out in turn, the camera
    fitted to the rest, and the fold's views explained with that camera (only their six pose
    parameters free). Returns (rms over all corners, rms per view); None with too few views."""
    n = len(views)
    k = min(folds, n // 3)
    if k < 2:
        return None
    order = np.random.default_rng(seed).permutation(n)
    res: List[Optional[np.ndarray]] = [None] * n
    for j in range(k):
        test = set(int(i) for i in order[j::k])
        f = models.fit(model, [views[i] for i in range(n) if i not in test], size, guess)
        for i in test:
            pose = models.solve_pose(model, f.K, f.dist, views[i].obj, views[i].img)
            if pose is None:
                raise ValueError("%s: a held-out view cannot be explained" % model)
            res[i] = models.project(model, f.K, f.dist, pose[0], pose[1], views[i].obj) - views[i].img
            if not np.all(np.isfinite(res[i])):
                raise ValueError("%s: a held-out view projects outside the model" % model)
    return models.rms(res), np.array([models.rms([r]) for r in res])


def bootstrap_std(model: str, views: Sequence[View], size, count: int, seed: int = 0,
                  guess: Optional[Fit] = None) -> Optional[Dict[str, float]]:
    """One standard deviation per parameter from refitting to views drawn with replacement."""
    rng = np.random.default_rng(seed + 7)
    rows = []
    for _ in range(count):
        idx = rng.integers(0, len(views), len(views))
        try:
            f = models.fit(model, [views[i] for i in idx], size, guess)
        except ValueError:
            continue
        rows.append([f.K[0, 0], f.K[1, 1], f.K[0, 2], f.K[1, 2]] + list(f.dist))
    if len(rows) < max(4, count // 2):
        return None
    sd = np.std(np.array(rows), axis=0, ddof=1)
    return dict(zip(("fx", "fy", "cx", "cy") + models.DIST_NAMES[model], [float(x) for x in sd]))


def coverage_grid(views: Sequence[View], size) -> np.ndarray:
    """Corners per cell: (rows, GRID_COLS)."""
    w, h = size
    rows = max(1, int(round(GRID_COLS * h / w)))
    grid = np.zeros((rows, GRID_COLS), dtype=np.int64)
    for v in views:
        c = np.clip((v.img[:, 0] + 0.5) * GRID_COLS / w, 0, GRID_COLS - 1).astype(int)
        r = np.clip((v.img[:, 1] + 0.5) * rows / h, 0, rows - 1).astype(int)
        np.add.at(grid, (r, c), 1)
    return grid


def _coverage(grid: np.ndarray) -> Tuple[float, float, List[str]]:
    have = grid >= 2
    ring = np.ones_like(have)
    ring[1:-1, 1:-1] = False
    rows, cols = have.shape
    empty = []
    for rn, rs in (("top", slice(0, rows // 3)), ("middle", slice(rows // 3, rows - rows // 3)),
                   ("bottom", slice(rows - rows // 3, rows))):
        for cn, cs in (("left", slice(0, cols // 3)), ("centre", slice(cols // 3, cols - cols // 3)),
                       ("right", slice(cols - cols // 3, cols))):
            if have[rs, cs].mean() < 0.5:
                empty.append("%s %s" % (rn, cn))
    return float(have.mean()), float(have[ring].mean()), empty


@dataclass
class Result:
    calibration: Calibration
    fit: Fit
    views: List[View]  # the ones the fit used, bad corners removed
    residuals: List[np.ndarray]
    grid: np.ndarray
    held_out_per_view: Optional[np.ndarray] = None
    log: List[str] = field(default_factory=list)


# what each grade needs: error on held-out views (px), share of the picture's cells and of its
# edge with corners, views used, focal length's relative 1 sigma, share of views tilted over 15 degrees.
# The number of views is a floor, not the measure: the error shrinks as 1/sqrt(views) (with the
# A3 board on a 1080p camera, 0.1 px corner noise, views spread well: picture-to-ray error 1.09 px
# from 10 views, 0.70 from 20, 0.41 from 40, 0.41 from 60), and the focal length's sigma is what
# says whether these views were enough: noisier pictures need more of them.
# "bias" is the focal length's disagreement between fits to the halves of the board (board_halves):
# what the fit's own sigma cannot see, a board that is not flat or not printed true.
GRADE_NEEDS = {
    "A": {"e": 0.25, "cov": 0.85, "border": 0.6, "views": 20, "sf": 0.001, "bias": 0.003, "tilt": 0.3},
    "B": {"e": 0.5, "cov": 0.7, "border": 0.35, "views": 15, "sf": 0.003, "bias": 0.01, "tilt": 0.2},
    "C": {"e": 1.0, "cov": 0.4, "border": 0.0, "views": 8, "sf": 0.01, "bias": 0.03, "tilt": 0.0},
}


def grade_inputs(q: dict) -> dict:
    """The quantities GRADE_NEEDS is about, from a calibration's quality."""
    e = q["rms_held_out"] if q["rms_held_out"] is not None else q["rms_train"] * 1.5
    halves = q.get("board_halves")
    return {"e": e, "cov": q["coverage"], "border": q["border_coverage"], "views": q["views_used"],
            "sf": q["sigma_f_rel"] if q["sigma_f_rel"] is not None else 1.0,
            "bias": halves["fx_relative"] if halves else 0.0, "tilt": q["tilted_share"]}


def grade(q: dict) -> Tuple[str, List[str]]:
    """The grade and what stands between it and the next one. Thresholds are engineering
    practice for a 1 to 2 megapixel camera, not a standard: an L4 programme has its own."""
    have, need = grade_inputs(q), GRADE_NEEDS
    text = {
        "e": lambda h, n: "error on held-out views %.2f px, needs %.2f or less: sharper, steadier pictures "
                          "of a flat board" % (h, n),
        "cov": lambda h, n: "corners in %.0f%% of the picture, needs %.0f%%" % (100 * h, 100 * n),
        "border": lambda h, n: "corners along %.0f%% of the picture's edge, needs %.0f%%: let the board run "
                               "out of the picture on every side" % (100 * h, 100 * n),
        "views": lambda h, n: "%d views, needs %d" % (h, n),
        "sf": lambda h, n: "focal length known to %.2f%%, needs %.2f%%: more tilted views, near and far"
                           % (100 * h, 100 * n),
        "tilt": lambda h, n: "%.0f%% of the views tilted more than 15 degrees, needs %.0f%%" % (100 * h, 100 * n),
        "bias": lambda h, n: "the halves of the board give focal lengths %.2f%% apart, needs %.2f%%: the board is "
                             "not flat (or not printed true); more pictures do not help" % (100 * h, 100 * n),
    }

    def missing(g):
        out = []
        for k, n in need[g].items():
            bad = have[k] > n if k in ("e", "sf", "bias") else have[k] < n
            if bad:
                out.append(text[k](have[k], n))
        return out

    folds = [] if q.get("valid_over_picture", True) else [
        "the model turns back before the picture's corners: do not undistort there. Views that reach the "
        "corners, or --model fisheye for a wide lens"]
    for g, nxt in (("A", None), ("B", "A"), ("C", "B")):
        if not missing(g):
            if folds and g in "AB":  # good where it was measured, not to be used beyond
                return "C", folds
            return g, folds + (["for %s: %s" % (nxt, m) for m in missing(nxt)] if nxt else [])
    return "D", folds + ["for C: %s" % m for m in missing("C")]


def board_halves(model: str, views: Sequence[View], size, full: Fit, min_corners: int = 8) -> Optional[dict]:
    """The camera fitted four more times, each time to the corners of one half of the board
    only (left, right, top, bottom), and how far those cameras are from the one fitted to all.

    On a flat, truly printed board the halves agree to within their own noise (3 sigma of each
    half's fit is taken off, what is left is reported). A sheet that bulges,
    or a print stretched one way, gives each half its own camera: the fit to all of it then
    sits between them with a small sigma, wrong by about their disagreement. No number of
    pictures removes that, and nothing else in the fit shows it. None with too few views."""
    xy = np.concatenate([v.obj[:, :2] for v in views])
    mid = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    fits = []
    for axis in (0, 1):
        for side in (-1.0, 1.0):
            part = []
            for v in views:
                m = (v.obj[:, axis] - mid[axis]) * side >= 0
                if m.sum() >= min_corners and _spans_a_plane(v.obj[m]):
                    part.append(v.keep(m))
            if len(part) < max(6, len(views) // 2):
                return None
            try:
                fits.append(models.fit(model, part, size, full))
            except ValueError:
                return None
    K = np.array([[f.K[0, 0], f.K[1, 1], f.K[0, 2], f.K[1, 2]] for f in fits])
    ref = np.array([full.K[0, 0], full.K[1, 1], full.K[0, 2], full.K[1, 2]])
    # a half determines the camera less well than the whole: only what its own 3 sigma does not
    # explain counts (flat board, rendered: nothing left; the sheet on the wall: 1%)
    sd = np.array([[f.std[n] if f.std else 0.0 for n in ("fx", "fy", "cx", "cy")] for f in fits])
    d = np.maximum(np.abs(K - ref) - 3.0 * sd, 0.0).max(axis=0)
    return {"fx_relative": float(max(d[0] / ref[0], d[1] / ref[1])), "cx_px": float(d[2]), "cy_px": float(d[3]),
            "fx_px": [float(K[:, 0].min()), float(K[:, 0].max())],
            "cx_range_px": [float(K[:, 2].min()), float(K[:, 2].max())],
            "cy_range_px": [float(K[:, 3].min()), float(K[:, 3].max())]}


def calibrate(spec: BoardSpec, views: Sequence[View], size, candidates: Sequence[str] = models.MODELS,
              folds: int = 5, seed: int = 0, bootstrap: int = 0, min_corners: int = 8,
              views_given: Optional[int] = None, source: Optional[dict] = None,
              log: Callable[[str], None] = lambda s: None) -> Result:
    if len(views) < 3:
        raise ValueError("the board was found in %d pictures; a calibration needs 3, a good one 20" % len(views))
    size = (int(size[0]), int(size[1]))
    candidates = tuple(candidates)

    # 1. every candidate on everything, to see which of them the lens is like
    first: Dict[str, Fit] = {}
    for m in sorted(candidates, key=lambda m: models.SIMPLICITY[m]):
        try:
            guess = next((first[p] for p in ("pinhole-radtan4", "pinhole-radtan") if p in first), None)
            first[m] = models.fit(m, views, size, guess)
            log("  %-17s fits all %d views to %.3f px" % (m, len(views), first[m].rms))
        except ValueError as e:
            log("  %-17s %s" % (m, e))
    if not first:
        raise ValueError("no camera model fits these views")
    best = min(f.rms for f in first.values())
    ref = min((m for m in first if first[m].rms <= 1.1 * best), key=lambda m: models.SIMPLICITY[m])
    guess = first.get("pinhole-radtan4") or first.get("pinhole-radtan") or first[ref]

    # 2. out with what disagrees
    used, _f, dropped_points, dropped_views = clean(ref, views, size, min_corners, guess)
    points_given = sum(len(v) for v in views)
    log("  kept %d of %d views, dropped %d of %d corners (judged with %s)"
        % (len(used), len(views), points_given - sum(len(v) for v in used), points_given, ref))

    # 3. the model, by views held out
    table = []
    for m in first:
        row = {"model": m, "rms_train": None, "rms_held_out": None, "chosen": False}
        try:
            row["rms_train"] = models.fit(m, used, size, guess).rms
            cv = cross_validate(m, used, size, folds, seed, guess)
            if cv is not None:
                row["rms_held_out"] = cv[0]
            log("  %-17s train %.3f px, held out %s" % (
                m, row["rms_train"], "%.3f px" % cv[0] if cv else "- (too few views)"))
        except ValueError as e:
            row["note"] = str(e)
            log("  %-17s %s" % (m, e))
        table.append(row)
    scored = [r for r in table if r["rms_held_out"] is not None]
    if scored:
        b = min(r["rms_held_out"] for r in scored)
        ok = [r for r in scored if r["rms_held_out"] <= b * (1 + SELECT_TOLERANCE) + 0.002]
        chosen = min(ok, key=lambda r: models.SIMPLICITY[r["model"]])
    else:  # too few views to hold any out: the plainest model that fitted
        fitted = [r for r in table if r["rms_train"] is not None]
        if not fitted:
            raise ValueError("no camera model fits these views")
        chosen = min(fitted, key=lambda r: models.SIMPLICITY[r["model"]])
    chosen["chosen"] = True
    model = chosen["model"]

    # 4. the camera, and how sure
    f = models.fit(model, used, size, guess)
    cv = cross_validate(model, used, size, folds, seed, guess) if scored else None
    std, method = f.std, "from the fit's covariance"
    if bootstrap or std is None:
        n = bootstrap or 16
        bs = bootstrap_std(model, used, size, n, seed, guess)
        if bs is not None:
            std, method = bs, "bootstrap over the views, %d resamples" % n
    res = f.residuals(used)
    norms = np.concatenate([np.linalg.norm(r, axis=1) for r in res])
    grid = coverage_grid(used, size)
    cov, border, empty = _coverage(grid)
    tilts, dists = [], []
    centre = np.append(np.array(spec.size_mm) / 2000.0, 0.0)
    for r, t in zip(f.rvecs, f.tvecs):
        R = cv2.Rodrigues(r.reshape(3, 1))[0]
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))))
        dists.append(float(np.linalg.norm(R @ centre + t)))
    sf = None if std is None else max(std["fx"] / f.K[0, 0], std["fy"] / f.K[1, 1])
    halves = board_halves(model, used, size, f, min_corners)
    fov = models.fov_degrees(model, f.K, f.dist, size)
    quality = {
        "views_given": views_given if views_given is not None else len(views),
        "views_with_board": len(views),
        "views_used": len(used),
        "corners_used": int(sum(len(v) for v in used)),
        "corners_dropped": int(points_given - sum(len(v) for v in used)),
        "rms_train": f.rms,
        "rms_held_out": None if cv is None else cv[0],
        "coverage": cov,
        "border_coverage": border,
        "regions_without_corners": empty,
        "tilt_deg_max": max(tilts),
        "tilted_share": float(np.mean(np.array(tilts) > 15.0)),
        "distance_m": [min(dists), max(dists)],
        "sigma_f_rel": sf,
        "sigma_c_px": None if std is None else max(std["cx"], std["cy"]),
        "board_halves": halves,
        "valid_over_picture": models.valid_over_picture(model, f.K, f.dist, size),
    }
    # what to carry into a calculation: the fit's sigma, or the halves' disagreement where larger
    uncertainty = None
    if std is not None or halves is not None:
        s0 = (sf or 0.0, std["cx"] if std else 0.0, std["cy"] if std else 0.0)
        h0 = (halves["fx_relative"], halves["cx_px"], halves["cy_px"]) if halves else (0.0, 0.0, 0.0)
        uncertainty = {"fx_relative": float(max(s0[0], h0[0])), "cx_px": float(max(s0[1], h0[1])),
                       "cy_px": float(max(s0[2], h0[2])),
                       "method": "1 sigma: the larger of the fit's standard deviation and the disagreement "
                                 "between fits to the halves of the board"}
    g, todo = grade(quality)
    meta = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "camcal_version": __version__,
        "opencv_version": cv2.__version__,
        # what this OpenCV's ChArUco detector adds to a corner; taken off before the fit
        "charuco_offset_removed_px": [float(x) for x in boards.charuco_offset()] if spec.type == "charuco" else None,
        "grade": g,
        "grade_meaning": GRADES[g],
        "to_improve": todo,
        "rms_px": {"train": f.rms, "held_out": quality["rms_held_out"],
                   "p50": float(np.percentile(norms, 50)), "p95": float(np.percentile(norms, 95)),
                   "max": float(norms.max())},
        "std": None if std is None else dict(std, method=method),
        "uncertainty": uncertainty,
        # None where the model cannot be followed out to the picture's edge
        "fov_deg": {k: (v if np.isfinite(v) else None) for k, v in zip(("horizontal", "vertical", "diagonal"), fov)},
        "quality": quality,
        "model_selection": table,
        "views": [{"name": v.name, "corners": len(v), "rms_px": models.rms([r]),
                   "rms_held_out_px": None if cv is None else float(cv[1][i]),
                   "tilt_deg": tilts[i], "distance_m": dists[i]} for i, (v, r) in enumerate(zip(used, res))],
        "dropped_views": [{"name": n, "why": w} for n, w in dropped_views],
        "board": spec.to_dict(),
        "source": source or {},
    }
    cal = Calibration(model, size, f.K, f.dist, meta)
    return Result(cal, f, list(used), res, grid, None if cv is None else cv[1])


def fingerprint(paths: Sequence[str]) -> dict:
    """Which pictures a calibration was made from: their number and one SHA-256 over their names
    and contents, so that a calibration can be traced to its data and the data shown unchanged."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths, key=os.path.basename):
        h.update(os.path.basename(p).encode("utf-8") + b"\0")
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return {"pictures": len(paths), "sha256": h.hexdigest()}


def load_session(inputs: Sequence[str]) -> dict:
    """session.json of the first directory given, if it has one (written by `capture`)."""
    for p in inputs:
        s = os.path.join(p, "session.json") if os.path.isdir(p) else None
        if s and os.path.exists(s):
            with open(s) as f:
                return json.load(f)
    return {}


def summary(cal: Calibration) -> str:
    m = cal.meta
    q, r, std = m["quality"], m["rms_px"], m.get("std") or {}

    def pm(name, value, fmt="%.2f"):
        return (fmt + " ± " + fmt) % (value, std[name]) if name in std else fmt % value

    lines = [
        "model      %s, %dx%d" % (cal.model, cal.size[0], cal.size[1]),
        "focal      fx %s  fy %s px" % (pm("fx", cal.K[0, 0]), pm("fy", cal.K[1, 1])),
        "centre     cx %s  cy %s px" % (pm("cx", cal.K[0, 2]), pm("cy", cal.K[1, 2])),
        "distortion " + "  ".join("%s %s" % (n, pm(n, v, "%.5f")) for n, v in zip(models.DIST_NAMES[cal.model], cal.dist)),
        "field      %s x %s degrees (diagonal %s)" % tuple(
            "?" if m["fov_deg"][k] is None else "%.1f" % m["fov_deg"][k] for k in ("horizontal", "vertical", "diagonal")),
        "error      %.3f px rms on the views used%s; half the corners within %.3f px, 95%% within %.3f" % (
            r["train"], "" if r["held_out"] is None else ", %.3f px on views held out" % r["held_out"],
            r["p50"], r["p95"]),
        "views      %d used of %d with the board (%d pictures), %d corners, %d dropped" % (
            q["views_used"], q["views_with_board"], q["views_given"], q["corners_used"], q["corners_dropped"]),
        "coverage   %.0f%% of the picture, %.0f%% of its edge; %.0f%% of the views tilted over 15 degrees" % (
            100 * q["coverage"], 100 * q["border_coverage"], 100 * q["tilted_share"]),
    ]
    hv, u = q.get("board_halves"), m.get("uncertainty")
    if hv:
        lines.append("halves     fitted to each half of the board alone: fx %.1f to %.1f px (%.2f%% from the whole), "
                     "cx within %.1f, cy within %.1f px" % (hv["fx_px"][0], hv["fx_px"][1], 100 * hv["fx_relative"],
                                                            hv["cx_px"], hv["cy_px"]))
    if u:
        lines.append("carry      focal ± %.2f%%, cx ± %.1f, cy ± %.1f px into calculations (1 sigma)"
                     % (100 * u["fx_relative"], u["cx_px"], u["cy_px"]))
    if not q.get("valid_over_picture", True):
        lines.append("model      turns back before the picture's corners")
    lines += [
        "grade      %s: %s" % (m["grade"], m["grade_meaning"]),
    ]
    lines += ["           " + t for t in m["to_improve"]]
    return "\n".join(lines)
