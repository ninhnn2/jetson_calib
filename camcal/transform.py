"""The same camera for a picture that has been turned, resized or cropped.

A calibration belongs to one picture geometry: the sensor mode, the rotation and the size the
board was photographed in. Where the application's picture differs from that only by a
rotation in quarter turns, a resize or a crop, the camera follows exactly:

  rotate=90|180|270   counterclockwise, as ucpeer-detect's --rotate and cv2.rotate turn it
  scale=S | scale=SX,SY | resize=WxH
  crop=X,Y,W,H        the window's top left pixel and its size

A different sensor mode is only such a change if the driver says exactly which pixels it
bins and crops; where that is not known for certain, calibrate in the mode that is used.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from .export import Calibration
from .models import DIST_NAMES


def _std(cal: Calibration) -> dict:
    return dict(cal.meta.get("std") or {})


def _with(cal: Calibration, size, K, dist, std: dict, op: str) -> Calibration:
    meta = copy.deepcopy(cal.meta)
    if meta.get("std"):
        meta["std"] = std
    meta["derived"] = list(meta.get("derived", [])) + [op]
    return Calibration(cal.model, size, K, dist, meta)


def rotate90(cal: Calibration) -> Calibration:
    """A quarter turn counterclockwise: pixel (u, v) of a W x H picture goes to (v, W-1-u)."""
    w, h = cal.size
    fx, fy, cx, cy = cal.K[0, 0], cal.K[1, 1], cal.K[0, 2], cal.K[1, 2]
    K = np.array([[fy, 0, cy], [0, fx, w - 1 - cx], [0, 0, 1.0]])
    d = dict(zip(DIST_NAMES[cal.model], cal.dist))
    std = _std(cal)
    if "p1" in d:  # the tangential pair turns with the picture; the radial terms do not care
        d["p1"], d["p2"] = -d["p2"], d["p1"]
        if "p1" in std and "p2" in std:
            std["p1"], std["p2"] = std["p2"], std["p1"]
    for a, b in (("fx", "fy"), ("cx", "cy")):
        if a in std and b in std:
            std[a], std[b] = std[b], std[a]
    return _with(cal, (h, w), K, [d[n] for n in DIST_NAMES[cal.model]], std, "rotate=90")


def rotate(cal: Calibration, degrees: int) -> Calibration:
    if degrees % 90:
        raise ValueError("rotate by 0, 90, 180 or 270 degrees (counterclockwise)")
    for _ in range((degrees // 90) % 4):
        cal = rotate90(cal)
    return cal


def scale(cal: Calibration, sx: float, sy: float = None, size=None) -> Calibration:
    """The picture resized by sx, sy (pixel areas mapped onto pixel areas, as cv2.resize does)."""
    sy = sx if sy is None else sy
    if sx <= 0 or sy <= 0:
        raise ValueError("a scale is positive")
    w, h = cal.size
    size = size or (int(round(w * sx)), int(round(h * sy)))
    K = cal.K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    std = _std(cal)
    for n, s in (("fx", sx), ("cx", sx), ("fy", sy), ("cy", sy)):
        if n in std:
            std[n] *= s
    return _with(cal, size, K, cal.dist.copy(), std, "scale=%.6g,%.6g" % (sx, sy))


def crop(cal: Calibration, x: int, y: int, w: int, h: int) -> Calibration:
    W, H = cal.size
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
        raise ValueError("the window %d,%d %dx%d is not inside the %dx%d picture" % (x, y, w, h, W, H))
    K = cal.K.copy()
    K[0, 2] -= x
    K[1, 2] -= y
    return _with(cal, (w, h), K, cal.dist.copy(), _std(cal), "crop=%d,%d,%d,%d" % (x, y, w, h))


def apply(cal: Calibration, ops: Sequence[str]) -> Calibration:
    """The operations in the order given, each as written in this module's description."""
    for op in ops:
        name, _, arg = op.partition("=")
        try:
            if name == "rotate":
                cal = rotate(cal, int(arg))
            elif name == "scale":
                v = [float(x) for x in arg.split(",")]
                cal = scale(cal, v[0], v[1] if len(v) > 1 else None)
            elif name == "resize":
                w, h = (int(x) for x in arg.lower().split("x"))
                cal = scale(cal, w / cal.size[0], h / cal.size[1], (w, h))
            elif name == "crop":
                cal = crop(cal, *[int(x) for x in arg.split(",")])
            else:
                raise ValueError("no operation '%s' (rotate, scale, resize, crop)" % name)
        except (IndexError, TypeError) as e:
            raise ValueError("cannot read '%s': %s" % (op, e))
    return cal
