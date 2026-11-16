"""An existing calibration against pictures it was not made from, and pictures undistorted.

validate: the board is found in each picture, its pose solved with the camera held fixed, and
the corners' distance from where that camera puts them reported. Taken some days later, or
after the camera was handled, this says whether the calibration still holds.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from . import boards, models
from .boards import BoardSpec
from .export import Calibration


def validate(cal: Calibration, spec: BoardSpec, paths: Sequence[str], min_corners: int = 8,
             overlay_dir: Optional[str] = None, log: Callable[[str], None] = lambda s: None) -> dict:
    rows, res_all, left_out = [], [], 0
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
    for p in paths:
        name = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append({"name": name, "why_not": "not a picture"})
            continue
        if (img.shape[1], img.shape[0]) != cal.size:
            rows.append({"name": name, "why_not": "%dx%d, the calibration is for %dx%d"
                         % (img.shape[1], img.shape[0], cal.size[0], cal.size[1])})
            continue
        det = boards.detect(spec, img, min_corners)
        if det is None:
            rows.append({"name": name, "why_not": "board not found"})
            continue
        obj = det.object_points(spec)
        pose = models.solve_pose_robust(cal.model, cal.K, cal.dist, obj, det.image)
        if pose is None:
            rows.append({"name": name, "why_not": "no pose explains the corners"})
            continue
        kept = pose[2]
        proj = cal.project(pose[0], pose[1], obj)
        r = (proj - det.image)[kept]
        res_all.append(r)
        left_out += int((~kept).sum())
        rows.append({"name": name, "corners": int(kept.sum()), "left_out": int((~kept).sum()),
                     "rms_px": models.rms([r]), "distance_m": float(np.linalg.norm(pose[1]))})
        log("  %-24s %3d corners  %.3f px%s" % (name, kept.sum(), rows[-1]["rms_px"],
                                                 "  (%d misread, left out)" % (~kept).sum() if (~kept).any() else ""))
        if overlay_dir:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for a, b, k in zip(det.image, proj, kept):  # the error drawn 20 times its length
                a = tuple(int(round(v)) for v in a)
                if not k:  # a misread corner, left out: a ring
                    cv2.circle(out, a, 7, (40, 40, 220), 1, cv2.LINE_AA)
                    continue
                e = tuple(int(round(v)) for v in (np.array(a) + 20.0 * (b - np.array(a))))
                cv2.line(out, a, e, (40, 40, 220), 1, cv2.LINE_AA)
                cv2.circle(out, a, 2, (60, 200, 60), -1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(overlay_dir, os.path.splitext(name)[0] + "_check.jpg"), out)
    if not res_all:
        return {"views": rows, "views_used": 0, "verdict": "no picture with the board in it"}
    norms = np.concatenate([np.linalg.norm(r, axis=1) for r in res_all])
    rms = models.rms(res_all)
    ref = (cal.meta.get("rms_px") or {}).get("held_out") or (cal.meta.get("rms_px") or {}).get("train")
    out = {"views": rows, "views_used": len(res_all), "corners": int(len(norms)), "rms_px": rms,
           "p50_px": float(np.percentile(norms, 50)), "p95_px": float(np.percentile(norms, 95)),
           "max_px": float(norms.max()), "calibration_rms_px": ref, "corners_left_out": left_out}
    if ref is None:
        out["verdict"] = "%.3f px (the calibration does not say what it had)" % rms
    elif rms <= max(1.5 * ref, ref + 0.1):
        out["verdict"] = "holds: %.3f px here, %.3f px when calibrated" % (rms, ref)
    else:
        out["verdict"] = ("does NOT hold: %.3f px here, %.3f px when calibrated. The lens or the mount "
                          "has moved, the picture geometry differs, or these pictures are blurred"
                          % (rms, ref))
    return out


def undistort_files(cal: Calibration, paths: Sequence[str], out_dir: str, alpha: float = 0.0,
                    size=None, save_maps: bool = False) -> dict:
    """Each picture undistorted into out_dir, with the pinhole camera of the result
    (rectified.json) and, asked for, the remap tables (rectify_maps.npz: new_K, map_x, map_y)."""
    os.makedirs(out_dir, exist_ok=True)
    newK, mx, my = cal.rectify_maps(alpha, size)
    done: List[str] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None or (img.shape[1], img.shape[0]) != cal.size:
            continue
        name = os.path.splitext(os.path.basename(p))[0] + "_rect.png"
        cv2.imwrite(os.path.join(out_dir, name), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        done.append(name)
    out_size = (mx.shape[1], mx.shape[0])
    rect = Calibration("pinhole-radtan4", out_size, newK, np.zeros(4), {
        "derived": list(cal.meta.get("derived", [])) + ["undistort alpha=%g" % alpha],
        "note": "the undistorted picture: a pinhole camera, no distortion left"})
    rect.save(os.path.join(out_dir, "rectified.json"))
    if save_maps:
        np.savez_compressed(os.path.join(out_dir, "rectify_maps.npz"), new_K=newK, map_x=mx, map_y=my)
    return {"pictures": done, "new_K": newK, "size": out_size}
