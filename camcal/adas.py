"""What a driving function does with the calibration: the ground, and how wrong a distance is.

  ground      a pixel to the point of the road it shows (vehicle frame: X forward, Y left, Z up),
              and back. Holds on a flat road only.
  bev         the picture seen from above (inverse perspective mapping), with the remap tables
  budget      how far off a measured distance is, at each range, and what makes it so:
                ground-plane ranging   the range of the pixel where an object stands on the road.
                                       Its error grows with the square of the range: the angle
                                       below the horizon gets small, and every error in it
                                       (pitch, principal point, the detector's pixel) counts more.
                width ranging          the range from the width in pixels of an object of known
                                       width. Its error is the focal length's, at every range.
              The numbers going in are the calibration's `uncertainty` (the fit's 1 sigma or the
              disagreement between the halves of the board, whichever is larger), the
              extrinsics', and what is given on the command line. Each part is the change in the
              answer when that one quantity is off by its 1 sigma; the total is their root sum
              of squares (they are independent).
"""

from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .export import Calibration

_BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # camera axes looking along X, level


def camera_pose(height_m: float, pitch_down_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0,
                x_m: float = 0.0, y_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """(R_vehicle_camera, t_vehicle_camera) from the angles as `extrinsics` reports them."""
    rz = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(yaw_deg)]))[0]
    ry = cv2.Rodrigues(np.array([0.0, np.radians(pitch_down_deg), 0.0]))[0]
    rr = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(-roll_deg)]))[0]
    return rz @ ry @ _BASE @ rr, np.array([x_m, y_m, height_m], dtype=np.float64)


def load_extrinsics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pose_of(ext: dict) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(ext["R_vehicle_camera"], dtype=np.float64), np.asarray(ext["t_vehicle_camera_m"], dtype=np.float64)


def ground_to_pixel(cal: Calibration, R_vc: np.ndarray, t_vc: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Points of the road (X, Y) in metres to pixels. NaN behind the camera."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return cal.project(R_vc.T, -R_vc.T @ t_vc, np.c_[xy, np.zeros(len(xy))])


def pixel_to_ground(cal: Calibration, R_vc: np.ndarray, t_vc: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Pixels to the points of the road they show: (N, 2) X, Y in metres. NaN at and above the
    horizon, and where the camera model has no ray for the pixel."""
    n = cal.normalize(np.asarray(pixels, dtype=np.float64).reshape(-1, 2))
    ray = np.c_[n, np.ones(len(n))] @ R_vc.T
    with np.errstate(all="ignore"):
        s = -t_vc[2] / ray[:, 2]
        out = t_vc[:2] + s[:, None] * ray[:, :2]
    out[~(ray[:, 2] < -1e-9)] = np.nan
    return out


def bev_maps(cal: Calibration, R_vc: np.ndarray, t_vc: np.ndarray, ahead: Tuple[float, float] = (2.0, 30.0),
             side: float = 8.0, metres_per_pixel: float = 0.02):
    """(map_x, map_y, info) for cv2.remap: the road from above. The top row is X = ahead[1],
    the left column Y = +side; a pixel is metres_per_pixel on the road.
    Row r, column c show X = ahead[1] - (r + 0.5) m, Y = side - (c + 0.5) m (m = metres_per_pixel)."""
    rows = int(round((ahead[1] - ahead[0]) / metres_per_pixel))
    cols = int(round(2 * side / metres_per_pixel))
    X = ahead[1] - (np.arange(rows) + 0.5) * metres_per_pixel
    Y = side - (np.arange(cols) + 0.5) * metres_per_pixel
    gx, gy = np.meshgrid(X, Y, indexing="ij")
    px = ground_to_pixel(cal, R_vc, t_vc, np.c_[gx.ravel(), gy.ravel()])
    px = np.where(np.isfinite(px), px, -1.0).astype(np.float32)
    info = {"x_m": [float(ahead[0]), float(ahead[1])], "y_m": [-float(side), float(side)],
            "metres_per_pixel": float(metres_per_pixel), "size": [cols, rows],
            "layout": "row r, column c: X = x_m[1] - (r + 0.5) m, Y = y_m[1] - (c + 0.5) m"}
    return px[:, 0].reshape(rows, cols), px[:, 1].reshape(rows, cols), info


# --- the error budget ------------------------------------------------------------------------


def uncertainty_of(cal: Calibration) -> dict:
    """1 sigma of the focal length (relative) and of the principal point (pixels): the
    calibration's own `uncertainty` where it has one, else the fit's standard deviations."""
    u = cal.meta.get("uncertainty")
    if u:
        return {"fx_relative": float(u["fx_relative"]), "cx_px": float(u["cx_px"]), "cy_px": float(u["cy_px"]),
                "from": u.get("method", "the calibration's uncertainty")}
    std = cal.meta.get("std") or {}
    if "fx" in std:
        return {"fx_relative": float(max(std["fx"] / cal.K[0, 0], std["fy"] / cal.K[1, 1])),
                "cx_px": float(std["cx"]), "cy_px": float(std["cy"]),
                "from": "the fit's standard deviations only: too small if the board was not flat"}
    raise ValueError("the calibration says nothing of its uncertainty: give --sigma-f, --sigma-c")


def budget(cal: Calibration, height_m: float, pitch_down_deg: float, yaw_deg: float = 0.0, roll_deg: float = 0.0,
           ranges_m: Sequence[float] = (5, 10, 15, 20, 30, 40, 50, 75, 100), sigma: Optional[dict] = None,
           object_width_m: float = 1.8, lateral_m: float = 0.0) -> dict:
    """The error of a range measured with this camera, at each range. sigma: 1 sigma of
    f_rel, cx_px, cy_px, pitch_deg, roll_deg, yaw_deg, height_m, pixel_px (where the detector puts
    the object's foot), width_px (its width); what is not given comes from the calibration, the
    rest as DEFAULT_SIGMA. Ranges are along the road from the point under the camera."""
    s = dict(DEFAULT_SIGMA)
    u = None
    if any((sigma or {}).get(k) is None for k in ("f_rel", "cx_px", "cy_px")):
        u = uncertainty_of(cal)
        s.update(f_rel=u["fx_relative"], cx_px=u["cx_px"], cy_px=u["cy_px"])
    s.update({k: v for k, v in (sigma or {}).items() if v is not None})

    def measure(pixel, f_rel=0.0, cx=0.0, cy=0.0, pitch=0.0, roll=0.0, yaw=0.0, height=0.0):
        """What the function computes from the pixel when the truth differs from what it
        believes by these: (range along X, Y) from the point under the camera."""
        K = cal.K.copy()
        K[0, 0] *= 1 + f_rel
        K[1, 1] *= 1 + f_rel
        K[0, 2] += cx
        K[1, 2] += cy
        c = Calibration(cal.model, cal.size, K, cal.dist)
        R, t = camera_pose(height_m + height, pitch_down_deg + pitch, yaw_deg + yaw, roll_deg + roll)
        return pixel_to_ground(c, R, t, pixel)[0]

    R0, t0 = camera_pose(height_m, pitch_down_deg, yaw_deg, roll_deg)
    w, h = cal.size
    rows = []
    for d in ranges_m:
        px = ground_to_pixel(cal, R0, t0, np.array([[float(d), float(lateral_m)]]))[0]
        row = {"range_m": float(d), "pixel": [float(px[0]), float(px[1])],
               "in_picture": bool(np.all(np.isfinite(px)) and 0 <= px[0] <= w - 1 and 0 <= px[1] <= h - 1)}
        if not np.all(np.isfinite(px)):
            rows.append(row)
            continue
        parts, lat = {}, {}
        for name, kw, sg in (("focal", "f_rel", s["f_rel"]), ("cx", "cx", s["cx_px"]), ("cy", "cy", s["cy_px"]),
                             ("pitch", "pitch", s["pitch_deg"]), ("roll", "roll", s["roll_deg"]),
                             ("yaw", "yaw", s["yaw_deg"]), ("height", "height", s["height_m"])):
            e = [measure(px.reshape(1, 2), **{kw: sign * sg}) - [d, lateral_m] for sign in (1, -1)]
            e = np.abs(np.array(e))
            e = np.where(np.isfinite(e), e, np.inf).mean(axis=0)
            parts[name], lat[name] = float(e[0]), float(e[1])
        e = np.abs(np.array([measure(px.reshape(1, 2) + [0, sign * s["pixel_px"]]) - [d, lateral_m] for sign in (1, -1)]))
        parts["pixel"] = float(np.where(np.isfinite(e), e, np.inf).mean(axis=0)[0])
        e = np.abs(np.array([measure(px.reshape(1, 2) + [sign * s["pixel_px"], 0]) - [d, lateral_m] for sign in (1, -1)]))
        lat["pixel"] = float(np.where(np.isfinite(e), e, np.inf).mean(axis=0)[1])
        total = float(np.sqrt(sum(v * v for v in parts.values())))
        # the vehicle pitching while it drives: not the calibration's, shown beside it
        e = np.abs(np.array([measure(px.reshape(1, 2), pitch=sign * s["pitch_motion_deg"]) - [d, lateral_m]
                             for sign in (1, -1)]))
        motion = float(np.where(np.isfinite(e), e, np.inf).mean(axis=0)[0])
        # the width of an object there, in pixels, by the camera's own (distorted) projection
        ends = ground_to_pixel(cal, R0, t0, np.array([[d, lateral_m - object_width_m / 2], [d, lateral_m + object_width_m / 2]]))
        wpx = float(abs(ends[0, 0] - ends[1, 0])) if np.all(np.isfinite(ends)) else float("nan")
        rel = float(np.sqrt(s["f_rel"] ** 2 + (s["width_px"] / wpx) ** 2)) if wpx > 0 else float("nan")
        row.update(ground={"error_m": total, "relative": total / d, "parts_m": parts,
                           "with_pitch_motion_m": float(np.hypot(total, motion))},
                   lateral={"error_m": float(np.sqrt(sum(v * v for v in lat.values()))), "parts_m": lat},
                   width={"pixels": wpx, "error_m": rel * d, "relative": rel})
        rows.append(row)

    def reach(limit):  # the range up to which ground-plane ranging stays within the limit
        ds = np.arange(2.0, 300.5, 1.0)
        last = None
        for d in ds:
            px = ground_to_pixel(cal, R0, t0, np.array([[d, lateral_m]]))[0]
            if not np.all(np.isfinite(px)):
                break
            tot = 0.0
            for kw, sg in (("f_rel", s["f_rel"]), ("cx", s["cx_px"]), ("cy", s["cy_px"]), ("pitch", s["pitch_deg"]),
                           ("roll", s["roll_deg"]), ("yaw", s["yaw_deg"]), ("height", s["height_m"])):
                e = abs(measure(px.reshape(1, 2), **{kw: sg})[0] - d)
                tot += (e if np.isfinite(e) else np.inf) ** 2
            e = abs(measure(px.reshape(1, 2) + [0, s["pixel_px"]])[0] - d)
            tot += (e if np.isfinite(e) else np.inf) ** 2
            if np.sqrt(tot) / d > limit:
                break
            last = float(d)
        return last

    horizon = cal.project(R0.T, -R0.T @ t0, np.array([[1e7, lateral_m, height_m]]))[0]
    return {"camera": {"height_m": float(height_m), "pitch_down_deg": float(pitch_down_deg),
                       "yaw_deg": float(yaw_deg), "roll_deg": float(roll_deg)},
            "sigma": s, "calibration_uncertainty_from": u["from"] if u else "given",
            "object_width_m": float(object_width_m), "lateral_m": float(lateral_m),
            "horizon_row_px": float(horizon[1]), "rows": rows,
            "ground_within": {"5%": reach(0.05), "10%": reach(0.10)},
            "assumes": "a flat road; errors independent; 1 sigma each, combined as a root sum of squares"}


# 1 sigma of what the calibration does not say. The mounting angles and height are the
# extrinsics' (a ground board: 0.1 to 0.3 degrees, 5 to 10 mm with a careful tape measure).
DEFAULT_SIGMA = {"pitch_deg": 0.2, "roll_deg": 0.2, "yaw_deg": 0.2, "height_m": 0.01, "pixel_px": 1.0,
                 "width_px": 2.0, "pitch_motion_deg": 0.5}


def budget_text(b: dict) -> str:
    s, c = b["sigma"], b["camera"]
    out = ["camera     %.2f m above the road, looking %.1f degrees down (yaw %+.1f, roll %+.1f); horizon at row %.0f"
           % (c["height_m"], c["pitch_down_deg"], c["yaw_deg"], c["roll_deg"], b["horizon_row_px"]),
           "1 sigma    focal %.2f%%  cx %.1f  cy %.1f px  |  pitch %.2f  roll %.2f  yaw %.2f degrees  height %.0f mm  |"
           "  detector %.1f px (foot), %.1f px (width)" % (100 * s["f_rel"], s["cx_px"], s["cy_px"], s["pitch_deg"],
                                                           s["roll_deg"], s["yaw_deg"], 1000 * s["height_m"],
                                                           s["pixel_px"], s["width_px"]),
           "           the camera's: %s" % b["calibration_uncertainty_from"], "",
           "           ground-plane ranging                                                  | width of %.1f m      | sideways"
           % b["object_width_m"],
           " range     error          of which: focal  cy     pitch  height pixel | +pitching %.1f | pixels  error         | error"
           % s["pitch_motion_deg"]]
    for r in b["rows"]:
        if "ground" not in r:
            out.append("%5.0f m     behind the camera" % r["range_m"])
            continue
        g, p = r["ground"], r["ground"]["parts_m"]
        f = lambda v: "%6.2f" % v if np.isfinite(v) else "   inf"
        out.append("%5.0f m   %s m %5.1f%%          %s %s %s %s %s |   %s m     | %6.1f %s m %4.1f%% | %5.2f m%s" % (
            r["range_m"], f(g["error_m"]), 100 * g["relative"], f(p["focal"]), f(p["cy"]), f(p["pitch"]),
            f(p["height"]), f(p["pixel"]), f(g["with_pitch_motion_m"]), r["width"]["pixels"],
            f(r["width"]["error_m"]), 100 * r["width"]["relative"], r["lateral"]["error_m"],
            "" if r["in_picture"] else "   (outside the picture)"))
    w = b["ground_within"]
    lim = lambda v: "under 2 m already not" if v is None else "out to %.0f m" % v
    out += ["", "ground-plane ranging stays within 5%% %s, within 10%% %s (calibration and detector; the road flat,"
            % (lim(w["5%"]), lim(w["10%"])), "the vehicle not pitching). inf: the error reaches the horizon."]
    return "\n".join(out)
