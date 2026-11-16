"""Where the camera is on the vehicle, from one board lying flat on the ground.

The vehicle's frame is ISO 8855: X forward, Y left, Z up, the origin on the ground (under the
middle of the rear axle is usual; any point will do as long as the board is measured from it).
Lay the ChArUco board face up on level ground in the camera's view and measure:

  origin   X, Y of the board's own origin, the pattern's outer corner next to marker 0
           (the top left of the page as printed)
  yaw      the direction of the board's x axis (along squares_x), in degrees counterclockwise
           from the vehicle's X seen from above. Page top towards the vehicle's left and the
           board's x pointing forward: 0.
  z        the height of the printed surface above the ground (the backing's thickness)

The result is the camera's pose in the vehicle (rotation, position, height, pitch, yaw, roll)
and the ground's homography. It is as good as the tape measure: 5 mm and 0.3 degrees on the
board are 5 mm and 0.3 degrees in the camera. A larger board, further out, or the average
over several placements does better; for L4 this is a starting value for a surveyed or
target-room calibration, not a replacement for one.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import boards, models
from .boards import BoardSpec
from .export import Calibration


def board_to_vehicle(origin_xy: Sequence[float], yaw_deg: float, z: float = 0.0):
    """(R, t): vehicle point = R board point + t, for a board face up on the ground."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, s, 0.0],  # columns: the board's x, y and z (into the ground) in the vehicle
                  [s, -c, 0.0],
                  [0.0, 0.0, -1.0]])
    return R, np.array([origin_xy[0], origin_xy[1], z], dtype=np.float64)


def _angles(R_vc: np.ndarray) -> dict:
    """The camera's attitude from its axes in the vehicle: where it looks and how it is rolled."""
    right, down, ahead = R_vc[:, 0], R_vc[:, 1], R_vc[:, 2]
    return {
        # the optical axis: left of straight ahead is positive, below the horizon is positive
        "yaw_deg": float(np.degrees(np.arctan2(ahead[1], ahead[0]))),
        "pitch_down_deg": float(np.degrees(-np.arcsin(np.clip(ahead[2], -1, 1)))),
        # about the optical axis: positive when the picture's right side is the higher one
        "roll_deg": float(np.degrees(np.arctan2(right[2], -down[2]))),
    }


def _camera_in_vehicle(rvec, tvec, R_vb, t_vb):
    R_cb = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    R_vc = R_vb @ R_cb.T
    return R_vc, t_vb - R_vc @ np.asarray(tvec, dtype=np.float64).reshape(3)


def solve(cal: Calibration, spec: BoardSpec, detections: Sequence[boards.Detection],
          origin_xy: Sequence[float], yaw_deg: float, z: float = 0.0, trials: int = 200,
          seed: int = 0) -> dict:
    """The camera in the vehicle from the board's corners in one or more pictures of the same
    scene (their corners are used together)."""
    if spec.type != "charuco":
        raise ValueError("the extrinsics need a ChArUco board: a chessboard cannot tell which way round it lies")
    obj = np.concatenate([d.object_points(spec) for d in detections])
    img = np.concatenate([d.image for d in detections])
    pose = models.solve_pose_robust(cal.model, cal.K, cal.dist, obj, img)
    if pose is None:
        raise ValueError("no pose explains the board's corners")
    left_out = int((~pose[2]).sum())
    obj, img = obj[pose[2]], img[pose[2]]  # without the misread corners
    R_vb, t_vb = board_to_vehicle(origin_xy, yaw_deg, z)
    R_vc, t_vc = _camera_in_vehicle(pose[0], pose[1], R_vb, t_vb)
    res = cal.project(pose[0], pose[1], obj) - img
    rms = models.rms([res])
    out = {
        "frame": "ISO 8855 vehicle frame: X forward, Y left, Z up; camera frame: x right, y down, z forward",
        "R_vehicle_camera": R_vc,
        "t_vehicle_camera_m": t_vc,
        "T_vehicle_camera": np.vstack([np.hstack([R_vc, t_vc.reshape(3, 1)]), [0, 0, 0, 1.0]]),
        "height_m": float(t_vc[2]),
        "rms_px": rms,
        "corners": int(len(obj)),
        "corners_left_out": left_out,
        "pictures": len(detections),
        "board_distance_m": float(np.linalg.norm(pose[1])),
        "board": {"origin_xy_m": [float(origin_xy[0]), float(origin_xy[1])], "yaw_deg": float(yaw_deg),
                  "z_m": float(z)},
    }
    out.update(_angles(R_vc))
    # what the corners' noise does to the answer: the same solve on corners moved by it
    rng = np.random.default_rng(seed)
    sigma = max(rms, 0.05) / np.sqrt(2.0)
    rows = []
    for _ in range(trials):
        r, t = models._refine_pose_pixels(cal.model, cal.K, cal.dist, obj, img + rng.normal(0, sigma, img.shape),
                                          pose[0].copy(), pose[1].copy())
        Rv, tv = _camera_in_vehicle(r, t, R_vb, t_vb)
        a = _angles(Rv)
        rows.append([tv[0], tv[1], tv[2], a["yaw_deg"], a["pitch_down_deg"], a["roll_deg"]])
    sd = np.std(np.array(rows), axis=0, ddof=1) if rows else np.full(6, np.nan)
    out["std_from_corner_noise"] = dict(zip(("x_m", "y_m", "z_m", "yaw_deg", "pitch_deg", "roll_deg"),
                                            [float(v) for v in sd]))
    out["note"] = ("std_from_corner_noise is the picture's share only; add the tape measure's: the board's "
                   "position error moves the camera by as much, its yaw error turns the camera's yaw by as much")
    # ground (X, Y, 1) in metres -> pixel of the undistorted pinhole picture with this K
    R_cv, t_cv = R_vc.T, -R_vc.T @ t_vc
    out["ground_to_undistorted_pixel"] = cal.K @ np.column_stack([R_cv[:, 0], R_cv[:, 1], t_cv])
    return out


def vehicle_to_pixels(cal: Calibration, ext: dict, points_vehicle: np.ndarray) -> np.ndarray:
    R_vc = np.asarray(ext["R_vehicle_camera"])
    t_vc = np.asarray(ext["t_vehicle_camera_m"])
    return cal.project(R_vc.T, -R_vc.T @ t_vc, points_vehicle)


def draw_ground_grid(cal: Calibration, ext: dict, image: np.ndarray, reach_m: float = 20.0,
                     step_m: float = 1.0) -> np.ndarray:
    """The vehicle frame's grid on the ground drawn into the picture: lines of constant X and
    of constant Y every step_m, every fifth stronger, the X axis (Y = 0) green. Where it does
    not lie along the ground's own lines (a tape laid out, lane markings, tiles), the
    extrinsics are off."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    w, h = cal.size
    R_vc = np.asarray(ext["R_vehicle_camera"])
    t_vc = np.asarray(ext["t_vehicle_camera_m"])
    lo = np.floor((t_vc[:2] - reach_m) / step_m) * step_m
    hi = np.ceil((t_vc[:2] + reach_m) / step_m) * step_m
    along = np.arange(0.0, 2 * reach_m + step_m, 0.2)

    def line(pts, color, thick):
        pc = (pts - t_vc) @ R_vc  # into the camera's frame
        px = cal.project(np.eye(3), np.zeros(3), pc)
        ok = (pc[:, 2] > 0.3) & np.all(np.isfinite(px), axis=1)
        ok &= (px[:, 0] > -w) & (px[:, 0] < 2 * w) & (px[:, 1] > -h) & (px[:, 1] < 2 * h)
        for i in range(len(px) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(out, (int(round(px[i, 0])), int(round(px[i, 1]))),
                         (int(round(px[i + 1, 0])), int(round(px[i + 1, 1]))), color, thick, cv2.LINE_AA)

    blue, amber, green = (214, 120, 42), (25, 178, 250), (12, 163, 12)
    for axis in (0, 1):  # lines of constant X, then of constant Y
        for v in np.arange(lo[axis], hi[axis] + 1e-6, step_m):
            pts = np.zeros((len(along), 3))
            pts[:, axis] = v
            pts[:, 1 - axis] = lo[1 - axis] + along
            fifth = abs(v / (5 * step_m) - round(v / (5 * step_m))) < 1e-6
            color = green if (axis == 1 and abs(v) < 1e-6) else amber if fifth else blue
            line(pts, color, 2 if fifth else 1)
            if axis == 0 and fifth:
                p = vehicle_to_pixels(cal, ext, np.array([[v, 0.0, 0.0]]))[0]
                front = ((np.array([v, 0.0, 0.0]) - t_vc) @ R_vc)[2] > 0.3
                if front and np.all(np.isfinite(p)) and 0 <= p[0] < w - 80 and 20 <= p[1] < h:
                    cv2.putText(out, "X %g m" % v, (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.putText(out, "X %g m" % v, (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
