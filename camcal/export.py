"""A calibration as a file: camcal's own JSON, and the forms other software reads.

  calibration.json   everything: the camera, how sure, what it was measured from
  opencv.yaml        cv::FileStorage: camera_matrix, distortion_coefficients
  camera_info.yaml   ROS sensor_msgs/CameraInfo, as camera_calibration writes it
  kalibr.yaml        a Kalibr camchain (pinhole-radtan4 and fisheye only: what Kalibr, VINS,
                     OpenVINS and basalt name radtan and equidistant)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from . import models

FORMAT = "camcal-1"


@dataclass
class Calibration:
    model: str
    size: Tuple[int, int]  # width, height
    K: np.ndarray
    dist: np.ndarray  # as models.DIST_NAMES[model]
    meta: dict = field(default_factory=dict)  # rms, std, quality, grade, board, source, ...

    def __post_init__(self):
        models.check_model(self.model)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        self.size = (int(self.size[0]), int(self.size[1]))

    @property
    def cv_dist(self) -> np.ndarray:
        return models.cv_dist(self.model, self.dist)

    def to_dict(self) -> dict:
        d = {
            "format": FORMAT,
            "model": self.model,
            "image_size": list(self.size),
            "intrinsics": {"fx": self.K[0, 0], "fy": self.K[1, 1], "cx": self.K[0, 2], "cy": self.K[1, 2]},
            "distortion": dict(zip(models.DIST_NAMES[self.model], [float(x) for x in self.dist])),
            "conventions": "OpenCV: x right, y down, z forward; a pixel's centre at whole coordinates",
        }
        d.update(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        if d.get("format") != FORMAT:
            raise ValueError("not a camcal calibration (format %r)" % d.get("format"))
        i = d["intrinsics"]
        K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
        dist = [d["distortion"][n] for n in models.DIST_NAMES[models.check_model(d["model"])]]
        meta = {k: v for k, v in d.items()
                if k not in ("format", "model", "image_size", "intrinsics", "distortion", "conventions")}
        return cls(d["model"], tuple(d["image_size"]), K, np.array(dist), meta)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_plain)
            f.write("\n")

    # --- what an application does with it ----------------------------------------------
    def project(self, R, t, points: np.ndarray) -> np.ndarray:
        return models.project(self.model, self.K, self.dist, R, t, points)

    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        return models.normalize(self.model, self.K, self.dist, pixels)

    def rectified_matrix(self, alpha: float = 0.0, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """The pinhole camera of the undistorted picture. alpha 0: only pixels the lens saw
        (the picture cropped to them); 1: all of the original picture, black where it had none."""
        size = size or self.size
        if models.is_fisheye(self.model):
            K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.dist.reshape(4, 1), self.size, np.eye(3), balance=float(alpha), new_size=size)
        else:
            K = cv2.getOptimalNewCameraMatrix(self.K, self.cv_dist, self.size, float(alpha), size)[0]
        return _square_pixels(np.asarray(K, dtype=np.float64), size, alpha)

    def rectify_maps(self, alpha: float = 0.0, size: Optional[Tuple[int, int]] = None):
        """(new K, map x, map y): cv2.remap(picture, map x, map y, INTER_LINEAR) undistorts."""
        size = size or self.size
        newK = self.rectified_matrix(alpha, size)
        if models.is_fisheye(self.model):
            mx, my = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.dist.reshape(4, 1), np.eye(3), newK, size, cv2.CV_32FC1)
        else:
            mx, my = cv2.initUndistortRectifyMap(self.K, self.cv_dist, np.eye(3), newK, size, cv2.CV_32FC1)
        return newK, mx, my



def _square_pixels(K: np.ndarray, size: Tuple[int, int], alpha: float) -> np.ndarray:
    """OpenCV fits the undistorted picture to the output separately across and down, so its
    pixels come out oblong (fx 1382, fy 1532 for the IMX219 at alpha 0): the picture is
    stretched, and a detector sees every object out of shape. One focal length for both: the
    larger at alpha 0 (only pixels the lens saw: crop a little more one way), the smaller at
    alpha 1 (all of them: a little more black one way), the centre of the view kept."""
    fx, fy = K[0, 0], K[1, 1]
    f = (1.0 - alpha) * max(fx, fy) + alpha * min(fx, fy)
    cx0, cy0 = (size[0] - 1) / 2.0, (size[1] - 1) / 2.0
    xc, yc = (cx0 - K[0, 2]) / fx, (cy0 - K[1, 2]) / fy  # the direction at the output's centre
    return np.array([[f, 0.0, cx0 - xc * f], [0.0, f, cy0 - yc * f], [0.0, 0.0, 1.0]])

def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o).__name__)


def _row(values) -> str:
    return "[" + ", ".join(repr(float(v)) for v in np.asarray(values).reshape(-1)) + "]"


def write_opencv_yaml(cal: Calibration, path: str) -> None:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", cal.size[0])
    fs.write("image_height", cal.size[1])
    fs.write("camera_model", cal.model)
    fs.write("camera_matrix", cal.K)
    # cv::fisheye functions for the fisheye model, the ordinary ones for the others
    fs.write("distortion_coefficients", cal.cv_dist.reshape(1, -1))
    fs.release()


ROS_MODEL = {"pinhole-radtan4": "plumb_bob", "pinhole-radtan": "plumb_bob",
             "pinhole-rational": "rational_polynomial", "fisheye": "equidistant"}


def write_ros_yaml(cal: Calibration, path: str, camera_name: str = "camera") -> None:
    P = np.hstack([cal.rectified_matrix(0.0), np.zeros((3, 1))])
    with open(path, "w") as f:
        f.write("image_width: %d\nimage_height: %d\ncamera_name: %s\n" % (cal.size[0], cal.size[1], camera_name))
        f.write("camera_matrix:\n  rows: 3\n  cols: 3\n  data: %s\n" % _row(cal.K))
        f.write("distortion_model: %s\n" % ROS_MODEL[cal.model])
        d = cal.cv_dist
        f.write("distortion_coefficients:\n  rows: 1\n  cols: %d\n  data: %s\n" % (len(d), _row(d)))
        f.write("rectification_matrix:\n  rows: 3\n  cols: 3\n  data: %s\n" % _row(np.eye(3)))
        f.write("projection_matrix:\n  rows: 3\n  cols: 4\n  data: %s\n" % _row(P))


def write_kalibr_yaml(cal: Calibration, path: str, rostopic: str = "/camera/image_raw") -> bool:
    """False (and no file) for a model Kalibr has no name for."""
    name = {"pinhole-radtan4": "radtan", "fisheye": "equidistant"}.get(cal.model)
    if name is None:
        return False
    with open(path, "w") as f:
        f.write("cam0:\n  camera_model: pinhole\n")
        f.write("  intrinsics: %s\n" % _row([cal.K[0, 0], cal.K[1, 1], cal.K[0, 2], cal.K[1, 2]]))
        f.write("  distortion_model: %s\n  distortion_coeffs: %s\n" % (name, _row(cal.dist)))
        f.write("  resolution: [%d, %d]\n  rostopic: %s\n" % (cal.size[0], cal.size[1], rostopic))
    return True
