"""Camera models: a point in front of the camera to a pixel, back again, and the fit of one to
views of a board.

  pinhole-radtan4    k1 k2 p1 p2          what most SLAM and VIO systems take (Kalibr radtan)
  pinhole-radtan     k1 k2 p1 p2 k3       OpenCV's and ROS's default (plumb_bob)
  pinhole-rational   k1 k2 p1 p2 k3 k4 k5 k6   wide lenses a polynomial does not follow
  fisheye            k1 k2 k3 k4          Kannala-Brandt (OpenCV fisheye, Kalibr equidistant)

The camera frame is OpenCV's: x right, y down, z out along the optical axis. A pixel's centre is
at whole coordinates. distort() and undistort() are written here in NumPy, apart from OpenCV's:
they are what the synthetic pictures are rendered with, so a test of the calibration is not
OpenCV agreeing with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

MODELS = ("pinhole-radtan4", "pinhole-radtan", "pinhole-rational", "fisheye")
# fewest parameters first: the order a tie on held-out error is broken in
SIMPLICITY = {"pinhole-radtan4": 0, "fisheye": 1, "pinhole-radtan": 2, "pinhole-rational": 3}
DIST_NAMES = {
    "pinhole-radtan4": ("k1", "k2", "p1", "p2"),
    "pinhole-radtan": ("k1", "k2", "p1", "p2", "k3"),
    "pinhole-rational": ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    "fisheye": ("k1", "k2", "k3", "k4"),
}


def is_fisheye(model: str) -> bool:
    return model == "fisheye"


def check_model(model: str) -> str:
    if model not in MODELS:
        raise ValueError("no camera model '%s' (%s)" % (model, ", ".join(MODELS)))
    return model


def cv_dist(model: str, dist: Sequence[float]) -> np.ndarray:
    """The coefficients as OpenCV's functions take them for this model."""
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    n = len(DIST_NAMES[model])
    if len(d) < n:
        d = np.concatenate([d, np.zeros(n - len(d))])
    d = d[:n]
    if model == "pinhole-radtan4":
        return np.concatenate([d, [0.0]])
    return d


# --- the models, in NumPy ---------------------------------------------------------------


def distort(model: str, dist: Sequence[float], xn: np.ndarray) -> np.ndarray:
    """Ideal normalised coordinates (x/z, y/z) to the distorted ones: (..., 2) -> (..., 2)."""
    d = cv_dist(model, dist)
    x, y = xn[..., 0], xn[..., 1]
    r2 = x * x + y * y
    if is_fisheye(model):
        r = np.sqrt(r2)
        th = np.arctan(r)
        th2 = th * th
        thd = th * (1 + th2 * (d[0] + th2 * (d[1] + th2 * (d[2] + th2 * d[3]))))
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        return np.stack([x * s, y * s], axis=-1)
    k = np.zeros(8)
    k[: len(d)] = d
    k1, k2, p1, p2, k3, k4, k5, k6 = k
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def undistort(model: str, dist: Sequence[float], xd: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Distorted normalised coordinates back to the ideal ones. NaN where the model has no
    answer (a fisheye ray at 90 degrees or beyond, a point the iteration walks away from)."""
    d = cv_dist(model, dist)
    xd = np.asarray(xd, dtype=np.float64)
    if is_fisheye(model):
        thd = np.sqrt(xd[..., 0] ** 2 + xd[..., 1] ** 2)
        th = thd.copy()
        for _ in range(iterations):  # Newton on theta (1 + k1 theta^2 + ...) = theta_d
            th2 = th * th
            f = th * (1 + th2 * (d[0] + th2 * (d[1] + th2 * (d[2] + th2 * d[3])))) - thd
            df = 1 + th2 * (3 * d[0] + th2 * (5 * d[1] + th2 * (7 * d[2] + th2 * 9 * d[3])))
            th = th - f / np.where(np.abs(df) > 1e-9, df, 1e-9)
        bad = (th >= np.pi / 2 - 1e-3) | (th < 0)
        s = np.where(thd > 1e-12, np.tan(np.clip(th, 0, np.pi / 2 - 1e-3)) / np.maximum(thd, 1e-12), 1.0)
        out = np.stack([xd[..., 0] * s, xd[..., 1] * s], axis=-1)
        out[bad] = np.nan
        return out
    # Newton in two dimensions, the Jacobian by differences: a fixed point iteration walks
    # away where the distortion is strong, which is where a wide lens is measured
    x = xd.copy()
    h = 1e-6
    with np.errstate(all="ignore"):
        for _ in range(iterations):
            f = distort(model, d, x) - xd
            af = np.abs(f[np.isfinite(f)])  # (NumPy 1.21 has no nanmax(initial=))
            if af.size == 0 or af.max() < 1e-13:
                break
            ex = x.copy()
            ex[..., 0] += h
            ey = x.copy()
            ey[..., 1] += h
            jx = (distort(model, d, ex) - xd - f) / h  # d f / d x
            jy = (distort(model, d, ey) - xd - f) / h  # d f / d y
            det = jx[..., 0] * jy[..., 1] - jy[..., 0] * jx[..., 1]
            det = np.where(np.abs(det) > 1e-12, det, np.nan)
            sx = (jy[..., 1] * f[..., 0] - jy[..., 0] * f[..., 1]) / det
            sy = (jx[..., 0] * f[..., 1] - jx[..., 1] * f[..., 0]) / det
            # no further in one step than the point is from the axis, plus a little
            lim = 0.5 + np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
            n = np.sqrt(sx * sx + sy * sy)
            k = np.where(n > lim, lim / n, 1.0)
            x = x - np.stack([sx * k, sy * k], axis=-1)
        bad = ~(np.abs(distort(model, d, x) - xd).max(axis=-1) < 1e-7)
    x[bad] = np.nan
    return x


def project(model: str, K: np.ndarray, dist: Sequence[float], R: np.ndarray, t: np.ndarray,
            points: np.ndarray) -> np.ndarray:
    """Points in the board's (or the world's) frame to pixels. R is a matrix or a Rodrigues
    vector. NaN for a point behind the camera."""
    R = np.asarray(R, dtype=np.float64)
    if R.size == 3:
        R = cv2.Rodrigues(R.reshape(3, 1))[0]
    pc = np.asarray(points, dtype=np.float64) @ R.T + np.asarray(t, dtype=np.float64).reshape(1, 3)
    z = pc[:, 2:3]
    xn = pc[:, :2] / np.where(z > 1e-9, z, np.nan)
    xd = distort(model, dist, xn)
    return np.stack([K[0, 0] * xd[:, 0] + K[0, 2], K[1, 1] * xd[:, 1] + K[1, 2]], axis=-1)


def normalize(model: str, K: np.ndarray, dist: Sequence[float], pixels: np.ndarray) -> np.ndarray:
    """Pixels to ideal normalised coordinates: the ray (x, y, 1) each one looks along."""
    p = np.asarray(pixels, dtype=np.float64)
    xd = np.stack([(p[..., 0] - K[0, 2]) / K[0, 0], (p[..., 1] - K[1, 2]) / K[1, 1]], axis=-1)
    return undistort(model, dist, xd)


def valid_over_picture(model: str, K: np.ndarray, dist: Sequence[float], size: Tuple[int, int]) -> bool:
    """Does the model hold out to the picture's corners (and 5% beyond)? A polynomial fitted
    where there were corners can turn back further out: the picture then folds over itself,
    two directions land on one pixel, and undistortion there is wrong without any sign of it."""
    w, h = size
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [(w - 1) / 2, 0], [(w - 1) / 2, h - 1],
                        [0, (h - 1) / 2], [w - 1, (h - 1) / 2]], dtype=np.float64)
    rays = normalize(model, K, dist, corners)
    if not np.all(np.isfinite(rays)):
        return False
    t = np.linspace(0.0, 1.05, 64)[:, None, None]
    xd = distort(model, dist, rays[None, :, :] * t)
    r = np.sqrt((xd ** 2).sum(axis=-1))
    return bool(np.all(np.isfinite(r)) and np.all(np.diff(r, axis=0) > 0))


# --- the fit ----------------------------------------------------------------------------


@dataclass
class View:
    """One picture's corners: where they are on the board and where in the picture."""

    name: str
    ids: np.ndarray  # (N,)
    obj: np.ndarray  # (N, 3) metres, the board's frame
    img: np.ndarray  # (N, 2) pixels

    def __len__(self) -> int:
        return len(self.ids)

    def keep(self, mask: np.ndarray) -> "View":
        return View(self.name, self.ids[mask], self.obj[mask], self.img[mask])


@dataclass
class Fit:
    model: str
    size: Tuple[int, int]  # width, height
    K: np.ndarray
    dist: np.ndarray  # as DIST_NAMES[model]
    rms: float
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    std: Optional[dict] = None  # one standard deviation per parameter, where the solver gives it

    def residuals(self, views: Sequence[View]) -> List[np.ndarray]:
        return [project(self.model, self.K, self.dist, r, t, v.obj) - v.img
                for v, r, t in zip(views, self.rvecs, self.tvecs)]


_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 300, 1e-10)


def _pinhole_flags(model: str) -> int:
    if model == "pinhole-radtan4":
        return cv2.CALIB_FIX_K3
    if model == "pinhole-rational":
        return cv2.CALIB_RATIONAL_MODEL
    return 0


def _fisheye_flag(name: str) -> int:
    # OpenCV 4 keeps the fisheye flags in cv2.fisheye, OpenCV 5 with all the others
    ns = cv2.fisheye if hasattr(cv2.fisheye, "CALIB_" + name) else cv2
    return int(getattr(ns, "CALIB_" + name))


def _first_guess(views: Sequence[View], size: Tuple[int, int]) -> np.ndarray:
    """A camera matrix to start from: the focal length the boards' homographies agree on, the
    principal point at the picture's centre."""
    try:
        K = cv2.initCameraMatrix2D([v.obj.reshape(-1, 1, 3).astype(np.float32) for v in views],
                                   [v.img.reshape(-1, 1, 2).astype(np.float32) for v in views], tuple(size))
        K = np.asarray(K, dtype=np.float64)
        if np.all(np.isfinite(K)) and 0.2 * size[0] < K[0, 0] < 20 * size[0] and 0.2 * size[0] < K[1, 1] < 20 * size[0]:
            return K
    except cv2.error:
        pass
    f = float(max(size))
    return np.array([[f, 0, (size[0] - 1) / 2.0], [0, f, (size[1] - 1) / 2.0], [0, 0, 1.0]])


def _fit_pinhole(model: str, views: Sequence[View], size: Tuple[int, int], guess: Optional["Fit"] = None,
                 iterations: int = 200):
    """Least squares on the pixels over the camera and every view's pose, by Levenberg-Marquardt.

    The same problem and the same answer as cv2.calibrateCamera (the tests hold the two
    together), solved with its structure: a view's pose touches only that view's corners, so
    the poses are eliminated six at a time (Schur complement) and what is left to solve is as
    small as the camera, 8 to 12 unknowns. OpenCV solves the whole system at once, 6 unknowns
    a view; built without LAPACK, as JetPack's is, that takes 9 to 18 s for 45 views on an
    Orin Nano, and a calibration fits some 35 times. This takes a fraction of a second.

    Returns (K, dist, rvecs, tvecs, rms, std): std is one standard deviation of each camera
    parameter, from the corners' scatter about the fit.
    """
    n = len(DIST_NAMES[model])
    k = 4 + n
    obj = [np.ascontiguousarray(v.obj, dtype=np.float64).reshape(-1, 1, 3) for v in views]
    img = [np.asarray(v.img, dtype=np.float64).reshape(-1, 2) for v in views]
    if guess is not None:
        K0, d0 = guess.K.copy(), np.asarray(guess.dist, dtype=np.float64)[:n].copy()
        if len(d0) < n:
            d0 = np.concatenate([d0, np.zeros(n - len(d0))])
    else:
        K0, d0 = _first_guess(views, size), np.zeros(n)
    theta = np.concatenate([[K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]], d0])
    poses = np.empty((len(views), 6))
    for i, v in enumerate(views):
        pose = solve_pose(model, K0, d0, v.obj, v.img)
        if pose is None:
            raise ValueError("%s: no pose explains the view %s" % (model, v.name))
        poses[i, :3], poses[i, 3:] = pose

    def system(theta, poses, jac):
        K = np.array([[theta[0], 0, theta[2]], [0, theta[1], theta[3]], [0, 0, 1.0]])
        cost, parts = 0.0, []
        for i in range(len(obj)):
            p, J = cv2.projectPoints(obj[i], poses[i, :3], poses[i, 3:], K, theta[4:])
            r = (p.reshape(-1, 2) - img[i]).reshape(-1)
            cost += float(r @ r)
            if jac:
                Jp, Jc = J[:, :6], J[:, 6:6 + k]  # pose | fx fy cx cy, the coefficients
                parts.append((Jc.T @ Jc, Jc.T @ Jp, Jp.T @ Jp, Jc.T @ r, Jp.T @ r))
        return cost, parts

    def reduced(parts, lam):
        """The camera's equations with the poses eliminated: S, g, and what gives each pose back."""
        S, g, back = np.zeros((k, k)), np.zeros(k), []
        for A, B, D, a, d in parts:
            S += A
            g += a
            Di = np.linalg.inv(D + lam * np.diag(np.diag(D)) if lam else D)
            BD = B @ Di
            S -= BD @ B.T
            g -= BD @ d
            back.append((Di, B, d))
        return S, g, back

    def descend(theta, poses):
        lam = 1e-3
        cost, parts = system(theta, poses, True)
        if not np.isfinite(cost):
            raise ValueError("%s: the fit did not start" % model)
        small = 0
        for _ in range(iterations):
            try:
                S, g, back = reduced(parts, lam)
                S = S + lam * np.diag(sum(np.diag(p[0]) for p in parts))  # damped as the poses are
                scale = 1.0 / np.sqrt(np.maximum(np.diag(S), 1e-300))
                dth = -scale * np.linalg.solve(S * scale[:, None] * scale[None, :], g * scale)
                dp = np.array([-Di @ (d + B.T @ dth) for Di, B, d in back])
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e12)
                continue
            new_cost, new_parts = system(theta + dth, poses + dp, True)
            if np.isfinite(new_cost) and new_cost <= cost and theta[0] + dth[0] > 0 and theta[1] + dth[1] > 0:
                gain = (cost - new_cost) / max(cost, 1e-300)
                theta, poses, cost, parts = theta + dth, poses + dp, new_cost, new_parts
                lam = max(lam / 5.0, 1e-12)
                small = small + 1 if gain < 1e-13 else 0
                if small >= 2:
                    break
            else:
                if lam >= 1e12:
                    break
                lam *= 10.0
        return theta, poses, cost, parts

    def view_rms(K, d, r, t, i):
        e = project(model, K, d, r, t, views[i].obj) - views[i].img
        return float(np.sqrt((e * e).sum() / len(e))) if np.all(np.isfinite(e)) else np.inf

    with np.errstate(all="ignore"):
        theta, poses, cost, parts = descend(theta, poses)
        # A flat board seen nearly head on has two poses that look almost alike (tilted one way
        # or the other). The first guess of the camera can pick the wrong one for a view, and
        # least squares then bends the camera to it and stays there: 0.6 px where 0.14 was
        # to be had, the principal point 20 px off (OpenCV's calibrateCamera ends there too).
        # So: with the camera found, each view's other pose is tried, and the fit continued.
        for _ in range(3):
            K = np.array([[theta[0], 0, theta[2]], [0, theta[1], theta[3]], [0, 0, 1.0]])
            swapped = False
            for i, v in enumerate(views):
                now = view_rms(K, theta[4:], poses[i, :3], poses[i, 3:], i)
                for r, t in _both_poses(model, K, theta[4:], v.obj, v.img):
                    if view_rms(K, theta[4:], r, t, i) < 2.0 * now:  # worth the least squares
                        r, t = _refine_pose_pixels(model, K, theta[4:], v.obj, v.img, r, t)
                        e = view_rms(K, theta[4:], r, t, i)
                        if e < 0.9 * now and np.linalg.norm(r - poses[i, :3]) > 1e-3:
                            poses[i, :3], poses[i, 3:], now, swapped = r, t, e, True
            if not swapped:
                break
            theta, poses, cost, parts = descend(theta, poses)
        try:
            S, _g, _back = reduced(parts, 0.0)
            scale = 1.0 / np.sqrt(np.maximum(np.diag(S), 1e-300))
            cov = np.linalg.inv(S * scale[:, None] * scale[None, :]) * scale[:, None] * scale[None, :]
            total = sum(len(x) for x in img)
            dof = 2 * total - k - 6 * len(views)
            sd = np.sqrt(np.maximum(np.diag(cov), 0.0) * cost / dof) if dof > 0 else np.full(k, np.nan)
        except np.linalg.LinAlgError:
            sd = np.full(k, np.nan)
    total = sum(len(x) for x in img)
    K = np.array([[theta[0], 0, theta[2]], [0, theta[1], theta[3]], [0, 0, 1.0]])
    std = dict(zip(("fx", "fy", "cx", "cy") + DIST_NAMES[model], [float(x) for x in sd]))
    return K, theta[4:].copy(), [p[:3].copy() for p in poses], [p[3:].copy() for p in poses], \
        float(np.sqrt(cost / total)), std


def _fit_pinhole_opencv(model: str, views: Sequence[View], size: Tuple[int, int]):
    """cv2.calibrateCamera on the same views: what _fit_pinhole is held against in the tests."""
    obj = [v.obj.reshape(-1, 1, 3).astype(np.float32) for v in views]
    img = [v.img.reshape(-1, 1, 2).astype(np.float32) for v in views]
    rms, K, D, rvecs, tvecs, sd, _se, _pv = cv2.calibrateCameraExtended(
        obj, img, tuple(size), None, None, flags=_pinhole_flags(model), criteria=_CRITERIA)
    sd = np.asarray(sd).reshape(-1)
    names = ("fx", "fy", "cx", "cy") + DIST_NAMES["pinhole-rational"]
    std = {name: float(sd[i]) for i, name in enumerate(names) if name in ("fx", "fy", "cx", "cy") + DIST_NAMES[model]}
    return (np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64).reshape(-1)[:len(DIST_NAMES[model])],
            [np.asarray(r).reshape(3) for r in rvecs], [np.asarray(t).reshape(3) for t in tvecs], float(rms), std)


def fit(model: str, views: Sequence[View], size: Tuple[int, int], guess: Optional[Fit] = None) -> Fit:
    """The camera of this model that best explains the views (least squares on the pixels).
    Raises ValueError where the views do not determine one."""
    check_model(model)
    if len(views) < 3:
        raise ValueError("a calibration needs at least 3 views, has %d" % len(views))
    n = len(DIST_NAMES[model])
    try:
        if is_fisheye(model):
            # one row of N points: the only layout OpenCV 4 and 5 both take here
            obj = [v.obj.reshape(1, -1, 3).astype(np.float64) for v in views]
            img = [v.img.reshape(1, -1, 2).astype(np.float64) for v in views]
            flags = _fisheye_flag("RECOMPUTE_EXTRINSIC") | _fisheye_flag("FIX_SKEW")
            if guess is not None:
                K0 = guess.K.copy()
            else:  # 90 degrees across the picture: enough for the first poses
                f = size[0] / 2.0
                K0 = np.array([[f, 0, (size[0] - 1) / 2.0], [0, f, (size[1] - 1) / 2.0], [0, 0, 1.0]])
            flags |= _fisheye_flag("USE_INTRINSIC_GUESS")
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj, img, tuple(size), K0, np.zeros((4, 1)), flags=flags, criteria=_CRITERIA)
            std = None
        else:
            K, D, rvecs, tvecs, rms, std = _fit_pinhole(model, views, size, guess)
    except cv2.error as e:
        raise ValueError("%s: the views do not determine it (%s)" % (model, str(e).strip().splitlines()[-1]))
    D = np.asarray(D, dtype=np.float64).reshape(-1)[:n]
    K = np.asarray(K, dtype=np.float64)
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("%s: the fit did not converge" % model)
    return Fit(model, (int(size[0]), int(size[1])), K, D, float(rms),
               [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs],
               [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs], std)


def _both_poses(model: str, K: np.ndarray, dist: Sequence[float], obj: np.ndarray, img: np.ndarray):
    """The two poses a flat target can have for these corners (IPPE), unrefined."""
    xn = normalize(model, K, dist, img)
    ok = np.all(np.isfinite(xn), axis=1)
    if ok.sum() < 4:
        return []
    try:
        n, rvecs, tvecs, _e = cv2.solvePnPGeneric(obj[ok].astype(np.float64).reshape(-1, 1, 3), xn[ok].reshape(-1, 1, 2),
                                                  np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    return [(np.asarray(r, dtype=np.float64).reshape(3), np.asarray(t, dtype=np.float64).reshape(3))
            for r, t in zip(rvecs, tvecs) if np.all(np.isfinite(r)) and np.all(np.isfinite(t))]


def solve_pose(model: str, K: np.ndarray, dist: Sequence[float], obj: np.ndarray, img: np.ndarray
               ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The board's pose in the camera from one view, the camera given: (rvec, tvec) such that
    camera point = R board point + t; None where it cannot be found."""
    xn = normalize(model, K, dist, img)
    ok = np.all(np.isfinite(xn), axis=1)
    if ok.sum() < 4:
        return None
    sv = np.linalg.svd(obj[ok][:, :2] - obj[ok][:, :2].mean(axis=0), compute_uv=False)
    if sv[-1] < 0.02 * sv[0]:  # the corners in one line: a pose about that line is free
        return None
    o = obj[ok].astype(np.float64).reshape(-1, 1, 3)
    p = xn[ok].reshape(-1, 1, 2)
    eye = np.eye(3)
    try:
        # a planar target: the ambiguity-aware solver, then least squares on all the points
        done, rvec, tvec = cv2.solvePnP(o, p, eye, None, flags=cv2.SOLVEPNP_IPPE)
        if not done:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(o, p, eye, None, rvec, tvec, criteria=_CRITERIA)
    except cv2.error:
        return None
    # the last steps on the pixels themselves, where the noise is: normalised coordinates
    # stretch it unevenly across a distorted picture
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None
    rvec, tvec = _refine_pose_pixels(model, K, dist, obj[ok], img[ok], rvec.reshape(3), tvec.reshape(3))
    return rvec, tvec


def solve_pose_robust(model: str, K: np.ndarray, dist: Sequence[float], obj: np.ndarray, img: np.ndarray,
                      rounds: int = 3):
    """solve_pose, then again without the corners that stand out from the rest: further than
    4 times the median distance and 0.3 px (the rule the calibration cleans its views by).
    A misread corner is the detector's fault, not the camera's; a camera that is wrong moves
    all the corners, the median with them, and nothing is left out. (rvec, tvec, kept) or None."""
    keep = np.ones(len(obj), dtype=bool)
    pose = solve_pose(model, K, dist, obj, img)
    for _ in range(rounds):
        if pose is None:
            return None
        n = np.linalg.norm(project(model, K, dist, pose[0], pose[1], obj) - img, axis=1)
        n = np.where(np.isfinite(n), n, np.inf)
        bad = keep & (n > max(4.0 * float(np.median(n[keep])), 0.3))
        if not bad.any() or (keep & ~bad).sum() < 6:
            break
        keep &= ~bad
        pose = solve_pose(model, K, dist, obj[keep], img[keep])
    return None if pose is None else (pose[0], pose[1], keep)


def _refine_pose_pixels(model, K, dist, obj, img, rvec, tvec, iterations: int = 8):
    """Gauss-Newton on the six pose parameters against the pixels (numeric Jacobian)."""
    x = np.concatenate([rvec, tvec]).astype(np.float64)

    def res(v):
        return (project(model, K, dist, v[:3], v[3:], obj) - img).reshape(-1)

    r = res(x)
    if not np.all(np.isfinite(r)):
        return rvec, tvec
    for _ in range(iterations):
        J = np.empty((len(r), 6))
        for i in range(6):
            h = 1e-6 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            J[:, i] = (res(xp) - r) / h
        if not np.all(np.isfinite(J)):
            break
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        xn = x + step
        rn = res(xn)
        if not np.all(np.isfinite(rn)) or rn @ rn > r @ r:
            break
        x, r = xn, rn
        if np.abs(step).max() < 1e-10:
            break
    return x[:3], x[3:]


def rms(residuals: Sequence[np.ndarray]) -> float:
    """Root of the mean squared distance in pixels over every point: OpenCV's 'RMS'."""
    r = np.concatenate([np.asarray(x).reshape(-1, 2) for x in residuals]) if len(residuals) else np.zeros((0, 2))
    r = r[np.all(np.isfinite(r), axis=1)]
    return float(np.sqrt((r * r).sum(axis=1).mean())) if len(r) else float("nan")


def fov_degrees(model: str, K: np.ndarray, dist: Sequence[float], size: Tuple[int, int]) -> Tuple[float, float, float]:
    """Horizontal, vertical and diagonal field of view of the whole picture."""
    w, h = size
    cx, cy = K[0, 2], K[1, 2]

    def angle(p, q):
        a = normalize(model, K, dist, np.array([p, q], dtype=np.float64))
        if not np.all(np.isfinite(a)):
            return float("nan")
        u = np.append(a[0], 1.0)
        v = np.append(a[1], 1.0)
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))

    return (angle((0, cy), (w - 1, cy)), angle((cx, 0), (cx, h - 1)), angle((0, 0), (w - 1, h - 1)))
