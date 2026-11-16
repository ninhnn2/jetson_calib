"""python3 -m unittest discover -s tests -v   (from the project's directory; about a minute)

Everything is checked against something that is known: OpenCV's own functions, or pictures
rendered from a camera whose parameters were chosen here.
"""

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camcal import adas, boards, calibrate, capture, cli, export, extrinsics, models, sources, synth, transform, validate  # noqa: E402
from camcal.export import Calibration  # noqa: E402

SIZE = (1280, 720)
DIST = {"pinhole-radtan4": (-0.10, 0.04, 0.001, -0.0005), "fisheye": (-0.02, 0.005, -0.002, 0.0005)}


def dataset(model, fx, count, seed):
    spec = boards.board_for_paper("A4")
    cam = synth.camera(model, SIZE, fx, fx * 1.005, 650.0, 350.0, DIST[model])
    rng = np.random.default_rng(seed + 1)
    tex = synth._texture(spec)
    frames = [synth.render(spec, cam, r, t, 2, 0.7, 1.5, rng, tex) for r, t in synth.random_poses(spec, cam, count, seed)]
    views = []
    for i, f in enumerate(frames):
        d = boards.detect(spec, f, 8)
        if d is not None:
            views.append(models.View("v%02d" % i, d.ids, d.object_points(spec), d.image))
    return spec, cam, frames, views


class Boards(unittest.TestCase):
    def test_render_is_opencvs_board(self):
        spec = boards.board_for_paper("A4")  # a 25 mm square is 8 cells: 3.2 px per mm is 10 px a cell
        ours = spec.render(3.2)
        theirs = spec.cv_board().generateImage((ours.shape[1], ours.shape[0]), marginSize=0, borderBits=1)
        self.assertEqual(ours.shape, theirs.shape)
        # OpenCV 4.7 and 4.8 draw the markers a pixel off (the half pixel of charuco_offset)
        old = tuple(int(x) for x in cv2.__version__.split(".")[:2]) < (4, 9)
        self.assertGreater(np.mean(ours == theirs), 0.98 if old else 0.9995)

    def test_detector_offset_is_known(self):
        self.assertTrue(np.all(boards.charuco_offset() == 0.0), boards.charuco_offset())

    def test_corners_of_tilted_views(self):
        # the case OpenCV 4.8's own ChArUco step fails (1.2 px rms, 8 in 100 beyond 0.5 px)
        spec = boards.board_for_paper("A4")
        cam = synth.camera("pinhole-radtan4", SIZE, 900.0, 904.5, 650.0, 350.0, DIST["pinhole-radtan4"])
        rng = np.random.default_rng(7)
        tex = synth._texture(spec)
        errors = []
        for r, t in synth.random_poses(spec, cam, 30, 7):
            det = boards.detect(spec, synth.render(spec, cam, r, t, 2, 0.7, 1.5, rng, tex), 8)
            if det is not None:
                errors.append(det.image - models.project(cam.model, cam.K, cam.dist, r, t, det.object_points(spec)))
        e = np.concatenate(errors)
        n = np.linalg.norm(e, axis=1)
        self.assertGreater(len(n), 800)
        self.assertLess(np.abs(e.mean(axis=0)).max(), 0.01)  # no offset
        self.assertLess(np.sqrt((n ** 2).mean()), 0.09)
        self.assertLess(n.max(), 0.5)

    def test_corners_found_where_they_are(self):
        for kind in ("charuco", "chessboard"):
            spec = boards.board_for_paper("A4", kind)
            img = cv2.GaussianBlur(spec.render(4.0, 15.0), (0, 0), 0.8)
            det = boards.detect(spec, img)
            self.assertIsNotNone(det, kind)
            self.assertEqual(len(det), spec.corner_count())
            want = (det.object_points(spec)[:, :2] * 1000.0 + 15.0) * 4.0 - 0.5
            self.assertLess(np.abs(det.image - want).max(), 0.08, kind)

    def test_pdf_prints_at_true_size(self):
        if not shutil.which("pdftoppm"):
            self.skipTest("no pdftoppm")
        spec = boards.board_for_paper("A4")
        with tempfile.TemporaryDirectory() as d:
            boards.write_pdf(spec, os.path.join(d, "b.pdf"), "A4")
            subprocess.run(["pdftoppm", "-r", "254", "-gray", "-png", os.path.join(d, "b.pdf"), os.path.join(d, "b")],
                           check=True)
            page = cv2.imread(os.path.join(d, [n for n in os.listdir(d) if n.endswith(".png")][0]), 0)
        self.assertLessEqual(abs(page.shape[0] - 2970) + abs(page.shape[1] - 2100), 2)  # 10 px per mm
        det = boards.detect(spec, page)
        self.assertEqual(len(det), spec.corner_count())
        x0, y0 = boards._layout(spec, boards.PAPER_MM["A4"])
        want = (det.object_points(spec)[:, :2] * 1000.0 + (x0, y0)) * 10.0 - 0.5
        self.assertLess(np.abs(det.image - want).max(), 0.5)  # 0.05 mm

    def test_sheets_join_into_the_board(self):
        spec = boards.BoardSpec("charuco", 6, 6, boards.tile_square_mm("A3"), 97.5, "DICT_4X4_250")
        self.assertEqual(spec.square_mm, 130.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b.pdf")
            self.assertEqual(boards.write_pdf_tiles(spec, path, "A3"), 6)
            with open(path, "rb") as f:
                pdf = f.read()
        import re
        k = 72.0 / 25.4
        pages = pdf.split(b">>\nstream\n")[1:]
        self.assertEqual(len(pages), 6)
        back = []  # every black rectangle, put back where its sheet belongs
        for i, page in enumerate(pages):
            body = page.split(b"\nf\n")[0]
            ox, oy = (i % 3) * 260.0, (i // 3) * 390.0
            for m in re.finditer(rb"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", body):
                x, y, w, h = (float(v) / k for v in m.groups())
                x, y = x - (297.0 - 260.0) / 2 + ox, (420.0 - y - h) - 6.0 + oy
                self.assertTrue(-1e-3 <= x - ox and x + w - ox <= 260.0 + 1e-3 and y + h - oy <= 390.0 + 1e-3)
                back.append((round(x, 2), round(y, 2), round(w, 2), round(h, 2)))
        want = sorted((round(x, 2), round(y, 2), round(w, 2), round(h, 2)) for x, y, w, h in spec.black_rects())
        self.assertEqual(sorted(back), want)
        with self.assertRaises(ValueError):
            boards.write_pdf_tiles(boards.BoardSpec("charuco", 5, 7, 130.0, 97.5, "DICT_4X4_250"), "x.pdf", "A3")

    def test_small_squares_are_left_out(self):
        spec = boards.board_for_paper("A4")
        img = cv2.GaussianBlur(spec.render(1.0, 15.0), (0, 0), 0.6)  # 25 px a square
        self.assertIsNone(boards.detect(spec, img))
        self.assertIsNotNone(boards.detect(spec, img, min_square_px=0))


class Models(unittest.TestCase):
    def test_projection_is_opencvs(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-0.4, 0.4, (200, 3)) + [0, 0, 1.2]
        rvec, tvec = np.array([0.1, -0.2, 0.05]), np.array([0.02, -0.03, 0.1])
        K = np.array([[900.0, 0, 650], [0, 905, 350], [0, 0, 1]])
        for model, dist in (("pinhole-radtan4", [-0.1, 0.04, 0.001, -0.0005]),
                            ("pinhole-radtan", [-0.1, 0.04, 0.001, -0.0005, 0.01]),
                            ("pinhole-rational", [0.3, -0.1, 0.001, -0.0005, 0.02, 0.5, -0.05, 0.01]),
                            ("fisheye", [-0.02, 0.005, -0.002, 0.0005])):
            ours = models.project(model, K, dist, rvec, tvec, pts)
            if model == "fisheye":
                theirs = cv2.fisheye.projectPoints(pts.reshape(1, -1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1), K,
                                                   np.array(dist))[0]
            else:
                theirs = cv2.projectPoints(pts, rvec, tvec, K, models.cv_dist(model, dist))[0]
            self.assertLess(np.abs(ours - theirs.reshape(-1, 2)).max(), 1e-9, model)
            back = models.distort(model, dist, models.undistort(model, dist, models.distort(model, dist, pts[:, :2])))
            self.assertLess(np.abs(back - models.distort(model, dist, pts[:, :2])).max(), 1e-9, model)


class Poses(unittest.TestCase):
    def test_a_misread_corner_is_left_out(self):
        spec = boards.board_for_paper("A4")
        cam = synth.camera("pinhole-radtan4", SIZE, 900.0, 904.5, 650.0, 350.0, DIST["pinhole-radtan4"])
        rvec, tvec = np.array([0.3, -0.2, 0.05]), np.array([-0.08, -0.12, 0.5])
        obj = spec.corner_points()
        rng = np.random.default_rng(3)
        img = models.project(cam.model, cam.K, cam.dist, rvec, tvec, obj) + rng.normal(0, 0.05, (len(obj), 2))
        img[7] += (3.0, -2.0)
        plain = models.solve_pose(cam.model, cam.K, cam.dist, obj, img)
        r, t, kept = models.solve_pose_robust(cam.model, cam.K, cam.dist, obj, img)
        self.assertEqual(np.flatnonzero(~kept).tolist(), [7])
        self.assertLess(np.abs(t - tvec).max(), 2e-4)
        self.assertLess(np.abs(t - tvec).max(), np.abs(plain[1] - tvec).max())
        # a wrong camera moves every corner: nothing stands out, nothing is left out
        wrong = models.solve_pose_robust(cam.model, cam.K * [[1.05, 1, 1], [1, 1.05, 1], [1, 1, 1]], cam.dist, obj,
                                         img - [[3.0, -2.0]] * (np.arange(len(obj)) == 7)[:, None])
        self.assertLess((~wrong[2]).sum(), 3)


class Solver(unittest.TestCase):
    """The own least squares (fast where OpenCV has no LAPACK) against cv2.calibrateCamera."""

    def test_same_answer_as_opencv(self):
        spec = boards.board_for_paper("A4")
        obj = spec.corner_points()
        size = (1280, 720)
        cam = synth.camera("pinhole-radtan", size, 900.0, 904.0, 650.0, 350.0, (-0.25, 0.1, 0.001, -0.0005, -0.03))
        rng = np.random.default_rng(1)
        views = []
        for i, (r, t) in enumerate(synth.random_poses(spec, cam, 30, 3)):
            p = models.project(cam.model, cam.K, cam.dist, r, t, obj)
            ok = (p[:, 0] > 5) & (p[:, 0] < size[0] - 5) & (p[:, 1] > 5) & (p[:, 1] < size[1] - 5)
            if ok.sum() >= 12 and calibrate._spans_a_plane(obj[ok]):
                views.append(models.View("v%02d" % i, np.flatnonzero(ok), obj[ok],
                                         p[ok] + rng.normal(0, 0.05, (int(ok.sum()), 2))))
        self.assertGreater(len(views), 20)
        grid = np.stack(np.meshgrid(np.linspace(-0.6, 0.6, 9), np.linspace(-0.35, 0.35, 7)), -1).reshape(-1, 2)
        grid = np.c_[grid, np.ones(len(grid))]
        zero = np.zeros(3)
        for model in ("pinhole-radtan4", "pinhole-radtan"):
            K, d, rvecs, tvecs, rms, std = models._fit_pinhole(model, views, size)
            K2, d2, _r, _t, rms2, std2 = models._fit_pinhole_opencv(model, views, size)
            self.assertAlmostEqual(rms, rms2, delta=1e-6)
            # the same camera: the pixels the two give over the picture
            self.assertLess(np.abs(models.project(model, K, d, zero, zero, grid)
                                   - models.project(model, K2, d2, zero, zero, grid)).max(), 2e-3)
            for name in std:  # and the same uncertainty
                self.assertAlmostEqual(std[name] / std2[name], 1.0, delta=0.02, msg=name)
            # the poses it returns are the ones the rms is for
            err = np.concatenate([models.project(model, K, d, rvecs[i], tvecs[i], v.obj) - v.img
                                  for i, v in enumerate(views)])
            self.assertAlmostEqual(float(np.sqrt((err ** 2).sum() / len(err))), rms, delta=1e-9)
        got = models.fit("pinhole-radtan", views, size)
        self.assertLess(abs(got.K[0, 0] - 900.0), 4 * got.std["fx"] + 0.05)
        self.assertLess(np.abs(got.dist - cam.dist).max(), 0.02)


class Monitor(unittest.TestCase):
    def test_flatness_finds_a_bent_sheet(self):
        from camcal import monitor

        spec = boards.board_for_paper("A4")
        obj = spec.corner_points()
        size = (1280, 720)
        cam = synth.camera("pinhole-radtan4", size, 900.0, 900.0, 640.0, 360.0, (-0.2, 0.05, 0.0, 0.0))
        got = {}
        for bump in (0.0, 0.0015):  # the right part of the sheet lifted, up to 1.5 mm at its edge
            true = obj.copy()
            true[:, 2] = -bump * np.clip((obj[:, 0] / obj[:, 0].max() - 0.6) / 0.4, 0, 1) ** 2
            rng = np.random.default_rng(2)
            views = []
            for i, (r, t) in enumerate(synth.random_poses(spec, cam, 30, 5)):
                p = models.project(cam.model, cam.K, cam.dist, r, t, true)
                ok = (p[:, 0] > 5) & (p[:, 0] < size[0] - 5) & (p[:, 1] > 5) & (p[:, 1] < size[1] - 5)
                if ok.sum() >= 12 and calibrate._spans_a_plane(obj[ok]):
                    views.append(models.View("v%02d" % i, np.flatnonzero(ok), obj[ok],
                                             p[ok] + rng.normal(0, 0.05, (int(ok.sum()), 2))))
            res = calibrate.calibrate(spec, views, size, ("pinhole-radtan4",))
            got[bump] = monitor.flatness(res, spec)
        flat, bent = got[0.0], got[0.0015]
        self.assertFalse(monitor.uneven(flat[1]))
        self.assertTrue(monitor.uneven(bent[1]))
        third = flat[0].shape[1] // 3
        for grid, ratio in ((flat[0], (0.85, 1.2)), (bent[0], (1.5, 10.0))):
            r = np.nanmean(grid[:, -third:]) / np.nanmean(grid[:, :third])
            self.assertTrue(ratio[0] < r < ratio[1], r)

    def test_follows_a_session(self):
        from camcal import monitor

        with tempfile.TemporaryDirectory() as d, open(os.devnull, "w") as null:
            s = os.path.join(d, "s")
            out, sys.stdout = sys.stdout, null
            try:
                self.assertEqual(cli.main(["synth", "-o", s, "--count", "10", "--dist", "-0.1", "0.03"]), 0)
                png = os.path.join(d, "monitor.png")
                self.assertEqual(cli.main(["monitor", s, "--snapshot", png]), 0)
            finally:
                sys.stdout = out
            self.assertGreater(os.path.getsize(png), 20000)
            w = monitor.Watch(monitor.Session(s))
            w.step()
            st = w.snapshot()[1]
            self.assertEqual(len(st["names"]), 10)
            self.assertEqual(st["views"] + len(st["no_board"]), 10)
            self.assertIn(st["fit"]["grade"], "ABCD")
            self.assertEqual(len(st["fit"]["views"]) + len(st["fit"]["dropped"]), st["views"])
            os.remove(os.path.join(s, sorted(os.listdir(s))[0]))  # a picture dropped in the capture
            w.step()
            self.assertEqual(len(w.snapshot()[1]["names"]), 9)


def _points_only(spec, cam, count, seed, noise, lift=0.0):
    """Views as corner coordinates (no rendering); lift: the sheet's right part raised, metres."""
    obj = spec.corner_points()
    true = obj.copy()
    true[:, 2] = -lift * np.clip((obj[:, 0] / obj[:, 0].max() - 0.6) / 0.4, 0, 1) ** 2
    rng = np.random.default_rng(seed)
    w, h = cam.size
    views = []
    for i, (r, t) in enumerate(synth.random_poses(spec, cam, count, seed)):
        p = models.project(cam.model, cam.K, cam.dist, r, t, true)
        ok = (p[:, 0] > 5) & (p[:, 0] < w - 5) & (p[:, 1] > 5) & (p[:, 1] < h - 5)
        if ok.sum() >= 12 and calibrate._spans_a_plane(obj[ok]):
            views.append(models.View("v%02d" % i, np.flatnonzero(ok), obj[ok],
                                     p[ok] + rng.normal(0, noise, (int(ok.sum()), 2))))
    return views


class Trust(unittest.TestCase):
    """What stands between a small error on paper and a camera that is right."""

    IMX = ("pinhole-radtan4", (1920, 1080), 1603.3, 1603.4, 1006.7, 473.3, (-0.363, 0.164, 0.00036, 0.00065))

    def test_the_other_pose_of_a_flat_board(self):
        # these 30 views send cv2.calibrateCamera to 0.65 px and a principal point 10 px off
        cam = synth.camera(*self.IMX)
        rng = np.random.default_rng(8)  # the noise of the case as it was found
        obj = boards.board_for_paper("A3").corner_points()
        views = []
        for i, (r, t) in enumerate(synth.random_poses(boards.board_for_paper("A3"), cam, 30, seed=3008)):
            p = models.project(cam.model, cam.K, cam.dist, r, t, obj)
            ok = (p[:, 0] > 5) & (p[:, 0] < 1915) & (p[:, 1] > 5) & (p[:, 1] < 1075)
            if ok.sum() >= 12 and calibrate._spans_a_plane(obj[ok]):
                views.append(models.View("v%d" % i, np.flatnonzero(ok), obj[ok], p[ok] + rng.normal(0, 0.1, (int(ok.sum()), 2))))
        f = models.fit("pinhole-radtan4", views, cam.size)
        self.assertLess(f.rms, 0.16)
        self.assertLess(abs(f.K[0, 2] - 1006.7), 1.5)
        self.assertGreater(models._fit_pinhole_opencv("pinhole-radtan4", views, cam.size)[4], 0.4)

    def test_halves_of_the_board(self):
        cam = synth.camera(*self.IMX)
        spec = boards.board_for_paper("A3")
        flat = _points_only(spec, cam, 24, 5, 0.1)
        f = models.fit(cam.model, flat, cam.size)
        h = calibrate.board_halves(cam.model, flat, cam.size, f)
        self.assertLess(h["fx_relative"], 0.0005)
        bent = _points_only(spec, cam, 24, 5, 0.1, lift=0.003)  # 3 mm at the right edge
        f = models.fit(cam.model, bent, cam.size)
        h = calibrate.board_halves(cam.model, bent, cam.size, f)
        self.assertGreater(h["fx_relative"], 0.003)
        res = calibrate.calibrate(spec, bent, cam.size, ("pinhole-radtan4",))
        m = res.calibration.meta
        self.assertNotEqual(m["grade"], "A")
        self.assertGreater(m["uncertainty"]["fx_relative"], 3 * m["std"]["fx"] / res.calibration.K[0, 0])
        # the truth is inside what the calibration says to carry, which its sigma alone is not
        self.assertLess(abs(res.calibration.K[0, 0] / 1603.3 - 1), 3 * m["uncertainty"]["fx_relative"])

    def test_model_that_turns_back(self):
        K = np.array([[1603.3, 0, 1006.7], [0, 1603.4, 473.3], [0, 0, 1.0]])
        self.assertTrue(models.valid_over_picture("pinhole-radtan4", K, [-0.363, 0.164, 0, 0], (1920, 1080)))
        self.assertFalse(models.valid_over_picture("pinhole-radtan4", K, [-0.9, 0.0, 0, 0], (1920, 1080)))
        self.assertTrue(models.valid_over_picture("fisheye", K * [[0.5], [0.5], [1]] + [[0, 0, 500], [0, 0, 240], [0, 0, 0.0]],
                                                  [0.02, -0.01, 0, 0], (1920, 1080)))

    def test_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            for n, b in (("b.png", b"2"), ("a.png", b"1")):
                with open(os.path.join(d, n), "wb") as f:
                    f.write(b)
            one = calibrate.fingerprint([os.path.join(d, "a.png"), os.path.join(d, "b.png")])
            two = calibrate.fingerprint([os.path.join(d, "b.png"), os.path.join(d, "a.png")])
            self.assertEqual(one, two)
            with open(os.path.join(d, "a.png"), "wb") as f:
                f.write(b"3")
            self.assertNotEqual(one["sha256"], calibrate.fingerprint([os.path.join(d, "a.png"), os.path.join(d, "b.png")])["sha256"])


class Adas(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration("pinhole-radtan4", (1920, 1080), np.array([[1600.0, 0, 960], [0, 1600, 540], [0, 0, 1]]),
                               np.zeros(4), {"uncertainty": {"fx_relative": 0.002, "cx_px": 1.0, "cy_px": 1.0}})
        self.wide = Calibration("pinhole-radtan4", (1920, 1080), np.array([[1603.3, 0, 1006.7], [0, 1603.4, 473.3], [0, 0, 1]]),
                                np.array([-0.363, 0.164, 0.00036, 0.00065]))

    def test_pose_is_what_extrinsics_reports(self):
        R, t = adas.camera_pose(1.3, 7.0, -3.0, 1.5)
        a = extrinsics._angles(R)
        for k, want in (("pitch_down_deg", 7.0), ("yaw_deg", -3.0), ("roll_deg", 1.5)):
            self.assertAlmostEqual(a[k], want, places=9)

    def test_ground_and_back(self):
        R, t = adas.camera_pose(1.3, 7.0, -3.0, 1.5, 1.8, 0.2)
        pts = np.array([[6.0, 1.0], [20.0, -3.0], [60.0, 0.5]])
        px = adas.ground_to_pixel(self.wide, R, t, pts)
        self.assertLess(np.abs(adas.pixel_to_ground(self.wide, R, t, px) - pts).max(), 1e-6)
        self.assertTrue(np.all(np.isnan(adas.pixel_to_ground(self.wide, R, t, np.array([[960.0, 5.0]])))))  # sky

    def test_budget_is_the_textbook_one(self):
        h, d = 1.2, 30.0
        b = adas.budget(self.cal, h, 2.0, ranges_m=(d,), sigma={"pitch_deg": 0.1, "height_m": 0.01})
        g = b["rows"][0]["ground"]["parts_m"]
        self.assertAlmostEqual(g["pitch"], (d * d + h * h) / h * np.radians(0.1), delta=0.03 * g["pitch"])
        self.assertAlmostEqual(g["height"], d * 0.01 / h, delta=0.02 * g["height"])
        self.assertAlmostEqual(g["cy"], (d * d + h * h) / h / 1600.0, delta=0.05 * g["cy"])
        self.assertAlmostEqual(g["pixel"], g["cy"], delta=0.02 * g["cy"])
        w = b["rows"][0]["width"]
        self.assertAlmostEqual(w["pixels"], 1600 * 1.8 / np.hypot(d, h), delta=0.5)
        self.assertAlmostEqual(w["relative"], np.hypot(0.002, 2.0 / w["pixels"]), places=6)
        self.assertGreater(b["ground_within"]["10%"], b["ground_within"]["5%"])
        self.assertIn("ground-plane ranging stays within", adas.budget_text(b))
        # a worse calibration, a worse answer
        worse = adas.budget(self.cal, h, 2.0, ranges_m=(d,), sigma={"f_rel": 0.015, "cx_px": 8, "cy_px": 8})
        self.assertGreater(worse["rows"][0]["ground"]["error_m"], 3 * b["rows"][0]["ground"]["error_m"])

    def test_bev_shows_the_road_to_scale(self):
        R, t = adas.camera_pose(1.4, 12.0)
        mx, my, info = adas.bev_maps(self.wide, R, t, (3.0, 13.0), 4.0, 0.05)
        self.assertEqual(mx.shape, (200, 160))
        r, c = 37, 51  # the pixel that fills this cell is the road point the layout says
        want = adas.ground_to_pixel(self.wide, R, t, np.array([[13.0 - (r + 0.5) * 0.05, 4.0 - (c + 0.5) * 0.05]]))[0]
        self.assertLess(abs(mx[r, c] - want[0]) + abs(my[r, c] - want[1]), 1e-3)

    def test_commands(self):
        with tempfile.TemporaryDirectory() as d, open(os.devnull, "w") as null:
            cal = os.path.join(d, "calibration.json")
            self.cal.save(cal)
            cv2.imwrite(os.path.join(d, "road.png"), np.full((1080, 1920), 90, np.uint8))
            out, sys.stdout = sys.stdout, null
            try:
                self.assertEqual(cli.main(["budget", cal, "--height", "1.2", "--pitch", "3", "-o", os.path.join(d, "b.json")]), 0)
                self.assertEqual(cli.main(["ground", cal, "--height", "1.2", "--pitch", "3", "--pixel", "960,700",
                                           "--point", "10,0"]), 0)
                self.assertEqual(cli.main(["bev", cal, os.path.join(d, "road.png"), "--height", "1.2", "--pitch", "3",
                                           "--res", "0.05", "-o", os.path.join(d, "bev")]), 0)
                self.assertEqual(cli.main(["budget", cal]), 1)  # no mounting given
            finally:
                sys.stdout = out
            with open(os.path.join(d, "b.json")) as f:
                self.assertEqual(len(json.load(f)["rows"]), 9)
            self.assertTrue(os.path.exists(os.path.join(d, "bev", "road_bev.png")))
            self.assertTrue(os.path.exists(os.path.join(d, "bev", "bev_maps.npz")))


class Pinhole(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.cam, cls.frames, cls.views = dataset("pinhole-radtan4", 900.0, 34, 1)
        cls.res = calibrate.calibrate(cls.spec, cls.views, SIZE)
        cls.cal = cls.res.calibration

    def test_finds_the_camera(self):
        cal, std, K = self.cal, self.cal.meta["std"], self.cam.K
        self.assertEqual(cal.model, "pinhole-radtan4")  # the plainest model that explains the lens
        for name, got, want in (("fx", cal.K[0, 0], K[0, 0]), ("fy", cal.K[1, 1], K[1, 1]),
                                ("cx", cal.K[0, 2], K[0, 2]), ("cy", cal.K[1, 2], K[1, 2])):
            self.assertLess(abs(got - want), 4 * std[name], name)  # the stated error is honest
            self.assertLess(abs(got - want), 1.0, name)
        self.assertLess(cal.meta["rms_px"]["held_out"], 0.12)
        self.assertIn(cal.meta["grade"], "AB")

    def test_files(self):
        with tempfile.TemporaryDirectory() as d:
            names = cli.write_all(self.cal, d)
            self.assertIn("kalibr.yaml", names)
            back = Calibration.load(os.path.join(d, "calibration.json"))
            self.assertTrue(np.array_equal(back.K, self.cal.K) and np.array_equal(back.dist, self.cal.dist))
            self.assertEqual(back.meta["grade"], self.cal.meta["grade"])
            fs = cv2.FileStorage(os.path.join(d, "opencv.yaml"), cv2.FILE_STORAGE_READ)
            self.assertLess(np.abs(fs.getNode("camera_matrix").mat() - self.cal.K).max(), 1e-9)
            with open(os.path.join(d, "camera_info.yaml")) as f:
                self.assertIn("distortion_model: plumb_bob", f.read())

    def test_report(self):
        try:
            from camcal import report
        except ImportError:
            self.skipTest("no matplotlib")
        with tempfile.TemporaryDirectory() as d:
            report.write(self.res, self.spec, os.path.join(d, "report.html"))
            with open(os.path.join(d, "report.html"), encoding="utf-8") as f:
                text = f.read()
        self.assertEqual(text.count("data:image/png;base64,"), 6)

    def test_validate_tells_a_moved_lens(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, f in enumerate(self.frames[:8]):
                paths.append(os.path.join(d, "f%d.png" % i))
                cv2.imwrite(paths[-1], f)
            good = validate.validate(self.cal, self.spec, paths)
            self.assertTrue(good["verdict"].startswith("holds"), good["verdict"])
            moved = Calibration(self.cal.model, self.cal.size, self.cal.K, self.cal.dist * [1.3, 1, 1, 1], self.cal.meta)
            self.assertTrue(validate.validate(moved, self.spec, paths)["verdict"].startswith("does NOT"))

    def test_transform_follows_the_picture(self):
        pts = np.array([[0.1, -0.05, 1.0], [-0.2, 0.1, 0.8], [0.3, 0.2, 1.5]])
        w, h = self.cal.size
        p = self.cal.project(np.eye(3), np.zeros(3), pts)
        r90 = transform.rotate(self.cal, 90)  # the scene turns with the picture: (x, y, z) -> (y, -x, z)
        q = r90.project(np.eye(3), np.zeros(3), pts[:, [1, 0, 2]] * [1, -1, 1])
        self.assertLess(np.abs(q - np.stack([p[:, 1], w - 1 - p[:, 0]], axis=1)).max(), 1e-9)
        self.assertEqual(r90.size, (h, w))
        c = transform.apply(self.cal, ["crop=160,90,960,540", "resize=1920x1080"])
        q = c.project(np.eye(3), np.zeros(3), pts)
        self.assertLess(np.abs(q - ((p - [160, 90] + 0.5) * 2 - 0.5)).max(), 1e-9)
        # and on real pixels: a picture turned by OpenCV fits the turned calibration
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, f in enumerate(self.frames[:5]):
                paths.append(os.path.join(d, "f%d.png" % i))
                cv2.imwrite(paths[-1], cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE))
            self.assertLess(validate.validate(r90, self.spec, paths)["rms_px"], 0.12)

    def test_undistorted_picture_is_a_pinhole(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "f.png")
            cv2.imwrite(src, self.frames[3])
            validate.undistort_files(self.cal, [src], os.path.join(d, "out"), save_maps=True)
            rect = Calibration.load(os.path.join(d, "out", "rectified.json"))
            self.assertTrue(os.path.exists(os.path.join(d, "out", "rectify_maps.npz")))
            v = validate.validate(rect, self.spec, [os.path.join(d, "out", "f_rect.png")])
        self.assertEqual(v["views_used"], 1)
        self.assertLess(v["rms_px"], 0.15)
        self.assertEqual(rect.K[0, 0], rect.K[1, 1])  # square pixels: the picture is not stretched

    def test_undistorted_pixels_are_square_and_seen(self):
        # the A3 session's IMX219: OpenCV alone gives fx 1382, fy 1532 at alpha 0
        cal = Calibration("pinhole-radtan4", (1920, 1080), np.array([[1603.3, 0, 1006.7], [0, 1603.4, 473.3], [0, 0, 1]]),
                          np.array([-0.363, 0.164, 0.00036, 0.00065]))
        for alpha in (0.0, 0.5, 1.0):
            K, mx, my = cal.rectify_maps(alpha)
            self.assertEqual(K[0, 0], K[1, 1])
            seen = (mx >= 0) & (mx <= 1919) & (my >= 0) & (my <= 1079)
            if alpha == 0.0:  # every output pixel comes from the picture
                self.assertGreater(seen.mean(), 0.999)
            if alpha == 1.0:  # and every pixel of the picture is in the output
                for u, v in ((0, 0), (1919, 0), (0, 1079), (1919, 1079), (960, 0), (0, 540)):
                    x = cal.normalize(np.array([[u, v]], dtype=np.float64))[0]
                    p = K @ np.array([x[0], x[1], 1.0])
                    self.assertTrue(-1 <= p[0] <= 1920 and -1 <= p[1] <= 1080, (alpha, u, v, p))

    def test_capture_takes_still_new_views_once(self):
        with tempfile.TemporaryDirectory() as d:
            s = capture.Session(self.spec, d, target=5, log=lambda t: None)
            frame = next(f for f in self.frames if boards.detect(self.spec, f, 12) is not None)
            got = [s.consider(frame, now=10.0 + 0.1 * i) for i in range(4)]
            self.assertEqual([bool(g["taken"]) for g in got], [False, False, True, False])
            self.assertEqual(got[1]["text"], "hold still")
            self.assertTrue(got[3]["text"].startswith("have this one"))
            moving = np.roll(frame, 40, axis=1)  # 40 px in 0.1 s: not still, though new
            self.assertFalse(s.consider(moving, now=11.0)["taken"])
            self.assertFalse(s.consider(np.roll(frame, 80, axis=1), now=11.1)["taken"])
            with open(os.path.join(d, "session.json")) as f:
                self.assertEqual(json.load(f)["pictures"], 1)
            again = capture.Session(self.spec, d, log=lambda t: None)  # resumed: knows what it has
            self.assertEqual(again.names, ["img_0001.png"])
            self.assertFalse(any(again.consider(frame, now=20.0 + 0.1 * i)["taken"] for i in range(4)))
            self.assertEqual(s.draw(frame, got[2]).shape[:2], (720, 1280))
            self.assertEqual(s.drop_last(), "img_0001.png")
            self.assertEqual(os.listdir(d), ["session.json"])


class Fisheye(unittest.TestCase):
    def test_model_is_chosen_by_held_out_views(self):
        spec, cam, _frames, views = dataset("fisheye", 520.0, 34, 2)
        cal = calibrate.calibrate(spec, views, SIZE).calibration
        self.assertEqual(cal.model, "fisheye")
        self.assertLess(abs(cal.K[0, 0] - cam.K[0, 0]), 1.0)
        self.assertLess(abs(cal.K[0, 2] - cam.K[0, 2]), 1.0)
        self.assertLess(cal.meta["rms_px"]["held_out"], 0.15)
        self.assertEqual(cal.meta["std"]["method"].split(",")[0], "bootstrap over the views")


class Extrinsics(unittest.TestCase):
    def test_camera_in_the_vehicle(self):
        spec = boards.board_for_paper("A1", squares=cli.GROUND_SQUARES)
        cam = synth.camera("pinhole-radtan4", (1920, 1080), 1400, 1400, 950, 545, [-0.25, 0.08, 0.0005, -0.0003])
        cal = Calibration(cam.model, cam.size, cam.K, cam.dist)
        yaw, pitch, roll, pos = 4.0, 18.0, -2.0, np.array([1.5, 0.1, 1.2])
        rz = cv2.Rodrigues(np.array([0, 0, np.radians(yaw)]))[0]
        ry = cv2.Rodrigues(np.array([0, np.radians(pitch), 0.0]))[0]
        rr = cv2.Rodrigues(np.array([0, 0, np.radians(-roll)]))[0]
        R_vc = rz @ ry @ np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]]) @ rr
        R_vb, t_vb = extrinsics.board_to_vehicle((3.6, -0.2), 25.0, 0.005)
        R_cb, t_cb = R_vc.T @ R_vb, R_vc.T @ (t_vb - pos)
        img = synth.render(spec, cam, cv2.Rodrigues(R_cb)[0].ravel(), t_cb)
        det = boards.detect(spec, img, min_square_px=0)
        ext = extrinsics.solve(cal, spec, [det], (3.6, -0.2), 25.0, 0.005, trials=40)
        self.assertLess(np.abs(ext["t_vehicle_camera_m"] - pos).max(), 0.015)
        for name, want in (("yaw_deg", yaw), ("pitch_down_deg", pitch), ("roll_deg", roll)):
            self.assertLess(abs(ext[name] - want), 0.3, name)
        # a point on the ground lands where the camera sees it
        g = np.array([[6.0, 1.0, 0.0]])
        want = cal.project(R_vc.T, -R_vc.T @ pos, g)
        self.assertLess(np.abs(extrinsics.vehicle_to_pixels(cal, ext, g) - want).max(), 3.0)
        self.assertEqual(extrinsics.draw_ground_grid(cal, ext, img).shape, (1080, 1920, 3))


class RawStream(unittest.TestCase):
    @staticmethod
    def message(header, payload=b""):
        h = json.dumps(header).encode()
        return b"UCRS" + struct.pack(">II", len(h), len(payload)) + h + payload

    def test_frames_from_ucpeer_detect(self):
        a, b = socket.socketpair()  # no port is opened
        y = np.arange(48 * 32, dtype=np.uint32).reshape(32, 48).astype(np.uint8)
        nv12 = y.tobytes() + bytes(48 * 16)
        b.sendall(self.message({"type": "hello", "source": "csi:0", "format": "NV12", "fps": 30}))
        b.sendall(self.message({"type": "status", "fps": 29.7}))
        b.sendall(self.message({"type": "frame", "seq": 7, "width": 48, "height": 32, "format": "NV12"}, nv12))
        src = sources._Serve(a, {"source": "serve://test"}).start()
        frame, _ = src.read(timeout=5.0)
        self.assertTrue(np.array_equal(frame, y))
        self.assertEqual(src.info["server"]["source"], "csi:0")
        b.close()
        with self.assertRaises(RuntimeError):
            src.read(timeout=5.0)
        src.close()

    def test_other_bytes_are_refused(self):
        a, b = socket.socketpair()
        b.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
        with self.assertRaises(ConnectionError):
            sources.read_message(a)
        a.close(), b.close()

    def test_csi_pipeline(self):
        p = sources.csi_pipeline(0, 1920, 1080, 30, 180, 2)
        self.assertIn("sensor-id=0 sensor-mode=2 !", p)
        self.assertIn("width=1920,height=1080,framerate=30/1", p)
        self.assertIn("flip-method=2", p)


class CommandLine(unittest.TestCase):
    def test_board_synth_calibrate(self):
        with tempfile.TemporaryDirectory() as d, open(os.devnull, "w") as null:
            out, sys.stdout = sys.stdout, null
            try:
                self.assertEqual(cli.main(["board", "-o", d]), 0)
                self.assertEqual(cli.main(["board", "-o", d, "--ground"]), 0)
                s = os.path.join(d, "s")
                self.assertEqual(cli.main(["synth", "-o", s, "--count", "12", "--dist", "-0.1", "0.03"]), 0)
                self.assertEqual(cli.main(["calibrate", s, "--model", "slam", "--no-report"]), 0)
                r = os.path.join(s, "result")
                self.assertEqual(cli.main(["transform", os.path.join(r, "calibration.json"), "rotate=180",
                                           "-o", os.path.join(d, "t")]), 0)
                self.assertEqual(cli.main(["validate", os.path.join(r, "calibration.json"), s]), 0)
            finally:
                sys.stdout = out
            self.assertTrue(os.path.exists(os.path.join(d, "board_charuco_A4_7x10_25mm.pdf")))
            self.assertTrue(os.path.exists(os.path.join(d, "board_ground_A1_5x7_114mm.json")))
            a = Calibration.load(os.path.join(r, "calibration.json"))
            b = Calibration.load(os.path.join(d, "t", "calibration.json"))
            self.assertAlmostEqual(b.K[0, 2], a.size[0] - 1 - a.K[0, 2], places=9)
            self.assertEqual(b.meta["derived"], ["rotate=90", "rotate=90"])


if __name__ == "__main__":
    unittest.main()
