"""The command line: python3 -m camcal <command> ... (-h on each command says what it takes)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import __doc__ as ABOUT
from . import __version__, boards, calibrate, export, models
from .boards import BoardSpec
from .export import Calibration

GROUND_SQUARES = (5, 7)  # the board for the extrinsics: squares large enough to read lying on the ground


def _size(text: str) -> Tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("'%s' is not WIDTHxHEIGHT" % text)


def _pair(text: str) -> Tuple[float, float]:
    try:
        x, y = text.split(",")
        return float(x), float(y)
    except ValueError:
        raise argparse.ArgumentTypeError("'%s' is not X,Y" % text)


def _board(path: Optional[str], inputs=(), cal: Optional[Calibration] = None, square_mm: Optional[float] = None) -> BoardSpec:
    """The board: the file given, else the one the session or the calibration was made with."""
    if path:
        spec = BoardSpec.load(path)
    else:
        d = calibrate.load_session(inputs).get("board") or (cal.meta.get("board") if cal else None)
        if not d:
            raise ValueError("which board? --board board.json (written by `camcal board`)")
        spec = BoardSpec.from_dict(d)
    return spec.with_square_mm(square_mm) if square_mm else spec


def write_all(cal: Calibration, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cal.save(os.path.join(out_dir, "calibration.json"))
    export.write_opencv_yaml(cal, os.path.join(out_dir, "opencv.yaml"))
    export.write_ros_yaml(cal, os.path.join(out_dir, "camera_info.yaml"))
    names = ["calibration.json", "opencv.yaml", "camera_info.yaml"]
    kalibr = os.path.join(out_dir, "kalibr.yaml")
    if export.write_kalibr_yaml(cal, kalibr):
        names.append("kalibr.yaml")
    elif os.path.exists(kalibr):
        os.remove(kalibr)  # one of another model, from a run before
    return names


# --- the commands -------------------------------------------------------------------------


def cmd_board(a) -> int:
    os.makedirs(a.output, exist_ok=True)
    if a.screen:
        if not a.ppi:
            raise ValueError("--screen needs --ppi, the display's pixels per inch")
        spec, square_px = boards.board_for_screen(a.screen[0], a.screen[1], a.ppi, a.type, a.dictionary)
        img = np.full((a.screen[1], a.screen[0]), 255, np.uint8)
        pat = spec.render(square_px / spec.square_mm)
        y, x = (img.shape[0] - pat.shape[0]) // 2, (img.shape[1] - pat.shape[1]) // 2
        img[y:y + pat.shape[0], x:x + pat.shape[1]] = pat
        base = os.path.join(a.output, "board_screen_%dx%d" % a.screen)
        cv2.imwrite(base + ".png", img)
        spec.save(base + ".json")
        print("%s\n%s.png: show it pixel for pixel (100%%, no scaling), full brightness\n%s.json"
              % (spec.describe(), base, base))
        return 0
    if a.sheets:
        paper = (a.paper or "A3").upper()
        s = a.square_mm or boards.tile_square_mm(paper)
        spec = BoardSpec("charuco", a.sheets[0] * boards.TILE_SQUARES[0], a.sheets[1] * boards.TILE_SQUARES[1], s,
                         a.marker_mm or BoardSpec.default_marker_mm(s, a.dictionary), a.dictionary,
                         "ground board for the extrinsics, printed in sheets")
        base = os.path.join(a.output, "board_ground_%dx%d_sheets_%s_%dx%d_%gmm" % (
            a.sheets[0], a.sheets[1], paper, spec.squares_x, spec.squares_y, s))
        n = boards.write_pdf_tiles(spec, base + ".pdf", paper)
        spec.save(base + ".json")
        w, h = spec.size_mm
        print(spec.describe())
        print("%s.pdf   %d sheets of %s: print at 100%%, cut each on its outline, join them edge to edge on\n"
              "         something flat and rigid (%d across, %d down; each sheet says its row and column).\n"
              "         Joined: %.0f x %.0f mm" % (base, n, paper, a.sheets[0], a.sheets[1], w, h))
        print("%s.json  for --board; measure a square after printing: extrinsics --square-mm" % base)
        return 0
    paper = (a.paper or ("A1" if a.ground else "A4")).upper()
    squares = a.squares or (GROUND_SQUARES if a.ground else None)
    spec = boards.board_for_paper(paper, a.type, squares, a.square_mm, a.marker_mm, a.dictionary)
    if a.ground:
        spec.note = "ground board for the extrinsics"
    base = os.path.join(a.output, "board_%s_%s_%dx%d_%gmm" % (
        "ground" if a.ground else spec.type, paper, spec.squares_x, spec.squares_y, spec.square_mm))
    boards.write_pdf(spec, base + ".pdf", paper)
    boards.write_svg(spec, base + ".svg", paper)
    spec.save(base + ".json")
    print(spec.describe())
    print("%s.pdf   print at 100%% (no 'fit to page'), matt paper, glued flat on something rigid" % base)
    print("%s.json  the board for --board; after printing measure a square (the rule under the\n"
          "         pattern is 100 mm) and give what you measure: calibrate --square-mm" % base)
    return 0


def cmd_capture(a) -> int:
    from . import capture, sources

    spec = _board(a.board, [a.output])
    src = sources.open_source(a.source, a.size, a.fps, a.rotate, a.sensor_mode, a.focus)
    try:
        frame, _ = src.read(timeout=15.0)
        src.info["image_size"] = [frame.shape[1], frame.shape[0]]
        print("%s: %dx%d   board: %s" % (a.source, frame.shape[1], frame.shape[0], spec.describe()))
        headless = a.headless or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
        if headless and not a.headless:
            print("no display: headless. See it with: python3 -m camcal preview USER@HOST:%s" % a.output)
        s = capture.Session(spec, a.output, a.target, a.min_corners, a.steady, source=src.info)
        s.auto = not a.manual
        n = capture.run(s, src, headless, a.duration)
    finally:
        src.close()
    if n < 15:  # under 15 views a calibration is a guess
        print("%d pictures in %s: too few to calibrate (20 to 30 good ones make a calibration). "
              "Run capture again on the same directory to add to them" % (n, a.output))
    else:
        print("%d pictures in %s. Next: python3 -m camcal calibrate %s" % (n, a.output, a.output))
    return 0


def cmd_preview(a) -> int:
    from . import capture

    opts = (["-i", a.identity, "-o", "IdentitiesOnly=yes"] if a.identity else [])
    capture.preview(a.target, opts)
    return 0


def cmd_monitor(a) -> int:
    from . import monitor

    return monitor.main(a)


def cmd_calibrate(a) -> int:
    spec = _board(a.board, a.inputs, square_mm=a.square_mm)
    paths = calibrate.list_images(a.inputs)
    if not paths:
        raise ValueError("no pictures in %s" % ", ".join(a.inputs))
    print("board: %s\n%d pictures" % (spec.describe(), len(paths)))
    views, size, skipped = calibrate.detect_files(spec, paths, a.min_corners, print)
    for name, why in skipped:
        print("  %s: %s" % (name, why))
    session = calibrate.load_session(a.inputs)
    res = calibrate.calibrate(spec, views, size, calibrate.parse_models(a.model), bootstrap=a.bootstrap,
                              min_corners=a.min_corners, views_given=len(paths), source=session.get("source"),
                              log=print)
    cal = res.calibration
    cal.meta["dataset"] = calibrate.fingerprint(paths)
    out = a.output or os.path.join(a.inputs[0] if os.path.isdir(a.inputs[0]) else os.path.dirname(a.inputs[0]) or ".",
                                   "result")
    names = write_all(cal, out)
    if not a.no_report:
        try:
            from . import report

            report.write(res, spec, os.path.join(out, "report.html"))
            names.append("report.html")
        except ImportError as e:
            print("no report.html: %s (the report's charts need matplotlib)" % e)
    print()
    print(calibrate.summary(cal))
    print("\n%s: %s" % (out, ", ".join(names)))
    return 0


def cmd_validate(a) -> int:
    from . import validate

    cal = Calibration.load(a.calibration)
    spec = _board(a.board, a.inputs, cal, a.square_mm)
    paths = calibrate.list_images(a.inputs)
    out = validate.validate(cal, spec, paths, overlay_dir=a.overlay, log=print)
    for v in out["views"]:
        if "why_not" in v:
            print("  %-24s %s" % (v["name"], v["why_not"]))
    if out["views_used"]:
        print("%d views, %d corners: %.3f px rms; half within %.3f px, 95%% within %.3f, worst %.3f"
              % (out["views_used"], out["corners"], out["rms_px"], out["p50_px"], out["p95_px"], out["max_px"]))
    print(out["verdict"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=export._plain)
    return 0 if out["verdict"].startswith("holds") else 1


def cmd_undistort(a) -> int:
    from . import validate

    cal = Calibration.load(a.calibration)
    out = validate.undistort_files(cal, calibrate.list_images(a.inputs), a.output, a.alpha, a.size, a.maps)
    K = out["new_K"]
    print("%d pictures into %s, %dx%d: a pinhole camera fx %.2f fy %.2f cx %.2f cy %.2f (rectified.json%s)"
          % (len(out["pictures"]), a.output, out["size"][0], out["size"][1], K[0, 0], K[1, 1], K[0, 2], K[1, 2],
             ", rectify_maps.npz" if a.maps else ""))
    return 0


def cmd_extrinsics(a) -> int:
    from . import extrinsics

    cal = Calibration.load(a.calibration)
    spec = BoardSpec.load(a.board)
    if a.square_mm:
        spec = spec.with_square_mm(a.square_mm)
    dets, pictures = [], []
    for p in calibrate.list_images(a.inputs):
        img = cv2.imread(p)
        if img is None or (img.shape[1], img.shape[0]) != cal.size:
            print("  %s: not a %dx%d picture" % (os.path.basename(p), cal.size[0], cal.size[1]))
            continue
        # a board on the ground is seen at a shallow angle: its far squares are few pixels deep
        det = boards.detect(spec, img, 6, min_square_px=0)
        print("  %s: %s" % (os.path.basename(p), "board not found" if det is None else "%d corners" % len(det)))
        if det is not None:
            dets.append(det), pictures.append((p, img))
    if not dets:
        raise ValueError("the board was found in no picture: larger squares (camcal board --ground --paper A0), "
                         "nearer the camera, or more light")
    ext = extrinsics.solve(cal, spec, dets, a.origin, a.yaw, a.board_z)
    out = a.output or os.path.dirname(os.path.abspath(a.calibration))
    os.makedirs(out, exist_ok=True)
    ext["calibration"] = os.path.abspath(a.calibration)
    with open(os.path.join(out, "extrinsics.json"), "w") as f:
        json.dump(ext, f, indent=2, default=export._plain)
        f.write("\n")
    p, img = pictures[0]
    grid = os.path.join(out, os.path.splitext(os.path.basename(p))[0] + "_grid.jpg")
    cv2.imwrite(grid, extrinsics.draw_ground_grid(cal, ext, img))
    t, sd = ext["t_vehicle_camera_m"], ext["std_from_corner_noise"]
    print("camera in the vehicle (X forward, Y left, Z up), from %d corners, %.3f px rms:" % (ext["corners"], ext["rms_px"]))
    print("  position  X %+.3f  Y %+.3f  Z %+.3f m   (± %.3f %.3f %.3f from the corners' noise)"
          % (t[0], t[1], t[2], sd["x_m"], sd["y_m"], sd["z_m"]))
    print("  looks     yaw %+.2f (left +)  pitch %+.2f down  roll %+.2f degrees   (± %.2f %.2f %.2f)"
          % (ext["yaw_deg"], ext["pitch_down_deg"], ext["roll_deg"], sd["yaw_deg"], sd["pitch_deg"], sd["roll_deg"]))
    print("  the tape measure's error comes on top: yaw and X, Y are as good as the board was laid out;\n"
          "  height, pitch and roll do not depend on it")
    print("%s/extrinsics.json, %s: the grid must lie along the ground" % (out, grid))
    return 0


def _mounting(a, cal):
    """(R, t, extrinsics or None) from --extrinsics, or from --height and --pitch."""
    from . import adas

    if a.extrinsics:
        ext = adas.load_extrinsics(a.extrinsics)
        return adas.pose_of(ext) + (ext,)
    if a.height is None:
        raise ValueError("where is the camera? --extrinsics extrinsics.json, or --height and --pitch")
    return adas.camera_pose(a.height, a.pitch, a.yaw, a.roll) + (None,)


def cmd_budget(a) -> int:
    from . import adas

    cal = Calibration.load(a.calibration)
    R, t, ext = _mounting(a, cal)
    sigma = {"f_rel": None if a.sigma_f is None else a.sigma_f / 100.0, "cx_px": a.sigma_c, "cy_px": a.sigma_c,
             "pitch_deg": a.sigma_pitch, "roll_deg": a.sigma_roll, "yaw_deg": a.sigma_yaw,
             "height_m": a.sigma_height, "pixel_px": a.sigma_pixel, "pitch_motion_deg": a.pitching}
    if ext is not None:
        from . import extrinsics

        ang = extrinsics._angles(R)
        height, pitch, yaw, roll = float(t[2]), ang["pitch_down_deg"], ang["yaw_deg"], ang["roll_deg"]
        sd = ext.get("std_from_corner_noise") or {}
        # the corners' share only; the tape measure's (--sigma-...) is the larger part as a rule
        for k, name in (("pitch_deg", "pitch_deg"), ("roll_deg", "roll_deg"), ("yaw_deg", "yaw_deg"), ("height_m", "z_m")):
            if sigma[k] is None and name in sd:
                sigma[k] = max(float(sd[name]), adas.DEFAULT_SIGMA[k])
    else:
        height, pitch, yaw, roll = a.height, a.pitch, a.yaw, a.roll
    b = adas.budget(cal, height, pitch, yaw, roll, a.ranges or (5, 10, 15, 20, 30, 40, 50, 75, 100), sigma,
                    a.width, a.lateral)
    b["calibration"] = os.path.abspath(a.calibration)
    print(adas.budget_text(b))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(b, f, indent=2, default=export._plain)
            f.write("\n")
        print("\n%s" % a.output)
    return 0


def cmd_ground(a) -> int:
    from . import adas

    cal = Calibration.load(a.calibration)
    R, t, _ext = _mounting(a, cal)
    if not a.pixel and not a.point:
        raise ValueError("--pixel U,V and/or --point X,Y")
    for u, v in a.pixel or []:
        g = adas.pixel_to_ground(cal, R, t, np.array([[u, v]]))[0]
        print("pixel %.1f, %.1f  ->  %s" % (u, v, "above the horizon" if not np.all(np.isfinite(g)) else
                                            "X %+.3f m forward, Y %+.3f m left (%.3f m along the road from under the camera)"
                                            % (g[0], g[1], float(np.hypot(g[0] - t[0], g[1] - t[1])))))
    for x, y in a.point or []:
        p = adas.ground_to_pixel(cal, R, t, np.array([[x, y]]))[0]
        inside = np.all(np.isfinite(p)) and 0 <= p[0] <= cal.size[0] - 1 and 0 <= p[1] <= cal.size[1] - 1
        print("X %+.3f, Y %+.3f m  ->  %s" % (x, y, "behind the camera" if not np.all(np.isfinite(p)) else
                                              "pixel %.1f, %.1f%s" % (p[0], p[1], "" if inside else " (outside the picture)")))
    return 0


def cmd_bev(a) -> int:
    from . import adas

    cal = Calibration.load(a.calibration)
    R, t, _ext = _mounting(a, cal)
    mx, my, info = adas.bev_maps(cal, R, t, a.ahead, a.side, a.res)
    os.makedirs(a.output, exist_ok=True)
    np.savez_compressed(os.path.join(a.output, "bev_maps.npz"), map_x=mx, map_y=my,
                        x_m=info["x_m"], y_m=info["y_m"], metres_per_pixel=info["metres_per_pixel"])
    with open(os.path.join(a.output, "bev.json"), "w") as f:
        json.dump(dict(info, calibration=os.path.abspath(a.calibration)), f, indent=2)
        f.write("\n")
    done = 0
    for p in calibrate.list_images(a.inputs):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None or (img.shape[1], img.shape[0]) != cal.size:
            continue
        cv2.imwrite(os.path.join(a.output, os.path.splitext(os.path.basename(p))[0] + "_bev.png"),
                    cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        done += 1
    print("%s: bev_maps.npz (cv2.remap), bev.json, %d pictures; %dx%d, %g m a pixel, X %g to %g m, Y %+g to %+g m"
          % (a.output, done, info["size"][0], info["size"][1], a.res, a.ahead[0], a.ahead[1], a.side, -a.side))
    return 0


def cmd_transform(a) -> int:
    from . import transform

    cal = transform.apply(Calibration.load(a.calibration), a.ops)
    names = write_all(cal, a.output)
    print("%s, %dx%d: fx %.3f fy %.3f cx %.3f cy %.3f" % (
        cal.model, cal.size[0], cal.size[1], cal.K[0, 0], cal.K[1, 1], cal.K[0, 2], cal.K[1, 2]))
    print("%s: %s" % (a.output, ", ".join(names)))
    return 0


def cmd_synth(a) -> int:
    from . import synth

    spec = BoardSpec.load(a.board) if a.board else boards.board_for_paper(a.paper)
    w, h = a.size
    cam = synth.camera(a.model, a.size, a.fx or 0.7 * w, cx=w / 2 + 0.008 * w, cy=h / 2 - 0.01 * h, dist=a.dist or ())
    paths = synth.write_dataset(a.output, spec, cam, a.count, a.seed, a.noise)
    print("%d rendered pictures of %s in %s; the camera that made them is in truth.json"
          % (len(paths), spec.describe(), a.output))
    return 0


# --- the parser ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m camcal", description=ABOUT,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", metavar="command")

    def add(name, fn, text):
        s = sub.add_parser(name, help=text, description=text)
        s.set_defaults(fn=fn)
        return s

    s = add("board", cmd_board, "The calibration board as a PDF at its true size, with the JSON that describes it.")
    s.add_argument("-o", "--output", default=".", help="directory (default: here)")
    s.add_argument("--paper", help="A4 (default), A3, A2, A1, A0, LETTER")
    s.add_argument("--type", default="charuco", choices=("charuco", "chessboard"))
    s.add_argument("--squares", type=_size, help="squares across and down, e.g. 7x10 (default: by the paper)")
    s.add_argument("--square-mm", type=float, help="a square's side (default: what fills the paper)")
    s.add_argument("--marker-mm", type=float)
    s.add_argument("--dictionary", default="DICT_4X4_250")
    s.add_argument("--ground", action="store_true",
                   help="the board for `extrinsics`: 5x7 large squares, A1 unless --paper says otherwise")
    s.add_argument("--sheets", type=_size, metavar="AxB",
                   help="a large ground board from A sheets across and B down (2 by 3 squares a sheet), e.g. 3x2 of A3")
    s.add_argument("--screen", type=_size, help="for a flat display of this many pixels instead of paper")
    s.add_argument("--ppi", type=float, help="the display's pixels per inch (with --screen)")

    s = add("capture", cmd_capture, "Take the views: each is taken by itself when the board is still and somewhere new.")
    s.add_argument("source", help="csi | csi:N | v4l2:/dev/videoN | serve://HOST[:9000] | N | URL or file")
    s.add_argument("-o", "--output", required=True, help="the session's directory")
    s.add_argument("--board", help="board.json (default: the session's own, when resuming)")
    s.add_argument("--size", type=_size, default=(1920, 1080), help="the picture asked of the camera (1920x1080)")
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270), help="counterclockwise, as ucpeer-detect's")
    s.add_argument("--sensor-mode", type=int, help="csi: the sensor mode, where the size and rate fit more than one")
    s.add_argument("--focus", type=int, help="v4l2: the focus to fix (autofocus is always turned off)")
    s.add_argument("--target", type=int, default=30, help="views to take (30)")
    s.add_argument("--min-corners", type=int, default=12)
    s.add_argument("--steady", type=float, default=12.0, help="still means under this many pixels a second (12)")
    s.add_argument("--manual", action="store_true", help="take views only on the space key")
    s.add_argument("--headless", action="store_true", help="no window: preview.jpg for `camcal preview`")
    s.add_argument("--duration", type=float, help="stop after this many seconds")

    s = add("preview", cmd_preview, "Show a headless capture as it runs: a directory, or USER@HOST:DIRECTORY through ssh.")
    s.add_argument("target")
    s.add_argument("-i", "--identity", help="ssh key file")

    s = add("monitor", cmd_monitor, "A window that follows a capture as it runs: pictures, grade, what is still "
                                    "missing, the flatness of the sheet. Everything is worked out on this machine.")
    s.add_argument("target", help="the session: a directory, or USER@HOST:DIRECTORY (copied with rsync over ssh)")
    s.add_argument("-i", "--identity", help="ssh key file")
    s.add_argument("--square-mm", type=float, help="a square's side as measured on the print")
    s.add_argument("--cache", help="where the pictures are copied to (default ~/.cache/camcal/HOST_DIRECTORY)")
    s.add_argument("--start", action="store_true", help="start the capture there too; closing the window stops it")
    s.add_argument("--source", default="csi", help="with --start: the camera there (csi)")
    s.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270), help="with --start: counterclockwise")
    s.add_argument("--board", help="with --start: the board's JSON there, relative to --toolkit")
    s.add_argument("--steady", type=float, default=5.0,
                   help="with --start: still means under this many pixels a second (5: a rolling shutter)")
    s.add_argument("--views", type=int, default=30, help="with --start: views to take (30)")
    s.add_argument("--size", type=_size, help="with --start: the picture asked of the camera (capture's default 1920x1080)")
    s.add_argument("--sensor-mode", type=int, help="with --start, csi: the sensor mode (IMX219: 2 is 1920x1080 cropped, "
                                                   "3 is 1640x1232 the whole sensor binned)")
    s.add_argument("--fps", type=int, help="with --start: frames a second (capture's default 30)")
    s.add_argument("--toolkit", default="camera-calib", help="with --start: camcal's directory there")
    s.add_argument("--snapshot", help="draw the window once into this PNG and end")

    s = add("calibrate", cmd_calibrate, "Intrinsics from the pictures: calibration.json, the YAMLs and report.html.")
    s.add_argument("inputs", nargs="+", help="session directories or pictures")
    s.add_argument("--board", help="board.json (default: the session's)")
    s.add_argument("--square-mm", type=float, help="a square's side as measured on the print")
    s.add_argument("--model", default="auto",
                   help="auto (all four, the best on held-out views) | slam (pinhole-radtan4 or fisheye: what "
                        "Kalibr, VINS, OpenVINS and ORB-SLAM3 take) | names, comma separated: " + ", ".join(models.MODELS))
    s.add_argument("--bootstrap", type=int, default=0, help="σ from this many refits to resampled views")
    s.add_argument("--min-corners", type=int, default=8)
    s.add_argument("--no-report", action="store_true")
    s.add_argument("-o", "--output", help="directory (default: result/ in the session)")

    s = add("validate", cmd_validate, "Does a calibration still hold? Pictures it was not made from; exit status 1 if not.")
    s.add_argument("calibration")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--board"), s.add_argument("--square-mm", type=float)
    s.add_argument("--overlay", help="directory for pictures with every corner's error drawn 20 times its length")
    s.add_argument("--json", help="the numbers, to this file")

    s = add("undistort", cmd_undistort, "Undistorted pictures, the pinhole camera they have, and the remap tables.")
    s.add_argument("calibration")
    s.add_argument("inputs", nargs="+")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--alpha", type=float, default=0.0, help="0: only what the lens saw (default); 1: all of the picture")
    s.add_argument("--size", type=_size, help="the undistorted picture's size (default: the same)")
    s.add_argument("--maps", action="store_true", help="also rectify_maps.npz: new_K, map_x, map_y for cv2.remap")

    s = add("extrinsics", cmd_extrinsics,
            "The camera in the vehicle's frame (X forward, Y left, Z up) from a ChArUco board flat on the ground.")
    s.add_argument("calibration")
    s.add_argument("inputs", nargs="+", help="pictures of the one scene")
    s.add_argument("--board", required=True, help="the ground board's JSON (camcal board --ground)")
    s.add_argument("--square-mm", type=float)
    s.add_argument("--origin", type=_pair, required=True,
                   help="X,Y in metres of the pattern's outer corner beside marker 0 (the page's top left)")
    s.add_argument("--yaw", type=float, required=True,
                   help="the direction of the board's x axis (along its short side's squares), degrees "
                        "counterclockwise from the vehicle's X")
    s.add_argument("--board-z", type=float, default=0.0, help="the printed surface above the ground, metres")
    s.add_argument("-o", "--output", help="directory (default: the calibration's)")

    def mounting(s):
        s.add_argument("--extrinsics", help="extrinsics.json (written by `camcal extrinsics`)")
        s.add_argument("--height", type=float, help="without --extrinsics: the camera above the road, metres")
        s.add_argument("--pitch", type=float, default=0.0, help="... looking this many degrees down (0)")
        s.add_argument("--yaw", type=float, default=0.0, help="... and to the left (0)")
        s.add_argument("--roll", type=float, default=0.0)

    s = add("budget", cmd_budget, "ADAS error budget: how far off a measured range is at each range, and what makes it so.")
    s.add_argument("calibration")
    mounting(s)
    s.add_argument("--ranges", type=float, nargs="+", help="metres (5 10 15 20 30 40 50 75 100)")
    s.add_argument("--width", type=float, default=1.8, help="width ranging: the object's width, metres (1.8, a car)")
    s.add_argument("--lateral", type=float, default=0.0, help="the object's Y, metres left of the camera (0)")
    s.add_argument("--sigma-f", type=float, help="focal length, percent (default: the calibration's uncertainty)")
    s.add_argument("--sigma-c", type=float, help="principal point, pixels (default: the calibration's)")
    s.add_argument("--sigma-pitch", type=float, help="degrees (0.2, or the extrinsics' if larger)")
    s.add_argument("--sigma-roll", type=float), s.add_argument("--sigma-yaw", type=float)
    s.add_argument("--sigma-height", type=float, help="metres (0.01)")
    s.add_argument("--sigma-pixel", type=float, help="where the detector puts the object's foot, pixels (1)")
    s.add_argument("--pitching", type=float, help="the vehicle's pitch while driving, degrees, shown beside (0.5)")
    s.add_argument("-o", "--output", help="the numbers, to this JSON file")

    s = add("ground", cmd_ground, "A pixel to the point of the (flat) road it shows, and a point of the road to its pixel.")
    s.add_argument("calibration")
    mounting(s)
    s.add_argument("--pixel", type=_pair, action="append", metavar="U,V")
    s.add_argument("--point", type=_pair, action="append", metavar="X,Y", help="metres: X forward, Y left")

    s = add("bev", cmd_bev, "The road from above (inverse perspective mapping): pictures, and the remap tables.")
    s.add_argument("calibration")
    s.add_argument("inputs", nargs="*", help="pictures to map")
    mounting(s)
    s.add_argument("--ahead", type=_pair, default=(2.0, 30.0), help="X from,to in metres (2,30)")
    s.add_argument("--side", type=float, default=8.0, help="Y from +side to -side, metres (8)")
    s.add_argument("--res", type=float, default=0.02, help="metres a pixel (0.02)")
    s.add_argument("-o", "--output", required=True, help="directory")

    s = add("transform", cmd_transform, "The calibration for the same camera's picture rotated, resized or cropped.")
    s.add_argument("calibration")
    s.add_argument("ops", nargs="+", help="in order: rotate=90|180|270 (counterclockwise)  scale=S  resize=WxH  crop=X,Y,W,H")
    s.add_argument("-o", "--output", required=True, help="directory")

    s = add("synth", cmd_synth, "Rendered pictures of a board from a camera that is known: a test with the answer.")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--model", default="pinhole-radtan4", choices=models.MODELS)
    s.add_argument("--size", type=_size, default=(1280, 720))
    s.add_argument("--fx", type=float)
    s.add_argument("--dist", type=float, nargs="*", help="the model's coefficients, in its order")
    s.add_argument("--count", type=int, default=40), s.add_argument("--seed", type=int, default=0)
    s.add_argument("--noise", type=float, default=1.5, help="sensor noise, grey levels")
    s.add_argument("--paper", default="A4"), s.add_argument("--board")
    return p


def main(argv=None) -> int:
    p = parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 2
    try:
        return a.fn(a)
    except (ValueError, RuntimeError, OSError) as e:
        print("camcal %s: %s" % (a.command, e), file=sys.stderr)
        return 1
