"""monitor: a window that follows a capture while it runs, on the machine you sit at.

    python3 -m camcal monitor machineai@192.168.1.240:calib/csi0 -i ~/.ssh/id_ed25519_machineai \\
        --square-mm 29.133 [--start --board boards/board_charuco_A3_9x12_30mm.json --rotate 180]

What it shows, refreshed as pictures come in:
  the camera             preview.jpg of the headless capture, as `preview` shows it
  the count              pictures taken, used, dropped by the fit, without the board
  the grade              the camera fitted again on every picture so far (calibrate's own steps:
                         the model chosen on views held out, bad corners and views dropped), and
                         what still stands between it and grade A
  error per picture      the picture that is worse than the others
  coverage               where in the picture corners have been measured
  flatness of the sheet  the error at each place of the sheet over all pictures: about the same
                         everywhere on a flat sheet, more where it bulges

The pictures are copied here with rsync over ssh and everything is worked out here: the Jetson
runs only the capture. No port is opened anywhere. --start starts the capture there as well and
stops it (Ctrl-C) when the window is closed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import boards, calibrate, capture, models
from .boards import BoardSpec
from .models import View
from .report import BLUE, EMPTY, INK, MUTED, RAMP, RULE, STATUS, SURFACE

MIN_VIEWS = 4  # fewer and nothing is fitted yet

MEANING = {"A": "đủ cho SLAM, VIO và hình học tầm xa", "B": "đủ cho ADAS L2 và 3D object detection",
           "C": "chỉ để thử", "D": "chưa dùng được"}
# what each of calibrate.GRADE_NEEDS asks for, and what to do about it
ROWS = (
    ("e", "Lỗi trên ảnh để riêng", "ảnh nét, giữ yên khi chụp, bảng thật phẳng"),
    ("cov", "Phủ khung hình", "đưa bảng tới các ô xám của bản đồ phủ"),
    ("border", "Phủ mép ảnh", "cho bảng tràn ra mép ảnh ở cả bốn phía"),
    ("tilt", "Ảnh nghiêng trên 15°", "nghiêng bảng 30 đến 45° theo mỗi hướng"),
    ("sf", "Tiêu cự biết tới (1σ)", "thêm ảnh nghiêng, cả gần lẫn xa"),
    ("bias", "fx lệch giữa các nửa bảng", "làm phẳng bảng (chụp thêm không giảm được lỗi này)"),
    ("views", "Số ảnh dùng được", "chụp thêm"),
)


def _f(x: float, d: int = 2) -> str:
    return ("%.*f" % (d, x)).replace(".", ",")


def _pct(x: float, d: int = 0) -> str:
    return _f(100.0 * x, d) + "%"


def _value(k: str, x: float) -> str:
    return {"e": lambda: _f(x) + " px", "views": lambda: "%d" % x,
            "sf": lambda: _pct(x, 2), "bias": lambda: _pct(x, 2)}.get(k, lambda: _pct(x))()


def _need(k: str, n: float) -> str:
    return ("≤ " if k in ("e", "sf", "bias") else "≥ ") + _value(k, n)


def _met(k: str, have: float, need: float) -> bool:
    return have <= need if k in ("e", "sf", "bias") else have >= need


# --- the session: here, or there through ssh ------------------------------------------------


class Session:
    """The capture's directory: a directory here, or USER@HOST:DIR copied here with rsync."""

    def __init__(self, target: str, identity: Optional[str] = None, cache: Optional[str] = None):
        host, sep, path = target.partition(":")
        self.where = target
        self.remote = bool(sep) and not os.path.exists(target)
        self.ssh = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5"]
        if identity:
            self.ssh += ["-i", os.path.expanduser(identity), "-o", "IdentitiesOnly=yes"]
        if self.remote:
            self.host, self.path = host, path.rstrip("/") or "."
            name = "%s_%s" % (host.replace("@", "_"), self.path.strip("/~").replace("/", "_"))
            self.dir = cache or os.path.join(os.path.expanduser("~/.cache/camcal"), name)
        else:
            self.host, self.path, self.dir = None, target, target

    def sync(self) -> Optional[str]:
        """The pictures and session.json copied here. What went wrong, or None."""
        if not self.remote:
            return None if os.path.isdir(self.dir) else "chưa có thư mục %s" % self.dir
        os.makedirs(self.dir, exist_ok=True)
        cmd = ["rsync", "-a", "--delete", "--include=img_*.png", "--include=session.json", "--exclude=*",
               "-e", " ".join(shlex.quote(x) for x in self.ssh), "%s:%s/" % (self.host, self.path), self.dir + "/"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return "rsync quá 5 phút"
        if p.returncode in (0, 24):  # 24: a file went away while copying (a picture dropped)
            return None
        if "No such file" in p.stderr:
            return "đợi capture tạo %s trên board" % self.path
        lines = [x for x in p.stderr.strip().splitlines() if x.strip()]
        return lines[0] if lines else "rsync dừng với mã %d" % p.returncode


class Preview:
    """The newest preview.jpg of the session (RGB), read in a thread, and when it came."""

    def __init__(self, session: Session):
        self.s, self.rgb, self.at, self.stop, self.proc = session, None, 0.0, False, None

    def start(self) -> "Preview":
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def read_once(self) -> None:
        p = os.path.join(self.s.dir, "preview.jpg")
        if os.path.exists(p):
            with open(p, "rb") as f:
                self._got(f.read())

    def _run(self) -> None:
        while not self.stop:
            try:
                self._remote() if self.s.remote else self._local()
            except Exception:  # a dropped connection: try again
                pass
            time.sleep(2.0)

    def _local(self) -> None:
        p, last = os.path.join(self.s.dir, "preview.jpg"), 0
        while not self.stop:
            try:
                t = os.stat(p).st_mtime_ns
                if t != last:
                    with open(p, "rb") as f:
                        self._got(f.read())
                    last = t
            except OSError:
                pass
            time.sleep(0.2)

    def _remote(self) -> None:  # one ssh session that sends the file each time it changes
        cmd = self.s.ssh + [self.s.host, "python3 -u -c %s %s" % (
            shlex.quote(capture._REMOTE), shlex.quote(self.s.path + "/preview.jpg"))]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        try:
            while not self.stop:
                head = self.proc.stdout.read(4)
                if len(head) < 4:
                    return
                self._got(self.proc.stdout.read(int.from_bytes(head, "big")))
        finally:
            self.proc.terminate()

    def _got(self, data: bytes) -> None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.rgb, self.at = np.ascontiguousarray(img[:, :, ::-1]), time.time()

    def close(self) -> None:
        self.stop = True
        if self.proc is not None:
            self.proc.terminate()


class RemoteCapture:
    """`camcal capture` started on the Jetson through ssh, and stopped with Ctrl-C."""

    def __init__(self, session: Session, source: str, rotate: int, board: Optional[str], steady: float,
                 views: int, toolkit: str, log_path: str, size: Optional[Tuple[int, int]] = None,
                 sensor_mode: Optional[int] = None, fps: Optional[int] = None):
        def there(p):  # a path relative to the home directory there
            return shlex.quote(p) if p.startswith("/") else '"$HOME"/' + shlex.quote(p.lstrip("~/") or ".")

        cmd = "cd %s && exec python3 -m camcal capture %s -o %s --rotate %d --steady %g --target %d --headless" % (
            there(toolkit), shlex.quote(source), there(session.path), rotate, steady, views)
        if board:
            cmd += " --board " + shlex.quote(board)
        if size:
            cmd += " --size %dx%d" % tuple(size)
        if sensor_mode is not None:
            cmd += " --sensor-mode %d" % sensor_mode
        if fps:
            cmd += " --fps %d" % fps
        self.log_path = log_path
        self.log = open(log_path, "ab")
        # -tt: a terminal there, so that Ctrl-C (and a dropped connection) reach the capture
        self.proc = subprocess.Popen(session.ssh + ["-tt", session.host, cmd], stdin=subprocess.PIPE,
                                     stdout=self.log, stderr=subprocess.STDOUT)

    def running(self) -> bool:
        return self.proc.poll() is None

    def last_line(self) -> str:
        try:
            with open(self.log_path, "rb") as f:
                lines = [x.strip() for x in f.read()[-4000:].decode("utf-8", "replace").splitlines()]
        except OSError:
            return ""
        lines = [x for x in lines if x and not x.startswith(("GST_", "CONSUMER", "Opening", "NvMM", "Using", "["))]
        return lines[-1] if lines else ""

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"\x03")
                self.proc.stdin.flush()
                self.proc.wait(8)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.terminate()
        self.log.close()


# --- what is worked out ----------------------------------------------------------------------


def flatness(res: calibrate.Result, spec: BoardSpec, least: int = 4) -> Tuple[np.ndarray, Optional[dict]]:
    """Per inner corner of the sheet (rows down, columns across, as printed), the rms of its
    errors over the views that saw it at least `least` times. A flat sheet has about the same
    everywhere; a sheet that bulges somewhere has more there, whatever the view, while the fit
    spreads the rest of the bulge over the whole sheet. (The mean error vector would not do:
    a bulge moves a corner in a direction that turns with the camera, and averages out.)
    Returns (grid of px, NaN where seen too seldom; {median, p90} over the corners, or None)."""
    nx, ny = spec.squares_x - 1, spec.squares_y - 1
    acc: Dict[int, List[float]] = {}
    for v, r in zip(res.views, res.residuals):
        for i, e in zip(v.ids, r):
            acc.setdefault(int(i), []).append(float(e @ e))
    grid = np.full((ny, nx), np.nan)
    for i, es in acc.items():
        if len(es) >= least and 0 <= i < nx * ny:
            grid[i // nx, i % nx] = float(np.sqrt(np.mean(es)))
    seen = grid[np.isfinite(grid)]
    if seen.size < 6:
        return grid, None
    return grid, {"median": float(np.median(seen)), "p90": float(np.percentile(seen, 90))}


def uneven(stats: Optional[dict]) -> bool:
    """The error gathers in one part of the sheet: its worst tenth 1.4 times the median and
    0.05 px over it. Noise alone gives about 1.1 (a flat sheet, rendered: 1.13; 1.5 mm of
    bulge on one side: 1.62)."""
    return bool(stats) and stats["p90"] > 1.4 * stats["median"] and stats["p90"] - stats["median"] > 0.05


def summarize(res: calibrate.Result, spec: BoardSpec) -> dict:
    cal = res.calibration
    m = cal.meta
    grid, sheet = flatness(res, spec)
    return {"grade": m["grade"], "model": cal.model, "K": cal.K, "std": m.get("std") or {},
            "quality": m["quality"], "have": calibrate.grade_inputs(m["quality"]),
            "views": {v["name"]: v["rms_px"] for v in m["views"]},
            "dropped": {d["name"]: d["why"] for d in m["dropped_views"]},
            "flat": grid, "sheet": sheet}


class Watch:
    """Copies, looks at each new picture once, fits again when the set of views changes."""

    def __init__(self, session: Session, square_mm: Optional[float] = None, min_corners: int = 8):
        self.s, self.square_mm, self.min_corners = session, square_mm, min_corners
        self.lock = threading.Lock()
        self.version = 0
        self.state: dict = {"message": "đang nối tới %s" % session.where}
        self.stop = False
        self._seen: Dict[str, tuple] = {}  # name: ((mtime, bytes), View or None, why not)
        self._spec_key = None
        self._fitted: Optional[tuple] = None

    def publish(self, **kw) -> None:
        with self.lock:
            self.state = dict(self.state, **kw)
            self.version += 1

    def snapshot(self) -> Tuple[int, dict]:
        with self.lock:
            return self.version, dict(self.state)

    def run(self) -> None:
        while not self.stop:
            t = time.time()
            try:
                self.step()
            except Exception as e:  # shown, and tried again
                self.publish(message="lỗi: %s" % e, fitting=False)
            time.sleep(max(0.5, 2.0 - (time.time() - t)))

    def step(self) -> None:
        err = self.s.sync()
        path = os.path.join(self.s.dir, "session.json")
        if not os.path.exists(path):
            self.publish(message=err or "đợi capture bắt đầu ở %s" % self.s.where)
            return
        with open(path) as f:
            sess = json.load(f)
        spec = BoardSpec.from_dict(sess["board"])
        if self.square_mm:
            spec = spec.with_square_mm(self.square_mm)
        key = json.dumps(spec.to_dict(), sort_keys=True)
        if key != self._spec_key:
            self._seen.clear()
            self._spec_key, self._fitted = key, None
        paths = calibrate.list_images([self.s.dir])
        here = {os.path.basename(p) for p in paths}
        for n in [n for n in self._seen if n not in here]:
            del self._seen[n]
        size = tuple(sess["image_size"]) if sess.get("image_size") else None
        for p in paths:
            n = os.path.basename(p)
            st = os.stat(p)
            sig = (st.st_mtime_ns, st.st_size)
            if n in self._seen and self._seen[n][0] == sig:
                continue
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # not all there yet
            size = size or (img.shape[1], img.shape[0])
            det = boards.detect(spec, img, self.min_corners)
            if det is None:
                self._seen[n] = (sig, None, "không thấy bảng")
            elif not calibrate._spans_a_plane(det.object_points(spec)):
                self._seen[n] = (sig, None, "góc trên ít hơn 3 hàng")
            else:
                self._seen[n] = (sig, View(n, det.ids, det.object_points(spec), det.image), None)
        names = sorted(self._seen)
        views = [self._seen[n][1] for n in names if self._seen[n][1] is not None]
        key = tuple(v.name for v in views)
        due = len(views) >= MIN_VIEWS and key != self._fitted
        self.publish(message=err, spec=spec, size=size, names=names, views=len(views),
                     no_board=[n for n in names if self._seen[n][1] is None],
                     grid=calibrate.coverage_grid(views, size) if views and size else None,
                     fitting=due, fit=self.state.get("fit") if len(views) >= MIN_VIEWS else None,
                     synced=time.time())
        if due:
            t = time.time()
            try:
                res = calibrate.calibrate(spec, views, size, models.MODELS, min_corners=self.min_corners,
                                          views_given=len(names))
                fit = summarize(res, spec)
            except ValueError as e:
                fit = {"error": str(e)}
            fit.update(names=key, seconds=time.time() - t)
            self._fitted = key
            self.publish(fit=fit, fitting=False)


# --- the window --------------------------------------------------------------------------------


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.6, "axes.labelcolor": MUTED,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "grid.color": RULE, "grid.linewidth": 0.6, "axes.grid": True,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5, "text.color": INK,
        "font.family": "DejaVu Sans", "toolbar": "None",
    })


class Window:
    def __init__(self, watch: Watch, preview: Preview, remote: Optional[RemoteCapture], command: str):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        _style(plt)
        self.plt, self.watch, self.preview, self.remote, self.command = plt, watch, preview, remote, command
        self.ramp = LinearSegmentedColormap.from_list("camcal_blue", RAMP).with_extremes(bad=EMPTY)
        self.fig = fig = plt.figure(figsize=(15.5, 9.2))
        try:
            fig.canvas.manager.set_window_title("camcal monitor: %s" % watch.s.where)
        except AttributeError:
            pass
        gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.3, left=0.035, right=0.955,
                              top=0.955, bottom=0.075)
        top = gs[0].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.06)
        low = gs[1].subgridspec(1, 3, wspace=0.32)
        self.ax_cam = fig.add_subplot(top[0])
        self.ax_txt = fig.add_subplot(top[1])
        self.ax_err = fig.add_subplot(low[0])
        self.ax_cov = fig.add_subplot(low[1])
        self.ax_flat = fig.add_subplot(low[2])
        # colour bars in axes of their own: clearing an axes drops the axes inset in it
        self.cax_cov = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        self.cax_flat = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        self.im = None
        self.shown_at = -1.0
        self.drawn = -1
        self.second = -1
        self.state: dict = {}
        self.ax_cam.set_axis_off()

    # one tick of the window's timer: only what changed is drawn again
    def tick(self) -> None:
        now = time.time()
        changed = False
        if self.preview.at != self.shown_at and self.preview.rgb is not None:
            self.draw_camera()
            changed = True
        version, state = self.watch.snapshot()
        if version != self.drawn:
            self.drawn, self.state = version, state
            self.draw_status(), self.draw_errors(), self.draw_coverage(), self.draw_flatness()
            changed = True
        if int(now) != self.second:
            self.second = int(now)
            self.title_camera(now)
            changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def draw_all(self) -> None:
        self.drawn, self.state = self.watch.snapshot()
        if self.preview.rgb is not None:
            self.draw_camera()
        self.title_camera(time.time())
        self.draw_status(), self.draw_errors(), self.draw_coverage(), self.draw_flatness()

    def draw_camera(self) -> None:
        img = self.preview.rgb
        if self.im is None or self.im.get_array().shape != img.shape:
            self.ax_cam.clear()
            self.im = self.ax_cam.imshow(img, interpolation="antialiased")
            self.ax_cam.set_axis_off()
        else:
            self.im.set_data(img)
        self.shown_at = self.preview.at

    def title_camera(self, now: float) -> None:
        ax = self.ax_cam
        if self.preview.rgb is None:
            ax.set_title("Camera: đợi ảnh preview từ capture")
            return
        age = now - self.preview.at
        text = "Camera (ảnh mới %s s trước)" % _f(age, 0) if age < 5 else \
            "Camera: không có ảnh mới %d s, capture đã dừng?" % age
        if self.remote is not None and not self.remote.running():
            text += "  ·  capture trên board đã dừng: %s" % self.remote.last_line()[:90]
        ax.set_title(text)

    def draw_status(self) -> None:
        ax, st = self.ax_txt, self.state
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)

        def t(x, y, s, **kw):
            kw.setdefault("color", INK)
            return ax.text(x, y, s, transform=ax.transAxes, va="top", ha=kw.pop("ha", "left"), **kw)

        names = st.get("names") or []
        fit = st.get("fit") or {}
        need = calibrate.GRADE_NEEDS["A"]
        t(0, 1.0, "Ảnh đã chụp", color=MUTED, fontsize=10)
        t(0, 0.955, "%d" % len(names), fontsize=28, fontweight="bold")
        t(0.16, 0.93, "cần %d ảnh dùng được trở lên cho hạng A" % need["views"], color=MUTED, fontsize=9.5)
        used = len(fit.get("views", {}))
        line = "✓ dùng được %d    ✕ bị loại %d    ○ không thấy bảng %d" % (
            used, len(fit.get("dropped", {})), len(st.get("no_board") or []))
        waiting = st.get("views", 0) - used - len(fit.get("dropped", {}))
        if fit and waiting > 0:
            line += "    … chưa tính %d" % waiting
        t(0, 0.84, line, fontsize=10)
        y = 0.79
        if st.get("fitting"):
            t(0, y, "đang tính lại với %d ảnh có bảng…" % st.get("views", 0), color=MUTED, fontsize=9.5)
        y -= 0.04
        if "grade" in fit:
            from matplotlib.patches import FancyBboxPatch

            g = fit["grade"]
            col, icon = STATUS[g]
            ax.add_patch(FancyBboxPatch((0, y - 0.105), 0.11, 0.1, boxstyle="round,pad=0,rounding_size=0.02",
                                        transform=ax.transAxes, facecolor=col, edgecolor="none"))
            t(0.055, y - 0.012, g, ha="center", fontsize=22, fontweight="bold", color="white")
            t(0.14, y - 0.005, "%s Hạng %s" % (icon, g), fontsize=14, fontweight="bold")
            t(0.14, y - 0.06, MEANING[g], color=MUTED, fontsize=9.5)
            y -= 0.13
            have = fit["have"]
            t(0.6, y, "cần cho A", color=MUTED, fontsize=8.5)
            y -= 0.032
            todo = []
            for k, label, advice in ROWS:
                ok = _met(k, have[k], need[k])
                t(0, y, label, fontsize=9.5)
                t(0.4, y, _value(k, have[k]), fontsize=9.5, fontweight="bold")
                t(0.6, y, _need(k, need[k]), color=MUTED, fontsize=9.5)
                t(0.8, y, "✓" if ok else "✕", color=STATUS["A" if ok else "D"][0], fontsize=10.5, fontweight="bold")
                if not ok:
                    todo.append(advice)
                y -= 0.04
            sx = fit["std"].get("fx")
            t(0, y - 0.005, "Mô hình %s · fx %s%s px · %d ảnh, tính trong %s s" % (
                fit["model"], _f(fit["K"][0, 0], 1), " ± " + _f(sx, 2) if sx is not None else "",
                used, _f(fit["seconds"], 1)), color=MUTED, fontsize=9)
            y -= 0.045
            sheet = fit.get("sheet")
            if sheet is not None:
                t(0, y, "Lỗi theo chỗ trên tờ giấy: trung vị %s px, 10%% chỗ tệ nhất %s px" % (
                    _f(sheet["median"]), _f(sheet["p90"])), fontsize=9.5)
                if uneven(sheet):
                    y -= 0.036
                    t(0, y, "⚠ lỗi dồn vào vùng đậm của bản đồ dưới phải: tờ giấy không phẳng ở đó",
                      fontsize=9, fontweight="bold")
                y -= 0.05
            if g == "A":
                t(0, y, "✓ ĐỦ ẢNH: dừng capture rồi calibrate", fontsize=12, fontweight="bold", color=STATUS["A"][0])
            else:
                t(0, y, "Chưa đủ cho hạng A. Việc cần làm:", fontsize=10, fontweight="bold")
                for a in list(dict.fromkeys(todo))[:3]:
                    y -= 0.038
                    t(0.02, y, "• " + a, fontsize=9.5)
        elif "error" in fit:
            t(0, y, "Chưa tính được: %s" % fit["error"], fontsize=9.5, wrap=True)
        else:
            t(0, y, "Cần ít nhất %d ảnh có bảng để tính thử" % MIN_VIEWS, color=MUTED, fontsize=10)
        if st.get("message"):
            t(0, 0.05, "⚠ %s" % st["message"], fontsize=9)
        t(0, 0.008, self.command, color=MUTED, fontsize=8, family="DejaVu Sans Mono")

    def draw_errors(self) -> None:
        from matplotlib.ticker import MaxNLocator

        ax, st = self.ax_err, self.state
        ax.clear()
        ax.set_title("Lỗi từng ảnh (px, rms)")
        names = st.get("names") or []
        fit = st.get("fit") or {}
        per, dropped, none = fit.get("views", {}), fit.get("dropped", {}), set(st.get("no_board") or [])

        def num(n):
            digits = "".join(c for c in n if c.isdigit())
            return int(digits) if digits else names.index(n) + 1

        used = [(num(n), per[n]) for n in names if n in per]
        top = max([0.6] + [1.15 * h for _, h in used])
        if used:
            ax.bar([x for x, _ in used], [h for _, h in used], width=0.72, color=BLUE, edgecolor=SURFACE,
                   linewidth=1.0, label="dùng được")
        for xs, y, kw in (([num(n) for n in names if n in dropped], top * 0.94,
                           dict(marker="x", color=STATUS["D"][0], s=36, label="bị loại")),
                          ([num(n) for n in names if n in none], 0.0,
                           dict(marker="o", facecolors="none", edgecolors=MUTED, s=22, label="không thấy bảng")),
                          ([num(n) for n in names if n not in per and n not in dropped and n not in none], 0.0,
                           dict(marker="|", color=MUTED, s=60, label="chưa tính"))):
            if xs:
                ax.scatter(xs, [y] * len(xs), clip_on=False, zorder=3, **kw)
        for v, g in ((0.25, "A"), (0.5, "B")):
            if v < top:
                ax.axhline(v, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
                ax.annotate("%s ≤ %s" % (g, _f(v)), (1.0, v), xycoords=("axes fraction", "data"),
                            xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.set_ylim(0, top)
        if names:
            ax.set_xlim(0.3, max(num(n) for n in names) + 0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("ảnh số")
        ax.grid(axis="x", visible=False)
        handles = ax.get_legend_handles_labels()[0]
        if len(handles) > 1:
            ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.0), handletextpad=0.3, columnspacing=1.0)

    @staticmethod
    def _beside(ax, cax) -> None:
        ax.apply_aspect()
        p = ax.get_position()
        cax.set_position([p.x1 + 0.006, p.y0, 0.007, p.height])
        cax.set_axis_on()

    def draw_coverage(self) -> None:
        ax, cax, st = self.ax_cov, self.cax_cov, self.state
        ax.clear(), cax.clear()
        grid, size = st.get("grid"), st.get("size")
        if grid is None or size is None:
            ax.set_title("Độ phủ khung hình")
            ax.set_axis_off(), cax.set_axis_off()
            return
        w, h = size
        cov, border, _empty = calibrate._coverage(grid)
        rows, cols = grid.shape
        g = np.ma.masked_less(grid.astype(float), 2)
        m = ax.pcolormesh(np.linspace(0, w, cols + 1), np.linspace(0, h, rows + 1), g, cmap=self.ramp,
                          edgecolors=SURFACE, linewidth=1.2, vmin=2, vmax=max(3.0, float(grid.max())))
        ax.set_facecolor(EMPTY)
        ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.set_aspect("equal"), ax.grid(False)
        ax.set_title("Độ phủ khung hình %s · mép %s (xám: chưa có góc)" % (_pct(cov), _pct(border)))
        ax.tick_params(labelsize=7.5)
        self._beside(ax, cax)
        cb = self.fig.colorbar(m, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label("góc / ô", fontsize=8)

    def draw_flatness(self) -> None:
        ax, cax, st = self.ax_flat, self.cax_flat, self.state
        ax.clear(), cax.clear()
        fit, spec = st.get("fit") or {}, st.get("spec")
        grid = fit.get("flat")
        if grid is None or spec is None or not np.isfinite(grid).any():
            ax.set_title("Độ phẳng tờ giấy")
            ax.set_axis_off(), cax.set_axis_off()
            return
        s = spec.square_mm
        ny, nx = grid.shape
        vmax = max(0.5, float(np.nanmax(grid)))
        m = ax.pcolormesh((np.arange(nx + 1) + 0.5) * s, (np.arange(ny + 1) + 0.5) * s,
                          np.ma.masked_invalid(grid), cmap=self.ramp, vmin=0.0, vmax=vmax,
                          edgecolors=SURFACE, linewidth=1.0)
        from matplotlib.patches import Rectangle

        ax.add_patch(Rectangle((0, 0), spec.squares_x * s, spec.squares_y * s, fill=False, edgecolor=MUTED,
                               linewidth=0.8))
        ax.set_facecolor(EMPTY)
        ax.set_xlim(-0.2 * s, (spec.squares_x + 0.2) * s), ax.set_ylim((spec.squares_y + 0.2) * s, -0.2 * s)
        ax.set_aspect("equal"), ax.grid(False)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("mm trên tờ giấy, như bản in (góc marker 0 ở trên trái)")
        ax.set_title("Lỗi theo chỗ trên tờ giấy (px rms)")
        self._beside(ax, cax)
        cb = self.fig.colorbar(m, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label("px", fontsize=8)


def main(a) -> int:
    import matplotlib

    if a.snapshot:
        matplotlib.use("Agg")
    elif sys.platform == "darwin":
        matplotlib.use("macosx")
    session = Session(a.target, a.identity, a.cache)
    watch = Watch(session, a.square_mm)
    square = " --square-mm %g" % a.square_mm if a.square_mm else ""
    where = session.path if session.remote else session.dir
    command = "đủ ảnh thì trên board: python3 -m camcal calibrate %s%s" % (where, square)
    remote = None
    if a.start:
        if not session.remote:
            raise ValueError("--start is for USER@HOST:DIRECTORY")
        os.makedirs(session.dir, exist_ok=True)
        log = os.path.join(os.path.dirname(session.dir), os.path.basename(session.dir) + ".capture.log")
        remote = RemoteCapture(session, a.source, a.rotate, a.board, a.steady, a.views, a.toolkit, log,
                               a.size, a.sensor_mode, a.fps)
        print("capture started there; its output: %s" % log)
    preview = Preview(session)
    try:
        if a.snapshot:  # once, into a picture: for a test, or to send
            watch.step()
            preview.read_once()
            if session.remote:
                preview.start()
                end = time.time() + 6.0
                while preview.rgb is None and time.time() < end:
                    time.sleep(0.2)
            win = Window(watch, preview, remote, command)
            win.draw_all()
            win.fig.savefig(a.snapshot, dpi=110)
            print(a.snapshot)
            return 0
        preview.start()
        threading.Thread(target=watch.run, daemon=True).start()
        win = Window(watch, preview, remote, command)
        timer = win.fig.canvas.new_timer(interval=250)
        timer.add_callback(win.tick)
        timer.start()
        win.plt.show()
        return 0
    finally:
        watch.stop = True
        preview.close()
        if remote is not None:
            remote.stop()
