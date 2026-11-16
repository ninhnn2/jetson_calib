"""Taking the views: the camera fixed, the board moved in front of it.

A view is taken by itself when the board is still and is somewhere it has not been: still,
because a rolling-shutter sensor (the IMX219 is one) bends a board that moves while the frame
is read out; somewhere new, because forty pictures of one pose are one picture. What is still
missing is said in the picture: the cells of the picture without corners, tilt, near and far.

With a screen: a window (space takes the view now, d drops the last one, a turns the automatic
taking off and on, q or Esc ends). Without one: --headless writes preview.jpg into the session
for `camcal preview` to show on another machine through ssh; it ends at the target number.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import boards, calibrate
from .boards import BoardSpec, Detection
from .models import View

STEADY_PX_S = 12.0  # the board's corners move less than this many pixels a second
NOVELTY = 0.10  # least distance of a view from every view taken, in the units of describe()
TILTED = 0.12  # a view counts as tilted from this much foreshortening between opposite sides
_NAME = re.compile(r"^img_(\d{4})\.png$")


def describe(spec: BoardSpec, det: Detection, size) -> Optional[np.ndarray]:
    """A view in a few numbers, so that two views can be told apart: where the board is in the
    picture, how large, how its opposite sides differ (tilt) and how it is turned."""
    q = boards.board_quad(spec, det)
    if q is None or not np.all(np.isfinite(q)):
        return None
    w, h = size
    top, right = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
    bottom, left = np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0])
    if min(top, right, bottom, left) < 1.0:
        return None
    d1, d2 = q[2] - q[0], q[3] - q[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    a = np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0])
    return np.array([q[:, 0].mean() / w, q[:, 1].mean() / h, np.sqrt(area / (w * h)),
                     0.5 * np.log(left / right), 0.5 * np.log(top / bottom),
                     0.15 * np.cos(a), 0.15 * np.sin(a)])


class Session:
    """The pictures of one calibration in a directory: img_NNNN.png and session.json."""

    def __init__(self, spec: BoardSpec, out_dir: str, target: int = 30, min_corners: int = 12,
                 steady_px_s: float = STEADY_PX_S, novelty: float = NOVELTY, source: Optional[dict] = None,
                 log: Callable[[str], None] = print):
        self.spec, self.dir, self.target = spec, out_dir, target
        self.min_corners, self.steady_px_s, self.novelty = min_corners, steady_px_s, novelty
        self.source = source or {}
        self.log = log
        self.size = None
        self.names: List[str] = []
        self.views: List[View] = []
        self.desc: List[np.ndarray] = []
        self.auto = True
        self._prev = None  # (time, detection) of the frame before
        self._steady = 0
        self._last_take = 0.0
        self._flash = 0.0
        os.makedirs(out_dir, exist_ok=True)
        self._resume()

    # --- the directory --------------------------------------------------------------------
    def _resume(self) -> None:
        for n in sorted(os.listdir(self.dir)):
            if not _NAME.match(n):
                continue
            img = cv2.imread(os.path.join(self.dir, n), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if self.size is None:
                self.size = (img.shape[1], img.shape[0])
            det = boards.detect(self.spec, img, self.min_corners)
            d = describe(self.spec, det, self.size) if det is not None else None
            self._add(n, det, d)
        if self.names:
            self.log("resumed: %d pictures already in %s" % (len(self.names), self.dir))

    def _add(self, name: str, det: Optional[Detection], d: Optional[np.ndarray]) -> None:
        self.names.append(name)
        ok = det is not None and d is not None
        self.views.append(View(name, det.ids, det.object_points(self.spec), det.image) if ok
                          else View(name, np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 2))))
        self.desc.append(d if ok else np.full(7, np.nan))

    def _write_session(self) -> None:
        s = {"board": self.spec.to_dict(), "source": self.source,
             "image_size": list(self.size) if self.size else None,
             "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "pictures": len(self.names)}
        tmp = os.path.join(self.dir, ".session.json.tmp")
        with open(tmp, "w") as f:
            json.dump(s, f, indent=2)
            f.write("\n")
        os.replace(tmp, os.path.join(self.dir, "session.json"))

    def take(self, frame: np.ndarray, det: Detection) -> str:
        nxt = 1 + max([int(_NAME.match(n).group(1)) for n in self.names], default=0)
        name = "img_%04d.png" % nxt
        cv2.imwrite(os.path.join(self.dir, name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self._add(name, det, describe(self.spec, det, self.size))
        self._write_session()
        self._last_take = self._flash = time.monotonic()
        return name

    def drop_last(self) -> Optional[str]:
        if not self.names:
            return None
        name = self.names.pop()
        self.views.pop(), self.desc.pop()
        try:
            os.remove(os.path.join(self.dir, name))
        except OSError:
            pass
        self._write_session()
        return name

    # --- one frame ------------------------------------------------------------------------
    def grid(self) -> np.ndarray:
        return calibrate.coverage_grid(self.views, self.size)

    def _what_next(self) -> str:
        if len(self.names) >= self.target:
            return "enough views: finish (q), or add more"
        _cov, _border, empty = calibrate._coverage(self.grid())
        d = np.array([x for x in self.desc if np.all(np.isfinite(x))]).reshape(-1, 7)
        n = len(self.names)
        if n >= 6 and len(d):
            if np.mean(np.max(np.abs(d[:, 3:5]), axis=1) > TILTED) < 0.4:
                return "tilt the board 30 to 45 degrees, each way"
            if d[:, 2].max() < 0.5:
                return "closer: let the board fill the picture"
            if d[:, 2].min() > 0.35:
                return "further away too"
        if empty:
            return "board to the " + ", ".join(empty[:3])
        return "any new position or tilt"

    def consider(self, frame: np.ndarray, now: Optional[float] = None) -> dict:
        """Looks at one frame and takes it if it is a view worth having. Returns what it saw:
        det, taken (the file's name or None), text (what the operator should do)."""
        now = time.monotonic() if now is None else now
        size = (frame.shape[1], frame.shape[0])
        if self.size is None:
            self.size = size
        elif size != self.size:
            raise ValueError("the picture is %dx%d, this session's are %dx%d: another directory for another "
                             "geometry" % (size + self.size))
        det = boards.detect(self.spec, frame, self.min_corners)
        if det is not None and not calibrate._spans_a_plane(det.object_points(self.spec)):
            det = None
        out = {"det": det, "taken": None}
        if det is None:
            self._prev, self._steady = None, 0
            out["text"] = "no board: " + self._what_next()
            return out
        speed = None
        if self._prev is not None and now > self._prev[0]:
            _a, ia, ib = np.intersect1d(self._prev[1].ids, det.ids, return_indices=True)
            if len(ia) >= 4:
                move = np.median(np.linalg.norm(self._prev[1].image[ia] - det.image[ib], axis=1))
                speed = float(move / (now - self._prev[0]))
        self._prev = (now, det)
        self._steady = self._steady + 1 if speed is not None and speed < self.steady_px_s else 0
        out["speed_px_s"] = speed
        d = describe(self.spec, det, self.size)
        known = np.array([x for x in self.desc if np.all(np.isfinite(x))]).reshape(-1, 7)
        new = d is not None and (len(known) == 0 or np.linalg.norm(known - d, axis=1).min() > self.novelty)
        out["new"] = bool(new)
        if not new:
            out["text"] = "have this one: " + self._what_next()
        elif self._steady < 2:
            out["text"] = "hold still"
        elif not self.auto:
            out["text"] = "space takes it"
        elif now - self._last_take < 0.6:
            out["text"] = "..."
        else:
            out["taken"] = self.take(frame, det)
            out["text"] = "taken %d" % len(self.names)
        return out

    # --- what the operator sees -------------------------------------------------------------
    def draw(self, frame: np.ndarray, state: dict, width: int = 1280) -> np.ndarray:
        img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        s = min(1.0, width / img.shape[1])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else img.copy()
        h, w = img.shape[:2]
        if self.size is not None:  # the cells that have their corners, in green
            have = (self.grid() >= 2).astype(np.uint8)
            mask = cv2.resize(have, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            tint = img.copy()
            tint[mask] = (0.7 * tint[mask] + 0.3 * np.array([60, 200, 60])).astype(np.uint8)
            img = tint
            rows, cols = have.shape
            for c in range(1, cols):
                cv2.line(img, (c * w // cols, 0), (c * w // cols, h), (90, 90, 90), 1)
            for r in range(1, rows):
                cv2.line(img, (0, r * h // rows), (w, r * h // rows), (90, 90, 90), 1)
        det = state.get("det")
        if det is not None:
            color = (60, 200, 60) if state.get("new") else (160, 160, 160)
            for p in det.image * s:
                cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 3, color, -1, cv2.LINE_AA)
        if time.monotonic() - self._flash < 0.35:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 200, 60), 14)
        head = "%d/%d  %s" % (len(self.names), self.target, state.get("text", ""))
        for text, y in ((head, 34), ("space take  d drop last  a auto %s  q end" % ("on" if self.auto else "off"), h - 14)):
            scale = 0.9 if y < 100 else 0.55
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def run(session: Session, source, headless: bool = False, duration: Optional[float] = None,
        log: Callable[[str], None] = print) -> int:
    """Takes views from the source until the operator ends it (window), or until the target is
    reached, the time is up or Ctrl-C (headless). Returns the number of pictures in the session."""
    title = "camcal capture"
    preview = os.path.join(session.dir, "preview.jpg")
    start = last_preview = time.monotonic()
    said = None
    try:
        while True:
            frame, _seq = source.read()
            state = session.consider(frame)
            if state["taken"]:
                log("  %s  %d corners  (%d/%d)" % (state["taken"], len(state["det"]), len(session.names),
                                                   session.target))
            now = time.monotonic()
            if headless:
                if now - last_preview > 0.4:
                    tmp = os.path.join(session.dir, ".preview.tmp.jpg")
                    cv2.imwrite(tmp, session.draw(frame, state, 960), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    os.replace(tmp, preview)
                    last_preview = now
                    if state["text"] != said and not state["taken"]:
                        said = state["text"]
                        log("  " + said)
                if len(session.names) >= session.target:
                    break
            else:
                cv2.imshow(title, session.draw(frame, state))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" ") and state["det"] is not None:
                    log("  %s  %d corners  (by hand)" % (session.take(frame, state["det"]), len(state["det"])))
                elif key == ord("d"):
                    log("  dropped %s" % session.drop_last())
                elif key == ord("a"):
                    session.auto = not session.auto
            if duration is not None and now - start > duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not headless:
            cv2.destroyAllWindows()
        elif os.path.exists(preview):
            os.remove(preview)
    return len(session.names)


# --- the headless session's preview, on another machine ---------------------------------------

# sends preview.jpg each time it changes; ends with the ssh session (its parent then changes),
# not only when a write finds the pipe closed: a stopped capture writes no more files
_REMOTE = ("import sys,time,os\np=sys.argv[1]\nm=0\npp=os.getppid()\nwhile os.getppid()==pp:\n"
           "    try:\n        t=os.stat(p).st_mtime_ns\n"
           "        if t!=m:\n            b=open(p,'rb').read();m=t\n"
           "            sys.stdout.buffer.write(len(b).to_bytes(4,'big')+b);sys.stdout.buffer.flush()\n"
           "    except OSError: pass\n    time.sleep(0.2)\n")


def preview(target: str, ssh_options: Optional[List[str]] = None) -> None:
    """Shows preview.jpg of a headless session as it changes. target is a directory here, or
    user@host:directory: then one ssh session reads the file there (no port is opened)."""
    import subprocess

    host, sep, path = target.partition(":")
    if not sep or os.path.exists(target):
        last = 0
        while True:
            p = os.path.join(target, "preview.jpg")
            if os.path.exists(p) and os.stat(p).st_mtime_ns != last:
                last = os.stat(p).st_mtime_ns
                img = cv2.imread(p)
                if img is not None:
                    cv2.imshow("camcal preview", img)
            if cv2.waitKey(150) & 0xFF in (ord("q"), 27):
                return
    import shlex

    cmd = ["ssh"] + list(ssh_options or []) + [host, "python3 -u -c %s %s" % (
        shlex.quote(_REMOTE), shlex.quote(path.rstrip("/") + "/preview.jpg"))]
    import threading

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    box = {"jpg": None, "end": False}

    def pump():  # the pipe is read here so that the window keeps answering
        while True:
            head = proc.stdout.read(4)
            if len(head) < 4:
                box["end"] = True
                return
            box["jpg"] = proc.stdout.read(int.from_bytes(head, "big"))

    threading.Thread(target=pump, daemon=True).start()
    try:
        while not box["end"]:
            data, box["jpg"] = box["jpg"], None
            if data:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow("camcal preview", img)
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                return
    finally:
        proc.terminate()
        cv2.destroyAllWindows()
