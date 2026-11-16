"""Where the pictures come from.

  csi[:N]              the Jetson's CSI camera N through Argus (GStreamer nvarguscamerasrc)
  v4l2:/dev/videoN     a USB camera; its autofocus is turned off, a calibration is for one focus
  serve://host[:port]  the raw stream of `ucpeer-detect --serve` (port 9000): the very picture
                       the application works on, its rotation and sensor mode included
  N                    camera N as OpenCV numbers them (a laptop's own camera: 0)
  anything else        what cv2.VideoCapture opens: an RTSP URL, a video file

A calibration is for one geometry of the picture. Calibrate with the size, the sensor mode and
the rotation the application uses; serve:// takes all three from the application itself.
--rotate is counterclockwise, as ucpeer-detect's.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

SERVE_PORT = 9000
_MAX_HEADER = 64 * 1024
_MAX_PAYLOAD = 64 << 20
_ROTATE = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def csi_pipeline(sensor: int, width: int, height: int, fps: int, rotate: int = 0,
                 sensor_mode: Optional[int] = None) -> str:
    """The GStreamer pipeline that gives the CSI camera's picture as BGR, only the newest frame
    kept. nvvidconv turns it on the way: flip-method 1, 2, 3 are 90, 180, 270 counterclockwise."""
    flip = {0: 0, 90: 1, 180: 2, 270: 3}[rotate]
    mode = "" if sensor_mode is None else " sensor-mode=%d" % sensor_mode
    return ("nvarguscamerasrc sensor-id=%d%s ! video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1 ! "
            "nvvidconv flip-method=%d ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false" % (sensor, mode, width, height, fps, flip))


# --- the raw stream of ucpeer-detect --serve ----------------------------------------------


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("the stream ended")
        got += k
    return bytes(buf)


def read_message(sock: socket.socket) -> Tuple[dict, bytes]:
    """One message: "UCRS", header bytes and payload bytes (u32 big endian), JSON header, payload."""
    prefix = _read_exact(sock, 12)
    if prefix[:4] != b"UCRS":
        raise ConnectionError("not a ucpeer-detect raw stream")
    hlen, plen = struct.unpack(">II", prefix[4:])
    if hlen > _MAX_HEADER or plen > _MAX_PAYLOAD:
        raise ConnectionError("a message of %d + %d bytes: not this stream" % (hlen, plen))
    header = json.loads(_read_exact(sock, hlen).decode("utf-8"))
    return header, _read_exact(sock, plen) if plen else b""


def nv12_luma(header: dict, payload: bytes) -> np.ndarray:
    """The Y plane of a frame message: the grey picture exactly as the sensor pipeline gave it."""
    w, h = int(header["width"]), int(header["height"])
    if header.get("format", "NV12") != "NV12" or len(payload) < w * h:
        raise ConnectionError("a frame of %d bytes is not %dx%d NV12" % (len(payload), w, h))
    return np.frombuffer(payload, np.uint8, w * h).reshape(h, w).copy()


# --- sources ----------------------------------------------------------------------------


class Source:
    """A camera read in its own thread: read() gives the newest frame and never an old one."""

    def __init__(self, info: dict):
        self.info = info
        self._lock = threading.Condition()
        self._frame = None
        self._seq = 0
        self._taken = 0
        self._error: Optional[str] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Source":
        self._thread.start()
        return self

    def _grab(self) -> Optional[np.ndarray]:  # one frame, blocking; None at the end
        raise NotImplementedError

    def _release(self) -> None:
        pass

    def _run(self) -> None:
        try:
            while not self._stop:
                frame = self._grab()
                if frame is None:
                    self._error = "the camera stopped giving pictures"
                    break
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._lock.notify_all()
        except Exception as e:  # the reader's thread must say why it ended
            self._error = str(e)
        finally:
            with self._lock:
                self._stop = True
                self._lock.notify_all()
            self._release()

    def read(self, timeout: float = 5.0) -> Tuple[np.ndarray, int]:
        """(frame, its number). Raises RuntimeError when the source has ended or gives nothing."""
        end = time.time() + timeout
        with self._lock:
            while self._seq == self._taken and not self._stop:
                left = end - time.time()
                if left <= 0:
                    raise RuntimeError("no picture in %.0f s from %s" % (timeout, self.info.get("source")))
                self._lock.wait(left)
            if self._seq == self._taken:
                raise RuntimeError(self._error or "the source ended")
            self._taken = self._seq
            return self._frame, self._seq

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=3.0)


class _Capture(Source):
    def __init__(self, cap, info: dict, rotate: int = 0):
        super().__init__(info)
        self._cap = cap
        self._rot = _ROTATE.get(rotate)

    def _grab(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.rotate(frame, self._rot) if self._rot is not None else frame

    def _release(self):
        self._cap.release()


class _Serve(Source):
    def __init__(self, sock: socket.socket, info: dict):
        super().__init__(info)
        self._sock = sock

    def _grab(self):
        while True:
            header, payload = read_message(self._sock)
            kind = header.get("type")
            if kind == "hello":
                self.info["server"] = {k: header.get(k) for k in ("source", "format", "fps")}
            elif kind == "frame":
                return nv12_luma(header, payload)

    def _release(self):
        try:
            self._sock.close()
        except OSError:
            pass


def open_source(text: str, size: Tuple[int, int] = (1920, 1080), fps: int = 30, rotate: int = 0,
                sensor_mode: Optional[int] = None, focus: Optional[int] = None) -> Source:
    if rotate not in (0, 90, 180, 270):
        raise ValueError("--rotate 0, 90, 180 or 270 (counterclockwise)")
    w, h = size
    info = {"source": text, "requested_size": [w, h], "fps": fps, "rotate": rotate}
    if text == "csi" or text.startswith("csi:"):
        sensor = int(text[4:] or 0) if text.startswith("csi:") else 0
        pipe = csi_pipeline(sensor, w, h, fps, rotate, sensor_mode)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("the CSI camera %d did not open. Another program may be using it "
                               "(ucpeer-detect: ./run.sh --stop), or this OpenCV has no GStreamer" % sensor)
        info.update(kind="csi", sensor=sensor, sensor_mode=sensor_mode, pipeline=pipe)
        return _Capture(cap, info).start()  # turned by nvvidconv already
    if text.startswith("serve://"):
        host, _, port = text[8:].partition(":")
        sock = socket.create_connection((host, int(port or SERVE_PORT)), timeout=5.0)
        sock.settimeout(10.0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        info.update(kind="serve", note="the application's own picture: rotation and mode are the server's")
        info.pop("requested_size"), info.pop("rotate"), info.pop("fps")
        return _Serve(sock, info).start()
    if text.startswith("v4l2:"):
        cap = cv2.VideoCapture(text[5:], cv2.CAP_V4L2)
        kind = "v4l2"
    elif text.isdigit():
        cap = cv2.VideoCapture(int(text))
        kind = "camera"
    else:
        cap = cv2.VideoCapture(text)
        kind = "stream"
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % text)
    if kind != "stream":
        if kind == "v4l2":  # 1080p at 30 from a USB camera is MJPEG; raw YUYV there is 5 fps
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, focus)
        info["focus"] = cap.get(cv2.CAP_PROP_FOCUS)
        info["autofocus"] = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    info.update(kind=kind)
    return _Capture(cap, info, rotate).start()
