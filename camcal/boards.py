"""The calibration target: what it is, how it is printed, and how it is found in a picture.

A board is squares_x by squares_y squares of square_mm. Its frame is the one OpenCV gives a
board: the origin at the pattern's top left corner as printed, x along squares_x, y along
squares_y (down the page), z into the board. A point of it is a chessboard corner between four
squares; corner i is at ((i % (squares_x - 1)) + 1, (i // (squares_x - 1)) + 1) squares.

  charuco     a chessboard with an ArUco marker in every white square. Each corner is known by
              the markers around it, so a view with half the board outside the picture still
              counts: that is what fills the picture's corners, where the distortion is.
  chessboard  the plain one. Needs all of it in the picture. (For the extrinsics use charuco:
              a chessboard cannot tell which way round it lies.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

PAPER_MM = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "A0": (841.0, 1189.0),
    "LETTER": (215.9, 279.4),
}
# squares (x, y) and square size per paper: an odd by even count, so that a chessboard of them
# has one way round, and a margin no printer clips
PAPER_DEFAULT = {
    "A4": (7, 10, 25.0),
    "A3": (9, 12, 30.0),
    "A2": (9, 12, 42.0),
    "A1": (9, 12, 60.0),
    "A0": (9, 12, 85.0),
    "LETTER": (7, 9, 25.0),
}


def dictionary_bits(name: str) -> int:
    """Bits along one side of a marker of this dictionary (DICT_4X4_250: 4)."""
    return int(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)).markerSize)


@dataclass
class BoardSpec:
    type: str = "charuco"  # charuco | chessboard
    squares_x: int = 7
    squares_y: int = 10
    square_mm: float = 25.0
    marker_mm: float = 18.75  # charuco: the marker's side, its black border included
    dictionary: str = "DICT_4X4_250"
    note: str = ""
    _detector: object = field(default=None, init=False, repr=False, compare=False)

    # --- construction -------------------------------------------------------------------
    def __post_init__(self):
        if self.type not in ("charuco", "chessboard"):
            raise ValueError("board type '%s' (charuco, chessboard)" % self.type)
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a board needs at least 3 x 3 squares")
        if self.type == "charuco":
            if not hasattr(cv2.aruco, self.dictionary):
                raise ValueError("no ArUco dictionary '%s'" % self.dictionary)
            if not 0.3 * self.square_mm <= self.marker_mm < self.square_mm:
                raise ValueError("marker_mm must be between 0.3 and 1 square")
            capacity = len(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary)).bytesList)
            if self.marker_count() > capacity:
                raise ValueError(
                    "%s has %d markers, this board needs %d" % (self.dictionary, capacity, self.marker_count())
                )

    @staticmethod
    def default_marker_mm(square_mm: float, dictionary: str) -> float:
        """A marker with one of its own cells of white around it: square = bits + 4 cells."""
        bits = dictionary_bits(dictionary)
        return square_mm * (bits + 2) / (bits + 4)

    def to_dict(self) -> dict:
        d = {"type": self.type, "squares_x": self.squares_x, "squares_y": self.squares_y,
             "square_mm": round(self.square_mm, 4)}
        if self.type == "charuco":
            d.update(marker_mm=round(self.marker_mm, 4), dictionary=self.dictionary)
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BoardSpec":
        known = {k: d[k] for k in ("type", "squares_x", "squares_y", "square_mm", "marker_mm", "dictionary", "note") if k in d}
        return cls(**known)

    @classmethod
    def load(cls, path: str) -> "BoardSpec":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d.get("board", d))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"board": self.to_dict()}, f, indent=2)
            f.write("\n")

    def with_square_mm(self, square_mm: float) -> "BoardSpec":
        """The same board as it came out of the printer: every length scaled with the square."""
        s = square_mm / self.square_mm
        return BoardSpec(self.type, self.squares_x, self.squares_y, square_mm, self.marker_mm * s, self.dictionary, self.note)

    def describe(self) -> str:
        if self.type == "charuco":
            return "ChArUco %dx%d, square %.2f mm, marker %.2f mm, %s" % (
                self.squares_x, self.squares_y, self.square_mm, self.marker_mm, self.dictionary)
        return "chessboard %dx%d squares, square %.2f mm" % (self.squares_x, self.squares_y, self.square_mm)

    # --- geometry -----------------------------------------------------------------------
    @property
    def size_mm(self) -> Tuple[float, float]:
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm

    def corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def marker_count(self) -> int:
        return len(self._marker_squares())

    def corner_points(self) -> np.ndarray:
        """Every chessboard corner in the board's frame, metres: (corner_count, 3)."""
        nx = self.squares_x - 1
        i = np.arange(self.corner_count())
        s = self.square_mm / 1000.0
        return np.stack([(i % nx + 1) * s, (i // nx + 1) * s, np.zeros(len(i))], axis=1)

    def outline(self) -> np.ndarray:
        """The pattern's four outer corners, metres, clockwise from the origin."""
        w, h = self.size_mm
        return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64) / 1000.0

    def _square_is_black(self, ix: int, iy: int) -> bool:
        # OpenCV's ChArUco pattern (not the legacy one): the top left square is black
        return (ix + iy) % 2 == 0

    def _marker_squares(self) -> List[Tuple[int, int]]:
        """The white squares, row by row: marker id k is in the k-th of them."""
        return [(ix, iy) for iy in range(self.squares_y) for ix in range(self.squares_x)
                if not self._square_is_black(ix, iy)]

    def cv_board(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_mm / 1000.0, self.marker_mm / 1000.0, d)

    def black_rects(self) -> List[Tuple[float, float, float, float]]:
        """All that is printed black: (x, y, w, h) in mm from the pattern's top left, y down.

        A marker is its black cells, run by run, so that nothing is resampled on the way to
        paper: what a printer gets is rectangles.
        """
        s = self.square_mm
        rects = [(ix * s, iy * s, s, s) for iy in range(self.squares_y) for ix in range(self.squares_x)
                 if self._square_is_black(ix, iy)]
        if self.type != "charuco":
            return rects
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        cells = d.markerSize + 2
        cell = self.marker_mm / cells
        off = (s - self.marker_mm) / 2.0
        for k, (ix, iy) in enumerate(self._marker_squares()):
            bits = d.generateImageMarker(k, cells)  # one pixel a cell, the border with it
            x0, y0 = ix * s + off, iy * s + off
            for r in range(cells):
                c = 0
                while c < cells:
                    if bits[r, c] != 0:
                        c += 1
                        continue
                    e = c
                    while e < cells and bits[r, e] == 0:
                        e += 1
                    rects.append((x0 + c * cell, y0 + r * cell, (e - c) * cell, cell))
                    c = e
        return rects

    def render(self, px_per_mm: float, margin_mm: float = 0.0) -> np.ndarray:
        """The board as a picture (white margin around it), 8 bit. Exact where a marker's cell
        is a whole number of pixels."""
        w, h = self.size_mm
        m = int(round(margin_mm * px_per_mm))
        img = np.full((int(round(h * px_per_mm)) + 2 * m, int(round(w * px_per_mm)) + 2 * m), 255, np.uint8)
        for x, y, rw, rh in self.black_rects():
            x0, y0 = int(round(x * px_per_mm)) + m, int(round(y * px_per_mm)) + m
            x1, y1 = int(round((x + rw) * px_per_mm)) + m, int(round((y + rh) * px_per_mm)) + m
            img[y0:y1, x0:x1] = 0
        return img


def board_for_paper(paper: str, type: str = "charuco", squares: Optional[Tuple[int, int]] = None,
                    square_mm: Optional[float] = None, marker_mm: Optional[float] = None,
                    dictionary: str = "DICT_4X4_250") -> BoardSpec:
    """The board that fills a sheet of this paper (or the given squares on it)."""
    paper = paper.upper()
    if paper not in PAPER_MM:
        raise ValueError("paper '%s' (%s)" % (paper, ", ".join(PAPER_MM)))
    dx, dy, ds = PAPER_DEFAULT[paper]
    nx, ny = squares or (dx, dy)
    if square_mm is None:
        if squares is None:
            square_mm = ds
        else:  # the largest whole millimetre that leaves the margins
            pw, ph = PAPER_MM[paper]
            square_mm = float(int(min((pw - 2 * MARGIN_MM) / nx, (ph - 2 * MARGIN_MM - FOOTER_MM) / ny)))
    if marker_mm is None:
        marker_mm = BoardSpec.default_marker_mm(square_mm, dictionary)
    return BoardSpec(type, nx, ny, square_mm, marker_mm, dictionary)


def board_for_screen(width_px: int, height_px: int, ppi: float, type: str = "charuco",
                     dictionary: str = "DICT_4X4_250", fill: float = 0.92) -> Tuple[BoardSpec, int]:
    """A board for a flat display: every cell a whole number of pixels, so nothing is blurred.
    Returns it with the pixels of one square. Its size in mm is as good as the ppi given; the
    intrinsics do not depend on that, only distances do."""
    bits = dictionary_bits(dictionary)
    cells = bits + 4  # one square
    landscape = width_px >= height_px
    nx, ny = (10, 7) if landscape else (7, 10)
    unit = int(min(width_px * fill / (nx * cells), height_px * fill / (ny * cells)))
    if unit < 2:
        raise ValueError("the screen is too small for a %dx%d board" % (nx, ny))
    square_px = unit * cells
    square_mm = square_px * 25.4 / ppi
    spec = BoardSpec(type, nx, ny, square_mm, square_mm * (bits + 2) / cells, dictionary,
                     note="shown on a %dx%d display at %.1f ppi" % (width_px, height_px, ppi))
    return spec, square_px


# --- printing ---------------------------------------------------------------------------

MARGIN_MM = 10.0  # least white around the pattern
FOOTER_MM = 14.0  # the line of text and the 100 mm check under it


def _layout(spec: BoardSpec, paper_mm: Tuple[float, float]):
    """Where the pattern sits on the page (mm from the page's top left), landscape if it fits
    only so."""
    w, h = spec.size_mm
    pw, ph = paper_mm
    if w > pw - 2 * MARGIN_MM or h > ph - 2 * MARGIN_MM - FOOTER_MM:
        if h <= pw - 2 * MARGIN_MM and w <= ph - 2 * MARGIN_MM - FOOTER_MM:
            raise ValueError("the board fits this paper only turned: swap squares_x and squares_y")
        raise ValueError("a %.0f x %.0f mm board does not fit %.0f x %.0f mm paper with its margins" % (w, h, pw, ph))
    x0 = (pw - w) / 2.0
    y0 = (ph - FOOTER_MM - h) / 2.0
    return x0, y0


def write_pdf(spec: BoardSpec, path: str, paper: str = "A4") -> None:
    """A one page PDF of the board at its true size: vector rectangles, nothing rasterised.
    Under it a line that says what it is and a 100 mm rule to check the print against."""
    paper_mm = PAPER_MM[paper.upper()]
    x0, y0 = _layout(spec, paper_mm)
    k = 72.0 / 25.4  # mm -> PDF points
    ph = paper_mm[1]
    ops = ["0 g"]
    # one path, one fill: a renderer draws cells that touch without a seam between them
    for x, y, w, h in spec.black_rects():
        ops.append("%.4f %.4f %.4f %.4f re" % ((x0 + x) * k, (ph - (y0 + y) - h) * k, w * k, h * k))
    ops.append("f")
    # the 100 mm check: a rule with a tick every 10 mm
    rx, ry = x0, y0 + spec.size_mm[1] + 6.0
    ops.append("0.2 w 0 G")
    ops.append("%.4f %.4f m %.4f %.4f l S" % (rx * k, (ph - ry) * k, (rx + 100.0) * k, (ph - ry) * k))
    for i in range(11):
        t = 2.0 if i % 5 == 0 else 1.0
        ops.append("%.4f %.4f m %.4f %.4f l S" % ((rx + 10.0 * i) * k, (ph - ry) * k, (rx + 10.0 * i) * k, (ph - ry - t) * k))
    text = "%s | print at 100%% (no fit to page) | the rule is 100 mm: measure it" % spec.describe()
    esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    ops.append("BT /F1 7 Tf %.4f %.4f Td (%s) Tj ET" % (rx * k, (ph - ry - 5.0) * k, esc))
    content = "\n".join(ops).encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] /Contents 4 0 R "
         "/Resources << /Font << /F1 5 0 R >> >> >>" % (paper_mm[0] * k, paper_mm[1] * k)).encode("ascii"),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % (i + 1) + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1) + b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)


TILE_SQUARES = (2, 3)  # squares on one sheet of a board printed in sheets: across, down


def tile_square_mm(paper: str) -> float:
    """The largest square, in whole 5 mm, of which 2 by 3 fit a sheet with room to print."""
    pw, ph = PAPER_MM[paper.upper()]
    return float(int(min((pw - 2 * MARGIN_MM) / TILE_SQUARES[0], (ph - 2 * MARGIN_MM - 6.0) / TILE_SQUARES[1]) / 5) * 5)


def write_pdf_tiles(spec: BoardSpec, path: str, paper: str = "A3") -> int:
    """A board larger than the printer's paper, as one page per sheet: each sheet carries 2 by
    3 whole squares, so the sheets meet along the squares' edges and a cut that is a little off
    shows at once. A thin outline says where to cut; the corner of every sheet names its place
    (row and column from the board's top left, the corner beside marker 0). Returns the pages."""
    tx, ty = TILE_SQUARES
    if spec.squares_x % tx or spec.squares_y % ty:
        raise ValueError("a board in sheets has %d squares across and %d down a sheet" % (tx, ty))
    pw, ph = PAPER_MM[paper.upper()]
    s = spec.square_mm
    w, h = tx * s, ty * s
    if w > pw - 10 or h > ph - 24:
        raise ValueError("%d x %d squares of %.0f mm do not fit a sheet of %s" % (tx, ty, s, paper))
    x0, y0 = (pw - w) / 2.0, 6.0
    k = 72.0 / 25.4
    cols, rows = spec.squares_x // tx, spec.squares_y // ty
    rects = spec.black_rects()
    pages = []
    for r in range(rows):
        for c in range(cols):
            ox, oy = c * w, r * h
            ops = ["0 g"]
            for x, y, rw, rh in rects:
                if ox - 1e-6 <= x < ox + w - 1e-6 and oy - 1e-6 <= y < oy + h - 1e-6:
                    ops.append("%.4f %.4f %.4f %.4f re" % ((x0 + x - ox) * k, (ph - (y0 + y - oy) - rh) * k, rw * k, rh * k))
            ops.append("f")
            ops.append("0.15 w 0.5 G %.4f %.4f %.4f %.4f re S" % (x0 * k, (ph - y0 - h) * k, w * k, h * k))  # cut here
            ry = y0 + h + 7.0
            ops.append("0.2 w 0 G %.4f %.4f m %.4f %.4f l S" % (x0 * k, (ph - ry) * k, (x0 + 100.0) * k, (ph - ry) * k))
            for i in range(11):
                t = 2.0 if i % 5 == 0 else 1.0
                ops.append("%.4f %.4f m %.4f %.4f l S" % ((x0 + 10.0 * i) * k, (ph - ry) * k, (x0 + 10.0 * i) * k, (ph - ry - t) * k))
            text = ("SHEET row %d of %d, column %d of %d (row 1, column 1 is the top left) | %s | print at 100%%, cut on "
                    "the outline | the rule is 100 mm" % (r + 1, rows, c + 1, cols, spec.describe()))
            esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append("BT /F1 6 Tf %.4f %.4f Td (%s) Tj ET" % (x0 * k, (ph - ry - 5.0) * k, esc))
            pages.append("\n".join(ops).encode("ascii"))
    n = len(pages)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            ("<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join("%d 0 R" % (4 + 2 * i) for i in range(n)), n)).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, content in enumerate(pages):
        objs.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] /Contents %d 0 R "
                     "/Resources << /Font << /F1 3 0 R >> >> >>" % (pw * k, ph * k, 5 + 2 * i)).encode("ascii"))
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % (i + 1) + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1) + b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)
    return n


def write_svg(spec: BoardSpec, path: str, paper: str = "A4") -> None:
    paper_mm = PAPER_MM[paper.upper()]
    x0, y0 = _layout(spec, paper_mm)
    d = "".join("M%.4f %.4fh%.4fv%.4fh%.4fz" % (x0 + x, y0 + y, w, h, -w) for x, y, w, h in spec.black_rects())
    ry = y0 + spec.size_mm[1] + 6.0
    ticks = "".join("M%.4f %.4fv%.1f" % (x0 + 10.0 * i, ry, 2.0 if i % 5 == 0 else 1.0) for i in range(11))
    with open(path, "w") as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" viewBox="0 0 %g %g">\n'
            '<rect width="100%%" height="100%%" fill="#fff"/>\n<path d="%s" fill="#000"/>\n'
            '<path d="M%.4f %.4fh100%s" stroke="#000" stroke-width="0.07" fill="none"/>\n'
            '<text x="%.4f" y="%.4f" font-family="Helvetica,Arial,sans-serif" font-size="2.5">%s | print at 100%% | '
            "the rule is 100 mm</text>\n</svg>\n"
            % (paper_mm[0], paper_mm[1], paper_mm[0], paper_mm[1], d, x0, ry, ticks, x0, ry + 5.0, spec.describe()))


# --- detection --------------------------------------------------------------------------


@dataclass
class Detection:
    """The board's corners found in one picture."""

    ids: np.ndarray  # (N,) corner numbers
    image: np.ndarray  # (N, 2) pixels, float64
    markers: int = 0  # charuco: markers read

    def __len__(self) -> int:
        return len(self.ids)

    def object_points(self, spec: BoardSpec) -> np.ndarray:
        return spec.corner_points()[self.ids]


MIN_SQUARE_PX = 30.0  # charuco: a corner whose squares are smaller in the picture is dropped
_WINDOW = 0.7  # the refinement window's half side, as a share of the clear distance around a corner
_AGREE_PX = 0.25  # two window sizes must put a corner within this of each other
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
_OFFSET: List[np.ndarray] = []


def _marker_detector(spec: BoardSpec):
    if spec._detector is None:
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # OpenCV up to 4.8 takes this window in pixels whatever the marker's size: at 5 a marker
        # corner slides onto the chessboard corner 4 px away. 4.9 on scales it by the marker.
        dp.cornerRefinementWinSize = 3
        dp.cornerRefinementMaxIterations = 50
        dp.cornerRefinementMinAccuracy = 0.01
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
        spec._detector = cv2.aruco.ArucoDetector(d, dp)
    return spec._detector


def _clear_distance(p: np.ndarray, quad: np.ndarray) -> float:
    """From a point to the nearest edge of a marker's outline (4, 2), pixels."""
    a, b = quad, np.roll(quad, -1, axis=0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())


def _refine(gray: np.ndarray, pts: np.ndarray, wins: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts)
    for w in np.unique(wins):  # one call for each window size
        k = wins == w
        out[k] = cv2.cornerSubPix(gray, pts[k].reshape(-1, 1, 2).astype(np.float32),
                                  (int(w), int(w)), (-1, -1), _SUBPIX).reshape(-1, 2)
    return out


def _charuco_corners(spec: BoardSpec, gray: np.ndarray, min_square_px: float):
    """(ids, points, markers read) of a ChArUco board, the corners measured here.

    OpenCV's own ChArUco step is not used: in 4.8 (JetPack 6) it returns corners half a pixel
    off and, on tilted views, 6 in 100 of them more than half a pixel wrong (rendered truth,
    1280x720: 0.035 px median but 1.2 px rms; OpenCV 5 on the same pictures 0.064 px rms).
    Only its marker reader is: a marker says which corner this is and about where. Then

      - a corner is placed by the two markers beside it (their eight corners, one homography),
      - refined on the picture in a window that stays clear of those markers, its half side
        0.7 of the distance to the nearest marker edge as that edge lies in the picture,
      - refined again with another window size: a corner the two do not agree on is left out.

    The same code and the same result on OpenCV 4.8 and 5: 0.05 to 0.07 px rms, none of
    2600 corners beyond 0.5 px.
    """
    mc, mids, _rej = _marker_detector(spec).detectMarkers(gray)
    if mids is None or len(mids) == 0:
        return None
    s, mm = spec.square_mm, spec.marker_mm
    g = (s - mm) / 2.0
    squares = spec._marker_squares()
    where = {q: i for i, q in enumerate(squares)}
    found = {}
    for c, i in zip(mc, mids.reshape(-1)):
        if 0 <= int(i) < len(squares):
            found.setdefault(int(i), c.reshape(4, 2).astype(np.float64))
    nx = spec.squares_x - 1
    h, w_img = gray.shape[:2]
    min_clear = min_square_px * g / s
    ids, start, wins = [], [], []
    for cid in range(spec.corner_count()):
        cx, cy = cid % nx + 1, cid // nx + 1
        src, quads = [], []
        for ix, iy in ((cx - 1, cy - 1), (cx, cy - 1), (cx - 1, cy), (cx, cy)):
            m = where.get((ix, iy))
            if m is None or m not in found:
                continue
            x0, y0 = ix * s + g, iy * s + g
            src += [(x0, y0), (x0 + mm, y0), (x0 + mm, y0 + mm), (x0, y0 + mm)]
            quads.append(found[m])
        if len(quads) < 2:  # both white squares at the corner must be read
            continue
        src, dst = np.array(src), np.concatenate(quads)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            continue
        side = min(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).min() for q in quads)
        back = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        if np.linalg.norm(back - dst, axis=1).max() > max(2.0, 0.1 * side):
            continue  # the two markers do not lie on one board: one of them is misread
        p = cv2.perspectiveTransform(np.array([[[cx * s, cy * s]]], dtype=np.float64), H).reshape(2)
        clear = min(_clear_distance(p, q) for q in quads)
        if clear < min_clear:
            continue
        win = int(np.clip(np.floor(_WINDOW * clear), 2, 12))
        if not (win + 2 <= p[0] < w_img - win - 2 and win + 2 <= p[1] < h - win - 2):
            continue
        ids.append(cid)
        start.append(p)
        wins.append(win)
    if not ids:
        return None
    ids, start, wins = np.array(ids, dtype=np.int64), np.array(start), np.array(wins)
    pts = _refine(gray, start, wins)
    again = _refine(gray, pts, np.where(wins > 2, wins - 1, 3))
    ok = np.linalg.norm(again - pts, axis=1) <= _AGREE_PX
    return ids[ok], pts[ok].astype(np.float64), len(found)


def charuco_offset() -> np.ndarray:
    """What the corner finder of this installation adds to a corner, in pixels: measured once
    on a board drawn here, whose corners are known exactly, and taken off in detect().

    Everything in OpenCV that a calibration feeds (projectPoints, undistort, remap, solvePnP)
    puts a pixel's centre at whole coordinates. The ChArUco step of OpenCV 4.7 and 4.8 does
    not, by half a pixel, which is half a pixel on the principal point. The corners are
    measured here instead (_charuco_corners) and this comes out as nothing; it stays as the
    proof, written into every calibration, and as the guard should an OpenCV differ again.
    """
    if not _OFFSET:
        spec = BoardSpec("charuco", 5, 5, 48.0, 36.0, "DICT_4X4_50")  # a cell is 6 mm: 1 px per mm is exact
        img = cv2.GaussianBlur(spec.render(1.0, 24.0), (0, 0), 0.8)
        got = _charuco_corners(spec, img, 0.0)
        if got is None or len(got[0]) < spec.corner_count():
            raise RuntimeError("this OpenCV (%s) does not find a ChArUco board drawn for it" % cv2.__version__)
        e = got[1] - (spec.corner_points()[got[0]][:, :2] * 1000.0 + 24.0 - 0.5)
        off = np.median(e, axis=0)
        if np.abs(e - off).max() > 0.05:
            raise RuntimeError("this OpenCV (%s) places ChArUco corners inconsistently" % cv2.__version__)
        half = np.round(off * 2.0) / 2.0
        _OFFSET.append(half if np.abs(off - half).max() < 0.02 else off)
    return _OFFSET[0]


def local_square_px(spec: BoardSpec, ids: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """How long one square is in the picture at each corner, along the direction it is shortest
    (a tilted board's far side). From the board's homography: good to a few percent under
    distortion, which is all a threshold needs."""
    obj = spec.corner_points()[ids][:, :2]
    if len(ids) < 4:
        return np.full(len(ids), np.inf)
    H, _ = cv2.findHomography(obj, pts, 0)
    if H is None:
        return np.full(len(ids), np.inf)
    h = spec.square_mm / 2000.0
    out = np.full(len(ids), np.inf)
    for dx, dy in ((h, 0.0), (0.0, h)):
        a = cv2.perspectiveTransform((obj - (dx, dy)).reshape(-1, 1, 2), H).reshape(-1, 2)
        b = cv2.perspectiveTransform((obj + (dx, dy)).reshape(-1, 1, 2), H).reshape(-1, 2)
        out = np.minimum(out, np.linalg.norm(b - a, axis=1))
    return out


def detect(spec: BoardSpec, image: np.ndarray, min_corners: int = 6,
           min_square_px: float = MIN_SQUARE_PX) -> Optional[Detection]:
    """The board's corners in a picture (grey or BGR), to a fraction of a pixel; None where
    there are fewer than min_corners.

    A ChArUco corner is refined in a window that must stay clear of the markers beside it.
    Where a square is under about 30 pixels (the clear distance under a few pixels) the
    window is too small to average the noise out, and the corner is measured several times
    worse. Those corners are left out rather than weighed down; min_square_px=0 keeps them
    (a board on the ground far from the camera).
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if spec.type == "charuco":
        got = _charuco_corners(spec, gray, min_square_px)
        if got is None or len(got[0]) < min_corners:
            return None
        ids, pts = got[0], got[1] - charuco_offset()
        if min_square_px > 0:  # the square's size from the corners themselves: the markers'
            keep = local_square_px(spec, ids, pts) >= min_square_px  # outlines are half a pixel coarse
            ids, pts = ids[keep], pts[keep]
            if len(ids) < min_corners:
                return None
        return Detection(ids, pts, got[2])
    size = (spec.squares_x - 1, spec.squares_y - 1)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
    if not ok:
        return None
    pts = corners.reshape(-1, 2).astype(np.float64)
    # one way round: the first corner the one nearer the picture's top left
    if pts[0, 1] + pts[0, 0] > pts[-1, 1] + pts[-1, 0]:
        pts = pts[::-1].copy()
    return Detection(np.arange(len(pts), dtype=np.int64), pts)


def board_quad(spec: BoardSpec, det: Detection) -> Optional[np.ndarray]:
    """Where the pattern's four outer corners are in the picture, from the corners found (also
    when the board runs out of the picture): (4, 2), or None."""
    if len(det) < 4:
        return None
    obj = det.object_points(spec)[:, :2].astype(np.float64)
    H, _ = cv2.findHomography(obj, det.image, 0)
    if H is None:
        return None
    q = cv2.perspectiveTransform(spec.outline()[:, :2].reshape(-1, 1, 2), H)
    return q.reshape(4, 2)


def draw_detection(image: np.ndarray, det: Optional[Detection], color=(60, 200, 60)) -> None:
    if det is None:
        return
    r = max(2, image.shape[1] // 480)
    for p in det.image:
        cv2.circle(image, (int(round(p[0])), int(round(p[1]))), r, color, -1, cv2.LINE_AA)
