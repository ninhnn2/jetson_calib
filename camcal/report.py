"""report.html: the calibration to look at. One file, the charts inside it.

What each chart is for:
  corners per cell      where in the picture the camera has been measured, and where not
  residuals             the cloud of corner errors: round and centred when nothing is wrong
  error per view        the view that is worse than the others
  error against radius  a model too poor for the lens shows as error growing to the edge
  mean error per cell   what is left over in a pattern (not noise): arrows that agree
  distortion            how far the lens moves each pixel from where a pinhole would put it
"""

from __future__ import annotations

import base64
import html
import io
from typing import List

import numpy as np

from . import models
from .boards import BoardSpec

SURFACE, INK, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, DEEP, EMPTY = "#2a78d6", "#0d366b", "#f0efec"
RAMP = ["#cde2fb", "#9cc5f5", "#5e9fe8", "#2a78d6", "#1a56a6", "#0d366b"]
STATUS = {"A": ("#0ca30c", "✓"), "B": ("#fab219", "▲"), "C": ("#ec835a", "◆"), "D": ("#d03b3b", "✕")}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.6, "axes.labelcolor": MUTED,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelsize": 8.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "grid.color": RULE, "grid.linewidth": 0.6, "axes.grid": True,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 8.5, "text.color": INK,
        "lines.linewidth": 2.0,
    })
    return plt


def _ramp():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("camcal_blue", RAMP).with_extremes(bad=EMPTY)


def _png(plt, fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _coverage(plt, result) -> str:
    w, h = result.calibration.size
    g = np.ma.masked_less(result.grid.astype(float), 2)
    rows, cols = result.grid.shape
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * h / w + 0.9))
    m = ax.pcolormesh(np.linspace(0, w, cols + 1), np.linspace(0, h, rows + 1), g, cmap=_ramp(),
                      edgecolors=SURFACE, linewidth=1.5, vmin=2)
    ax.set_facecolor(EMPTY)
    ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.set_aspect("equal"), ax.grid(False)
    ax.set_title("Corners per cell (grey: fewer than 2)")
    ax.set_xlabel("pixel"), ax.set_ylabel("pixel")
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.03).outline.set_visible(False)
    return _png(plt, fig)


def _scatter(plt, result) -> str:
    r = np.concatenate(result.residuals)
    n = np.linalg.norm(r, axis=1)
    p50, p95 = np.percentile(n, 50), np.percentile(n, 95)
    lim = max(1.6 * p95, 0.05)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(r[:, 0], r[:, 1], s=5, color=BLUE, alpha=0.25, linewidths=0)
    t = np.linspace(0, 2 * np.pi, 200)
    for rad, name in ((p50, "half within %.3f px" % p50), (p95, "95%% within %.3f px" % p95)):
        ax.plot(rad * np.cos(t), rad * np.sin(t), color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(name, (rad * 0.71, rad * 0.71), xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=7.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0, alpha=0.85))
    ax.set_xlim(-lim, lim), ax.set_ylim(lim, -lim), ax.set_aspect("equal")
    ax.set_title("Corner errors, %d corners" % len(r))
    ax.set_xlabel("error along x, px"), ax.set_ylabel("error along y, px")
    return _png(plt, fig)


def _views(plt, result) -> str:
    rv = np.array([models.rms([r]) for r in result.residuals])
    x = np.arange(1, len(rv) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(x, rv, width=0.72, color=BLUE, label="fitted with this view")
    if result.held_out_per_view is not None:
        ax.scatter(x, result.held_out_per_view, s=16, color=DEEP, zorder=3, edgecolors=SURFACE, linewidths=0.8,
                   label="this view held out")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
    ax.set_xlim(0.3, len(rv) + 0.7), ax.set_ylim(0, None), ax.grid(axis="x", visible=False)
    ax.set_title("Error per view"), ax.set_xlabel("view, in the order taken"), ax.set_ylabel("rms, px")
    return _png(plt, fig)


def _radius(plt, result) -> str:
    cal = result.calibration
    pts = np.concatenate([v.img for v in result.views])
    n = np.linalg.norm(np.concatenate(result.residuals), axis=1)
    rad = np.linalg.norm(pts - cal.K[:2, 2], axis=1)
    edges = np.unique(np.percentile(rad, np.linspace(0, 100, 11)))
    mid, med, p95 = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rad >= a) & (rad <= b)
        if m.sum() >= 5:
            mid.append(0.5 * (a + b)), med.append(np.median(n[m])), p95.append(np.percentile(n[m], 95))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(mid, p95, color=DEEP, marker="o", markersize=4, label="95th percentile")
    ax.plot(mid, med, color=BLUE, marker="o", markersize=4, label="median")
    ax.set_ylim(0, None), ax.legend(loc="upper left")
    ax.set_title("Error against distance from the centre")
    ax.set_xlabel("distance from the principal point, px"), ax.set_ylabel("error, px")
    return _png(plt, fig)


def _field(plt, result) -> str:
    cal = result.calibration
    w, h = cal.size
    rows, cols = result.grid.shape
    pts = np.concatenate([v.img for v in result.views])
    res = np.concatenate(result.residuals)
    c = np.clip((pts[:, 0] + 0.5) * cols / w, 0, cols - 1).astype(int)
    r = np.clip((pts[:, 1] + 0.5) * rows / h, 0, rows - 1).astype(int)
    sx, sy, cnt = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
    np.add.at(sx, (r, c), res[:, 0]), np.add.at(sy, (r, c), res[:, 1]), np.add.at(cnt, (r, c), 1)
    ok = cnt >= 5
    mx, my = np.where(ok, sx / np.maximum(cnt, 1), 0), np.where(ok, sy / np.maximum(cnt, 1), 0)
    big = float(np.max(np.hypot(mx, my))) if ok.any() else 0.0
    unit = max(big, 0.02)  # the longest arrow is one cell long
    cell = w / cols
    gx, gy = (np.arange(cols) + 0.5) * cell, (np.arange(rows) + 0.5) * h / rows
    X, Y = np.meshgrid(gx, gy)
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * h / w + 0.9))
    ax.quiver(X[ok], Y[ok], mx[ok] / unit * cell, my[ok] / unit * cell, angles="xy", scale_units="xy", scale=1,
              color=BLUE, width=0.004)
    ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.set_aspect("equal")
    ax.set_title("Mean error per cell (an arrow one cell long: %.3f px)" % unit)
    ax.set_xlabel("pixel"), ax.set_ylabel("pixel")
    return _png(plt, fig)


def _distortion(plt, result) -> str:
    cal = result.calibration
    w, h = cal.size
    xs, ys = np.linspace(0, w - 1, 64), np.linspace(0, h - 1, max(8, int(64 * h / w)))
    X, Y = np.meshgrid(xs, ys)
    px = np.stack([X.ravel(), Y.ravel()], axis=1)
    xn = cal.normalize(px)
    ideal = xn * [cal.K[0, 0], cal.K[1, 1]] + cal.K[:2, 2]
    d = np.ma.masked_invalid(np.linalg.norm(ideal - px, axis=1).reshape(X.shape))
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * h / w + 0.9))
    m = ax.contourf(X, Y, d, levels=12, cmap=_ramp())
    cs = ax.contour(X, Y, d, levels=6, colors=INK, linewidths=0.5)
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    ax.plot([cal.K[0, 2]], [cal.K[1, 2]], marker="+", color=INK, markersize=9)
    ax.set_facecolor(EMPTY)
    ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.set_aspect("equal"), ax.grid(False)
    ax.set_title("Distortion, pixels (+ is the centre)")
    ax.set_xlabel("pixel"), ax.set_ylabel("pixel")
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.03).outline.set_visible(False)
    return _png(plt, fig)


def charts(result) -> List[tuple]:
    """[(name, title, PNG as base64)] for a calibration result."""
    plt = _plt()
    return [(n, t, f(plt, result)) for n, t, f in (
        ("coverage", "Where the camera was measured", _coverage),
        ("distortion", "What the lens does", _distortion),
        ("residuals", "Corner errors", _scatter),
        ("radius", "Error from centre to edge", _radius),
        ("views", "Error per view", _views),
        ("field", "What is left over, by place", _field))]


_CSS = """
body{margin:0;background:#fcfcfb;color:#0b0b0b;font:15px/1.5 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:34px 0 10px}
p{margin:6px 0;color:#52514e} .lead{color:#0b0b0b}
.grade{display:inline-flex;align-items:center;gap:8px;border:1px solid #e4e3df;border-radius:8px;padding:8px 14px;margin:12px 0}
.grade b{font-size:20px} .dot{font-size:18px}
table{border-collapse:collapse;font-size:13.5px;margin:6px 0} td,th{padding:4px 12px 4px 0;text-align:left;border-bottom:1px solid #e4e3df}
th{color:#52514e;font-weight:600} td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;align-items:start}
figure{margin:0} img{max-width:100%;height:auto;border:1px solid #e4e3df;border-radius:6px}
figcaption{color:#52514e;font-size:13px;margin-top:4px}
ul{margin:6px 0;padding-left:20px;color:#52514e} .scroll{overflow-x:auto}
details{margin-top:8px} summary{cursor:pointer;color:#52514e}
"""

_CAPTION = {
    "coverage": "Every cell should have corners, the ones along the edge too: the distortion is largest there, "
                "and where there are no corners the model is a guess.",
    "distortion": "The lens against a perfect pinhole. Lines are pixels of displacement.",
    "residuals": "A round cloud about zero is noise. A stretched or shifted one is something systematic: "
                 "a board that is not flat, motion, a wrong board size.",
    "radius": "Flat is right. Rising to the edge: the model has too few coefficients for this lens, or the "
              "edge has too few corners.",
    "views": "A view well above the others was moving or bent. Held out and fitted should be close: "
             "far apart means the model fits the views, not the lens.",
    "field": "Arrows that point one way over a region are an error the model keeps. Arrows in all "
             "directions are noise.",
}


def _fmt(v, f="%.3f"):
    return "–" if v is None else f % v


def write(result, spec: BoardSpec, path: str) -> None:
    cal = result.calibration
    m = cal.meta
    q, r, std = m["quality"], m["rms_px"], m.get("std") or {}
    e = html.escape
    color, icon = STATUS[m["grade"]]
    out = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
           "initial-scale=1'><title>Camera calibration</title><style>%s</style><main>" % _CSS]
    out.append("<h1>Camera calibration</h1><p>%s · %dx%d · %s · camcal %s, OpenCV %s</p>" % (
        e(cal.model), cal.size[0], cal.size[1], e(m["created"]), e(m["camcal_version"]), e(m["opencv_version"])))
    out.append("<div class='grade'><span class='dot' style='color:%s'>%s</span><b>Grade %s</b><span>%s</span></div>"
               % (color, icon, m["grade"], e(m["grade_meaning"])))
    if m["to_improve"]:
        out.append("<ul>%s</ul>" % "".join("<li>%s</li>" % e(t) for t in m["to_improve"]))
    out.append("<p>The grade is engineering guidance for a single camera's intrinsics, not a certification.</p>")

    out.append("<h2>The camera</h2><div class='scroll'><table><tr><th>parameter</th><th class='n'>value</th>"
               "<th class='n'>± one σ</th></tr>")
    vals = [("fx", cal.K[0, 0], "%.3f"), ("fy", cal.K[1, 1], "%.3f"), ("cx", cal.K[0, 2], "%.3f"),
            ("cy", cal.K[1, 2], "%.3f")] + [(n, v, "%.6f") for n, v in zip(models.DIST_NAMES[cal.model], cal.dist)]
    for n, v, f in vals:
        out.append("<tr><td>%s</td><td class='n'>%s</td><td class='n'>%s</td></tr>" % (n, f % v, _fmt(std.get(n), f)))
    out.append("</table></div><p>σ %s. Field of view %s × %s degrees.</p>" % (
        e(std.get("method", "not known")), _fmt(m["fov_deg"]["horizontal"], "%.1f"), _fmt(m["fov_deg"]["vertical"], "%.1f")))

    out.append("<h2>How good</h2><div class='scroll'><table>")
    rows = [
        ("error on the views used", "%.3f px rms" % r["train"]),
        ("error on views held out", _fmt(r["held_out"], "%.3f px rms")),
        ("half the corners within / 95% within / worst", "%.3f / %.3f / %.3f px" % (r["p50"], r["p95"], r["max"])),
        ("views used / with the board / pictures", "%d / %d / %d" % (q["views_used"], q["views_with_board"], q["views_given"])),
        ("corners used / dropped", "%d / %d" % (q["corners_used"], q["corners_dropped"])),
        ("picture covered / its edge covered", "%.0f%% / %.0f%%" % (100 * q["coverage"], 100 * q["border_coverage"])),
        ("views tilted over 15 degrees / largest tilt", "%.0f%% / %.0f degrees" % (100 * q["tilted_share"], q["tilt_deg_max"])),
        ("board distance", "%.2f to %.2f m" % tuple(q["distance_m"])),
        ("focal length known to", _fmt(None if q["sigma_f_rel"] is None else 100 * q["sigma_f_rel"], "%.3f%%")),
        ("focal length between the halves of the board",
         "-" if not q.get("board_halves") else "%.1f to %.1f px (%.2f%% from the whole board's)" % (
             q["board_halves"]["fx_px"][0], q["board_halves"]["fx_px"][1], 100 * q["board_halves"]["fx_relative"])),
        ("to carry into calculations (1 sigma)",
         "-" if not m.get("uncertainty") else "focal ± %.2f%%, cx ± %.1f px, cy ± %.1f px" % (
             100 * m["uncertainty"]["fx_relative"], m["uncertainty"]["cx_px"], m["uncertainty"]["cy_px"])),
        ("model holds out to the picture's corners", "yes" if q.get("valid_over_picture", True) else "NO"),
        ("board", spec.describe()),
    ]
    out += ["<tr><td>%s</td><td>%s</td></tr>" % (e(a), e(b)) for a, b in rows]
    out.append("</table></div>")

    out.append("<h2>The model, chosen by views held out</h2><div class='scroll'><table><tr><th>model</th>"
               "<th class='n'>fitted, px</th><th class='n'>held out, px</th><th></th></tr>")
    for row in m["model_selection"]:
        out.append("<tr><td>%s</td><td class='n'>%s</td><td class='n'>%s</td><td>%s</td></tr>" % (
            e(row["model"]), _fmt(row["rms_train"]), _fmt(row["rms_held_out"]),
            "✓ chosen" if row["chosen"] else e(row.get("note", ""))))
    out.append("</table></div><p>Of the models within 5% of the best on held-out views, the simplest.</p>")

    out.append("<h2>Charts</h2><div class='grid'>")
    for name, title, png in charts(result):
        out.append("<figure><img alt='%s' src='data:image/png;base64,%s'><figcaption>%s</figcaption></figure>"
                   % (e(title), png, e(_CAPTION[name])))
    out.append("</div>")

    out.append("<h2>The views</h2><details><summary>%d used, %d dropped: the table</summary><div class='scroll'>"
               "<table><tr><th>view</th><th class='n'>corners</th><th class='n'>rms, px</th>"
               "<th class='n'>held out, px</th><th class='n'>tilt, °</th><th class='n'>distance, m</th></tr>"
               % (len(m["views"]), len(m["dropped_views"])))
    for v in m["views"]:
        out.append("<tr><td>%s</td><td class='n'>%d</td><td class='n'>%.3f</td><td class='n'>%s</td>"
                   "<td class='n'>%.0f</td><td class='n'>%.2f</td></tr>" % (
                       e(v["name"]), v["corners"], v["rms_px"], _fmt(v["rms_held_out_px"]), v["tilt_deg"], v["distance_m"]))
    out.append("</table>")
    if m["dropped_views"]:
        out.append("<table><tr><th>dropped</th><th>why</th></tr>%s</table>" % "".join(
            "<tr><td>%s</td><td>%s</td></tr>" % (e(d["name"]), e(d["why"])) for d in m["dropped_views"]))
    out.append("</div></details></main></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
