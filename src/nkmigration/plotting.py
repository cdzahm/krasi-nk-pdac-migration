"""Dot-plot panels: a violin of the distribution, up to 300 sampled cells as
points, and the median with interquartile range."""
from __future__ import annotations

import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import stats  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,  # keep text editable in Illustrator
})

FIGSIZE = (3.30, 2.20)          # inches
AXIS_TITLE_PT, AXIS_NUM_PT, STAR_PT, TITLE_PT, N_PT = 9, 8, 9, 8, 7
LEFT_IN, RIGHT_IN, TITLE_IN = 0.62, 0.08, 0.17
LABEL_LINE_IN, N_ROW_IN, BRACKET_GAP_IN = 0.14, 0.15, 0.04
BRACKET_STEP_IN = 1.25 * STAR_PT / 72
DOT_SIZE, MEDIAN_MS = 1.6, 4
COLOR = "#3B6EA5"
MAX_DOTS = 300


def _rng(key: str) -> np.random.Generator:
    """Seeded from the panel name, so each panel's sampled points are fixed."""
    return np.random.default_rng(zlib.crc32(key.encode()))


def dot_panel(groups: list[str], values: list[np.ndarray], ylabel: str, title: str,
              key: str, ceiling: float | None = None, bounded01: bool = False,
              connect_medians: bool = False):
    """Draw one panel and return (figure, statistics).

    With `ceiling`, values above it are drawn as points at the ceiling and
    never dropped. The violin shows the values up to the ceiling, and the
    medians and tests use the full data.
    """
    rng = _rng(key)
    n_pairs = len(groups) * (len(groups) - 1) // 2
    top = TITLE_IN + BRACKET_GAP_IN + n_pairs * BRACKET_STEP_IN
    bottom = 0.18 * LABEL_LINE_IN / 0.23 + LABEL_LINE_IN + N_ROW_IN
    fw, fh = FIGSIZE
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([LEFT_IN / fw, bottom / fh, (fw - LEFT_IN - RIGHT_IN) / fw,
                       (fh - bottom - top) / fh])
    ax_h = ax.get_position().height * fh

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.93", lw=0.6)
    n_y = -(0.06 * LABEL_LINE_IN / 0.23 + LABEL_LINE_IN) / ax_h
    over = {}
    medians = []
    for i, (g, v) in enumerate(zip(groups, values)):
        v = np.asarray(v, float)
        shown, body = v, v
        if ceiling is not None:
            over[g] = int((v > ceiling).sum())
            shown, body = np.minimum(v, ceiling), v[v <= ceiling]
        if len(body) > 8:
            for b in ax.violinplot([body], positions=[i], widths=0.55,
                                   showextrema=False)["bodies"]:
                b.set_facecolor(COLOR)
                b.set_edgecolor(COLOR)
                b.set_alpha(0.20)
                b.set_linewidth(0.7)
        if len(v):
            idx = (np.arange(len(v)) if len(v) <= MAX_DOTS
                   else rng.choice(len(v), MAX_DOTS, replace=False))
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(idx)), shown[idx], s=DOT_SIZE,
                       alpha=0.24, color=COLOR, linewidths=0, zorder=2, clip_on=False)
            med = float(np.median(v))
            q1, q3 = np.percentile(v, [25, 75])
            if ceiling is not None:
                med, q1, q3 = min(med, ceiling), min(q1, ceiling), min(q3, ceiling)
            ax.plot([i, i], [q1, q3], color="0.25", lw=1.4, zorder=4, solid_capstyle="butt")
            ax.plot([i], [med], marker="D", ms=MEDIAN_MS, color="0.95",
                    markeredgecolor="0.2", markeredgewidth=1.1, zorder=5)
            medians.append(med)
        else:
            medians.append(np.nan)
        ax.annotate(f"n={len(v):,}", (i, n_y), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=N_PT, color="0.4", annotation_clip=False)
    if connect_medians:
        ax.plot(range(len(values)), medians, color="0.35", lw=1.1, ls="--", zorder=3, alpha=0.8)

    if ceiling is not None:
        ax.set_ylim(0, ceiling)
    elif bounded01:
        ax.set_ylim(-0.03, 1.0)
    else:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - 0.04 * (y1 - y0), y1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=AXIS_NUM_PT)
    ax.set_xlim(-0.65, len(groups) - 0.35)
    ax.set_ylabel(ylabel, fontsize=AXIS_TITLE_PT)
    ax.tick_params(axis="y", labelsize=AXIS_NUM_PT)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 1 - 0.07 / fh, title, ha="center", va="top", fontsize=TITLE_PT, color="0.25")

    result = stats.compare([np.asarray(v, float) for v in values])
    trans = ax.get_xaxis_transform()
    for k, pr in enumerate(result["pairs"]):
        y = 1.0 + BRACKET_GAP_IN / ax_h + k * BRACKET_STEP_IN / ax_h
        a, b = pr["i"], pr["j"]
        ax.plot([a, a, b, b], [y - 0.014, y, y, y - 0.014], transform=trans,
                color="0.35", lw=0.9, clip_on=False)
        ax.annotate(pr["shown"], ((a + b) / 2, y + 0.004), xycoords=trans, ha="center",
                    va="bottom", fontsize=STAR_PT, color="0.15", annotation_clip=False)
    result["above_ceiling"] = over
    return fig, result
