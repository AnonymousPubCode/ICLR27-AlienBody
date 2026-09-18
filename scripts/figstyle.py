"""Shared visual style for every AlienBody paper figure.

One palette, one type setting, one set of axis conventions, so that all
figures read as a single family alongside the LaTeX tables (which use the
same accent colours via ``\\definecolor`` in ``main.tex``).

Usage::

    from figstyle import *        # applies rcParams on import
    fig, ax = plt.subplots(figsize=(W_COL, 2.4))
    ...
    save(fig, "planning_wall")    # -> fig/planning_wall.pdf (+ .png preview)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent            # papers/alienbody
FIG = PAPER / "fig"
CAPTURES = FIG / "game_captures"
RESULTS = HERE.parent / "results"

# ICLR text width is 5.5in; figures are usually included at \columnwidth.
W_COL = 5.5
W_HALF = 2.65

# ─────────────────────────────────────────────────────────────────────
# Palette (mirrors \definecolor in main.tex)
# ─────────────────────────────────────────────────────────────────────
INK = "#1F1F1F"          # near-black for text / oracle
MUTED = "#8C8C8C"        # secondary text, anonymous condition
FAINT = "#C9C9C9"        # gridlines, reference rules
PAPER_BG = "#F4F1EC"     # warm off-white for highlight bands

ACCENT = "#2F5D8A"       # deep blue    (primary model: GPT-4o, table bars)
GREEN = "#2A7F62"        # teal green   (named-true, Gemini, FM_heur)
RED = "#B5443A"          # brick red    (misleading, failure)
AMBER = "#C77B3A"        # amber        (GPT-5.1, category hints)
VIOLET = "#6B5B95"       # violet       (DeepSeek)
SLATE = "#5C6B7A"        # slate grey   (Qwen / local models)

# Name-prior conditions
C_ANON, C_TRUE, C_MIS, C_CAT = MUTED, GREEN, RED, AMBER

# Models
C_GPT4O, C_GEMINI, C_DS, C_GPT51, C_QWEN = ACCENT, GREEN, VIOLET, AMBER, SLATE
C_FMB, C_ORACLE = GREEN, INK

# Sequential ramp for heat-tinted matrices (white -> ACCENT)
HEAT = mpl.colors.LinearSegmentedColormap.from_list(
    "alienbody_heat", ["#FFFFFF", "#DCE6F0", "#9DB8D4", "#5A85B0", ACCENT], N=256
)

# ─────────────────────────────────────────────────────────────────────
# rcParams
# ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def ygrid(ax, **kw):
    """Light dotted horizontal gridlines behind the data."""
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color=FAINT, **kw)
    ax.set_axisbelow(True)


def panel_label(ax, text, x=0.0, y=1.02, **kw):
    """Bold '(a)'-style label anchored at the top-left of the axes."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", **kw)


def bar_labels(ax, bars, fmt="{:.0f}", dy=1.2, fontsize=6.8, color=INK, **kw):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=fontsize, color=color, **kw)


def save(fig, stem: str, png: bool = True, pad: float = 0.02):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=pad)
    if png:
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"Saved: {out}")


__all__ = [n for n in dir() if not n.startswith("_")]
