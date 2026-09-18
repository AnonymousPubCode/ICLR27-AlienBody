"""Shared visual style for every AlienBody paper figure.

One palette, one type setting, one set of axis conventions, so that all
figures read as a single family alongside the LaTeX tables (which use the
same accent colours via ``\\definecolor`` in ``main.tex``).

Usage::

    from figstyle import *        # applies rcParams on import
    fig, ax = plt.subplots(figsize=(W_COL, 2.4))
    ...
    save(fig, "planning_wall")    # -> build/fig/planning_wall.pdf (+ .png preview)

Output isolation (R7/G7): ``save`` writes to the **untracked** scratch dir
``papers/alienbody/build/fig`` by default, so a stray regeneration cannot dirty
the tracked ``fig/`` tree that the LaTeX sources read.  Writing into the tracked
tree is explicit, via ``--out fig`` on the command line or the environment
variable ``ALIENBODY_FIG_OUT``::

    python scripts/gen_fig_teaser.py --out fig          # tracked paper tree
    ALIENBODY_FIG_OUT=fig python scripts/gen_fig_teaser.py
    python scripts/gen_fig_teaser.py --out /tmp/preview # any other directory
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent            # papers/alienbody
FIG = PAPER / "fig"                   # tracked tree (\includegraphics reads this)
BUILD_FIG = PAPER / "build" / "fig"   # untracked scratch dir (root .gitignore: build/)
CAPTURES = FIG / "game_captures"
RESULTS = HERE.parent / "results"

#: tokens accepted by --out / $ALIENBODY_FIG_OUT meaning "the tracked fig/ tree"
TRACKED_TOKENS = {"fig", "tracked"}

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
C_GEMMA = "#6E7F8A"       # cool grey (Gemma; distinct from Qwen slate)
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


def figure_out_dir(argv=None, environ=None) -> Path:
    """Resolve the directory :func:`save` writes figures to (R7/G7).

    Precedence: ``--out DIR`` / ``--out=DIR`` on the command line, then
    ``$ALIENBODY_FIG_OUT``, then the untracked default ``build/fig``.  The
    tokens ``fig`` / ``tracked`` select the tracked ``fig/`` tree explicitly.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    token = env.get("ALIENBODY_FIG_OUT")
    for i, arg in enumerate(argv):
        if arg == "--out" and i + 1 < len(argv):
            token = argv[i + 1]
            break
        if arg.startswith("--out="):
            token = arg.split("=", 1)[1]
            break
    if not token:
        return BUILD_FIG
    token = token.strip()
    if token in TRACKED_TOKENS:
        return FIG
    return Path(token).expanduser()


def save(fig, stem: str, png: bool = True, pad: float = 0.02, out=None):
    """Write ``stem.pdf`` (+ ``.png``) into :func:`figure_out_dir`.

    Pass ``out`` to force a directory (``out=FIG`` for the tracked tree).
    """
    out_dir = FIG if out == "tracked" else (Path(out) if out is not None
                                            else figure_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}.pdf"
    fig.savefig(target, bbox_inches="tight", pad_inches=pad)
    if png:
        fig.savefig(target.with_suffix(".png"), dpi=200, bbox_inches="tight",
                    pad_inches=pad)
    plt.close(fig)
    tag = "  [tracked paper tree]" if out_dir == FIG else ""
    print(f"Saved: {target}{tag}")


__all__ = [n for n in dir() if not n.startswith("_") and n not in {"os", "sys"}]
