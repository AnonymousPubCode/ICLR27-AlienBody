#!/usr/bin/env python3
"""Figure 3: shared family SR matrix and F4 comparison.

Default output: build/fig; --out fig publishes to the paper.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.gridspec import GridSpec

from figstyle import *  # noqa: F401,F403

COLS = ["F1", "F2", "F3", "F4", "F5", "F6"]
WALL = 3

# Provenance of the numbers below. They are transcribed results from the
# results/ tree, not computed at plot time; per-row sources, relative to it:
#   GPT-4o           gpt4o_l3_f{1..6}/gateway_gpt-4o/            -> metrics.sr_pct
#   Gemini 3.1 Pro   gemini_l3_f{1..6}/gateway_gemini-3-pro-preview/   (image)
#                    most cells -> metrics.sr_pct; the F4 cell is that run's
#                    success count (n_success = 2 of 38 recorded episodes,
#                    whereas its recorded sr_pct is 5.26), because that run was
#                    restarted and is not comparable to the clean n=50 rows
#   DeepSeek V4 Pro  dsv4pro_l3_fixed/gateway_dsv4-lh/  (text; F2 42.86 printed as 43)
#                    F4 from dsv4pro_l3_f4/, which has 0 successes across all
#                    219 recorded episodes (50 unique environments)
#   Qwen3-VL-32B     action_collapse/qwen32bvl_l3.json -> per_family.FN.sr_pct
#   FM_heur          p07_fmb_all_families/fmb/ for F1-F3 and F5;
#                    fmb_regression_20260904/fmb/ for F4 (44/50);
#                    f6_recheck_20260911/fmb_f6_fixed/fmb/ for F6 (2/50).
#                    This row mixes runs: no single invocation produced all six.
#   Oracle BFS       exact BFS over the ground-truth transition system (upper bound)
# F4_BARS only: GPT-5.1 -> gpt51_l3_f4/ (2/50); Gemma-4-26B is 0/50 recorded on
# the h200 host (results_wall/gemma26b_text), not in the local results/ tree.

TABLE_ROWS = [
    ("GPT-4o",          [100, 60, 72, 0, 66, 26], False),
    ("Gemini 3.1 Pro",  [100, 100, 100, 2, 100, 29], False),
    ("DeepSeek V4 Pro", [92, 43, 66, 0, 28, 8], False),
    ("Qwen3-VL-32B",    [86, 44, 82, 0, 20, 16], False),
    (r"FM$_{\mathrm{heur}}$", [100, 92, 80, 88, 0, 4], True),
    ("Oracle BFS",      [100, 100, 100, 100, 100, 100], True),
]
N_CTX = 4

F4_BARS = [
    ("GPT-4o", C_GPT4O, 0),
    ("Gemini 3.1 Pro", C_GEMINI, 2),
    ("DeepSeek V4 Pro", C_DS, 0),
    ("Qwen3-VL-32B", C_QWEN, 0),
    ("GPT-5.1", C_GPT51, 4),
    ("Gemma-4-26B", C_GEMMA, 0),
    (r"FM$_{\mathrm{heur}}$", C_FMB, 88),
    ("Oracle BFS", C_ORACLE, 100),
]
N_WALL = 6  # in-context / F4-only probes above the search divider
PROBE_NAMES = {"GPT-5.1", "Gemma-4-26B"}


W, H = 5.5, 1.91
# Preserve type sizes; tighten vertical geometry rather than scaling the PDF.
FIG_HEIGHT = 1.70
MUTED, RULE = "#657078", "#D8DDDF"


def text(fig, x, y, value, **kw):
    return fig.text(x / W, y / H, value, va="center", **kw)


def line(fig, x0, x1, y, **kw):
    fig.add_artist(plt.Line2D([x0 / W, x1 / W], [y / H] * 2,
                             transform=fig.transFigure, **kw))


def build():
    probes = []
    for name, _, value in F4_BARS:
        if name in PROBE_NAMES:
            vals = [None] * 6
            vals[WALL] = value
            probes.append((name, vals, False))
    rows = TABLE_ROWS[:4] + probes + TABLE_ROWS[4:]
    expected = {name: val for name, _, val in F4_BARS}
    assert len(rows) == 8
    assert all(vals[WALL] == expected[name] for name, vals, _ in rows)
    ys = [1.36, 1.225, 1.09, .955, .765, .630, .440, .305]

    fig = plt.figure(figsize=(W, FIG_HEIGHT), facecolor="white")
    text(fig, .04, 1.79, "(a) L3 success by family (%)", fontsize=8.2, weight="bold")
    text(fig, 3.99, 1.79, "(b) F4 contrast", fontsize=8.2, weight="bold")
    centers = [1.38 + .40 * j for j in range(6)]
    fig.add_artist(patches.Rectangle(
        ((centers[WALL] - .19) / W, .22 / H), .38 / W, 1.43 / H,
        facecolor="#F8EEEB", edgecolor="none", transform=fig.transFigure, zorder=0))
    text(fig, .04, 1.545, "Agent / method", fontsize=6.6, color=MUTED)
    for j, name in enumerate(COLS):
        text(fig, centers[j], 1.545, name, ha="center", fontsize=7.0,
             weight="bold" if j == WALL else "normal", color=RED if j == WALL else INK)
    line(fig, .04, 3.64, 1.65, color=INK, lw=.65)
    line(fig, .04, 3.64, 1.445, color=RULE, lw=.6)
    line(fig, .04, 3.64, .22, color=INK, lw=.65)

    bx = fig.add_axes([3.99 / W, .22 / H, 1.19 / W, 1.225 / H])
    bx.set(xlim=(-4, 104), ylim=(.22, 1.445), xticks=[0, 50, 100], yticks=[])
    bx.xaxis.grid(True, color=RULE, lw=.5, zorder=0)
    bx.tick_params(axis="x", labelsize=6.3, length=0, pad=3)
    for side in ("left", "right", "top"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set(color="#7D878C", linewidth=.55)
    text(fig, 4.58, 1.545, "Success rate (%)", ha="center", fontsize=6.6, color=MUTED)

    for i, ((name, vals, search), y) in enumerate(zip(rows, ys)):
        shown = name + "*" if name in PROBE_NAMES else name
        color = GREEN if i == 6 else INK if i == 7 else RED
        text(fig, .04, y, shown, fontsize=6.5,
             fontstyle="italic" if search else "normal", color=INK)
        for j, value in enumerate(vals):
            shown_value = "\u2014" if value is None else f"{value:g}"
            text(fig, centers[j], y, shown_value, ha="center", fontsize=6.6,
                 color="#B0B7BC" if value is None else color if j == WALL else INK,
                 weight="bold" if j == WALL else "normal")
        value = vals[WALL]
        bx.plot([0, 100], [y, y], color="#EDF0F2", lw=.55, zorder=1)
        bx.plot([0, value], [y, y], color=color, lw=1.7, zorder=3, solid_capstyle="butt")
        bx.scatter([value], [y], s=20, facecolor=color, edgecolor="white", lw=.55, zorder=4)
        bx.text(1.065, y, f"{value:g}", transform=bx.get_yaxis_transform(),
                va="center", ha="left", fontsize=6.6, color=color, clip_on=False)

    for y in (.8625, .5375):
        line(fig, .04, 3.64, y, color=RULE, lw=.65)
        bx.axhline(y, color=RULE, lw=.65, zorder=2)
    fig.add_artist(plt.Line2D([3.82 / W] * 2, [.20 / H, 1.65 / H],
                             transform=fig.transFigure, color=RULE, lw=.5))
    text(fig, .04, .065, "* F4-only probes; \u2014 not evaluated. Italics: search over a model.",
         fontsize=6.0, color=MUTED)
    out = figure_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "planning_wall.pdf"
    fig.savefig(target, metadata={"Title": "AlienBody: planning wall"})
    fig.savefig(target.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    build()
