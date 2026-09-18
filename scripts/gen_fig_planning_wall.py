#!/usr/bin/env python3
"""Figure: the planning wall at L3 (exact mapping given).

(a) Heat-tinted matrix of L3 SR% (agents x families). The F4 column is the
    wall: every in-context agent is at 0-2% while both plan-over-model
    rows (FM_heur, Oracle BFS) stay high.
(b) F4 only, agent by agent, with the in-context band (<= 6%) marked.

Output: fig/planning_wall.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from figstyle import *  # noqa: F401,F403

FAMILIES = ["F1\nDir.", "F2\nSpat.", "F3\nCond.", "F4\nRel.", "F5\nComp.", "F6\nTemp."]

# (label, per-family L3 SR%, group)  -- group 0 = in-context VLM/LLM, 1 = plan-over-model
ROWS = [
    ("GPT-4o (image)",          [100, 60, 72, 0, 66, 26], 0),
    ("Gemini 3.1 Pro (text)",   [100, 100, 100, 2, 100, 29], 0),
    ("DeepSeek V4 Pro (text)",  [92, 43, 66, 0, 28, 8], 0),
    ("Qwen3-VL-32B (image)",    [86, 44, 82, 0, 20, 16], 0),
    (r"FM$_{\mathrm{heur}}$ (induce + BFS)", [100, 92, 80, 88, 0, 4], 1),
    ("Oracle BFS",              [100, 100, 100, 100, 100, 100], 1),
]

# F4 only (label, SR%, colour)
F4 = [
    ("GPT-4o (image)", 0, C_GPT4O),
    ("GPT-4o (text)", 0, C_GPT4O),
    ("Gemini 3.1 Pro", 2, C_GEMINI),
    ("DeepSeek V4 Pro", 0, C_DS),
    ("GPT-5.1 (thinking)", 4, C_GPT51),
    ("Qwen3-VL-32B", 0, C_QWEN),
    (r"FM$_{\mathrm{heur}}$", 88, C_FMB),
    ("Oracle BFS", 100, C_ORACLE),
]

WALL_COL = 3
GAP = 0.35  # vertical gap between the two row groups in (a)


def draw_matrix(ax):
    n_rows, n_cols = len(ROWS), len(FAMILIES)
    # y position of each row (top -> bottom), with a gap before group 1
    ys, y = [], 0.0
    for i, (_, _, g) in enumerate(ROWS):
        if i > 0 and g != ROWS[i - 1][2]:
            y += GAP
        ys.append(y)
        y += 1.0
    total_h = y

    for (label, vals, g), y0 in zip(ROWS, ys):
        for j, v in enumerate(vals):
            face = HEAT(v / 100.0)
            ax.add_patch(mpatches.Rectangle(
                (j, y0), 1, 1, facecolor=face, edgecolor="white", linewidth=1.2))
            lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
            ax.text(j + 0.5, y0 + 0.5, f"{v:g}", ha="center", va="center",
                    fontsize=7.3, color="white" if lum < 0.55 else INK,
                    fontweight="bold" if j == WALL_COL else "normal")
        ax.text(-0.15, y0 + 0.5, label, ha="right", va="center", fontsize=7.3,
                color=INK if g == 1 else "#333333",
                fontstyle="italic" if g == 1 else "normal")

    # column headers (axis is inverted: smaller y is higher on the page)
    for j, fam in enumerate(FAMILIES):
        ax.text(j + 0.5, -0.12, fam, ha="center", va="bottom", fontsize=7.3,
                linespacing=1.1, color=RED if j == WALL_COL else INK,
                fontweight="bold" if j == WALL_COL else "normal")

    # wall column outline + label
    ax.add_patch(mpatches.Rectangle(
        (WALL_COL, 0), 1, total_h, fill=False, edgecolor=RED, linewidth=1.1, zorder=5))
    ax.text(WALL_COL + 0.5, total_h + 0.14, "planning wall", ha="center", va="top",
            fontsize=7, color=RED, fontstyle="italic")

    # group brace labels on the right
    g0 = [y for (_, _, g), y in zip(ROWS, ys) if g == 0]
    g1 = [y for (_, _, g), y in zip(ROWS, ys) if g == 1]
    for grp, txt in ((g0, "in-context\nagent"), (g1, "search over\nmodel")):
        ymid = (min(grp) + max(grp) + 1) / 2
        ax.plot([n_cols + 0.12, n_cols + 0.12], [min(grp) + 0.08, max(grp) + 0.92],
                color=FAINT, lw=0.8)
        ax.text(n_cols + 0.24, ymid, txt, ha="left", va="center", fontsize=6.6,
                color=MUTED, linespacing=1.1)

    ax.set_xlim(-0.2, n_cols + 1.6)
    ax.set_ylim(total_h + 0.55, -0.95)  # inverted: first row on top
    ax.set_aspect("equal")
    ax.axis("off")


def draw_f4(ax):
    y = np.arange(len(F4))
    vals = [v for _, v, _ in F4]
    cols = [c for _, _, c in F4]

    ax.axvspan(0, 6, color=PAPER_BG, zorder=0, lw=0)
    ax.text(6.8, -0.62, "in-context ≤ 6%", fontsize=6.6, color=MUTED,
            fontstyle="italic", va="center", ha="left")

    ax.barh(y, vals, height=0.56, color=cols, edgecolor="none", zorder=2)
    for yi, v in zip(y, vals):
        inside = v >= 60
        ax.text(v - 2 if inside else v + 1.8, yi, f"{v}%", va="center",
                ha="right" if inside else "left", fontsize=6.9,
                color="white" if inside else INK,
                fontweight="bold" if v >= 88 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _, _ in F4], fontsize=7)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("F4 Relational SR (%)", labelpad=2)
    ax.set_xlim(0, 108)
    ax.set_ylim(len(F4) - 0.4, -1.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color=FAINT)
    ax.set_axisbelow(True)


def build():
    fig = plt.figure(figsize=(W_COL, 2.55))
    # manual boxes: (a) needs room for row labels on its left, (b) for tick labels
    ax_a = fig.add_axes([0.20, 0.02, 0.42, 0.86])
    ax_b = fig.add_axes([0.79, 0.16, 0.20, 0.71])
    draw_matrix(ax_a)
    draw_f4(ax_b)
    fig.text(0.005, 0.965, "(a)  L3 success by family (SR %)", fontsize=8.5,
             fontweight="bold", ha="left", va="top")
    fig.text(0.665, 0.965, "(b)  F4 only", fontsize=8.5, fontweight="bold",
             ha="left", va="top")
    save(fig, "planning_wall")


if __name__ == "__main__":
    build()
