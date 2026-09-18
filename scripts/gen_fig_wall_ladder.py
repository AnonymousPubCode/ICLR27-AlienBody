#!/usr/bin/env python3
"""Figure: F4 interface ladder (slope chart). The exact-SR table is typeset
in LaTeX next to this plot, so the plot carries only the shape of the jump.

Output: fig/wall_ladder.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import *  # noqa: F401,F403

INTERFACES = ["L3 only", "+ chat\n$\\mathtt{next\\_state}$", "+ code\nREPL", "Oracle\nBFS"]
SERIES = [
    ("GPT-4o",          [0, 26, 98, 100], C_GPT4O, "o", "-"),
    ("DeepSeek V4 Pro", [0, 16, 100, 100], C_DS, "s", "--"),
    ("GPT-5.1",         [4, 10, 90, 100], C_GPT51, "^", "--"),
]


def build():
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    x = np.arange(len(INTERFACES))

    # in-context region vs. code region
    ax.axvspan(-0.5, 1.5, color=PAPER_BG, lw=0, zorder=0)
    ax.text(0.5, 118, "search in dialogue", ha="center", va="center", fontsize=6.6,
            color=MUTED, fontstyle="italic")
    ax.text(2.5, 118, "search as code", ha="center", va="center", fontsize=6.6,
            color=MUTED, fontstyle="italic")

    for label, ys, col, mk, ls in SERIES:
        ax.plot(x, ys, ls, color=col, lw=1.6 if ls == "-" else 1.2, zorder=3)
        ax.plot(x, ys, mk, color=col, ms=4.6, markerfacecolor="white",
                markeredgewidth=1.3, zorder=4)

    # sparse value labels: the chat step (small gains) and the code step (the jump)
    ax.annotate("26", (1, 26), xytext=(-5, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.8, color=C_GPT4O, fontweight="bold")
    ax.annotate("16", (1, 16), xytext=(6, -1), textcoords="offset points",
                ha="left", va="top", fontsize=6.8, color=C_DS)
    ax.annotate("10", (1, 10), xytext=(6, -8), textcoords="offset points",
                ha="left", va="top", fontsize=6.8, color=C_GPT51)
    ax.annotate("98", (2, 98), xytext=(-6, -2), textcoords="offset points",
                ha="right", va="top", fontsize=6.8, color=C_GPT4O, fontweight="bold")
    ax.annotate("100", (2, 100), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.8, color=C_DS)
    ax.annotate("90", (2, 90), xytext=(7, -3), textcoords="offset points",
                ha="left", va="top", fontsize=6.8, color=C_GPT51)

    ax.set_xticks(x)
    ax.set_xticklabels(INTERFACES, fontsize=7, linespacing=1.1)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("F4 Relational SR (%)", labelpad=3)
    ax.set_ylim(-4, 126)
    ax.set_xlim(-0.5, 3.5)
    ax.tick_params(axis="x", length=0)
    ygrid(ax)

    # legend: markers only (the lines are already colour-coded)
    handles = [plt.Line2D([], [], color=c, marker=m, ls=l, ms=4, markerfacecolor="white",
                          markeredgewidth=1.2, lw=1.2) for _, _, c, m, l in SERIES]
    ax.legend(handles, [s[0] for s in SERIES], loc="center left",
              bbox_to_anchor=(0.0, 0.66), fontsize=6.6, handlelength=1.8)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.2)
    save(fig, "wall_ladder")


if __name__ == "__main__":
    build()
