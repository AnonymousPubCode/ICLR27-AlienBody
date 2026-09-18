#!/usr/bin/env python3
"""Figure (appendix): three Phase-1 exploration patterns on Family 1.

Cumulative unique actions discovered vs. Phase-1 steps, reconstructed from
the measured (P1, EC) of each agent:
  Premature  (Qwen3.5-4B,  image): P1 ~ 0.9, EC ~ 0.24 -> 1 step, 1 action
  Redundant  (Qwen3.5-397B, image): P1 ~ 16.6, EC ~ 0.85 -> all 4 after ~4 re-tests each
  Structured (Systematic):          P1 = 4, EC = 1.00     -> one new action per step

Output: fig/exploration_patterns.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from figstyle import *  # noqa: F401,F403

PREMATURE = ([0, 1, 2, 3], [0, 1, 1, 1])
REDUNDANT = (list(range(0, 18)),
             [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4])
STRUCTURED = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
BUDGET = 20


def build():
    fig, ax = plt.subplots(figsize=(W_COL * 0.62, 2.05))

    ax.axvline(4, color=FAINT, lw=0.7, ls="--", zorder=1)
    ax.axvline(BUDGET, color=FAINT, lw=0.7, ls=":", zorder=1)
    ax.text(4, 4.28, "optimal (4 steps)", ha="center", va="bottom", fontsize=6.4, color=MUTED)
    ax.text(BUDGET, 4.28, "budget", ha="center", va="bottom", fontsize=6.4, color=MUTED)

    for (xs, ys), col, lw, name, sub in [
        (REDUNDANT, C_GPT4O, 1.5, "Redundant", "Qwen3.5-397B, image"),
        (PREMATURE, RED, 1.5, "Premature", "Qwen3.5-4B, image"),
        (STRUCTURED, GREEN, 1.8, "Structured", "Systematic script"),
    ]:
        ax.step(xs, ys, where="post", color=col, lw=lw, zorder=3, solid_capstyle="round")
        ax.plot(xs[-1], ys[-1], "o", color=col, ms=3.6, zorder=4)

    # direct labels at the end of each trace
    ax.text(3.4, 0.82, "Premature\nQwen3.5-4B, stops at 1 of 4", ha="left", va="top",
            fontsize=6.6, color=RED, linespacing=1.15)
    ax.text(13.4, 3.55, "Redundant\nQwen3.5-397B, ~4 tries per action", ha="left",
            va="top", fontsize=6.6, color=C_GPT4O, linespacing=1.15)
    ax.text(4.5, 3.5, "Structured\nSystematic, 1 new action / step", ha="left", va="center",
            fontsize=6.6, color=GREEN, linespacing=1.15)

    ax.set_xlabel("Phase-1 step")
    ax.set_ylabel("Unique actions discovered")
    ax.set_xlim(-0.3, 21.5)
    ax.set_ylim(-0.15, 4.75)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_yticks([0, 1, 2, 3, 4])
    ygrid(ax)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.17)
    save(fig, "exploration_patterns")


if __name__ == "__main__":
    build()
