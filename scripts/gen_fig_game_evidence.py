#!/usr/bin/env python3
"""Figure (appendix): the name prior on screen.

Frames from recorded trajectories (deterministic replay) at matched steps:
Kirby's Dream Land (named vs. anonymous, step 299) and Crafter (named vs.
anonymous, step 150). Frames keep their native aspect ratio; both frames in a
row share the same width.

Output: fig/game_evidence.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from figstyle import *  # noqa: F401,F403

ROWS = [
    ("Kirby's Dream Land, step 299 / 300", [
        ("kirby_named_step299.png", "named", "scroll 36, 12 checkpoints", C_TRUE),
        ("kirby_anonymous_step299.png", "anonymous", "scroll 1, still at spawn", C_ANON),
    ]),
    ("Crafter, step 150 / 300", [
        ("crafter_named_step150.png", "named", "4 achievements unlocked", C_TRUE),
        ("crafter_anonymous_step150.png", "anonymous", "0 achievements", C_ANON),
    ]),
]
LETTERS = "abcd"


def build():
    fig = plt.figure(figsize=(W_COL * 0.8, 3.75))
    # height ratios follow the native aspect ratios (Kirby 160x144, Crafter 512x512)
    gs = fig.add_gridspec(2, 2, height_ratios=[144 / 160, 1.0], hspace=0.42, wspace=0.10,
                          left=0.02, right=0.98, top=0.93, bottom=0.07)
    k = 0
    for r, (row_title, cells) in enumerate(ROWS):
        for c, (fname, cond, note, col) in enumerate(cells):
            ax = fig.add_subplot(gs[r, c])
            im = np.asarray(Image.open(CAPTURES / fname).convert("RGB"))
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.4)
            ax.set_title(f"({LETTERS[k]}) {cond}", fontsize=7.6, color=col, loc="left", pad=3)
            ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", va="top",
                    fontsize=7, color=INK)
            k += 1
        # row title centred over the pair
        pos_l = gs[r, 0].get_position(fig)
        pos_r = gs[r, 1].get_position(fig)
        fig.text((pos_l.x0 + pos_r.x1) / 2, pos_l.y1 + 0.035, row_title, ha="center",
                 va="bottom", fontsize=7.6, color=MUTED, fontstyle="italic")
    save(fig, "game_evidence")


if __name__ == "__main__":
    build()
