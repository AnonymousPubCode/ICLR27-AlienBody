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
    ("Crafter, v2 protocol, step 149 / 150", [
        ("crafter_v2_named_seed0_step149.png", "named", "0 achievements, no item collected", C_TRUE),
        ("crafter_v2_anonymous_seed0_step149.png", "anonymous", "2 achievements, wood collected", C_ANON),
    ]),
]
LETTERS = "abcd"


def build():
    # Four native-aspect frames in one compact row, grouped by game.
    W, H = 5.5, 1.88
    fig = plt.figure(figsize=(W,H), facecolor='white')
    for group, title in enumerate(["Kirby's Dream Land | step 299", "Crafter (v2) | step 149"]):
        x = .04 + group*2.76
        fig.text(x/W,1.78/H,title,fontsize=8,weight='bold',va='center')
    fig.add_artist(plt.Line2D([2.75/W]*2,[.10/H,1.83/H],transform=fig.transFigure,
                              color='#D8DDDF',lw=.5))
    for k,(fname,cond,note,col) in enumerate([cell for _,cells in ROWS for cell in cells]):
        x = .05 + (k//2)*2.76 + (k%2)*1.34
        fig.text(x/W,1.56/H,f'({LETTERS[k]}) {cond}',fontsize=6.8,color=col,va='center')
        im=np.asarray(Image.open(CAPTURES/fname).convert('RGB'))
        iw=1.25
        ih=iw*im.shape[0]/im.shape[1]
        ax=fig.add_axes([x/W,(.25+(1.25-ih)/2)/H,iw/W,ih/H])
        ax.imshow(im,interpolation='nearest',aspect='equal')
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True);sp.set_color(col);sp.set_linewidth(1.1)
            sp.set_linestyle('-' if cond=='named' else '--')
        fig.text((x+iw/2)/W,.12/H,note,fontsize=5.7,color=INK,ha='center',va='center')
    out=figure_out_dir();out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'game_evidence.pdf')
    fig.savefig(out/'game_evidence.png',dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    build()
