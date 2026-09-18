#!/usr/bin/env python3
"""Figure: name-prior ablation — panel (a) three labelings x two models,
panel (b) the six-condition controlled replication on the frozen suite.

Output: build/fig/name_prior.pdf by default; --out fig publishes to the paper.

Panel (a) numbers are the pooled two-batch GPT-4o/Gemini runs recomputed in
``results/stat_recheck/stat_recheck_output.txt`` §4: the anonymous baseline is
75 (F1) / 25 (F2) from the two local waves (82/68 and 26/24), NOT the 74/23
printed in an earlier revision — that value implies a third wave that exists
nowhere on disk.  True/misleading reproduce exactly as the batch average
(F1 82/50, F2 40/9).

Panel (b) is the E-C2 six-condition run (Qwen3.5-9B text, L0, frozen
``replication_primary`` F1-F6 x 100 = 600 envs per condition, H200 vLLM),
with the A800 anonymous leg as a cross-host control.  This model sits at the
benchmark floor (pooled SR 1.0-2.2%), so the panel is drawn on its own axis;
the point it makes is that the manipulation is inert at this scale, which is
consistent with the induction wall and is not evidence against panel (a).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from figstyle import *  # noqa: F401,F403

# (model, family, (anonymous, named-true, named-misleading))  SR %
GROUPS = [
    ("GPT-4o", "F1", (75.0, 82.0, 50.0)),
    ("GPT-4o", "F2", (25.0, 40.0, 9.0)),
    ("Gemini 3.1 Pro", "F1", (96.0, 100.0, 96.0)),
    ("Gemini 3.1 Pro", "F2", (96.0, 100.0, 82.0)),
]
COND_COLS = [C_ANON, C_TRUE, C_MIS]
COND_LABS = ["anonymous", "named-true", "named-misleading"]

# E-C2 six conditions, Qwen3.5-9B, 600 envs each (SR %, successes / 600)
SIX = [
    ("anonymous", 1.67, 10),
    ("true", 1.67, 10),
    ("misleading", 1.00, 6),
    ("nonce\nmatched", 1.50, 9),
    ("synonym\ntrue", 2.17, 13),
    ("id\npermuted", 1.67, 10),
]
SIX_A800_ANON = (1.17, 7)          # cross-host control, same suite


from matplotlib.colors import to_rgb

W, H = 5.5, 2.30
RULE, SECONDARY = "#D8DDDF", "#5F6970"
COLORS = [C_ANON, C_TRUE, C_MIS]


def label(fig, x, y, value, **kw):
    return fig.text(x / W, y / H, value, va="center", **kw)


def soft_color(color, strength=.34):
    """Opaque tint: keep the grid from showing through the column interiors."""
    return tuple(1 - strength * (1 - channel) for channel in to_rgb(color))


def rounded_top_bar(ax, x, value, width, color):
    """1.1-point upper corners, flat clipped base, and exact data-coordinate top."""
    # Convert corner dimensions to screen space so x/y units cannot distort them.
    sx = ax.get_position().width * W * 72 / (ax.get_xlim()[1] - ax.get_xlim()[0])
    sy = ax.get_position().height * H * 72 / (ax.get_ylim()[1] - ax.get_ylim()[0])
    rx, ry = 1.1 / sx, 1.1 / sy
    bar = patches.FancyBboxPatch(
        (x - width / 2, -ry), width, value + ry,
        boxstyle=f"round,pad=0,rounding_size={rx}",
        mutation_aspect=sx / sy, facecolor=soft_color(color),
        edgecolor=color, linewidth=.65, zorder=3, clip_on=True,
    )
    ax.add_patch(bar)
    bar.set_clip_path(ax.patch)


def build():
    fig = plt.figure(figsize=(W, H), facecolor="white")
    label(fig, .04, 2.18, "(a) Label sensitivity", fontsize=8.2, weight="bold")
    label(fig, 3.18, 2.18, "(b) Frozen-suite replication", fontsize=8.2, weight="bold")
    fig.add_artist(plt.Line2D([3.05 / W] * 2, [.04 / H, 2.03 / H],
                             color=RULE, lw=.5, transform=fig.transFigure))

    ax = fig.add_axes([.32 / W, .48 / H, 2.60 / W, 1.24 / H])
    centers = np.array([0., 1.15, 2.60, 3.75])
    width = .28
    ax.set(xlim=(-.58, 4.33), ylim=(0, 112), yticks=[0, 50, 100])
    for group, center in zip(GROUPS, centers):
        for k, (value, color) in enumerate(zip(group[2], COLORS)):
            x = center + (k - 1) * width
            rounded_top_bar(ax, x, value, width * .84, color)
            ax.text(x, value + 2.0, f"{value:g}", fontsize=6.5,
                    ha="center", va="bottom", color=INK)
    ax.set_xticks(centers, ["F1", "F2", "F1", "F2"], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, labelsize=6.6, pad=3)
    label(fig, .04, 1.83, "SR (%)", fontsize=6.3, color=SECONDARY)
    ax.yaxis.grid(True, color=RULE, lw=.5, zorder=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set(color="#7D878C", linewidth=.55)
    for center, name, modality in ((.575, "GPT-4o", "image + text"),
                                   (3.175, "Gemini 3.1 Pro", "image + text")):
        xf = .32 + (center + .58) / 4.91 * 2.60
        label(fig, xf, 1.97, name, fontsize=7.2, weight="bold", ha="center")
        label(fig, xf, 1.83, modality, fontsize=6.5, color=SECONDARY, ha="center")
    # A subtle divider separates models without background bands.
    ax.axvline(1.88, color=RULE, lw=.5, zorder=0)
    ax.legend(handles=[patches.Patch(facecolor=soft_color(c), edgecolor=c, linewidth=.65, label=n)
                       for c, n in zip(COLORS, ["Anonymous", "True names", "Misleading"])],
              loc="upper center", bbox_to_anchor=(.5, -.26), ncol=3,
              fontsize=6.5, handlelength=.9, handletextpad=.4, columnspacing=.9,
              borderaxespad=0, frameon=False)

    label(fig, 4.31, 1.97, "Qwen3.5-9B / text", fontsize=7.2, weight="bold", ha="center")
    label(fig, 4.31, 1.83, "F1--F6; 600 environments per condition", fontsize=6.3,
          color=SECONDARY, ha="center")
    bx = fig.add_axes([4.06 / W, .48 / H, 1.04 / W, 1.24 / H])
    ys = np.arange(6)[::-1]
    names = ["Anonymous", "True names", "Misleading", "Nonce-matched", "Synonym-true", "ID-permuted"]
    for i, ((_, sr, successes), y, name) in enumerate(zip(SIX, ys, names)):
        color = COLORS[i] if i < 3 else SLATE
        # Row guides align the label, estimate, and count. Marker centers encode SR;
        # they are point estimates, not uncertainty intervals.
        bx.plot([0, 3], [y, y], color="#EDF0F2", lw=.65, zorder=1,
                solid_capstyle="butt")
        bx.plot([0, sr], [y, y], color=color, lw=1.7, zorder=3,
                solid_capstyle="butt")
        bx.scatter([sr], [y], s=23, facecolor=color, edgecolor="white",
                   linewidth=.65, zorder=5)
        bx.text(-.13, y, name, ha="right", va="center", fontsize=6.5,
                transform=bx.get_yaxis_transform(), clip_on=False)
        bx.text(1.075, y, f"{successes}/600", ha="left", va="center", fontsize=6.5,
                transform=bx.get_yaxis_transform(), clip_on=False)
    bx.axvline(SIX_A800_ANON[0], color="#68747C", lw=.85,
               ls=(0, (2.5, 2)), zorder=2)
    bx.set(xlim=(0, 3), ylim=(-.55, 5.55), xticks=[0, 1, 2, 3], yticks=[])
    bx.xaxis.grid(True, color=RULE, lw=.5, zorder=0)
    bx.tick_params(axis="x", length=0, labelsize=6.5, pad=4)
    for side in ("left", "right", "top"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set(color="#7D878C", linewidth=.55)
    label(fig, 4.58, .23, "SR (%) / independent scale", fontsize=6.3,
          color=SECONDARY, ha="center")
    fig.add_artist(plt.Line2D([3.20 / W, 3.40 / W], [.085 / H] * 2,
                             color="#68747C", lw=.85, ls=(0, (2.5, 2)),
                             transform=fig.transFigure))
    label(fig, 3.46, .085, "A800 anonymous control: 7/600", fontsize=6.3, color=SECONDARY)
    out = figure_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "name_prior.pdf"
    fig.savefig(target, metadata={"Title": "AlienBody: name-prior ablation"})
    fig.savefig(target.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    build()
