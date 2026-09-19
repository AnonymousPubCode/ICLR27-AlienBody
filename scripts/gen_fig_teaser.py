#!/usr/bin/env python3
"""Figure 1: label intervention, measured success, and game illustrations.

Drawn on a 6.1 x 1.72-inch canvas: the four-frame game panel adds width while
keeping the source figure's height fixed. LaTeX fits it to the column width.
Diagrams, bars, and text remain vector objects; game images preserve native
aspect ratio without cropping.

Numbers match gen_fig_name_prior.py and stat_recheck/stat_recheck_output.txt
section 4: two equal-sized GPT-4o image-evaluation batches, each reusing the
same 50 environment IDs. These are pooled point estimates, not new trials.
Panel (a) is explicitly a schematic F1 mapping, not an observed F2 episode.

The game images are illustrative replay frames ONLY. The archival Crafter
renderer used seed 0 for seed-1 action logs; Kirby selected the best named
rollout. Therefore these images must not be presented as paired, representative,
or verified evaluation trajectories. The four images are restored at the
user's request for side-by-side inspection, not as quantitative evidence.

Preview: python code/scripts/gen_fig_teaser.py
Publish: python code/scripts/gen_fig_teaser.py --out fig
"""
from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from figstyle import CAPTURES, C_ANON, C_MIS, C_TRUE, INK, figure_out_dir


# Order shared by the action matrix and the bars.
CONDITIONS = (
    ("Anonymous", C_ANON, ("act_0", "act_1", "act_2", "act_3")),
    ("True names", C_TRUE, ("right", "left", "up", "down")),
    ("Misleading", C_MIS, ("left", "right", "down", "up")),
)
GPT4O = {"F1": (75.0, 82.0, 50.0), "F2": (25.0, 40.0, 9.0)}

WIDTH, HEIGHT = 6.10, 1.72
SECONDARY = "#5F6970"
RULE = "#D8DDDF"
GOLD = "#C9A65A"


def text(fig, x, y, value, **kwargs):
    """Place text using inches, independently of subplot bounds."""
    return fig.text(x / WIDTH, y / HEIGHT, value, va="center", **kwargs)


def axes(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])


def panel_title(fig, x, letter, title):
    text(fig, x, 1.635, f"({letter})", fontsize=8.3, fontweight="bold")
    text(fig, x + 0.22, 1.635, title, fontsize=8.3, fontweight="bold")


def panel_a(fig):
    ax = axes(fig, 0.04, 0.05, 2.02, 1.72)
    ax.set(xlim=(0, 2.02), ylim=(0, 1.72), aspect="equal")
    ax.axis("off")

    # A compact grid gives context; no trajectory or result is implied.
    gx, gy, cell, gap = 0.045, 1.12, 0.106, 0.016
    for col in range(3):
        for row in range(3):
            ax.add_patch(patches.FancyBboxPatch(
                (gx + col * (cell + gap), gy + row * (cell + gap)),
                cell, cell, boxstyle="round,pad=0,rounding_size=0.018",
                facecolor="#F7F8F8", edgecolor="#CCD2D5", linewidth=0.5,
            ))
    ax.add_patch(patches.Circle(
        (gx + cell / 2, gy + 2 * (cell + gap) + cell / 2),
        0.048, facecolor=INK, edgecolor="none"))
    ax.add_patch(patches.Rectangle(
        (gx + 2 * (cell + gap) + 0.03, gy + 0.03),
        0.082, 0.082, facecolor=GOLD, edgecolor="none"))
    ax.text(gx + 0.23, gy - 0.08, "F1 example", ha="center", va="center",
            fontsize=6.4, color=SECONDARY)
    ax.annotate("", xy=(0.70, 1.36), xytext=(0.52, 1.36),
                arrowprops={"arrowstyle": "-|>", "lw": 0.7,
                            "color": SECONDARY, "mutation_scale": 6})
    ax.text(0.77, 1.42, "Fixed dynamics", fontsize=7.3, va="center")
    ax.text(0.77, 1.295, "Same model", fontsize=7.3, va="center")
    ax.text(0.77, 1.17, "Only labels vary", fontsize=7.3, va="center",
            fontweight="bold")

    # Aligned columns denote fixed effects. Misleading labels form a
    # derangement; anonymous IDs show one schematic example permutation.
    bx, bw, bgap, bh = 0.68, 0.31, 0.028, 0.22
    centers = [bx + i * (bw + bgap) + bw / 2 for i in range(4)]
    ax.text(0.04, 0.915, "Effect", fontsize=6.7, color=SECONDARY, va="center")
    for x, direction in zip(centers, (r"$\rightarrow$", r"$\leftarrow$",
                                       r"$\uparrow$", r"$\downarrow$")):
        ax.text(x, 0.915, direction, ha="center", va="center", fontsize=10)
    ax.plot([0.04, 2.005], [0.810, 0.810], lw=0.5, color=RULE)
    for y, (name, color, labels) in zip((0.545, 0.295, 0.045), CONDITIONS):
        ax.text(0.04, y + bh / 2, name, fontsize=6.9, color=color,
                va="center", fontweight="bold")
        for x, label in zip(centers, labels):
            ax.add_patch(patches.FancyBboxPatch(
                (x - bw / 2, y), bw, bh,
                boxstyle="round,pad=0,rounding_size=0.025",
                facecolor=color, alpha=0.09, edgecolor="none"))
            ax.add_patch(patches.FancyBboxPatch(
                (x - bw / 2, y), bw, bh,
                boxstyle="round,pad=0,rounding_size=0.025",
                facecolor="none", edgecolor=color, linewidth=0.55))
            ax.text(x, y + bh / 2, label, ha="center", va="center",
                    color=color, fontsize=6.0,
                    family=["Consolas", "DejaVu Sans Mono"])


def panel_b(fig):
    text(fig, 2.36, 1.509, "Success rate (%)", fontsize=6.9, color=SECONDARY)
    ax = axes(fig, 2.42, 0.42, 1.31, 0.95)
    centers = np.array([0.0, 1.15])
    width = 0.265
    for k, (_, color, _) in enumerate(CONDITIONS):
        values = [GPT4O[family][k] for family in ("F1", "F2")]
        bars = ax.bar(centers + (k - 1) * width, values, width * 0.89,
                      color=color, edgecolor="none", zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2.5,
                    f"{value:g}", ha="center", va="bottom", fontsize=7.0)
    ax.set(xlim=(-0.52, 1.67), ylim=(0, 88), yticks=[0, 50], xticks=[])
    ax.yaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", labelsize=6.5, length=0, pad=3)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set(color="#7D878C", linewidth=0.55)
    for x, family, subtitle in zip(centers, ("F1", "F2"), ("Cardinal", "Egocentric")):
        ax.text(x, -0.105, family, transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=7.3, fontweight="bold")
        ax.text(x, -0.23, subtitle, transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=6.5, color=SECONDARY)


def panel_c(fig):
    # These are selected archival replays, not verified paired trajectories.
    # Their side-by-side placement supports qualitative inspection only.
    x0, width, gap = 4.42, 0.75, 0.09
    columns = (("anonymous", "Anonymous", C_ANON), ("named", "Named", C_TRUE))
    for col, (_, label, color) in enumerate(columns):
        text(fig, x0 + col * (width + gap) + width / 2, 1.509, label,
             ha="center", fontsize=7.0, fontweight="bold", color=color)
    for game, step, top in (("kirby", 299, 1.449), ("crafter", 150, 0.762)):
        for col, (condition, _, color) in enumerate(columns):
            path = CAPTURES / f"{game}_{condition}_step{step}.png"
            with Image.open(path) as source:
                pixels = np.asarray(source.convert("RGB"))
            height = width * pixels.shape[0] / pixels.shape[1]
            ax = axes(fig, x0 + col * (width + gap), top - height, width, height)
            ax.imshow(pixels, interpolation="nearest", aspect="equal")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.1)
            if col == 0:
                text(fig, x0 - 0.12, top - height / 2,
                     f"{game.title()}, t = {step}", ha="center", rotation=90,
                     fontsize=6.0, color=SECONDARY)
    # Disclaimer lives rotated in the left margin so the panel stays flat.
    text(fig, 3.98, 0.73, "Illustrative replay frames", ha="center",
         rotation=90, fontsize=6.1, color=SECONDARY, style="italic")


def build():
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    panel_title(fig, 0.04, "a", "Label intervention")
    panel_title(fig, 2.27, "b", "GPT-4o success")
    panel_title(fig, 4.00, "c", "Game examples")
    for x in (2.16, 3.88):
        fig.add_artist(plt.Line2D([x / WIDTH, x / WIDTH], [0.06 / HEIGHT, 1.51 / HEIGHT],
                                 transform=fig.transFigure, color=RULE, linewidth=0.5))
    panel_a(fig)
    panel_b(fig)
    panel_c(fig)
    out = figure_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "teaser.pdf"
    # Fixed physical bounds keep this source canvas at exactly 6.1 x 1.72 in.
    fig.savefig(target, facecolor="white",
                metadata={"Title": "AlienBody: label intervention and success"})
    fig.savefig(target.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {target}")


if __name__ == "__main__":
    build()
