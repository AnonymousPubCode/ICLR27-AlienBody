#!/usr/bin/env python3
"""Figure 1 (teaser): the name-prior fallacy.

  (a) one environment, three label strips (named-true / named-misleading /
      anonymous) and the SR each strip yields for GPT-4o on F2
  (b) GPT-4o SR under the three labelings on F1 / F2
  (c) the same confound on two real games (equal square frames)

Output: fig/teaser.pdf (+ .png preview)
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

from figstyle import *  # noqa: F401,F403

GPT4O = {"F1": (74.0, 82.0, 50.0), "F2": (23.0, 40.0, 9.0)}
C_AGENT, C_GOAL, C_CELL, C_GRID = INK, "#C9A65A", "#FAF9F6", "#D5D2CC"


def _fit_square(path, size=320, bg=(246, 244, 240)):
    im = Image.open(path).convert("RGB")
    scale = min(size / im.width, size / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas)


def _rounded(ax, xy, w, h, fc, ec="none", lw=0.0, r=0.08, **kw):
    ax.add_patch(mpatches.FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, **kw))


def panel_a(ax):
    ax.set_xlim(0, 10.0)
    ax.set_ylim(-0.1, 5.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- shared mini grid (left) --
    ox, oy, s = 0.15, 1.15, 0.85
    for i in range(3):
        for j in range(3):
            _rounded(ax, (ox + i * s, oy + j * s), s * 0.88, s * 0.88, C_CELL, C_GRID, 0.7, r=0.06)
    ax.add_patch(mpatches.Circle((ox + 0.44 * s, oy + 2.44 * s), 0.22, facecolor=C_AGENT))
    ax.add_patch(mpatches.Rectangle((ox + 2.2 * s, oy + 0.2 * s), 0.48 * s, 0.48 * s,
                                    facecolor=C_GOAL, edgecolor="none"))
    cx = ox + 1.44 * s
    ax.text(cx, oy - 0.3, "one environment", fontsize=6.4, ha="center", va="top",
            color=MUTED, style="italic")
    ax.text(cx, oy + 3 * s + 0.05, "F2 Spatial", fontsize=6.4, ha="center", va="bottom",
            color=MUTED)

    # -- three label strips (right) --
    rows = [
        (3.45, "named-true", ["right", "left", "up", "down"], 40.0, C_TRUE),
        (1.95, "named-misleading", ["left", "right", "down", "up"], 9.0, C_MIS),
        (0.45, "anonymous", ["act_0", "act_1", "act_2", "act_3"], 23.0, C_ANON),
    ]
    x0, bw, bh, gap = 3.35, 1.12, 0.6, 0.08
    for y, title, labs, sr, col in rows:
        ax.text(x0, y + bh + 0.08, title, fontsize=6.6, color=col, ha="left", va="bottom")
        for k, lab in enumerate(labs):
            x = x0 + k * (bw + gap)
            _rounded(ax, (x, y), bw, bh, col, r=0.08)
            ax.text(x + bw / 2, y + bh / 2, lab, fontsize=5.6, color="white",
                    ha="center", va="center", family=["Consolas", "DejaVu Sans Mono"])
        ax.text(9.95, y + bh / 2, f"{sr:g}%", fontsize=9.5, fontweight="bold",
                color=col, ha="right", va="center")
    ax.text(9.95, 3.45 + bh + 0.08, "GPT-4o SR", fontsize=6.0, color=MUTED,
            ha="right", va="bottom")

    ax.annotate("", xy=(3.2, 2.45), xytext=(ox + 3 * s - 0.05, 2.45),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=6))


def panel_b(ax):
    fams = ["F1 Adapt", "F2 Spatial"]
    x = np.arange(len(fams))
    w = 0.25
    conds = [("anonymous", C_ANON), ("named-true", C_TRUE), ("named-misleading", C_MIS)]
    for k, (lab, col) in enumerate(conds):
        vals = [GPT4O[f][k] for f in ("F1", "F2")]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, edgecolor="none",
                      zorder=3, label=lab)
        bar_labels(ax, bars, dy=1.5, fontsize=6.2)
    ax.set_ylabel("SR (%)", labelpad=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(fams, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=6.5)
    ygrid(ax)
    # colours are keyed in panel (a); no separate legend needed


def panel_c(fig, gs_cell):
    sub = gs_cell.subgridspec(2, 2, wspace=0.06, hspace=0.12)
    frames = [
        (CAPTURES / "kirby_anonymous_step299.png", "anonymous", C_ANON),
        (CAPTURES / "kirby_named_step299.png", "named", C_TRUE),
        (CAPTURES / "crafter_anonymous_step150.png", "anonymous", C_ANON),
        (CAPTURES / "crafter_named_step150.png", "named", C_TRUE),
    ]
    row_titles = ["Kirby, t = 299", "Crafter, t = 150"]
    for idx, (path, label, col) in enumerate(frames):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(sub[r, c])
        ax.imshow(_fit_square(path), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.3)
        if r == 0:
            ax.set_title(label, fontsize=6.6, color=col, pad=2)
        if c == 0:
            ax.text(-0.07, 0.5, row_titles[r], transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=6.2, color=MUTED)


def build():
    # drawn at the true column width so that type renders at its nominal size
    fig = plt.figure(figsize=(W_COL, 1.95))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.75, 1.0, 1.05],
                  wspace=0.22, left=0.005, right=0.995, top=0.86, bottom=0.10)

    ax_a = fig.add_subplot(gs[0]); panel_a(ax_a)
    ax_b = fig.add_subplot(gs[1]); panel_b(ax_b)
    panel_c(fig, gs[2])

    for cell, txt in zip(gs, ["(a) Same environment, only the labels change",
                              "(b) Labels move success",
                              "(c) The same confound in two games"]):
        pos = cell.get_position(fig)
        x = pos.x0 - (0.075 if txt.startswith("(b)") else 0.0)
        fig.text(x, 0.955, txt, fontsize=7.4, fontweight="bold", ha="left", va="center")
    save(fig, "teaser")


if __name__ == "__main__":
    build()
