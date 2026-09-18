#!/usr/bin/env python3
"""Figure (appendix): name-prior ablation, GPT-4o vs. Gemini 3.1 Pro.

Same environments, three button-label conditions: anonymous (default),
named-true (labels = real mapping), named-misleading (every label wrong).

Output: fig/name_prior.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import *  # noqa: F401,F403

# (anonymous, named-true, named-misleading) per family
GPT4O = {"F1": (74.0, 82.0, 50.0), "F2": (23.0, 40.0, 9.0)}
GEMINI = {"F1": (96.0, 100.0, 96.0), "F2": (96.0, 100.0, 82.0)}

CONDS = [("anonymous", C_ANON), ("named-true", C_TRUE), ("named-misleading", C_MIS)]
FAMS = ["F1 Adapt", "F2 Spatial"]


def panel(ax, data, title):
    x = np.arange(len(FAMS))
    w = 0.25
    for k, (lab, col) in enumerate(CONDS):
        vals = [data[f][k] for f in ("F1", "F2")]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, edgecolor="none",
                      zorder=3, label=lab)
        bar_labels(ax, bars, dy=1.5)
    # true -> misleading swing, printed once per family above the group
    for i, f in enumerate(("F1", "F2")):
        _, t, m = data[f]
        ax.text(x[i], 106, f"swing $-${t - m:.0f}pp", ha="center", va="bottom",
                fontsize=6.6, color=C_MIS, fontstyle="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(FAMS)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
    ygrid(ax)


def build():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W_COL * 0.85, 2.1), sharey=True)
    panel(ax_a, GPT4O, "(a) GPT-4o (image)")
    panel(ax_b, GEMINI, "(b) Gemini 3.1 Pro (text)")
    ax_a.set_ylabel("SR (%)")
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines["left"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CONDS]
    fig.legend(handles, [l for l, _ in CONDS], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), fontsize=7, handlelength=1.1,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.24, wspace=0.12)
    save(fig, "name_prior")


if __name__ == "__main__":
    build()
