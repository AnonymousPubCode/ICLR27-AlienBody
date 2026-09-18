#!/usr/bin/env python3
"""Figure: verification-guided induction (FM_ind) on F4/F5/F6.

Three small-multiple slope panels (one per family). Lines: the two LLM
proposers across the Z1 -> Z3 -> Z4 ladder. Horizontal rules: the two
search-only references (Enumerate+Verify, AFMB) that no LLM loop reaches.

Output: fig/fmb_inductive.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import *  # noqa: F401,F403

STAGES = ["Z1\nzero-shot", "Z3\n+verify", "Z4\n+fallback"]

# family -> (GPT-4o, DeepSeek V4, Enumerate+Verify, AFMB)
FAMILIES = [
    ("F4 Relational",    [2, 47, 94],  [8, 40, 94],  98, 100),
    ("F5 Compositional", [26, 65, 62], [30, 85, 80], 100, 100),
    ("F6 Temporal",      [24, 20, 8],  [6, 34, 12],  58, 74),
]
SERIES = [("GPT-4o", C_GPT4O, "o", "-"), ("DeepSeek V4 Pro", C_DS, "s", "--")]


def label_points(ax, x, ya, yb):
    """Value labels; the higher series goes above, the lower below."""
    for xi, a, b in zip(x, ya, yb):
        if a == b:  # shared point: label to the left, clear of the reference rules
            ax.annotate(f"{a}", (xi, a), xytext=(-7, -1), textcoords="offset points",
                        ha="right", va="center", fontsize=6.5, color=INK)
            continue
        hi, lo = (a, b) if a > b else (b, a)
        c_hi, c_lo = (C_GPT4O, C_DS) if a > b else (C_DS, C_GPT4O)
        ax.annotate(f"{hi}", (xi, hi), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=c_hi)
        ax.annotate(f"{lo}", (xi, lo), xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5, color=c_lo)


def build():
    fig, axes = plt.subplots(1, 3, figsize=(W_COL, 2.05), sharey=True)
    x = np.arange(3)

    for k, (ax, (fam, gpt, ds, ev, afmb)) in enumerate(zip(axes, FAMILIES)):
        # search-only references
        if ev == afmb:
            ax.axhline(afmb, color=C_FMB, lw=0.9, ls="-", zorder=1)
            ax.text(-0.42, afmb + 2.5, f"Enum+Verify / AFMB  {afmb}", fontsize=6.3,
                    color=C_FMB, ha="left", va="bottom")
        else:
            ax.axhline(afmb, color=C_FMB, lw=0.9, ls="-", zorder=1)
            ax.axhline(ev, color=MUTED, lw=0.8, ls=":", zorder=1)
            ax.text(-0.42, afmb + 2.5, f"AFMB  {afmb}", fontsize=6.3, color=C_FMB,
                    ha="left", va="bottom")
            ax.text(-0.42, ev - 2.5, f"Enum+Verify  {ev}", fontsize=6.3, color=MUTED,
                    ha="left", va="top")

        for ys, (_, col, mk, ls) in zip((gpt, ds), SERIES):
            ax.plot(x, ys, ls, color=col, lw=1.5 if ls == "-" else 1.2, zorder=3)
            ax.plot(x, ys, mk, color=col, ms=4.4, markerfacecolor="white",
                    markeredgewidth=1.3, zorder=4)
        label_points(ax, x, gpt, ds)

        ax.set_title(fam, fontsize=8, fontweight="bold", pad=5, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=6.8, linespacing=1.1)
        ax.set_xlim(-0.45, 2.45)
        ax.tick_params(axis="x", length=0)
        ygrid(ax)
        if k > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

    axes[0].set_ylim(-10, 118)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("SR (%)", labelpad=3)

    # single legend, in the empty lower-right of the F4 panel
    handles = [plt.Line2D([], [], color=c, marker=m, ls=l, ms=4, markerfacecolor="white",
                          markeredgewidth=1.2, lw=1.2) for _, c, m, l in SERIES]
    axes[0].legend(handles, [s[0] for s in SERIES], loc="lower right",
                   bbox_to_anchor=(1.02, 0.02), fontsize=6.5, handlelength=1.8,
                   labelspacing=0.3)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.2, wspace=0.12)
    save(fig, "fmb_inductive")


if __name__ == "__main__":
    build()
