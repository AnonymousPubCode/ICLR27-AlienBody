#!/usr/bin/env python3
"""Figure: the F4 wall is neither an artifact nor generic collapse.

(a) Oracle-plan agreement per family (Gemini 3.1 Pro text vs. Qwen3-VL-32B
    image, both at L3). Small numerals above the bars: SR%.
(b) Phase-2 action entropy per family: the two models fail F4 with opposite
    signatures (active search vs. degenerate repetition).

The induction-frontier claim (trained variants flat at 2% SR) lives in
Table tab:induction_frontier (method section), not here.

Inputs: code/results/action_collapse/{gemini_l3,qwen32bvl_l3}.json
Output: fig/wall_evidence.pdf
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from figstyle import *  # noqa: F401,F403

FAMS = [f"F{i}" for i in range(1, 7)]
WALL_IDX = [3, 5]  # F4, F6 (the two hard families)

# ── (a)/(b) data ──────────────────────────────────────────────────────
with open(RESULTS / "action_collapse" / "gemini_l3.json") as f:
    GEM = json.load(f)["per_family"]
with open(RESULTS / "action_collapse" / "qwen32bvl_l3.json") as f:
    Q32 = json.load(f)["per_family"]

MODELS = [("Gemini 3.1 Pro (text)", GEM, C_GEMINI), ("Qwen3-VL-32B (image)", Q32, C_QWEN)]


def grouped(ax, key, ylabel, ylim, sr_labels=False, fmt="{:.0f}"):
    x = np.arange(len(FAMS))
    w = 0.36
    for i in WALL_IDX:
        ax.axvspan(i - 0.5, i + 0.5, color=PAPER_BG, lw=0, zorder=0)
    for k, (name, data, col) in enumerate(MODELS):
        vals = [data[f][key] for f in FAMS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.92, color=col, edgecolor="none",
                      zorder=3, label=name)
        if sr_labels:
            for b, f in zip(bars, FAMS):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                        f"{data[f]['sr_pct']:.0f}", ha="center", va="bottom",
                        fontsize=5.8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMS)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(axis="x", length=0)
    ygrid(ax)


def build():
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(W_COL, 2.15), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.38})

    grouped(ax_a, "oracle_agree_rate_pct", "Oracle-plan agreement (%)", (0, 110), sr_labels=True)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    grouped(ax_b, "entropy_p2_mean", "Phase-2 action entropy", (0, 1.0))
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # failure-signature callouts in (b), placed directly above the two F4 bars
    gem_f4 = GEM["F4"]["entropy_p2_mean"]
    q_f4 = Q32["F4"]["entropy_p2_mean"]
    ax_b.text(3 - 0.18, gem_f4 + 0.03, "active\nsearch", ha="center", va="bottom",
              fontsize=6.0, color=C_GEMINI, linespacing=1.05)
    ax_b.text(3 + 0.2, q_f4 + 0.03, "one\naction", ha="center", va="bottom",
              fontsize=5.8, color=C_QWEN, linespacing=1.05, zorder=6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS]
    fig.legend(handles, [m[0] for m in MODELS], loc="lower left", ncol=2,
               bbox_to_anchor=(0.18, -0.02), fontsize=6.6, handlelength=1.0,
               columnspacing=1.4)

    panel_label(ax_a, "(a) plan agreement", x=-0.22, y=1.03)
    panel_label(ax_b, "(b) failure signature", x=-0.22, y=1.03)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22)
    save(fig, "wall_evidence")


if __name__ == "__main__":
    build()
