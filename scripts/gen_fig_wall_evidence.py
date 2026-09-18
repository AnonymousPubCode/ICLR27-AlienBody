#!/usr/bin/env python3
"""Figure (appendix): the F4 wall is neither an artifact nor a data problem.

(a) Oracle-plan agreement per family (Gemini 3.1 Pro text vs. Qwen3-VL-32B
    image, both at L3). Small numerals above the bars: SR%.
(b) Phase-2 action entropy per family: the two models fail F4 with opposite
    signatures (active search vs. degenerate repetition).
(c) Induction frontier: every trained F4 induction variant sits at 2% SR
    (one shared degenerate environment) across a 13x range of training data,
    against the heuristic FMB (88%) and Oracle BFS (100%) ceilings.

Inputs: code/results/action_collapse/{gemini_l3,qwen32bvl_l3}.json
Output: fig/wall_evidence.pdf  (replaces collapse_analysis.pdf + scaling_curve.pdf)
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

# ── (c) data: (F4 training examples, SR%) per trained variant ────────
LORA_9B = [(200, 2.0), (500, 2.0), (1000, 2.0), (2000, 2.0), (2700, 2.0)]
POINTS = [  # (x, y, label, marker, x-jitter factor)
    (2000, 2.0, "Full FT, 4B (text)", "^", 1.10),
    (2000, 2.0, "CoT LoRA, 9B (text)", "D", 0.91),
    (200, 2.0, "LoRA, 27B (text)", "s", 0.90),
    (200, 2.0, "LoRA, 32B (image)", "v", 1.11),
    (212, 2.0, "LoRA multi-obs, 9B (text)", "P", 1.0),
]
HEURISTIC, ORACLE, RANDOM = 88.0, 100.0, 2.0


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


def frontier(ax):
    xs = [p[0] for p in LORA_9B]
    ys = [p[1] for p in LORA_9B]
    ax.plot(xs, ys, "-", color=C_GPT4O, lw=1.2, zorder=3)
    ax.plot(xs, ys, "o", color=C_GPT4O, ms=4, markerfacecolor="white",
            markeredgewidth=1.1, zorder=4, label="LoRA, 9B (text)")
    for x, y, lab, mk, jit in POINTS:
        ax.plot([x * jit], [y], mk, color=SLATE, ms=4.2, markerfacecolor="white",
                markeredgewidth=1.0, zorder=4, label=lab)

    for yv, txt, col in [(ORACLE, f"Oracle BFS  {ORACLE:.0f}%", C_ORACLE),
                         (HEURISTIC, f"FM$_{{\\mathrm{{heur}}}}$  {HEURISTIC:.0f}%", C_FMB)]:
        ax.axhline(yv, color=col, lw=0.9, ls="--", zorder=2)
        ax.text(3600, yv - 2.5, txt, ha="right", va="top", fontsize=6.6, color=col)
    ax.axhline(RANDOM, color=FAINT, lw=0.8, ls=":", zorder=1)
    ax.text(3600, RANDOM + 3, "trained variants: all 2% (random)", ha="right", va="bottom",
            fontsize=6.6, color=MUTED, fontstyle="italic")

    ax.set_xscale("log")
    ax.set_xlim(150, 3800)
    ax.set_xticks([200, 500, 1000, 2000])
    ax.set_xticklabels(["200", "500", "1k", "2k"])
    ax.minorticks_off()
    ax.set_xlabel("F4 training examples", labelpad=2)
    ax.set_ylabel("F4 SR (%)", labelpad=2)
    ax.set_ylim(-5, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ygrid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.48), fontsize=6.0,
              handlelength=1.2, labelspacing=0.35, handletextpad=0.5)


def build():
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(W_COL, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.42})

    grouped(ax_a, "oracle_agree_rate_pct", "Oracle-plan agreement (%)", (0, 110), sr_labels=True)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    grouped(ax_b, "entropy_p2_mean", "Phase-2 action entropy", (0, 1.0))
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    frontier(ax_c)

    # failure-signature callouts in (b), placed directly above the two F4 bars
    gem_f4 = GEM["F4"]["entropy_p2_mean"]
    q_f4 = Q32["F4"]["entropy_p2_mean"]
    ax_b.text(3 - 0.18, gem_f4 + 0.03, "active\nsearch", ha="center", va="bottom",
              fontsize=6.0, color=C_GEMINI, linespacing=1.05)
    ax_b.text(3 + 0.2, q_f4 + 0.03, "one\naction", ha="center", va="bottom",
              fontsize=5.8, color=C_QWEN, linespacing=1.05, zorder=6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS]
    fig.legend(handles, [m[0] for m in MODELS], loc="lower left", ncol=2,
               bbox_to_anchor=(0.06, -0.01), fontsize=6.6, handlelength=1.0,
               columnspacing=1.4)

    panel_label(ax_a, "(a) plan agreement", x=-0.28, y=1.03)
    panel_label(ax_b, "(b) failure signature", x=-0.28, y=1.03)
    panel_label(ax_c, "(c) induction frontier", x=-0.22, y=1.03)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.24)
    save(fig, "wall_evidence")


if __name__ == "__main__":
    build()
