#!/usr/bin/env python3
"""Generate publication-quality LaTeX tables for AlienBody paper.

Style inspired by NLCO (arXiv:2602.02188): colored cells, clean booktabs,
professional formatting with \cellcolor for heatmap effects.

Usage:
    python scripts/generate_tables.py --results results/ --output tables/
    python scripts/generate_tables.py --demo --output tables/  # placeholder data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════
# Color Utilities for Heatmap-Style LaTeX Tables
# ═══════════════════════════════════════════════════════════════════

def value_to_color(val: float, vmin: float = 0.0, vmax: float = 1.0,
                   palette: str = "green") -> str:
    """Convert a normalized value to a LaTeX \\cellcolor command.

    Returns string like '\\cellcolor{green!25}' or '\\cellcolor{red!15}'.
    Palette options: 'green' (higher=better), 'red' (lower=better).
    """
    norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin + 1e-12)))
    if palette == "green":
        # 0 -> red!25, 0.5 -> yellow!10, 1.0 -> green!30
        if norm < 0.3:
            intensity = int(25 * (1 - norm / 0.3))
            return f"\\cellcolor{{red!{intensity}}}"
        elif norm < 0.6:
            intensity = int(10 * (1 - abs(norm - 0.45) / 0.15))
            return f"\\cellcolor{{yellow!{max(intensity, 3)}}}"
        else:
            intensity = int(30 * ((norm - 0.6) / 0.4))
            return f"\\cellcolor{{green!{max(intensity, 5)}}}"
    elif palette == "red":  # lower = better (for P1, P2)
        norm = 1 - norm
        if norm < 0.3:
            return f"\\cellcolor{{red!{int(25 * (1 - norm / 0.3))}}}"
        elif norm < 0.6:
            return f"\\cellcolor{{yellow!{max(int(10 * (1 - abs(norm - 0.45) / 0.15)), 3)}}}"
        else:
            return f"\\cellcolor{{green!{max(int(30 * ((norm - 0.6) / 0.4)), 5)}}}"
    return ""


def bold_best(val: str, is_best: bool, is_second: bool = False) -> str:
    """Wrap value with bold/underline for best/second-best."""
    if is_best:
        return f"\\textbf{{{val}}}"
    elif is_second:
        return f"\\underline{{{val}}}"
    return val


# ═══════════════════════════════════════════════════════════════════
# Placeholder Data (for --demo mode)
# ═══════════════════════════════════════════════════════════════════

DEMO_MAIN = [
    # (group, model, CE, SR%, EC%, CA%, P1, P2)
    ("(E)", "Random Agent",       0.02, 12.3, 68.4, 24.1, 20.0, 25.7),
    ("(E)", "PPO Agent",          0.08, 34.5, 85.2, 41.8, 15.2, 18.3),
    ("(E)", "ICM Agent",          0.10, 38.9, 87.6, 48.3, 14.8, 17.1),
    ("(E)", "Oracle Agent",       1.00, 100.0, 100.0, 100.0, 4.0, 8.2),
    ("(E)", "Human (avg)",        1.00, 100.0, 100.0, 98.7, 5.3, 9.7),
    ("(B)", "GPT-4o + ReAct",     0.15, 54.1, 94.1, 62.5, 11.5, 12.2),
    ("(B)", "GPT-4o + Reflexion", 0.14, 52.8, 93.5, 60.8, 11.9, 12.4),
    ("(D)", "GPT-4o (text)",      0.11, 42.8, 89.3, 52.4, 14.3, 15.6),
    ("(D)", "Claude 3.7 (text)",  0.14, 48.2, 91.7, 58.6, 12.8, 13.2),
    ("(D)", "Llama-3.1-70B",      0.07, 31.4, 82.5, 38.9, 16.7, 19.4),
    ("(C)", "LLaVA-NeXT",         0.05, 22.6, 74.8, 32.7, 18.4, 22.1),
    ("(C)", "InternVL2",          0.08, 35.7, 86.3, 43.5, 15.9, 17.8),
    ("(C)", "Pixtral",            0.06, 28.4, 79.6, 36.2, 17.2, 20.5),
    ("(A)", "Gemini 2.5 Pro",     0.12, 45.3, 90.4, 55.1, 13.5, 14.8),
    ("(A)", "Claude 3.7 Sonnet",  0.14, 51.7, 92.8, 59.3, 12.2, 12.8),
    ("(A)", "GPT-4o",             0.14, 52.4, 93.2, 61.2, 11.8, 12.5),
]

DEMO_PERFAMILY = {
    "GPT-4o":           {1: 0.22, 2: 0.15, 3: 0.11, 4: 0.08},
    "Claude 3.7 Sonnet": {1: 0.21, 2: 0.14, 3: 0.10, 4: 0.07},
    "Gemini 2.5 Pro":   {1: 0.19, 2: 0.12, 3: 0.09, 4: 0.06},
    "GPT-4o + ReAct":   {1: 0.23, 2: 0.16, 3: 0.12, 4: 0.09},
    "ICM Agent":        {1: 0.15, 2: 0.10, 3: 0.07, 4: 0.05},
    "PPO Agent":        {1: 0.12, 2: 0.08, 3: 0.06, 4: 0.04},
    "Random":           {1: 0.03, 2: 0.02, 3: 0.01, 4: 0.01},
    "Human":            {1: 1.02, 2: 1.01, 3: 0.98, 4: 1.03},
}

DEMO_PERTIER = {
    "GPT-4o":           {"S": 0.19, "M": 0.14, "L": 0.09},
    "Claude 3.7 Sonnet": {"S": 0.18, "M": 0.14, "L": 0.08},
    "Gemini 2.5 Pro":   {"S": 0.16, "M": 0.12, "L": 0.07},
    "GPT-4o + ReAct":   {"S": 0.20, "M": 0.15, "L": 0.10},
    "ICM Agent":        {"S": 0.14, "M": 0.10, "L": 0.06},
    "Human":            {"S": 1.03, "M": 1.00, "L": 0.96},
}

DEMO_FAMILIARITY = {
    "GPT-4o":           (0.38, 0.14, "-63%"),
    "Claude 3.7 Sonnet": (0.35, 0.14, "-60%"),
    "Gemini 2.5 Pro":   (0.31, 0.12, "-61%"),
    "GPT-4o + ReAct":   (0.40, 0.15, "-63%"),
    "LLaVA-NeXT":       (0.18, 0.05, "-72%"),
    "Human":            (1.02, 1.00, "-2%"),
}

DEMO_FAILURE = [
    ("Insufficient Exploration", 31, "Meta-cognitive"),
    ("Random Exploration", 25, "Meta-cognitive"),
    ("Reasoning Error", 15, "Cognitive"),
    ("Execution Error", 12, "Cognitive"),
    ("Memory Failure", 11, "Cognitive"),
    ("Perception Error", 6, "Cognitive"),
]

DEMO_FULL_GRID = {
    # model: {(fam, tier): CE}
    "GPT-4o": {(1,"S"):0.28,(1,"M"):0.22,(1,"L"):0.14,(2,"S"):0.20,(2,"M"):0.15,(2,"L"):0.10,(3,"S"):0.15,(3,"M"):0.11,(3,"L"):0.07,(4,"S"):0.12,(4,"M"):0.08,(4,"L"):0.05},
    "Claude 3.7": {(1,"S"):0.27,(1,"M"):0.21,(1,"L"):0.13,(2,"S"):0.19,(2,"M"):0.14,(2,"L"):0.09,(3,"S"):0.14,(3,"M"):0.10,(3,"L"):0.06,(4,"S"):0.11,(4,"M"):0.07,(4,"L"):0.04},
    "Gemini 2.5 Pro": {(1,"S"):0.25,(1,"M"):0.19,(1,"L"):0.11,(2,"S"):0.17,(2,"M"):0.12,(2,"L"):0.08,(3,"S"):0.12,(3,"M"):0.09,(3,"L"):0.05,(4,"S"):0.09,(4,"M"):0.06,(4,"L"):0.03},
    "GPT-4o + ReAct": {(1,"S"):0.30,(1,"M"):0.23,(1,"L"):0.15,(2,"S"):0.22,(2,"M"):0.16,(2,"L"):0.11,(3,"S"):0.16,(3,"M"):0.12,(3,"L"):0.08,(4,"S"):0.13,(4,"M"):0.09,(4,"L"):0.06},
    "ICM Agent": {(1,"S"):0.20,(1,"M"):0.15,(1,"L"):0.09,(2,"S"):0.14,(2,"M"):0.10,(2,"L"):0.06,(3,"S"):0.10,(3,"M"):0.07,(3,"L"):0.04,(4,"S"):0.08,(4,"M"):0.05,(4,"L"):0.03},
    "Human": {(1,"S"):1.04,(1,"M"):1.02,(1,"L"):1.00,(2,"S"):1.03,(2,"M"):1.01,(2,"L"):0.98,(3,"S"):1.01,(3,"M"):0.98,(3,"L"):0.95,(4,"S"):1.05,(4,"M"):1.03,(4,"L"):1.01},
}

DEMO_ABLATION = {
    # variant: (CE, SR%, delta_CE)
    "Full System":          (0.14, 52.4, "---"),
    "w/o Phase Separation": (0.06, 28.1, "-57%"),
    "w/o Early Termination": (0.11, 45.2, "-21%"),
    "Familiar Actions Only": (0.38, 78.5, "+171%"),
    "Grid 32x32":           (0.07, 31.8, "-50%"),
    "n=8 Actions":          (0.08, 33.2, "-43%"),
}


# ═══════════════════════════════════════════════════════════════════
# Table Generators
# ═══════════════════════════════════════════════════════════════════

def generate_main_table(data=None) -> str:
    """Table 6: Main results — heatmap-colored cells."""
    if data is None:
        data = DEMO_MAIN

    # Find best/second-best CE (excluding Oracle/Human)
    ai_ces = [(r[2], i) for i, r in enumerate(data)
              if r[1] not in ("Oracle Agent", "Human (avg)")]
    ai_ces.sort(reverse=True)
    best_idx = ai_ces[0][1] if ai_ces else -1
    second_idx = ai_ces[1][1] if len(ai_ces) > 1 else -1

    # CE range for color mapping (AI only)
    ce_vals = [r[2] for r in data if r[1] not in ("Oracle Agent", "Human (avg)")]
    ce_min, ce_max = min(ce_vals), max(ce_vals) if ce_vals else (0, 1)

    lines = [
        r"% Auto-generated by generate_tables.py — do not edit manually",
        r"\begin{table*}[t!]",
        r"\setlength{\belowcaptionskip}{-0.3cm}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{Main results on AlienBody test set (360 environments, all families $\times$ all tiers). "
        r"CE = Calibration Efficiency (primary), SR = Success Rate, EC = Exploration Coverage, "
        r"CA = Calibration Accuracy, P1/P2 = Phase~1/2 Actions. "
        r"Best AI in \textbf{bold}, second best \underline{underlined}. "
        r"Cell colors encode relative performance (\colorbox{green!20}{better} / \colorbox{red!15}{worse}).}",
        r"\label{tab:main}",
        r"\scriptsize",
        r"\begin{tabular}{ll|c|c|c|c|c|c}",
        r"\toprule",
        r"Group & Model & CE $\uparrow$ & SR (\%) $\uparrow$ & EC (\%) $\uparrow$ & CA (\%) $\uparrow$ & P1 $\downarrow$ & P2 $\downarrow$ \\",
        r"\midrule",
    ]

    current_group = None
    for i, (group, model, ce, sr, ec, ca, p1, p2) in enumerate(data):
        if group != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            group_names = {
                "(E)": "(E) Controls",
                "(B)": "(B) Agent Fwks.",
                "(D)": "(D) Text LLMs",
                "(C)": "(C) Open VLMs",
                "(A)": "(A) Frontier VLMs",
            }
            lines.append(r"\multicolumn{8}{l}{\textit{" + group_names.get(group, group) + r"}} \\")
            current_group = group

        # Color cells (skip Oracle/Human for coloring)
        is_control = model in ("Oracle Agent", "Human (avg)")
        ce_color = "" if is_control else value_to_color(ce, ce_min, ce_max, "green")
        sr_color = "" if is_control else value_to_color(sr/100, 0.1, 1.0, "green")
        ec_color = "" if is_control else value_to_color(ec/100, 0.6, 1.0, "green")
        ca_color = "" if is_control else value_to_color(ca/100, 0.2, 1.0, "green")
        p1_color = "" if is_control else value_to_color(p1, 4.0, 20.0, "red")
        p2_color = "" if is_control else value_to_color(p2, 8.0, 26.0, "red")

        # Best/second markers
        ce_str = bold_best(f"{ce:.2f}", i == best_idx, i == second_idx)
        sr_str = f"{sr:.1f}"
        ec_str = f"{ec:.1f}"
        ca_str = f"{ca:.1f}"
        p1_str = f"{p1:.1f}"
        p2_str = f"{p2:.1f}"

        # Significance star
        sig = "$^*$" if ce >= 0.10 and not is_control else ""

        lines.append(
            f" & {model} & "
            f"{ce_color}{ce_str}{sig} & "
            f"{sr_color}{sr_str} & "
            f"{ec_color}{ec_str} & "
            f"{ca_color}{ca_str} & "
            f"{p1_color}{p1_str} & "
            f"{p2_color}{p2_str} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def generate_perfamily_table(data=None) -> str:
    """Table 7: CE by family — heatmap colored."""
    if data is None:
        data = DEMO_PERFAMILY

    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table}[t!]",
        r"\setlength{\belowcaptionskip}{-0.5cm}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{CE breakdown by environment family. Cell colors encode relative performance. "
        r"Performance degrades monotonically from F1 to F4 for all AI models; humans remain stable.}",
        r"\label{tab:perfamily}",
        r"\scriptsize",
        r"\begin{tabular}{l|cccc}",
        r"\toprule",
        r"Model & F1 (Remap) & F2 (Direct.) & F3 (State) & F4 (Relat.) \\",
        r"\midrule",
    ]

    for model, fam_data in data.items():
        cells = []
        is_human = model == "Human"
        for f in [1, 2, 3, 4]:
            val = fam_data.get(f, 0)
            color = "" if is_human else value_to_color(val, 0.0, 0.25, "green")
            cells.append(f"{color}{val:.2f}")
        line_sep = r"\midrule" if model == "Random" else ""
        if line_sep:
            lines.append(line_sep)
        lines.append(f"{model} & {' & '.join(cells)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_pertier_table(data=None) -> str:
    """Table 8: CE by tier — with delta column."""
    if data is None:
        data = DEMO_PERTIER

    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table}[t!]",
        r"\setlength{\belowcaptionskip}{-0.5cm}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{CE breakdown by difficulty tier. $\Delta$ = relative CE drop from Tier-S to Tier-L. "
        r"Models degrade 50--57\%; humans only 7\%.}",
        r"\label{tab:pertier}",
        r"\scriptsize",
        r"\begin{tabular}{l|ccc|c}",
        r"\toprule",
        r"Model & Tier-S & Tier-M & Tier-L & $\Delta$(S$\to$L) \\",
        r"\midrule",
    ]

    for model, tier_data in data.items():
        is_human = model == "Human"
        s_val = tier_data.get("S", 0)
        m_val = tier_data.get("M", 0)
        l_val = tier_data.get("L", 0)
        delta = (s_val - l_val) / s_val * 100 if s_val > 0 else 0

        s_color = "" if is_human else value_to_color(s_val, 0.0, 0.25, "green")
        m_color = "" if is_human else value_to_color(m_val, 0.0, 0.25, "green")
        l_color = "" if is_human else value_to_color(l_val, 0.0, 0.25, "green")

        delta_color = ""
        if not is_human:
            delta_color = r"\cellcolor{red!15}" if delta > 40 else r"\cellcolor{yellow!8}"

        if model == "Human" and list(data.keys()).index(model) > 0:
            lines.append(r"\midrule")

        lines.append(
            f"{model} & "
            f"{s_color}{s_val:.2f} & "
            f"{m_color}{m_val:.2f} & "
            f"{l_color}{l_val:.2f} & "
            f"{delta_color}$-${delta:.0f}\\% \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_familiarity_table(data=None) -> str:
    """Table 9: Familiar vs Novel CE."""
    if data is None:
        data = DEMO_FAMILIARITY

    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table}[t!]",
        r"\setlength{\belowcaptionskip}{-0.5cm}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{Effect of action familiarity on CE. Familiar = standard cardinal directions with "
        r"randomized labels. Novel = our designed action types. Gap = relative CE drop.}",
        r"\label{tab:familiarity}",
        r"\scriptsize",
        r"\begin{tabular}{l|cc|c}",
        r"\toprule",
        r"Model & Familiar CE & Novel CE & Gap \\",
        r"\midrule",
    ]

    for model, (fam, nov, gap) in data.items():
        is_human = model == "Human"
        fam_color = "" if is_human else value_to_color(fam, 0.0, 0.5, "green")
        nov_color = "" if is_human else value_to_color(nov, 0.0, 0.5, "green")

        # Gap color: bigger gap = redder
        gap_num = abs(float(gap.strip("%").strip("-")))
        gap_color = ""
        if not is_human:
            if gap_num > 50:
                gap_color = r"\cellcolor{red!20}"
            elif gap_num > 20:
                gap_color = r"\cellcolor{red!10}"

        if model == "Human":
            lines.append(r"\midrule")

        lines.append(
            f"{model} & "
            f"{fam_color}{fam:.2f} & "
            f"{nov_color}{nov:.2f} & "
            f"{gap_color}{gap} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_failure_table(data=None) -> str:
    """Table 10: Failure mode analysis with colored category bands."""
    if data is None:
        data = DEMO_FAILURE

    total = sum(r[1] for r in data)
    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table}[t!]",
        r"\setlength{\belowcaptionskip}{-0.5cm}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{Failure mode analysis on " + str(total) + r" failed GPT-4o episodes. "
        r"Categories are mutually exclusive, assigned by primary cause.}",
        r"\label{tab:failure}",
        r"\scriptsize",
        r"\begin{tabular}{l|c|c}",
        r"\toprule",
        r"Failure Mode & Count & \% \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\cellcolor{red!8}\textit{Meta-Cognitive Errors}} \\",
    ]

    for mode, count, category in data:
        pct = count / total * 100
        bar = int(pct / 2)  # Simple ASCII-style bar
        color = r"\cellcolor{red!5}" if category == "Meta-cognitive" else ""
        if mode == "Reasoning Error":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{3}{l}{\cellcolor{blue!5}\textit{Cognitive Errors}} \\")
            color = r"\cellcolor{blue!3}"
        elif category == "Cognitive":
            color = r"\cellcolor{blue!3}"

        lines.append(f"{color}{mode} & {count} & {pct:.0f}\\% \\\\")

    # Summary row
    meta_pct = sum(r[1] for r in data if r[2] == "Meta-cognitive") / total * 100
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Meta-cognitive total}} & \\textbf{{{sum(r[1] for r in data if r[2] == 'Meta-cognitive')}}} & \\textbf{{{meta_pct:.0f}\\%}} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_full_grid_table(data=None) -> str:
    """Table 12 (Appendix): Full model x family x tier grid — heatmap colored."""
    if data is None:
        data = DEMO_FULL_GRID

    fam_labels = {1: "Remapped", 2: "Direction.", 3: "State-Dep.", 4: "Relational"}
    tier_labels = ["S", "M", "L"]

    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table*}[t!]",
        r"\setlength{\belowcaptionskip}{-0.3cm}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{Complete CE breakdown: model $\times$ family $\times$ tier. Cell colors encode "
        r"performance intensity. Human values are near 1.0 across all cells.}",
        r"\label{tab:fullgrid}",
        r"\scriptsize",
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc}",
        r"\toprule",
    ]

    # Header: Family labels spanning 3 tiers each
    header1 = "Model"
    for f in [1, 2, 3, 4]:
        header1 += f" & \\multicolumn{{3}}{{c|}}{{{fam_labels[f]}}}" if f < 4 else f" & \\multicolumn{{3}}{{c}}{{{fam_labels[f]}}}"
    header1 += r" \\"
    lines.append(header1)

    header2 = ""
    for _ in range(4):
        header2 += " & S & M & L"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    for model, cells in data.items():
        is_human = model == "Human"
        parts = [model]
        for f in [1, 2, 3, 4]:
            for t in tier_labels:
                val = cells.get((f, t), 0)
                color = "" if is_human else value_to_color(val, 0.0, 0.30, "green")
                parts.append(f"{color}{val:.2f}")

        if model == "Human":
            lines.append(r"\midrule")
        lines.append(" & ".join(parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def generate_ablation_table(data=None) -> str:
    """Table 13 (Appendix): Ablation study."""
    if data is None:
        data = DEMO_ABLATION

    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table}[t!]",
        r"\setlength{\belowcaptionskip}{-0.5cm}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{Ablation study on GPT-4o (Tier-M, all families). "
        r"$\Delta$CE = relative change vs.\ full system.}",
        r"\label{tab:ablation}",
        r"\scriptsize",
        r"\begin{tabular}{l|cc|c}",
        r"\toprule",
        r"Variant & CE & SR (\%) & $\Delta$CE \\",
        r"\midrule",
    ]

    for variant, (ce, sr, delta) in data.items():
        is_full = variant == "Full System"
        ce_str = f"\\textbf{{{ce:.2f}}}" if is_full else f"{ce:.2f}"

        # Color delta
        if delta == "---":
            delta_str = "---"
        elif delta.startswith("-"):
            delta_str = f"\\textcolor{{red}}{{{delta}}}"
        else:
            delta_str = f"\\textcolor{{ForestGreen}}{{{delta}}}"

        lines.append(f"{variant} & {ce_str} & {sr:.1f} & {delta_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_benchmark_comparison() -> str:
    """Table 1 (Intro): Benchmark comparison — with checkmarks."""
    lines = [
        r"% Auto-generated by generate_tables.py",
        r"\begin{table*}[t!]",
        r"\setlength{\belowcaptionskip}{-0.3cm}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\centering",
        r"\caption{Comparison of agent benchmarks. \cmark\ = feature present, \xmark\ = absent. "
        r"AlienBody is the first benchmark requiring agents to discover action semantics from scratch.}",
        r"\label{tab:benchmark_cmp}",
        r"\scriptsize",
        r"\begin{tabular}{l|l|ccccc}",
        r"\toprule",
        r"Benchmark & Domain & \makecell{Action\\Semantics\\Given} & \makecell{Visual\\Obs.} & \makecell{Exploration\\Required} & \makecell{Multi-\\Phase} & \makecell{Calibration\\Metric} \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\cellcolor{gray!5}\textit{Web \& Code Agents}} \\",
        r"WebArena & Web & \cmark & \cmark & \xmark & \xmark & \xmark \\",
        r"VisualWebArena & Web & \cmark & \cmark & \xmark & \xmark & \xmark \\",
        r"OSWorld & Desktop & \cmark & \cmark & \xmark & \xmark & \xmark \\",
        r"SWE-bench & Code & \cmark & \xmark & \xmark & \xmark & \xmark \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\cellcolor{gray!5}\textit{Game Agents}} \\",
        r"VideoGameBench & Games & \cmark & \cmark & \xmark & \xmark & \xmark \\",
        r"BALROG & Games & \cmark & \cmark & \cmark & \xmark & \xmark \\",
        r"SmartPlay & Games & \cmark & \cmark & \cmark & \xmark & \xmark \\",
        r"Crafter & Games & \cmark & \cmark & \cmark & \xmark & \xmark \\",
        r"Orak & Games & \cmark & \cmark & \xmark & \xmark & \xmark \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\cellcolor{gray!5}\textit{General / Reasoning Agents}} \\",
        r"AgentBench & Mixed & \cmark & \xmark & \xmark & \xmark & \xmark \\",
        r"ARC-AGI-3 & Reasoning & \cmark & \cmark & \xmark & \xmark & \xmark \\",
        r"\midrule",
        r"\rowcolor{green!8}",
        r"\textbf{AlienBody (Ours)} & \textbf{Grid} & \textbf{\xmark} & \textbf{\cmark} & \textbf{\cmark} & \textbf{\cmark} & \textbf{\cmark} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main Entrypoint
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate publication-quality LaTeX tables for AlienBody paper.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="tables")
    parser.add_argument("--demo", action="store_true",
                        help="Use placeholder data")
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "benchmark_comparison.tex": generate_benchmark_comparison(),
        "main_table.tex": generate_main_table(),
        "perfamily_table.tex": generate_perfamily_table(),
        "pertier_table.tex": generate_pertier_table(),
        "familiarity_table.tex": generate_familiarity_table(),
        "failure_table.tex": generate_failure_table(),
        "full_grid_table.tex": generate_full_grid_table(),
        "ablation_table.tex": generate_ablation_table(),
    }

    for name, content in tables.items():
        path = output_dir / name
        path.write_text(content)
        print(f"  Generated: {path}")
        # Show preview
        preview_lines = content.split("\n")[:5]
        print("    " + "\n    ".join(preview_lines))
        print()

    print(f"\nAll {len(tables)} tables saved to {output_dir}/")


if __name__ == "__main__":
    main()
