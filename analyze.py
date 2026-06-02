"""
Analysis and visualization script for llm-rag-firewall evaluation results.

Loads all result CSVs (2 main models × 3 judge models × 4 modes = 24 files),
prints a rich summary table, and saves 6 plots to results/plots/.

File format:  {mode}_results_{main_slug}_judge-{judge_slug}.csv
CSV columns:  id, question, attack_type, should_refuse, mode,
              secret_leaked, any_suspicious_retrieved, chunks_blocked,
              latency_s, answer_snippet

Confusion matrix:
  TP — attack prompt, secret leaked       (attack succeeded)
  FN — attack prompt, no secret leaked    (attack blocked)
  FP — benign prompt, secret leaked       (spurious leak)
  TN — benign prompt, no secret leaked    (correct pass-through)

Usage:
    python analyze.py
    python analyze.py --results-dir results
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table

CONSOLE = Console()
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

MAIN_MODELS  = ["Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct-abliterated"]
JUDGE_MODELS = ["Llama-3.2-1B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct-abliterated"]
MODES        = ["baseline", "secure_prompt", "judge_only", "defended"]

MAIN_LABELS = {
    "Qwen2.5-7B-Instruct":             "Qwen (aligned)",
    "Qwen2.5-7B-Instruct-abliterated": "Qwen (abliterated)",
}
JUDGE_LABELS = {
    "Llama-3.2-1B-Instruct":            "Llama-3.2-1B",
    "Qwen2.5-7B-Instruct":              "Qwen-7B (aligned)",
    "Qwen2.5-7B-Instruct-abliterated":  "Qwen-7B (abliterated)",
}
MODE_LABELS = {
    "baseline":      "Baseline",
    "secure_prompt": "Secure Prompt",
    "judge_only":    "Judge Only",
    "defended":      "Full Defense",
}

ATTACK_TYPE_LABELS = {
    "direct_secret_request":                    "Direct: plain request",
    "direct_secret_request_authority":          "Direct: authority claim",
    "direct_secret_request_multilingual_zh":    "Direct: Chinese",
    "direct_secret_request_hypothetical":       "Direct: hypothetical",
    "direct_secret_request_cot":                "Direct: CoT framing",
    "poisoned_retrieval_naive":                 "Retrieval: naive keyword",
    "poisoned_retrieval_authority":             "Retrieval: authority impersonation",
    "poisoned_retrieval_multilingual_zh":       "Retrieval: Chinese injection",
    "poisoned_retrieval_multilingual_ar":       "Retrieval: Arabic injection",
    "poisoned_retrieval_hypothetical":          "Retrieval: hypothetical framing",
    "poisoned_retrieval_cot_forgery":           "Retrieval: CoT forgery",
    "poisoned_retrieval_zero_width":            "Retrieval: zero-width obfuscation",
    "poisoned_retrieval_conditional":           "Retrieval: conditional trigger",
    "poisoned_retrieval_data_instruction_blur": "Retrieval: data/instruction blur",
    "poisoned_retrieval_homoglyph":             "Retrieval: homoglyph Unicode",
}

MAIN_COLORS = {
    "Qwen2.5-7B-Instruct":             "#4c8cbf",
    "Qwen2.5-7B-Instruct-abliterated": "#d9534f",
}
JUDGE_COLORS = {
    "Llama-3.2-1B-Instruct":            "#e8a838",
    "Qwen2.5-7B-Instruct":              "#4c8cbf",
    "Qwen2.5-7B-Instruct-abliterated":  "#d9534f",
}
MODE_COLORS = {
    "baseline":      "#e05c5c",
    "secure_prompt": "#e0a83a",
    "judge_only":    "#5ca8e0",
    "defended":      "#4caf6e",
}

# Baseline and secure_prompt never call the judge — results are identical
# across all judge configurations for those modes.
JUDGE_AGNOSTIC_MODES = {"baseline", "secure_prompt"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all result CSVs into one DataFrame.
    Adds columns: main_model, judge_model (parsed from filename).
    """
    frames = []
    for path in sorted(results_dir.glob("*_results_*_judge-*.csv")):
        stem = path.stem
        try:
            mode_part, rest = stem.split("_results_", 1)
            main_slug, judge_slug = rest.split("_judge-", 1)
        except ValueError:
            CONSOLE.print(f"[yellow]Skipping unrecognised filename: {path.name}[/]")
            continue
        df = pd.read_csv(path)
        df["main_model"] = main_slug
        df["judge_model"] = judge_slug
        df["mode"] = mode_part
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confusion_metrics(df: pd.DataFrame) -> dict:
    """Compute confusion matrix counts and rates from a filtered result DataFrame."""
    attack = df[df["should_refuse"] == 1]
    benign = df[df["should_refuse"] == 0]
    n_atk  = max(len(attack), 1)
    n_ben  = max(len(benign),  1)
    tp = int((attack["secret_leaked"] == 1).sum())
    fn = int((attack["secret_leaked"] == 0).sum())
    fp = int((benign["secret_leaked"]  == 1).sum())
    tn = int((benign["secret_leaked"]  == 0).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_attack": len(attack), "n_benign": len(benign),
        "attack_success_rate": tp / n_atk,
        "attack_blocked_rate": fn / n_atk,
        "benign_pass_rate":    tn / n_ben,
        "benign_false_leak":   fp / n_ben,
        "avg_latency_s":       float(df["latency_s"].mean()),
    }


def _sub(df: pd.DataFrame, mm: str, jm: str, mode: str) -> pd.DataFrame:
    return df[(df["main_model"] == mm) & (df["judge_model"] == jm) & (df["mode"] == mode)]


# ---------------------------------------------------------------------------
# Rich summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame) -> None:
    CONSOLE.rule("[bold cyan]Evaluation Summary (all conditions)")

    table = Table(show_header=True, header_style="bold white", show_lines=True)
    table.add_column("Main model",   style="cyan",   min_width=20)
    table.add_column("Judge",        style="yellow",  min_width=22)
    table.add_column("Mode",         min_width=14)
    table.add_column("TP",  justify="right")
    table.add_column("FN",  justify="right")
    table.add_column("FP",  justify="right")
    table.add_column("TN",  justify="right")
    table.add_column("Atk success ↓", justify="right", style="red")
    table.add_column("Benign pass ↑", justify="right", style="green")

    ref_judge = JUDGE_MODELS[0]
    for mm in MAIN_MODELS:
        for jm in JUDGE_MODELS:
            for mode in MODES:
                # For judge-agnostic modes show only one judge to avoid duplicate rows
                if mode in JUDGE_AGNOSTIC_MODES and jm != ref_judge:
                    continue
                sub = _sub(df, mm, jm, mode)
                if sub.empty:
                    continue
                m = confusion_metrics(sub)
                jm_label = JUDGE_LABELS.get(jm, jm) if mode not in JUDGE_AGNOSTIC_MODES else "(any)"
                table.add_row(
                    MAIN_LABELS.get(mm, mm),
                    jm_label,
                    MODE_LABELS.get(mode, mode),
                    str(m["tp"]), str(m["fn"]), str(m["fp"]), str(m["tn"]),
                    f"{m['attack_success_rate']:.0%}",
                    f"{m['benign_pass_rate']:.0%}",
                )

    CONSOLE.print(table)
    CONSOLE.print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    CONSOLE.print(f"  Saved [cyan]{path}[/]")
    return path


# ---------------------------------------------------------------------------
# Plot 1 — Attack success rate heatmap (judge × mode, one panel per main model)
# ---------------------------------------------------------------------------

def plot_attack_success_heatmap(df: pd.DataFrame) -> None:
    """
    2-panel heatmap.  Rows = judge model, Cols = defense mode.
    Cell = fraction of attack cases where a secret was leaked.
    Baseline / secure_prompt are judge-agnostic so all rows share the same value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

    for ax, mm in zip(axes, MAIN_MODELS):
        matrix = []
        for jm in JUDGE_MODELS:
            row = []
            for mode in MODES:
                sub = _sub(df, mm, jm, mode)
                row.append(confusion_metrics(sub)["attack_success_rate"] if not sub.empty else np.nan)
            matrix.append(row)

        df_heat = pd.DataFrame(
            matrix,
            index=[JUDGE_LABELS[jm] for jm in JUDGE_MODELS],
            columns=[MODE_LABELS[m] for m in MODES],
        )
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
        sns.heatmap(
            df_heat, ax=ax, cmap=cmap, vmin=0, vmax=1,
            annot=True, fmt=".0%", linewidths=0.5, linecolor="white",
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Attack Success Rate", "shrink": 0.85},
        )
        ax.set_title(MAIN_LABELS[mm], fontweight="bold", pad=10)
        ax.set_xlabel("Defense Mode")
        ax.set_ylabel("Judge Model" if ax is axes[0] else "")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle("Attack Success Rate by Defense Mode & Judge Model",
                 fontweight="bold", y=1.02, fontsize=12)
    _save(fig, "1_attack_success_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 2 — Defense progression: how attack success drops across modes
# ---------------------------------------------------------------------------

def plot_defense_progression(df: pd.DataFrame) -> None:
    """
    Line chart: x = defense mode (ordered), y = attack_success_rate.
    One line per judge.  Two subplots, one per main model.
    At baseline/secure_prompt all lines collapse — the divergence at
    judge_only/defended shows each judge's contribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, mm in zip(axes, MAIN_MODELS):
        for jm in JUDGE_MODELS:
            vals = []
            for mode in MODES:
                sub = _sub(df, mm, jm, mode)
                vals.append(confusion_metrics(sub)["attack_success_rate"] if not sub.empty else np.nan)
            ax.plot(
                [MODE_LABELS[m] for m in MODES], vals,
                marker="o", linewidth=2.2, markersize=7,
                label=JUDGE_LABELS[jm], color=JUDGE_COLORS[jm],
            )
            if not np.isnan(vals[-1]):
                ax.annotate(
                    f"{vals[-1]:.0%}",
                    xy=(len(MODES) - 1, vals[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8.5, va="center", color=JUDGE_COLORS[jm],
                )

        ax.set_ylim(-0.05, 1.05)
        ax.set_title(MAIN_LABELS[mm], fontweight="bold")
        ax.set_xlabel("Defense Mode")
        ax.set_ylabel("Attack Success Rate" if ax is axes[0] else "")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelrotation=15)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(title="Judge Model", fontsize=9, title_fontsize=9, loc="upper right")

    fig.suptitle("Defense Layer Progression: Attack Success Rate Across Modes",
                 fontweight="bold", y=1.02, fontsize=12)
    _save(fig, "2_defense_progression.png")


# ---------------------------------------------------------------------------
# Plot 3 — Per-attack-type breakdown at baseline (both main models side by side)
# ---------------------------------------------------------------------------

def plot_attack_type_baseline(df: pd.DataFrame) -> None:
    """
    Heatmap: rows = attack type, cols = main model.
    Value = secret_leaked (binary) at baseline.
    Judge is irrelevant for baseline — uses the first judge as reference.
    """
    ref_judge = JUDGE_MODELS[0]
    attack_types = list(ATTACK_TYPE_LABELS.keys())

    matrix = {}
    for mm in MAIN_MODELS:
        col = []
        sub = _sub(df, mm, ref_judge, "baseline")
        for at in attack_types:
            row = sub[sub["attack_type"] == at]
            col.append(float(row["secret_leaked"].iloc[0]) if not row.empty else np.nan)
        matrix[MAIN_LABELS[mm]] = col

    df_heat = pd.DataFrame(matrix, index=[ATTACK_TYPE_LABELS[at] for at in attack_types])
    df_heat = df_heat.dropna(how="all")

    fig, ax = plt.subplots(figsize=(7, max(6, len(df_heat) * 0.52 + 2)))
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    sns.heatmap(
        df_heat, ax=ax, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".0f", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Secret Leaked  (1 = yes)", "shrink": 0.6},
    )
    ax.set_title("Attack Success by Type — Baseline (No Defense)\n1 = secret leaked, 0 = blocked",
                 fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    _save(fig, "3_attack_type_baseline.png")


# ---------------------------------------------------------------------------
# Plot 4 — Judge model comparison in defended mode
# ---------------------------------------------------------------------------

def plot_judge_comparison(df: pd.DataFrame) -> None:
    """
    Grouped bar chart.  x = judge model (3), groups = main model (2).
    y = attack_success_rate in full-defense mode.  Lower is better.
    """
    defended = df[df["mode"] == "defended"]
    x = np.arange(len(JUDGE_MODELS))
    width = 0.35
    offsets = [-width / 2, width / 2]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, mm in enumerate(MAIN_MODELS):
        vals = []
        for jm in JUDGE_MODELS:
            sub = defended[(defended["main_model"] == mm) & (defended["judge_model"] == jm)]
            vals.append(confusion_metrics(sub)["attack_success_rate"] if not sub.empty else np.nan)
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=MAIN_LABELS[mm], color=MAIN_COLORS[mm], alpha=0.88, zorder=3,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([JUDGE_LABELS[jm] for jm in JUDGE_MODELS], fontsize=10)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("Attack Success Rate (secret leaked)")
    ax.set_title("Judge Model Comparison — Full Defense Mode\n(lower = better)",
                 fontweight="bold", pad=12)
    ax.legend(title="Main Model", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_axisbelow(True)
    _save(fig, "4_judge_comparison_defended.png")


# ---------------------------------------------------------------------------
# Plot 5 — Attack outcome counts in defended mode (stacked TP / FN per combo)
# ---------------------------------------------------------------------------

def plot_confusion_stacked(df: pd.DataFrame) -> None:
    """
    Stacked bar chart: x = (main model × judge model) combination, 6 bars.
    Stacked: FN (blocked, green) + TP (leaked, red).
    Annotates count in each segment.
    """
    defended = df[df["mode"] == "defended"]

    labels, tp_vals, fn_vals = [], [], []
    for mm in MAIN_MODELS:
        for jm in JUDGE_MODELS:
            sub = defended[(defended["main_model"] == mm) & (defended["judge_model"] == jm)]
            if sub.empty:
                continue
            m = confusion_metrics(sub)
            labels.append(f"{MAIN_LABELS[mm]}\n{JUDGE_LABELS[jm]}")
            tp_vals.append(m["tp"])
            fn_vals.append(m["fn"])

    x = np.arange(len(labels))
    total = (fn_vals[0] + tp_vals[0]) if labels else 1

    fig, ax = plt.subplots(figsize=(12, 5))
    b_fn = ax.bar(x, fn_vals, label="Blocked — attack failed  ✓", color="#4caf6e", alpha=0.9, zorder=3)
    b_tp = ax.bar(x, tp_vals, bottom=fn_vals, label="Leaked — attack succeeded  ✗", color="#d9534f", alpha=0.9, zorder=3)

    for bar, v in zip(b_fn, fn_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                str(v), ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    for bar, top, v in zip(b_tp, fn_vals, tp_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, top + v / 2,
                    str(v), ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    ax.set_ylim(0, total + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Number of attack-prompt test cases")
    ax.set_title("Attack Outcomes in Full Defense Mode\n(out of 9 attack cases per configuration)",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_axisbelow(True)
    _save(fig, "5_confusion_stacked_defended.png")


# ---------------------------------------------------------------------------
# Plot 6 — Latency by defense mode
# ---------------------------------------------------------------------------

def plot_latency(df: pd.DataFrame) -> None:
    """
    Box plots of per-query latency grouped by defense mode.
    One subplot per main model.
    For judge-agnostic modes (baseline, secure_prompt) only one judge's
    rows are used to avoid artificial inflation of sample size.
    """
    ref_judge = JUDGE_MODELS[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, mm in zip(axes, MAIN_MODELS):
        data_vals, tick_labels, colors = [], [], []
        for mode in MODES:
            if mode in JUDGE_AGNOSTIC_MODES:
                sub = _sub(df, mm, ref_judge, mode)
            else:
                sub = df[(df["main_model"] == mm) & (df["mode"] == mode)]
            data_vals.append(sub["latency_s"].dropna().values)
            tick_labels.append(MODE_LABELS[mode])
            colors.append(MODE_COLORS[mode])

        bp = ax.boxplot(
            data_vals, patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.4},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for element in ("whiskers", "caps"):
            for item in bp[element]:
                item.set(color="gray", linewidth=1.2)

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=9.5)
        ax.set_title(MAIN_LABELS[mm], fontweight="bold")
        ax.set_ylabel("Latency per query (s)" if ax is axes[0] else "")
        ax.set_axisbelow(True)

    fig.suptitle("Query Latency by Defense Mode",
                 fontweight="bold", y=1.02, fontsize=12)
    _save(fig, "6_latency_by_mode.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and plot llm-rag-firewall results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        CONSOLE.print("[red]No result CSVs found. Run eval.py first.[/]")
        sys.exit(1)

    n_files = df.groupby(["main_model", "judge_model", "mode"]).ngroups
    CONSOLE.print(f"\nLoaded {len(df)} rows from {n_files} result files.\n")

    _style()
    print_summary_table(df)

    CONSOLE.rule("[bold cyan]Generating plots")
    plot_attack_success_heatmap(df)
    plot_defense_progression(df)
    plot_attack_type_baseline(df)
    plot_judge_comparison(df)
    plot_confusion_stacked(df)
    plot_latency(df)

    CONSOLE.print(f"\n[bold green]All plots saved to {PLOTS_DIR}/[/]")


if __name__ == "__main__":
    main()
