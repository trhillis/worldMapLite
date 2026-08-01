"""Plot per-seed supervision, learning-curve, and recovery results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_tables(root, child):
    paths = sorted((root / child).glob("*.csv"))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()


def seed_plot(frame, x, y, seed_column, output, title, condition_columns=()):
    clean = frame.dropna(subset=[x, y])
    if clean.empty:
        return False
    plt.figure(figsize=(7, 5))
    condition_columns = list(condition_columns)
    grouping = [*condition_columns, seed_column]
    for label, values in clean.groupby(grouping):
        values = values.sort_values(x)
        label = label if isinstance(label, tuple) else (label,)
        condition = "/".join(map(str, label[:-1])) or "all"
        plt.plot(values[x], values[y], marker="o", alpha=0.25, label=f"{condition}, seed {label[-1]}")
    for condition, summary in clean.groupby(condition_columns or [lambda _: "all"]):
        summary = summary.groupby(x)[y].agg(["mean", "std"]).reset_index().sort_values(x)
        condition = condition if isinstance(condition, tuple) else (condition,)
        label = "/".join(map(str, condition))
        line = plt.plot(summary[x], summary["mean"], linewidth=2, label=f"{label} mean")[0]
        if summary["std"].notna().any():
            std = summary["std"].fillna(0)
            plt.fill_between(summary[x], summary["mean"] - std, summary["mean"] + std, color=line.get_color(), alpha=0.15)
    plt.title(title)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y.replace("_", " "))
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="analysis_results/sweep_plots")
    args = parser.parse_args()
    root = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    curves = read_tables(root, "learning_curves")
    recovery = read_tables(root, "recovery")
    if not curves.empty:
        if "evaluation_protocol" in curves:
            curves = curves[curves["evaluation_protocol"] == "held_out_pairs"]
        final = curves.sort_values("optimizer_updates").groupby(
            ["world", "supervision_budget", "model_seed"], as_index=False,
        ).tail(1)
        seed_plot(final, "supervision_budget", "held_out_pair_spearman", "model_seed", output / "held_out_pairs_vs_budget.png", "Held-out-pair performance", ["world"])
        seed_plot(final, "supervision_budget", "embedding_distance_spearman", "model_seed", output / "representation_vs_budget.png", "Representation quality", ["world"])
        seed_plot(curves, "optimizer_updates", "held_out_pair_spearman", "model_seed", output / "metrics_vs_updates.png", "Learning curves (selected budgets)", ["world", "supervision_budget"])
    if not recovery.empty:
        recovery_by_seed = recovery.groupby(
            ["world", "supervision_budget", "model_seed", "recovery_anchor_count"],
            as_index=False,
        )["evaluation_spearman"].mean()
        seed_plot(recovery_by_seed, "recovery_anchor_count", "evaluation_spearman", "model_seed", output / "recovery_vs_anchors.png", "Held-out-point recovery", ["world", "supervision_budget"])


if __name__ == "__main__":
    main()
