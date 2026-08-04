"""Plot supervision sweeps, learning dynamics, grokking diagnostics, and recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    paths = sorted(paths)
    if not paths:
        return pd.DataFrame()

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def read_learning_curves(root: Path) -> pd.DataFrame:
    """Read only the main sweep learning curves."""
    return read_csvs((root / "learning_curves").glob("*.csv"))


def read_recovery_results(root: Path) -> pd.DataFrame:
    """Read nested recovery outputs such as recovery/grid/recovery/*.csv."""
    return read_csvs((root / "recovery").glob("*/recovery/*.csv"))


def available_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def seed_plot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    seed_column: str,
    output: Path,
    title: str,
    condition_columns: tuple[str, ...] = (),
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> bool:
    required = [x, y, seed_column, *condition_columns]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        print(f"Skipping {output.name}: missing columns {missing}")
        return False

    clean = frame.dropna(subset=[x, y, seed_column, *condition_columns]).copy()
    if clean.empty:
        print(f"Skipping {output.name}: no valid rows")
        return False

    plt.figure(figsize=(8, 5.5))

    grouping = [*condition_columns, seed_column]

    for label, values in clean.groupby(grouping, dropna=False):
        values = values.sort_values(x)
        label_tuple = label if isinstance(label, tuple) else (label,)
        condition_values = label_tuple[:-1]
        seed_value = label_tuple[-1]

        condition_label = (
            " / ".join(map(str, condition_values))
            if condition_values
            else "all"
        )

        plt.plot(
            values[x],
            values[y],
            marker="o",
            linewidth=1,
            alpha=0.25,
            label=f"{condition_label}, seed {seed_value}",
        )

    if condition_columns:
        summary_groups = clean.groupby(list(condition_columns), dropna=False)
    else:
        summary_groups = [("all", clean)]

    for condition, condition_frame in summary_groups:
        summary = (
            condition_frame.groupby(x)[y]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(x)
        )

        condition_tuple = (
            condition if isinstance(condition, tuple) else (condition,)
        )
        condition_label = " / ".join(map(str, condition_tuple))

        line = plt.plot(
            summary[x],
            summary["mean"],
            marker="o",
            linewidth=2.5,
            label=f"{condition_label} mean",
        )[0]

        if summary["std"].notna().any():
            std = summary["std"].fillna(0)
            plt.fill_between(
                summary[x],
                summary["mean"] - std,
                summary["mean"] + std,
                color=line.get_color(),
                alpha=0.15,
            )

    plt.title(title)
    plt.xlabel(xlabel or x.replace("_", " "))
    plt.ylabel(ylabel or y.replace("_", " "))
    plt.grid(alpha=0.2)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")
    return True


def scatter_geometry_vs_generalization(
    frame: pd.DataFrame,
    output: Path,
) -> bool:
    required = [
        "embedding_distance_spearman",
        "held_out_pair_r2",
        "world",
        "supervision_budget",
    ]

    missing = [column for column in required if column not in frame.columns]
    if missing:
        print(f"Skipping {output.name}: missing columns {missing}")
        return False

    clean = frame.dropna(subset=required).copy()
    if clean.empty:
        return False

    plt.figure(figsize=(7, 5.5))

    for world, values in clean.groupby("world"):
        plt.scatter(
            values["embedding_distance_spearman"],
            values["held_out_pair_r2"],
            alpha=0.7,
            label=str(world),
        )

        for _, row in values.iterrows():
            plt.annotate(
                str(int(row["supervision_budget"])),
                (
                    row["embedding_distance_spearman"],
                    row["held_out_pair_r2"],
                ),
                fontsize=7,
                alpha=0.65,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.axhline(0, linewidth=1, alpha=0.4)
    plt.title("Embedding geometry and held-out generalization")
    plt.xlabel("Embedding–geodesic Spearman correlation")
    plt.ylabel("Held-out pair R²")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-dir",
        default="analysis_results/sweep_plots",
    )
    parser.add_argument(
        "--learning-budgets",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000],
        help="Budgets shown in learning-dynamics and grokking figures.",
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    curves = read_learning_curves(root)
    recovery = read_recovery_results(root)

    if curves.empty:
        print(f"No main learning-curve CSVs found in {root / 'learning_curves'}")
    else:
        if "evaluation_protocol" in curves.columns:
            curves = curves[
                curves["evaluation_protocol"] == "held_out_pairs"
            ].copy()

        required_group_columns = [
            "world",
            "supervision_budget",
            "model_seed",
            "optimizer_updates",
        ]
        missing = [
            column
            for column in required_group_columns
            if column not in curves.columns
        ]

        if missing:
            raise ValueError(
                f"Learning-curve data is missing required columns: {missing}"
            )

        # Final checkpoint from every world/budget/seed run.
        final = (
            curves.sort_values("optimizer_updates")
            .groupby(
                ["world", "supervision_budget", "model_seed"],
                as_index=False,
            )
            .tail(1)
        )

        # Main supervision-budget figures.
        seed_plot(
            final,
            "supervision_budget",
            "held_out_pair_spearman",
            "model_seed",
            output / "held_out_pair_spearman_vs_budget.png",
            "Held-out-pair performance versus supervision",
            ("world",),
            xlabel="Observed training pairs",
            ylabel="Held-out pair Spearman correlation",
        )

        if "held_out_pair_r2" in final.columns:
            seed_plot(
                final,
                "supervision_budget",
                "held_out_pair_r2",
                "model_seed",
                output / "held_out_pair_r2_vs_budget.png",
                "Held-out-pair R² versus supervision",
                ("world",),
                xlabel="Observed training pairs",
                ylabel="Held-out pair R²",
            )

        seed_plot(
            final,
            "supervision_budget",
            "embedding_distance_spearman",
            "model_seed",
            output / "embedding_geometry_vs_budget.png",
            "Representation quality versus supervision",
            ("world",),
            xlabel="Observed training pairs",
            ylabel="Embedding–geodesic Spearman correlation",
        )

        nearest_neighbor_column = available_column(
            final,
            [
                "nearest_neighbor_recall",
                "nearest_neighbour_recall",
                "nn_recall",
                "nearest_neighbor_recall_at_k",
            ],
        )

        if nearest_neighbor_column:
            seed_plot(
                final,
                "supervision_budget",
                nearest_neighbor_column,
                "model_seed",
                output / "nearest_neighbor_recall_vs_budget.png",
                "Local structure versus supervision",
                ("world",),
                xlabel="Observed training pairs",
                ylabel="Nearest-neighbour recall",
            )

        # Direct relationship between representation geometry and generalization.
        scatter_geometry_vs_generalization(
            final,
            output / "geometry_vs_generalization.png",
        )

        # Selected budgets for readable learning-dynamics figures.
        selected = curves[
            curves["supervision_budget"].isin(args.learning_budgets)
        ].copy()

        seed_plot(
            selected,
            "optimizer_updates",
            "held_out_pair_spearman",
            "model_seed",
            output / "held_out_spearman_vs_updates.png",
            "Held-out generalization over training",
            ("world", "supervision_budget"),
            xlabel="Optimizer updates",
            ylabel="Held-out pair Spearman correlation",
        )

        if "held_out_pair_r2" in selected.columns:
            seed_plot(
                selected,
                "optimizer_updates",
                "held_out_pair_r2",
                "model_seed",
                output / "held_out_r2_vs_updates.png",
                "Held-out R² over training",
                ("world", "supervision_budget"),
                xlabel="Optimizer updates",
                ylabel="Held-out pair R²",
            )

        seed_plot(
            selected,
            "optimizer_updates",
            "embedding_distance_spearman",
            "model_seed",
            output / "embedding_geometry_vs_updates.png",
            "Geometric representation over training",
            ("world", "supervision_budget"),
            xlabel="Optimizer updates",
            ylabel="Embedding–geodesic Spearman correlation",
        )

        if nearest_neighbor_column:
            seed_plot(
                selected,
                "optimizer_updates",
                nearest_neighbor_column,
                "model_seed",
                output / "nearest_neighbor_recall_vs_updates.png",
                "Local structure over training",
                ("world", "supervision_budget"),
                xlabel="Optimizer updates",
                ylabel="Nearest-neighbour recall",
            )

        # Training metric for grokking diagnostics.
        training_metric = available_column(
            selected,
            [
                "training_loss",
                "train_loss",
                "loss",
                "training_pair_mae",
                "train_pair_mae",
                "train_mae",
            ],
        )

        if training_metric:
            seed_plot(
                selected,
                "optimizer_updates",
                training_metric,
                "model_seed",
                output / "training_metric_vs_updates.png",
                "Training fit over time",
                ("world", "supervision_budget"),
                xlabel="Optimizer updates",
                ylabel=training_metric.replace("_", " "),
            )
        else:
            print(
                "No recognised training-loss or training-error column found. "
                "A grokking diagnosis requires comparing training fit with "
                "held-out performance over optimizer updates."
            )

    if recovery.empty:
        print(
            "No recovery CSVs found under "
            f"{root / 'recovery'}/*/recovery/"
        )
    else:
        required = [
            "world",
            "supervision_budget",
            "model_seed",
            "recovery_anchor_count",
            "evaluation_spearman",
        ]
        missing = [column for column in required if column not in recovery.columns]

        if missing:
            print(
                "Skipping recovery plots because recovery data is missing "
                f"columns: {missing}"
            )
        else:
            recovery_by_seed = (
                recovery.groupby(
                    [
                        "world",
                        "supervision_budget",
                        "model_seed",
                        "recovery_anchor_count",
                    ],
                    as_index=False,
                )["evaluation_spearman"]
                .mean()
            )

            seed_plot(
                recovery_by_seed,
                "recovery_anchor_count",
                "evaluation_spearman",
                "model_seed",
                output / "recovery_vs_anchor_count.png",
                "Held-out-point recovery versus anchor count",
                ("world", "supervision_budget"),
                xlabel="Recovery anchor count",
                ylabel="Recovery evaluation Spearman correlation",
            )

            seed_plot(
                recovery_by_seed,
                "supervision_budget",
                "evaluation_spearman",
                "model_seed",
                output / "recovery_vs_supervision_budget.png",
                "Held-out-point recovery versus base-model supervision",
                ("world", "recovery_anchor_count"),
                xlabel="Observed training pairs",
                ylabel="Recovery evaluation Spearman correlation",
            )


if __name__ == "__main__":
    main()