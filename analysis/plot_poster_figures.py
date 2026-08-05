"""Create clean, poster-ready figures from the supervision sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORLD_ORDER = ["grid", "mobius", "torus", "octahedron"]

WORLD_LABELS = {
    "grid": "Grid",
    "mobius": "Möbius strip",
    "torus": "Flat torus",
    "octahedron": "Octahedron",
}

WORLD_COLORS = {
    "grid": "#2ca02c",
    "mobius": "#d62728",
    "torus": "#8c564b",
    "octahedron": "#9467bd",
}

BUDGET_COLORS = {
    1000: "#1f77b4",
    3000: "#ff7f0e",
    10000: "#2ca02c",
}


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "lines.linewidth": 3.2,
        "lines.markersize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in sorted(paths):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable file {path}: {exc}")
            continue

        if not frame.empty:
            frame["source_file"] = str(path)
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise KeyError(
        "None of these columns were found:\n"
        f"{candidates}\n\nAvailable columns:\n{list(frame.columns)}"
    )


def filter_main_sweep(curves: pd.DataFrame) -> pd.DataFrame:
    frame = curves.copy()

    # Exclude point-holdout runs from the main supervision sweep.
    if "source_file" in frame:
        frame = frame[
            ~frame["source_file"].str.contains(
                "pointholdout", case=False, na=False
            )
        ]

    if "evaluation_protocol" in frame.columns:
        frame = frame[
            frame["evaluation_protocol"].eq("held_out_pairs")
        ]

    return frame


def final_rows(curves: pd.DataFrame) -> pd.DataFrame:
    required = [
        "world",
        "supervision_budget",
        "model_seed",
        "optimizer_updates",
    ]

    missing = [column for column in required if column not in curves.columns]
    if missing:
        raise KeyError(f"Missing columns required for final rows: {missing}")

    return (
        curves.sort_values("optimizer_updates")
        .groupby(
            ["world", "supervision_budget", "model_seed"],
            as_index=False,
        )
        .tail(1)
        .copy()
    )


def summarize(
    frame: pd.DataFrame,
    grouping: list[str],
    metric: str,
) -> pd.DataFrame:
    return (
        frame.dropna(subset=[metric])
        .groupby(grouping, as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )


def finish_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
) -> None:
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)

    if title:
        ax.set_title(title, pad=8)

    ax.grid(alpha=0.22)
    ax.tick_params(pad=3)

def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)
    print(f"Saved {path}")


def plot_metric_vs_budget(
    final: pd.DataFrame,
    metric: str,
    output: Path,
    title: str,
    ylabel: str,
) -> None:
    summary = summarize(
        final,
        ["world", "supervision_budget"],
        metric,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.3))

    for world in WORLD_ORDER:
        values = summary[
            summary["world"] == world
        ].sort_values("supervision_budget")

        if values.empty:
            continue

        x = values["supervision_budget"].to_numpy()
        mean = values["mean"].to_numpy()
        std = values["std"].fillna(0).to_numpy()

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=3.2,
            markersize=8,
            label=WORLD_LABELS[world],
            color=WORLD_COLORS[world],
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.14,
            color=WORLD_COLORS[world],
        )

    finish_axis(
        ax,
        xlabel="Observed training pairs",
        ylabel=ylabel,
        title=None,
    )

    ax.margins(x=0.025)

    ax.legend(
        title="Environment",
        frameon=True,
        framealpha=0.88,
        loc="lower right",
        borderpad=0.5,
        labelspacing=0.35,
        handlelength=2.2,
    )

    ax.yaxis.label.set_fontsize(15)
    ax.yaxis.label.set_y(0.47)

    fig.subplots_adjust(
        left=0.16,
        right=0.985,
        bottom=0.16,
        top=0.98,
    )

    save_figure(fig, output)

def plot_geometry_vs_generalization(
    final: pd.DataFrame,
    geometry_metric: str,
    generalization_metric: str,
    output: Path,
) -> None:
    means = (
        final.dropna(
            subset=[geometry_metric, generalization_metric]
        )
        .groupby(
            ["world", "supervision_budget"],
            as_index=False,
        )[[geometry_metric, generalization_metric]]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    for world in WORLD_ORDER:
        values = means[
            means["world"] == world
        ].sort_values("supervision_budget")

        if values.empty:
            continue

        ax.plot(
            values[geometry_metric],
            values[generalization_metric],
            marker="o",
            linewidth=3.0,
            markersize=8,
            label=WORLD_LABELS[world],
            color=WORLD_COLORS[world],
        )

        for _, row in values.iterrows():
            budget = int(row["supervision_budget"])

            # Only label the two most informative points.
            if budget not in {1000, 3000}:
                continue

            ax.annotate(
                f"{budget // 1000}k",
                (
                    row[geometry_metric],
                    row[generalization_metric],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=13,
                color=WORLD_COLORS[world],
            )

    ax.axhline(
        0,
        linewidth=1.2,
        color="black",
        alpha=0.35,
    )

    finish_axis(
        ax,
        xlabel="Embedding–geodesic Spearman corr.",
        ylabel="Held-out pair $R^2$",
        title=None,
    )

    ax.legend(
        title="Environment",
        frameon=False,
        loc="upper left",
        borderaxespad=0.25,
    )

    ax.set_xlim(-0.03, 0.99)
    ax.set_ylim(-0.68, 1.06)

    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        bottom=0.16,
        top=0.98,
    )

    save_figure(fig, output)

def load_recovery(results_dir: Path) -> pd.DataFrame:
    paths = list(
        results_dir.glob("recovery/*/recovery/*.csv")
    )

    recovery = read_csvs(paths)

    if recovery.empty:
        raise FileNotFoundError(
            "No recovery CSV files found under:\n"
            "results/recovery/<world>/recovery/"
        )

    return recovery


def plot_recovery_small_multiples(
    recovery: pd.DataFrame,
    output: Path,
) -> None:
    anchor_column = find_column(
        recovery,
        [
            "recovery_anchor_count",
            "anchor_count",
        ],
    )

    metric_column = find_column(
        recovery,
        [
            "evaluation_spearman",
            "recovery_evaluation_spearman",
            "recovery_spearman",
        ],
    )

    required = [
        "world",
        "supervision_budget",
        "model_seed",
        anchor_column,
        metric_column,
    ]

    clean = recovery.dropna(subset=required).copy()

    # Average repeated recovery rows within each seed.
    seed_means = (
        clean.groupby(
            [
                "world",
                "supervision_budget",
                "model_seed",
                anchor_column,
            ],
            as_index=False,
        )[metric_column]
        .mean()
    )

    # Mean and standard deviation across model seeds.
    summary = (
        seed_means.groupby(
            [
                "world",
                "supervision_budget",
                anchor_column,
            ],
            as_index=False,
        )
        .agg(
            mean=(metric_column, "mean"),
            std=(metric_column, "std"),
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.6),
        sharex=True,
        sharey=True,
    )

    for ax, world in zip(axes.flat, WORLD_ORDER):
        world_values = summary[
            summary["world"] == world
        ]

        for budget in [1000, 3000, 10000]:
            values = world_values[
                world_values["supervision_budget"] == budget
            ].sort_values(anchor_column)

            if values.empty:
                continue

            x = values[anchor_column].to_numpy()
            mean = values["mean"].to_numpy()
            std = values["std"].fillna(0).to_numpy()

            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2.8,
                markersize=6,
                color=BUDGET_COLORS[budget],
                label=f"{budget // 1000}k pairs",
            )

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=BUDGET_COLORS[budget],
                alpha=0.13,
            )

        # Smaller titles, positioned clearly above each subplot.
        ax.set_title(
            WORLD_LABELS[world],
            fontsize=17,
            fontweight="bold",
            pad=7,
        )

        ax.grid(alpha=0.22)
        ax.set_xticks([4, 8, 16])
        ax.set_ylim(-0.05, 1.05)

    # Use one shared x label and one shared y label.
    fig.supxlabel(
        "Known anchor distances",
        fontsize=18,
        y=0.025,
    )

    fig.supylabel(
        "Recovery Spearman correlation",
        fontsize=18,
        x=0.015,
    )

    # Hide redundant tick labels.
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)

    # One legend only.
    axes[0, 0].legend(
        title="Training supervision",
        frameon=True,
        framealpha=0.90,
        loc="center right",
        fontsize=12,
        title_fontsize=12,
        borderpad=0.4,
        labelspacing=0.25,
    )

    # More vertical separation between rows.
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.12,
        top=0.95,
        wspace=0.08,
        hspace=0.24,
    )

    save_figure(fig, output)

def plot_learning_dynamics(
    curves: pd.DataFrame,
    output: Path,
    budget: int,
) -> None:
    training_metric = find_column(
        curves,
        [
            "training_pair_r2",
            "train_pair_r2",
            "training_r2",
        ],
    )

    heldout_metric = find_column(
        curves,
        [
            "held_out_pair_r2",
            "held_out_r2",
            "evaluation_r2",
        ],
    )

    selected = curves[
        curves["supervision_budget"] == budget
    ].copy()

    selected = selected.dropna(
        subset=[
            "world",
            "model_seed",
            "optimizer_updates",
            training_metric,
            heldout_metric,
        ]
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.0, 6.6),
        sharex=True,
        sharey=True,
    )

    for ax, world in zip(axes.flat, WORLD_ORDER):
        values = selected[selected["world"] == world]

        if values.empty:
            ax.set_visible(False)
            continue

        seed_summary = (
            values.groupby(
                ["optimizer_updates", "model_seed"],
                as_index=False,
            )[[training_metric, heldout_metric]]
            .mean()
        )

        summary = (
            seed_summary.groupby(
                "optimizer_updates",
                as_index=False,
            )
            .agg(
                training_mean=(training_metric, "mean"),
                training_std=(training_metric, "std"),
                heldout_mean=(heldout_metric, "mean"),
                heldout_std=(heldout_metric, "std"),
            )
            .sort_values("optimizer_updates")
        )

        x = summary["optimizer_updates"].to_numpy()

        train_mean = summary["training_mean"].to_numpy()
        train_std = summary["training_std"].fillna(0).to_numpy()

        heldout_mean = summary["heldout_mean"].to_numpy()
        heldout_std = summary["heldout_std"].fillna(0).to_numpy()

        ax.plot(
            x,
            train_mean,
            linewidth=2.8,
            label="Training-pair $R^2$",
            color="#1f77b4",
        )

        ax.fill_between(
            x,
            train_mean - train_std,
            train_mean + train_std,
            color="#1f77b4",
            alpha=0.10,
        )

        ax.plot(
            x,
            heldout_mean,
            linewidth=2.8,
            label="Held-out-pair $R^2$",
            color="#d62728",
        )

        ax.fill_between(
            x,
            heldout_mean - heldout_std,
            heldout_mean + heldout_std,
            color="#d62728",
            alpha=0.14,
        )

        ax.axhline(
            0,
            linewidth=1,
            color="black",
            alpha=0.28,
        )

        ax.text(
            0.03,
            0.94,
            WORLD_LABELS[world],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

        ax.grid(alpha=0.22)
        ax.set_ylim(-0.75, 1.05)

    axes[1, 0].set_xlabel("Optimizer updates")
    axes[1, 1].set_xlabel("Optimizer updates")

    axes[0, 0].set_ylabel("$R^2$")
    axes[1, 0].set_ylabel("$R^2$")

    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)

    axes[0, 0].legend(
        frameon=True,
        framealpha=0.9,
        loc="lower right",
        borderpad=0.4,
        labelspacing=0.3,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.12,
        top=0.99,
        wspace=0.08,
        hspace=0.10,
    )

    save_figure(fig, output)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-dir",
        default="analysis_results/poster_figures",
    )
    parser.add_argument(
        "--learning-budget",
        type=int,
        default=3000,
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = read_csvs(
        list((results_dir / "learning_curves").glob("*.csv"))
    )

    if curves.empty:
        raise FileNotFoundError(
            "No CSV files found in results/learning_curves/"
        )

    curves = filter_main_sweep(curves)

    heldout_r2 = find_column(
        curves,
        [
            "held_out_pair_r2",
            "held_out_r2",
            "evaluation_r2",
        ],
    )

    geometry_metric = find_column(
        curves,
        [
            "embedding_distance_spearman",
            "embedding_geodesic_spearman",
            "embedding_geodesic_correlation",
        ],
    )

    final = final_rows(curves)

    plot_metric_vs_budget(
        final=final,
        metric=heldout_r2,
        output=output_dir / "01_held_out_r2_vs_budget.png",
        title="Generalization emerges after a supervision threshold",
        ylabel="Held-out pair $R^2$",
    )

    plot_metric_vs_budget(
        final=final,
        metric=geometry_metric,
        output=output_dir / "02_geometry_vs_budget.png",
        title="Geometric structure emerges with supervision",
        ylabel="Embedding–geodesic Spearman corr.",
    )

    plot_geometry_vs_generalization(
        final=final,
        geometry_metric=geometry_metric,
        generalization_metric=heldout_r2,
        output=output_dir / "03_geometry_vs_generalization.png",
    )

    recovery = load_recovery(results_dir)

    plot_recovery_small_multiples(
        recovery=recovery,
        output=output_dir / "04_recovery_vs_anchors.png",
    )

    plot_learning_dynamics(
        curves=curves,
        output=output_dir / "05_learning_dynamics.png",
        budget=args.learning_budget,
    )


if __name__ == "__main__":
    main()
