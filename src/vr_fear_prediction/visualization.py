"""The plots used in the README and notebook.

Kept to the handful that the results actually rely on, rather than a full gallery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid", context="talk")
PALETTE = "flare"


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_suds_and_physio_by_level(long_df: pd.DataFrame, physio_col: str = "EDA_MEAN") -> Path:
    """SUDS and one physiological measure across exposure levels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.lineplot(data=long_df, x="level", y="suds", errorbar="se", marker="o", ax=axes[0], color="#b3312e")
    axes[0].set_title("Self-reported fear (SUDS) by exposure level")
    axes[0].set_xlabel("VR exposure level (0 = baseline)")
    axes[0].set_ylabel("SUDS (0-10)")

    sns.lineplot(data=long_df, x="level", y=physio_col, errorbar="se", marker="o", ax=axes[1], color="#2e6fb3")
    axes[1].set_title(f"{physio_col} by exposure level")
    axes[1].set_xlabel("VR exposure level (0 = baseline)")
    axes[1].set_ylabel(physio_col)

    fig.suptitle("Subjective fear and physiology across VR spider exposure stages", y=1.03)
    return _save(fig, "suds_and_physiology_by_level.png")


def plot_dissociation(long_df: pd.DataFrame, physio_col: str = "EDA_MEAN_delta_baseline") -> Path:
    """Per-participant agreement/dissociation between physiological reactivity and
    subjective fear: z-scored mean SUDS change vs z-scored mean physiological change."""
    agg = (
        long_df[long_df["level"] > 0]
        .groupby("participant_id")
        .agg(mean_suds=("suds", "mean"), mean_physio=(physio_col, "mean"))
        .reset_index()
    )
    agg["suds_z"] = (agg["mean_suds"] - agg["mean_suds"].mean()) / agg["mean_suds"].std()
    agg["physio_z"] = (agg["mean_physio"] - agg["mean_physio"].mean()) / agg["mean_physio"].std()
    agg["dissociation"] = (agg["suds_z"] - agg["physio_z"]).abs()

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(agg["suds_z"].abs().max(), agg["physio_z"].abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray", linewidth=1, label="perfect agreement")
    sns.scatterplot(
        data=agg, x="suds_z", y="physio_z", hue="dissociation", palette=PALETTE, s=90, ax=ax, legend=False
    )
    most_dissociated = agg.nlargest(3, "dissociation")
    for _, row in most_dissociated.iterrows():
        ax.annotate(row["participant_id"], (row["suds_z"], row["physio_z"]), fontsize=10, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Mean self-reported fear during exposure (z-score)")
    ax.set_ylabel(f"Mean {physio_col}\nduring exposure (z-score)")
    ax.set_title("Agreement vs. dissociation:\nsubjective vs. physiological fear", fontsize=16)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    return _save(fig, "subjective_vs_physiological_dissociation.png")


def plot_paired_mae_difference(results_df: pd.DataFrame, title: str, filename: str) -> Path:
    """Distribution of the paired (context_only_MAE - combined_MAE) difference across
    repeated participant-level holdout splits.

    Positive values favor the combined model, negative values favor context-only. A
    distribution straddling zero means the comparison is unstable across which
    participants happen to be held out, not that one model is reliably better.
    """
    diff = results_df["diff_context_minus_combined"]
    frac_combined_better = float((diff > 0).mean())

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.histplot(diff, bins=20, color="#2e6fb3", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.text(0.97, 0.95, "combined better →", transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#2e6fb3")
    ax.text(0.03, 0.95, "← context-only better", transform=ax.transAxes, ha="left", va="top", fontsize=11, color="#b3312e")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Paired MAE difference (context_only_MAE - combined_MAE)")
    ax.set_ylabel("Number of splits")
    fig.text(
        0.5,
        -0.06,
        f"Combined had lower MAE in {frac_combined_better:.0%} of {len(diff)} repeated participant-level holdout "
        "splits (different random choices of which participants are held out).",
        ha="center",
        fontsize=10,
        wrap=True,
    )
    fig.tight_layout()
    return _save(fig, filename)


def plot_baseline_sensitivity_comparison(
    primary_df: pd.DataFrame, sensitivity_df: pd.DataFrame, filename: str = "baseline_sensitivity_comparison.png"
) -> Path:
    """Paired MAE difference distribution with vs. without the level-0 baseline
    observations, to show whether excluding baseline changes the comparison."""
    combined = pd.concat(
        [
            primary_df.assign(analysis="All levels 0-9\n(primary)"),
            sensitivity_df.assign(analysis="Levels 1-9 only\n(sensitivity)"),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.boxplot(data=combined, x="analysis", y="diff_context_minus_combined", hue="analysis", palette=PALETTE, legend=False, ax=ax)
    sns.stripplot(data=combined, x="analysis", y="diff_context_minus_combined", color="black", alpha=0.35, size=3, ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Paired MAE difference\n(context_only - combined)")
    ax.set_xlabel("")
    ax.set_title("Does excluding the baseline level\nchange the comparison?", fontsize=14)
    fig.tight_layout()
    return _save(fig, filename)
