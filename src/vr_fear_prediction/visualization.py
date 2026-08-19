"""Publication-style plots for exploratory analysis and model evaluation.

Kept deliberately small: a handful of plots that each answer a specific question the
README and model card refer to, rather than a large decorative gallery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    """SUDS and one physiological measure across exposure levels: does subjective fear
    track physiology as exposure intensity increases?"""
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


def plot_feature_set_comparison(metrics: dict) -> Path:
    """Bar chart comparing held-out-test MAE across feature sets and models."""
    rows = []
    for feature_set, models in metrics["feature_sets"].items():
        for model_name, result in models.items():
            if model_name == "best_model":
                continue
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "MAE": result["held_out_test"]["MAE"],
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=df, x="feature_set", y="MAE", hue="model", palette=PALETTE, ax=ax)
    ax.set_title("Held-out participant test MAE by feature set and model")
    ax.set_ylabel("MAE (SUDS points, lower is better)")
    ax.set_xlabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, "feature_set_model_comparison.png")


def plot_predicted_vs_observed(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    ax.scatter(y_true, y_pred, alpha=0.7, color="#2e6fb3", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Observed SUDS")
    ax.set_ylabel("Predicted SUDS")
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return _save(fig, filename)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> Path:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.7, color="#b3312e", edgecolor="white", linewidth=0.5)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Predicted SUDS")
    ax.set_ylabel("Residual (observed - predicted)")
    ax.set_title(title)
    return _save(fig, filename)


def plot_paired_mae_difference(results_df: pd.DataFrame, title: str, filename: str) -> Path:
    """Distribution of the paired (context_only_MAE - combined_MAE) difference across
    repeated participant-level holdout splits.

    Positive values favor the combined model; negative values favor context-only. This is
    an honest view of how much the context-vs-combined comparison depends on which
    participants happen to be held out -- it should NOT be read as picking a winner on its
    own if the distribution straddles zero.
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
        "splits (different random choices of which participants are held out). A distribution straddling zero "
        "means this comparison is unstable, not that one model is reliably better.",
        ha="center",
        fontsize=10,
        wrap=True,
    )
    fig.tight_layout()
    return _save(fig, filename)


def plot_baseline_sensitivity_comparison(
    primary_df: pd.DataFrame, sensitivity_df: pd.DataFrame, filename: str = "baseline_sensitivity_comparison.png"
) -> Path:
    """Compact comparison of the paired MAE difference distribution with vs. without the
    level-0 baseline (floor-effect) observations, to show whether excluding baseline
    materially changes the context-vs-combined comparison."""
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
    ax.set_title("Does excluding the baseline (floor-effect) level\nchange the context-vs-combined comparison?", fontsize=14)
    fig.tight_layout()
    return _save(fig, filename)


def plot_permutation_importance(importance_df: pd.DataFrame, title: str, filename: str, top_n: int = 15) -> Path:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2e6fb3")
    ax.set_xlabel("Permutation importance (increase in MAE when shuffled)")
    ax.set_title(title)
    return _save(fig, filename)
