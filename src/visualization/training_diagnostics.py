import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("training_diagnostics")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for server/subprocess use
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — diagnostic plots will be skipped")


def _check_mpl(func_name: str) -> bool:
    if not _MPL_AVAILABLE:
        logger.debug(f"{func_name}: matplotlib not available, skipping")
        return False
    return True


def plot_learning_curves(
    val_loss_curve: list,
    best_iteration: int,
    symbol: str,
    overfit_ratio: float,
    output_dir: Path,
) -> Path | None:
    if not _check_mpl("plot_learning_curves") or not val_loss_curve:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = list(range(len(val_loss_curve)))
    ax.plot(iterations, val_loss_curve, color="#2196F3", linewidth=1.5, label="Val LogLoss")
    ax.axvline(x=best_iteration, color="#F44336", linestyle="--", linewidth=1.2,
               label=f"Best iter={best_iteration}")

    # Flag overfitting region visually if val loss rises after best_iteration
    if best_iteration < len(val_loss_curve) - 1:
        ax.fill_betweenx(
            [min(val_loss_curve), max(val_loss_curve)],
            best_iteration, len(val_loss_curve) - 1,
            alpha=0.08, color="#F44336", label="Post-best (overfit zone)",
        )

    overfit_label = f"overfit_ratio={overfit_ratio:.3f}" if not math.isnan(overfit_ratio) else "overfit_ratio=n/a"
    ax.set_title(f"{symbol} — Learning Curve  [{overfit_label}]")
    ax.set_xlabel("Boosting Iteration")
    ax.set_ylabel("LogLoss (Val)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out_path = output_dir / f"{symbol}_15m_learning_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.debug(f"{symbol}: learning curve saved → {out_path}")
    return out_path


def plot_fold_performance(
    fold_da_list: list,
    fold_val_losses: list,
    symbol: str,
    output_dir: Path,
) -> Path | None:
    if not _check_mpl("plot_fold_performance") or not fold_da_list:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    folds = list(range(1, len(fold_da_list) + 1))
    da_mean = float(np.mean(fold_da_list))
    da_std = float(np.std(fold_da_list))

    # DA per fold
    ax1.bar(folds, fold_da_list, color="#4CAF50", alpha=0.75, edgecolor="white")
    ax1.axhline(da_mean, color="#F44336", linestyle="--", linewidth=1.2, label=f"Mean={da_mean:.3f}")
    ax1.axhline(da_mean + da_std, color="#FF9800", linestyle=":", linewidth=1, label=f"±1σ={da_std:.3f}")
    ax1.axhline(da_mean - da_std, color="#FF9800", linestyle=":", linewidth=1)
    ax1.set_title(f"{symbol} — DA per Fold")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Directional Accuracy")
    ax1.set_ylim(max(0, da_mean - 3 * da_std - 0.02), min(1, da_mean + 3 * da_std + 0.02))
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    # Flag if std > 0.05 — unstable
    if da_std > 0.05:
        ax1.set_title(f"{symbol} — DA per Fold  ⚠ HIGH VARIANCE (σ={da_std:.3f})", color="#F44336")

    # Val logloss per fold
    valid_losses = [v for v in fold_val_losses if not math.isnan(v)]
    if valid_losses:
        loss_mean = float(np.mean(valid_losses))
        ax2.bar(folds[:len(fold_val_losses)], fold_val_losses, color="#2196F3", alpha=0.75, edgecolor="white")
        ax2.axhline(loss_mean, color="#F44336", linestyle="--", linewidth=1.2, label=f"Mean={loss_mean:.4f}")
        ax2.set_title(f"{symbol} — Val LogLoss per Fold")
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("LogLoss (best iteration)")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    out_path = output_dir / f"{symbol}_15m_fold_performance.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.debug(f"{symbol}: fold performance saved → {out_path}")
    return out_path


def plot_calibration(
    val_proba_cal: np.ndarray,
    y_val: np.ndarray,
    symbol: str,
    output_dir: Path,
    n_bins: int = 10,
) -> Path | None:
    if not _check_mpl("plot_calibration") or len(val_proba_cal) == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_actual = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (val_proba_cal >= lo) & (val_proba_cal < hi)
        if mask.sum() > 0:
            bin_actual.append(float(y_val[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_actual.append(float("nan"))
            bin_counts.append(0)

    # ECE: weighted mean absolute deviation from diagonal
    valid = [(p, a, c) for p, a, c in zip(bin_centers, bin_actual, bin_counts) if not math.isnan(a)]
    total = sum(c for _, _, c in valid)
    ece = sum(abs(p - a) * c / total for p, a, c in valid) if total > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    valid_centers = [bin_centers[i] for i, a in enumerate(bin_actual) if not math.isnan(a)]
    valid_actual = [a for a in bin_actual if not math.isnan(a)]
    ax.plot(valid_centers, valid_actual, "o-", color="#2196F3", linewidth=1.5,
            markersize=6, label="Model calibration")

    # Bar chart of sample counts in background
    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_counts, width=(bins[1] - bins[0]) * 0.8,
            alpha=0.15, color="#9E9E9E", label="Sample count")
    ax2.set_ylabel("Sample count", color="#9E9E9E", fontsize=9)
    ax2.tick_params(axis="y", colors="#9E9E9E")

    ece_label = f"ECE={ece:.4f}" if not math.isnan(ece) else "ECE=n/a"
    ax.set_title(f"{symbol} — Calibration Plot  [{ece_label}]")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Positive Rate")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    out_path = output_dir / f"{symbol}_15m_calibration.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.debug(f"{symbol}: calibration plot saved → {out_path}")
    return out_path


def plot_per_symbol_summary(training_summary_csv: Path, output_dir: Path) -> Path | None:
    if not _check_mpl("plot_per_symbol_summary") or not training_summary_csv.exists():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(training_summary_csv)
    if df.empty or "da" not in df.columns:
        return None

    df = df.sort_values("da", ascending=True)
    tier_colors = {"A": "#4CAF50", "B": "#FF9800", "C": "#F44336"}
    colors = [tier_colors.get(str(t), "#9E9E9E") for t in df.get("tier", ["B"] * len(df))]

    fig, ax1 = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
    bars = ax1.barh(df["symbol"], df["da"], color=colors, alpha=0.8, edgecolor="white")
    ax1.axvline(0.55, color="#9E9E9E", linestyle=":", linewidth=1, label="0.55 threshold")
    ax1.set_xlabel("Directional Accuracy (Val)")
    ax1.set_title("Per-Symbol Training Summary")

    # Overfit ratio on secondary axis
    if "overfit_ratio" in df.columns:
        ax2 = ax1.twiny()
        valid_mask = df["overfit_ratio"].notna()
        ax2.plot(df.loc[valid_mask, "overfit_ratio"], df.loc[valid_mask, "symbol"],
                 "D", color="#E91E63", markersize=5, alpha=0.7, label="Overfit ratio")
        ax2.axvline(0.85, color="#E91E63", linestyle=":", linewidth=1, label="0.85 overfit threshold")
        ax2.set_xlabel("Overfit Ratio (train_logloss / val_logloss)", color="#E91E63")
        ax2.tick_params(axis="x", colors="#E91E63")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f"Tier {t}") for t, c in tier_colors.items()]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3, axis="x")

    out_path = output_dir / "per_symbol_summary.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info(f"Per-symbol summary plot saved → {out_path}")
    return out_path


def generate_all_diagnostics(
    symbol: str,
    val_loss_curve: list,
    best_iteration: int,
    fold_da_list: list,
    fold_val_losses: list,
    val_proba_cal: np.ndarray,
    y_val: np.ndarray,
    overfit_ratio: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(val_loss_curve, best_iteration, symbol, overfit_ratio, output_dir)
    plot_fold_performance(fold_da_list, fold_val_losses, symbol, output_dir)
    plot_calibration(val_proba_cal, y_val, symbol, output_dir)
    logger.info(f"{symbol}: diagnostic plots written to {output_dir}")
