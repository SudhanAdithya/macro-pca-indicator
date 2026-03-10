"""
viz/charts.py
Reusable charting utilities for the PCA Macro Slowdown Indicator project.
All functions return matplotlib Figure/Axes objects so callers can
further customize or save them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Tuple
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

# Apply clean, modern style
plt.rcParams.update({
    "figure.dpi":       config.FIG_DPI,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
})


# ---------------------------------------------------------------------------
# Recession shading helper
# ---------------------------------------------------------------------------

def shade_recessions(
    ax: plt.Axes,
    recession_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    color: str = None,
    alpha: float = 0.35,
    label: str = "NBER Recession",
) -> None:
    """
    Add grey shaded bands to an Axes for each NBER recession period.

    Parameters
    ----------
    ax                : matplotlib Axes to shade
    recession_periods : list of (start, end) Timestamp tuples
    color             : shade color (defaults to config.RECESSION_SHADE_COLOR)
    alpha             : transparency
    label             : legend label (added once)
    """
    if color is None:
        color = config.RECESSION_SHADE_COLOR

    hatch = getattr(config, "RECESSION_HATCH", None)

    for i, (start, end) in enumerate(recession_periods):
        ax.axvspan(
            start, end,
            color=color, alpha=alpha,
            hatch=hatch,
            label=label if i == 0 else "_nolegend_",
            zorder=2,  # Bring recessions forward
        )


def shade_regimes(
    ax: plt.Axes,
    regime_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    color: str = None,
    alpha: float = 0.25,
    label: str = "Slowdown Regime",
) -> None:
    """Add coloured shading for PCA-identified slowdown periods."""
    if color is None:
        color = config.SLOWDOWN_SHADE_COLOR

    for i, (start, end) in enumerate(regime_periods):
        ax.axvspan(
            start, end,
            color=color, alpha=alpha,
            label=label if i == 0 else "_nolegend_",
            zorder=1,  # Keep slowdowns behind recessions if they overlap
        )


# ---------------------------------------------------------------------------
# PC1 signal chart
# ---------------------------------------------------------------------------

def plot_pc1_signal(
    pc1: pd.Series,
    pc1_smooth: pd.Series = None,
    recession_periods: List[Tuple] = None,
    regime_periods_list: List[Tuple] = None,
    title: str = "PCA Macro Activity Indicator (PC1)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot PC1 over time with optional NBER shading and slowdown regime overlay.
    """
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)

    if recession_periods:
        shade_recessions(ax, recession_periods)

    if regime_periods_list:
        shade_regimes(ax, regime_periods_list)

    ax.plot(pc1.index, pc1.values, color="#2c7bb6", linewidth=1.2,
            alpha=0.7, label="PC1 (raw)")

    if pc1_smooth is not None:
        ax.plot(pc1_smooth.index, pc1_smooth.values, color="#d7191c",
                linewidth=2.0, label=f"PC1 ({len(pc1) // len(pc1_smooth.dropna())}-m smooth)", zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized Score")
    ax.legend(loc="upper right")
    ax.set_xlim(pc1.index[0], pc1.index[-1])

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Loadings bar chart
# ---------------------------------------------------------------------------

def plot_loadings(
    loadings_df: pd.DataFrame,
    component: str = "PC1",
    title: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of PC1 loadings, colored by sign.
    """
    if title is None:
        title = f"{component} Loadings — Contribution of Each Variable"

    vals = loadings_df[component].sort_values(ascending=True)
    colors = ["#d7191c" if v < 0 else "#2c7bb6" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(vals) * 0.55)))
    bars = ax.barh(vals.index, vals.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{component} Loading")
    ax.set_ylabel("Variable")

    # Value labels
    for bar, val in zip(bars, vals.values):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Scree plot
# ---------------------------------------------------------------------------

def plot_scree(
    variance_df: pd.DataFrame,
    title: str = "Variance Explained by Principal Components",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scree plot: individual and cumulative variance explained.

    Parameters
    ----------
    variance_df : DataFrame from pca.build_indicator.variance_explained()
                  Columns: component, var_explained_pct, cumulative_pct
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = range(len(variance_df))
    ax1.bar(x, variance_df["var_explained_pct"], color="#2c7bb6",
            alpha=0.7, label="Individual %")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(variance_df["component"])
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title(title, fontsize=12, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, variance_df["cumulative_pct"], "o-", color="#d7191c",
             linewidth=2, markersize=5, label="Cumulative %")
    ax2.axhline(80, color="#d7191c", linestyle="--", alpha=0.4, linewidth=0.8)
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    return fig, ax1


# ---------------------------------------------------------------------------
# Regime chart
# ---------------------------------------------------------------------------

def plot_regime_periods(
    pc1: pd.Series,
    pc1_smooth: pd.Series,
    regime_periods_list: List[Tuple],
    recession_periods: List[Tuple] = None,
    threshold: float = 0.0,
    title: str = "Macro Slowdown Regime vs. NBER Recessions",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Dual-panel: top = PC1 with shaded slowdown regions,
                bottom = regime binary indicator.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # --- Top panel: PC1 ---
    if recession_periods:
        shade_recessions(ax1, recession_periods)
    shade_regimes(ax1, regime_periods_list)

    ax1.plot(pc1.index, pc1.values, color="#2c7bb6", linewidth=1.0, alpha=0.6, label="PC1")
    ax1.plot(pc1_smooth.index, pc1_smooth.values, color="#d7191c", linewidth=2.0, label="PC1 (smoothed)")
    ax1.axhline(threshold, color="black", linewidth=0.8, linestyle="--",
                alpha=0.6, label=f"Threshold = {threshold}")
    ax1.set_ylabel("PC1 Score")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", ncol=2)

    # --- Bottom panel: binary regime ---
    regime_binary = pd.Series(0.0, index=pc1.index)
    for start, end in regime_periods_list:
        regime_binary[start:end] = 1.0
    ax2.fill_between(regime_binary.index, 0, regime_binary.values,
                     color=config.SLOWDOWN_SHADE_COLOR, step="post",
                     label="Slowdown (PCA)", alpha=0.8)
    if recession_periods:
        nber_binary = pd.Series(0.0, index=pc1.index)
        for start, end in recession_periods:
            nber_binary[start:end] = 0.5
        ax2.fill_between(nber_binary.index, 0, nber_binary.values,
                         color=config.RECESSION_SHADE_COLOR, step="post",
                         label="NBER Recession", alpha=0.7)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["Normal", "NBER", "Slowdown"])
    ax2.set_ylabel("Regime")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlation Matrix: PC1 & Financial Variables",
) -> Tuple[plt.Figure, plt.Axes]:
    """Seaborn heatmap of a correlation matrix. """
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_SQUARE)
    sns.heatmap(
        corr_df,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Lead-lag beta chart
# ---------------------------------------------------------------------------

def plot_lead_lag_betas(
    bivariate_df: pd.DataFrame,
    title: str = "Lead-Lag β Coefficients: Financial Variables → Future PC1",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Grouped bar chart of lead-lag regression β coefficients.

    Parameters
    ----------
    bivariate_df : long-form DataFrame from analysis.lead_lag.bivariate_lead_lag()
                   Columns: variable, horizon, beta, p_value
    """
    pivoted = bivariate_df.pivot(index="horizon", columns="variable", values="beta")
    sig     = bivariate_df.pivot(index="horizon", columns="variable", values="p_value")

    colors = sns.color_palette("tab10", n_colors=len(pivoted.columns))
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(pivoted.index))
    width = 0.8 / len(pivoted.columns)

    for i, (col, color) in enumerate(zip(pivoted.columns, colors)):
        offset = (i - len(pivoted.columns) / 2 + 0.5) * width
        betas  = pivoted[col].values
        pvals  = sig[col].values

        bars = ax.bar(x + offset, betas, width=width * 0.9, color=color,
                      alpha=0.75, label=col)

        # Mark statistically significant bars with *
        for j, (bar, p) in enumerate(zip(bars, pvals)):
            if p < 0.10:
                stars = "***" if p < 0.01 else ("**" if p < 0.05 else "*")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005 * np.sign(bar.get_height()),
                        stars, ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h = {h} mo" for h in pivoted.index])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("β Coefficient")
    ax.legend(title="Financial Variable", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Portfolio Performance Chart
# ---------------------------------------------------------------------------

def plot_portfolio_performance(
    results: pd.DataFrame,
    recession_periods: List[Tuple] = None,
    regime_periods: List[Tuple] = None,
    title: str = "Portfolio Performance: Dynamic Strategy vs. 60/40 Benchmark",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative returns of benchmark vs. strategy with regime shading.
    """
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    
    # Shade recessions/regimes
    if recession_periods:
        shade_recessions(ax, recession_periods)
    if regime_periods:
        shade_regimes(ax, regime_periods)
        
    ax.plot(results.index, results['cum_benchmark'], color="#333333", 
            linewidth=1.5, label="Benchmark (60/40)", alpha=0.8)
    
    if 'cum_binary' in results.columns:
        ax.plot(results.index, results['cum_binary'], color="#2c7bb6", 
                linewidth=1.5, label="Binary Strategy", alpha=0.7)
        
    if 'cum_continuous' in results.columns:
        ax.plot(results.index, results['cum_continuous'], color="#d7191c", 
                linewidth=2.0, label="Continuous Logistic Strategy")
    
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Total Return ($1 Start)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Risk Score and Weight Attribution
# ---------------------------------------------------------------------------

def plot_risk_and_weights(
    results: pd.DataFrame,
    title: str = "Continuous Strategy: Risk Score vs. Equity Weight",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the Risk Score and the resulting Equity Weight over time.
    """
    fig, ax1 = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    
    ax1.plot(results.index, results['risk_score'], color="#d7191c", 
             linewidth=1.5, label="Risk Score (0-1)")
    ax1.set_ylabel("Risk Score", color="#d7191c")
    ax1.tick_params(axis='y', labelcolor="#d7191c")
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    ax2.plot(results.index, results['weight_stocks'], color="#2c7bb6", 
             linewidth=2.0, label="Equity Weight (%)")
    ax2.set_ylabel("Equity Weight", color="#2c7bb6")
    ax2.tick_params(axis='y', labelcolor="#2c7bb6")
    ax2.set_ylim(0, 1.0)
    
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date")
    
    fig.tight_layout()
    return fig, ax1
