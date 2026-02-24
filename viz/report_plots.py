"""
viz/report_plots.py
Generates and saves all publication-quality charts to outputs/charts/.
Call generate_all_report_plots() from main.py after the pipeline runs.
"""

import os
import pandas as pd
import sys

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from viz.charts import (
    plot_pc1_signal,
    plot_loadings,
    plot_scree,
    plot_regime_periods,
    plot_correlation_heatmap,
    plot_lead_lag_betas,
)


def _save(fig, filename: str, charts_dir: str = config.CHARTS_DIR) -> str:
    """Save a figure to charts_dir and return the full path."""
    os.makedirs(charts_dir, exist_ok=True)
    path = os.path.join(charts_dir, filename)
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    return path


def generate_all_report_plots(
    pc1: pd.Series,
    pc1_smooth: pd.Series,
    loadings_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    regime_periods_list: list,
    recession_periods: list,
    corr_matrix: pd.DataFrame,
    bivariate_lead_lag_df: pd.DataFrame,
) -> dict:
    """
    Generate and save all report charts. Returns a dict of {name: filepath}.

    Parameters
    ----------
    pc1                  : PC1 series
    pc1_smooth           : smoothed PC1
    loadings_df          : from pca.build_indicator.loadings_table()
    variance_df          : from pca.build_indicator.variance_explained()
    regime_periods_list  : from pca.regime.regime_periods()
    recession_periods    : NBER recession (start, end) tuples
    corr_matrix          : correlation matrix DataFrame (pc1 + financial vars)
    bivariate_lead_lag_df: from analysis.lead_lag.bivariate_lead_lag()
    """
    saved = {}

    print("\nGenerating report charts...")

    # 1. PC1 signal chart
    fig, _ = plot_pc1_signal(
        pc1, pc1_smooth,
        recession_periods=recession_periods,
        regime_periods_list=regime_periods_list,
        title="PCA Macro Activity Indicator (PC1) — Monthly Signal",
    )
    saved["pc1_signal"] = _save(fig, "01_pc1_signal.png")
    fig.clear()

    # 2. PC1 loadings
    fig, _ = plot_loadings(loadings_df, component="PC1")
    saved["pc1_loadings"] = _save(fig, "02_pc1_loadings.png")
    fig.clear()

    # 3. Scree plot
    fig, _ = plot_scree(variance_df)
    saved["scree"] = _save(fig, "03_scree_plot.png")
    fig.clear()

    # 4. Regime chart
    fig, _ = plot_regime_periods(
        pc1, pc1_smooth, regime_periods_list, recession_periods,
    )
    saved["regime"] = _save(fig, "04_regime_chart.png")
    fig.clear()

    # 5. Correlation heatmap
    fig, _ = plot_correlation_heatmap(corr_matrix)
    saved["correlation"] = _save(fig, "05_correlation_heatmap.png")
    fig.clear()

    # 6. Lead-lag betas
    if bivariate_lead_lag_df is not None and len(bivariate_lead_lag_df) > 0:
        fig, _ = plot_lead_lag_betas(bivariate_lead_lag_df)
        saved["lead_lag"] = _save(fig, "06_lead_lag_betas.png")
        fig.clear()

    print(f"\nAll charts saved to: {config.CHARTS_DIR}/")
    return saved
