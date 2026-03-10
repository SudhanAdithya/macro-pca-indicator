"""
main.py
End-to-end pipeline for the PCA Macro Slowdown Indicator.

Usage:
    python main.py [--fetch] [--no-fetch]

  --fetch     : re-download data from FRED (default if data/raw is empty)
  --no-fetch  : skip FRED download; use existing data/raw CSVs

Outputs written to:
    data/processed/panel_clean.csv
    outputs/pc1_series.csv
    outputs/loadings.csv
    outputs/tables/variance_explained.csv
    outputs/tables/contemporaneous_correlations.csv
    outputs/tables/contemporaneous_regressions.csv
    outputs/tables/lead_lag_bivariate.csv
    outputs/tables/nber_comparison.csv
    outputs/tables/stationarity_tests.csv
    outputs/tables/portfolio_metrics.csv
    outputs/charts/01_pc1_signal.png  ... (6 charts total)

Key methodological notes (addressing professor feedback):
  - Z-scores use an expanding window (point-in-time, no look-ahead bias).
    mu and sigma at time t are computed from data through t-1 only.
  - PCA eigenvectors are derived from the TRANSFORMED (pre-Z-score) panel,
    not from the Z-score matrix, to avoid second moments of second moments.
    PC1 signal scores are then obtained by projecting Z-scores onto the
    PC1 eigenvector.
  - All portfolio weights are lagged by SIGNAL_LAG_MONTHS (default 1) so
    that signal observed at month t drives trades at month t+1.
  - 60/40 benchmark is rebalanced monthly back to target weights with
    transaction costs charged on one-way turnover.
  - Sharpe ratios use the 3-Month T-Bill rate (TB3MS) as risk-free rate.
  - Alpha is reported as Jensen's alpha (OLS regression intercept of
    strategy excess returns on benchmark excess returns), not simple
    return difference.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetch_data import (get_fred_client, fetch_all_macro, fetch_all_financial,
                              fetch_reference_series, load_raw_series, load_rf_rate)
from data.align import align_macro_to_monthly, align_financial_to_monthly, merge_panel, trim_to_overlap
from data.transform import apply_transformations, drop_leading_nans, check_stationarity
from data.standardize import standardize_panel, apply_sign_alignment, prepare_pca_matrix
from pca.build_indicator import (fit_pca_on_transformed, project_zscore_to_pc1,
                                  normalize_pc1_sign, loadings_table, variance_explained)
from pca.regime import smooth_pc1, classify_regime, regime_periods, nber_recession_periods, compare_with_nber
from analysis.financial_linkage import compute_correlations, contemporaneous_regressions, regression_summary_table
from analysis.lead_lag import bivariate_lead_lag
from analysis.portfolio_engine import PortfolioEngine, save_portfolio_results
from viz.report_plots import generate_all_report_plots


def ensure_dirs():
    for d in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR,
              config.OUTPUTS_DIR, config.CHARTS_DIR, config.TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Stage 1: Data collection
# ---------------------------------------------------------------------------

def stage_fetch(args) -> None:
    raw_files = os.listdir(config.RAW_DATA_DIR) if os.path.exists(config.RAW_DATA_DIR) else []
    should_fetch = args.fetch or len(raw_files) == 0
    if not should_fetch:
        print("Stage 1: Skipping FRED fetch (use --fetch to re-download).")
        return
    print("\n=== Stage 1: Fetching FRED Data ===")
    fred = get_fred_client()
    fetch_all_macro(fred)
    fetch_all_financial(fred)
    from data.fetch_data import fetch_portfolio_assets
    fetch_portfolio_assets()
    fetch_reference_series(fred)   # now also fetches TB3MS (risk-free rate)


# ---------------------------------------------------------------------------
# Stage 2: Align & merge
# ---------------------------------------------------------------------------

def stage_align() -> pd.DataFrame:
    print("\n=== Stage 2: Aligning & Merging Panel ===")
    macro_frames, fin_frames, ref_frames = {}, {}, {}

    for col in config.MACRO_SERIES:
        try:    macro_frames[col] = load_raw_series(col)
        except FileNotFoundError: print(f"  [MISS] {col}")

    for col in config.FINANCIAL_SERIES:
        try:    fin_frames[col] = load_raw_series(col)
        except FileNotFoundError: print(f"  [MISS] {col}")

    for col in config.REFERENCE_SERIES:
        try:    ref_frames[col] = load_raw_series(col)
        except FileNotFoundError: print(f"  [MISS] {col}")

    macro_m = align_macro_to_monthly(pd.DataFrame(macro_frames))
    fin_m   = align_financial_to_monthly(pd.DataFrame(fin_frames))
    ref_m   = pd.DataFrame(ref_frames) if ref_frames else None

    panel = merge_panel(macro_m, fin_m, ref_m)
    panel = trim_to_overlap(panel, min_coverage=0.6)
    save_csv(panel, os.path.join(config.PROCESSED_DATA_DIR, "panel_raw_merged.csv"))
    return panel


# ---------------------------------------------------------------------------
# Stage 3: Transform, standardize, sign-align
# ---------------------------------------------------------------------------

def stage_transform(panel: pd.DataFrame):
    """
    Returns BOTH the transformed-only panel (for PCA fitting) and the
    fully standardized panel (expanding-window Z-scores, for signal projection).
    """
    print("\n=== Stage 3: Transforming Variables ===")

    panel_t = apply_transformations(panel)
    pca_cols_present = [c for c in config.MACRO_PCA_COLS if c in panel_t.columns]
    panel_t = drop_leading_nans(panel_t, cols=pca_cols_present)
    print(f"  After transformation drop: {len(panel_t)} rows")

    # Stationarity check — validates level vs pct_mom choices
    print("\n--- Stationarity tests (ADF) ---")
    check_stationarity(panel_t, cols=pca_cols_present)

    print("\n--- Expanding-window Z-score standardization ---")
    all_model_cols = pca_cols_present + [c for c in config.FINANCIAL_COLS if c in panel_t.columns]
    panel_std = standardize_panel(panel_t, cols=all_model_cols)

    print("\n--- Sign alignment ---")

    panel_t_aligned = apply_sign_alignment(panel_t, cols=all_model_cols)
    panel_std = standardize_panel(panel_t_aligned, cols=all_model_cols)

    save_csv(panel_std, os.path.join(config.PROCESSED_DATA_DIR, "panel_clean.csv"))
    return panel_t_aligned, panel_std


# ---------------------------------------------------------------------------
# Stage 4: PCA  (two-step: fit on transformed, project Z-scores)
# ---------------------------------------------------------------------------

def stage_pca(panel_t_aligned: pd.DataFrame, panel_std: pd.DataFrame):
    print("\n=== Stage 4: Building PCA Indicator ===")

    # --- Step A: derive eigenvectors from TRANSFORMED (not Z-scored) data ---
    pca_cols_present = [c for c in config.MACRO_PCA_COLS if c in panel_t_aligned.columns]
    X_trans = panel_t_aligned[pca_cols_present].dropna()
    print("  Step A: fitting PCA eigenvectors on transformed panel...")
    pca, scaler = fit_pca_on_transformed(X_trans)

    # --- Step B: project expanding-window Z-scores onto PC1 ---
    print("  Step B: projecting Z-scores onto PC1 eigenvector...")
    Z = prepare_pca_matrix(panel_std, pca_cols=config.MACRO_PCA_COLS)
    pc1_raw = project_zscore_to_pc1(pca, Z)

    # Sign normalization
    nber = panel_std.get("nber_recession")
    pc1  = normalize_pc1_sign(pc1_raw, recession_mask=nber)

    loadings_df = loadings_table(pca, list(X_trans.columns))
    variance_df = variance_explained(pca)

    pc1.to_csv(os.path.join(config.OUTPUTS_DIR, "pc1_series.csv"), header=True)
    save_csv(loadings_df, os.path.join(config.OUTPUTS_DIR, "loadings.csv"))
    save_csv(variance_df, os.path.join(config.TABLES_DIR, "variance_explained.csv"))

    print("\nPC1 summary:")
    print(pc1.describe().round(3))
    return pc1, pca, loadings_df, variance_df, Z


# ---------------------------------------------------------------------------
# Stage 5: Regime classification
# ---------------------------------------------------------------------------

def stage_regime(pc1: pd.Series, panel_std: pd.DataFrame):
    print("\n=== Stage 5: Regime Classification ===")
    pc1_smooth    = smooth_pc1(pc1)
    regime_series = classify_regime(pc1, threshold=config.REGIME_THRESHOLD)
    slow_periods  = regime_periods(regime_series)
    nber = panel_std.get("nber_recession")
    rec_periods = nber_recession_periods(nber) if nber is not None else []
    if nber is not None:
        nber_aligned = nber.reindex(regime_series.index).fillna(0)
        comparison = compare_with_nber(regime_series, nber_aligned)
        print("\nNBER Comparison:"); print(comparison)
        save_csv(comparison, os.path.join(config.TABLES_DIR, "nber_comparison.csv"))
    regime_series.to_csv(os.path.join(config.OUTPUTS_DIR, "regime_series.csv"), header=True)
    return pc1_smooth, regime_series, slow_periods, rec_periods


# ---------------------------------------------------------------------------
# Stage 6: Financial linkage
# ---------------------------------------------------------------------------

def stage_financial_linkage(pc1: pd.Series, panel_std: pd.DataFrame):
    print("\n=== Stage 6: Financial Linkage Analysis ===")
    fin_cols = [c for c in config.FINANCIAL_COLS if c in panel_std.columns]
    if not fin_cols:
        print("  No financial columns found — skipping.")
        return None, None
    fin_panel = panel_std[fin_cols]
    corr_table = compute_correlations(pc1, fin_panel)
    save_csv(corr_table, os.path.join(config.TABLES_DIR, "contemporaneous_correlations.csv"))
    reg_results = contemporaneous_regressions(pc1, fin_panel)
    reg_table   = regression_summary_table(reg_results)
    save_csv(reg_table, os.path.join(config.TABLES_DIR, "contemporaneous_regressions.csv"))
    ll_df = bivariate_lead_lag(pc1, fin_panel, leads=[1, 3, 6])
    save_csv(ll_df, os.path.join(config.TABLES_DIR, "lead_lag_bivariate.csv"))
    combined     = pd.concat([pc1.rename("pc1"), fin_panel], axis=1).dropna()
    corr_matrix  = combined.corr()
    return corr_matrix, ll_df


# ---------------------------------------------------------------------------
# Stage 8: Portfolio Optimization
# ---------------------------------------------------------------------------

def stage_portfolio_optimization(pc1_smooth: pd.Series, regime_series: pd.Series):
    print("\n=== Stage 8: Portfolio Optimization ===")

    assets = {}
    for col in config.PORTFOLIO_ASSETS:
        try:    assets[col] = load_raw_series(f"asset_{col}")
        except FileNotFoundError:
            print(f"  [MISS] asset_{col} — run with --fetch."); return None, None

    asset_prices = pd.DataFrame(assets)

    # Load risk-free rate (TB3MS) — needed for correct Sharpe and Jensen's alpha
    rf_monthly = load_rf_rate()

    engine = PortfolioEngine(
        asset_prices=asset_prices,
        pc1_smooth=pc1_smooth,
        regimes=regime_series,
        rf_series=rf_monthly,
    )

    print("  Running backtest (Benchmark + Binary + Continuous)...")
    results = engine.run_backtest()

    print("  Calculating metrics (Sharpe with Rf, Jensen's alpha)...")
    metrics = engine.calculate_metrics(results)
    print(metrics.to_string())

    save_portfolio_results(results, metrics)
    return results, metrics


# ---------------------------------------------------------------------------
# Stage 7: Charts
# ---------------------------------------------------------------------------

def stage_charts(pc1, pc1_smooth, loadings_df, variance_df,
                 slow_periods, rec_periods, corr_matrix, ll_df,
                 portfolio_results=None):
    print("\n=== Stage 7: Generating Charts ===")
    generate_all_report_plots(
        pc1=pc1, pc1_smooth=pc1_smooth, loadings_df=loadings_df,
        variance_df=variance_df, regime_periods_list=slow_periods,
        recession_periods=rec_periods,
        corr_matrix=corr_matrix if corr_matrix is not None else pd.DataFrame(),
        bivariate_lead_lag_df=ll_df if ll_df is not None else pd.DataFrame(),
        portfolio_results= portfolio_results,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PCA Macro Slowdown Indicator Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fetch",    action="store_true")
    group.add_argument("--no-fetch", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    stage_fetch(args)
    panel                  = stage_align()
    panel_t_aligned, panel_std = stage_transform(panel)
    pc1, pca, loadings_df, variance_df, Z = stage_pca(panel_t_aligned, panel_std)
    pc1_smooth, regime_series, slow_periods, rec_periods = stage_regime(pc1, panel_std)
    corr_matrix, ll_df     = stage_financial_linkage(pc1, panel_std)
    portfolio_results, _   = stage_portfolio_optimization(pc1_smooth, regime_series)
    stage_charts(pc1, pc1_smooth, loadings_df, variance_df,
                 slow_periods, rec_periods, corr_matrix, ll_df,
                 portfolio_results=portfolio_results)
    print("\n✅  Pipeline complete. Outputs in:", config.OUTPUTS_DIR)


if __name__ == "__main__":
    main()