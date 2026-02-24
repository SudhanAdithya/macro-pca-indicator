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
    outputs/charts/01_pc1_signal.png  ... (6 charts total)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetch_data import get_fred_client, fetch_all_macro, fetch_all_financial, fetch_reference_series, load_raw_series
from data.align import align_macro_to_monthly, align_financial_to_monthly, merge_panel, trim_to_overlap
from data.transform import apply_transformations, drop_leading_nans
from data.standardize import standardize_panel, apply_sign_alignment, prepare_pca_matrix
from pca.build_indicator import run_pca, extract_pc1, normalize_pc1_sign, loadings_table, variance_explained
from pca.regime import smooth_pc1, classify_regime, regime_periods, nber_recession_periods, compare_with_nber
from analysis.financial_linkage import compute_correlations, contemporaneous_regressions, regression_summary_table
from analysis.lead_lag import bivariate_lead_lag
from viz.report_plots import generate_all_report_plots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    print("  Macro fundamentals:")
    fetch_all_macro(fred)
    print("  Financial variables:")
    fetch_all_financial(fred)
    print("  Reference series:")
    fetch_reference_series(fred)


# ---------------------------------------------------------------------------
# Stage 2: Align & merge
# ---------------------------------------------------------------------------

def stage_align() -> pd.DataFrame:
    print("\n=== Stage 2: Aligning & Merging Panel ===")

    macro_frames = {}
    for col in config.MACRO_SERIES:
        try:
            macro_frames[col] = load_raw_series(col)
        except FileNotFoundError:
            print(f"  [MISS] {col} — no raw file")
    macro_raw = pd.DataFrame(macro_frames)

    fin_frames = {}
    for col in config.FINANCIAL_SERIES:
        try:
            fin_frames[col] = load_raw_series(col)
        except FileNotFoundError:
            print(f"  [MISS] {col} — no raw file")
    fin_raw = pd.DataFrame(fin_frames)

    ref_frames = {}
    for col in config.REFERENCE_SERIES:
        try:
            ref_frames[col] = load_raw_series(col)
        except FileNotFoundError:
            print(f"  [MISS] {col} — no raw file")
    ref_raw = pd.DataFrame(ref_frames) if ref_frames else None

    macro_m = align_macro_to_monthly(macro_raw)
    fin_m   = align_financial_to_monthly(fin_raw)
    ref_m   = ref_raw

    panel = merge_panel(macro_m, fin_m, ref_m)
    panel = trim_to_overlap(panel, min_coverage=0.6)

    save_csv(panel, os.path.join(config.PROCESSED_DATA_DIR, "panel_raw_merged.csv"))
    return panel


# ---------------------------------------------------------------------------
# Stage 3: Transform, standardize, sign-align
# ---------------------------------------------------------------------------

def stage_transform(panel: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Stage 3: Transforming Variables ===")

    panel_t = apply_transformations(panel)
    # Drop leading NaN rows introduced by transformations
    pca_cols_present = [c for c in config.MACRO_PCA_COLS if c in panel_t.columns]
    panel_t = drop_leading_nans(panel_t, cols=pca_cols_present)
    print(f"  After transformation drop: {len(panel_t)} rows")

    print("\n--- Standardizing ---")
    all_model_cols = pca_cols_present + [c for c in config.FINANCIAL_COLS if c in panel_t.columns]
    panel_std = standardize_panel(panel_t, cols=all_model_cols)

    print("\n--- Sign alignment ---")
    panel_std = apply_sign_alignment(panel_std, cols=all_model_cols)

    save_csv(panel_std, os.path.join(config.PROCESSED_DATA_DIR, "panel_clean.csv"))
    return panel_std


# ---------------------------------------------------------------------------
# Stage 4: PCA
# ---------------------------------------------------------------------------

def stage_pca(panel_std: pd.DataFrame):
    print("\n=== Stage 4: Building PCA Indicator ===")

    Z = prepare_pca_matrix(panel_std, pca_cols=config.MACRO_PCA_COLS)

    pca, scores = run_pca(Z)
    pc1_raw = extract_pc1(pca, scores, Z)

    # Sign normalization using NBER if available
    nber = panel_std.get("nber_recession")
    pc1 = normalize_pc1_sign(pc1_raw, recession_mask=nber)

    loadings_df  = loadings_table(pca, list(Z.columns))
    variance_df  = variance_explained(pca)

    # Save outputs
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
        print("\nNBER Comparison:")
        print(comparison)
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
        print("  No financial columns found — skipping linkage analysis.")
        return None, None

    fin_panel = panel_std[fin_cols]

    # Correlations
    print("\n--- Contemporaneous correlations ---")
    corr_table = compute_correlations(pc1, fin_panel)
    print(corr_table)
    save_csv(corr_table, os.path.join(config.TABLES_DIR, "contemporaneous_correlations.csv"))

    # OLS regressions
    print("\n--- Contemporaneous regressions (financial ~ PC1) ---")
    reg_results = contemporaneous_regressions(pc1, fin_panel, direction="financial_on_pc1")
    reg_table   = regression_summary_table(reg_results)
    save_csv(reg_table, os.path.join(config.TABLES_DIR, "contemporaneous_regressions.csv"))

    # Lead-lag
    print("\n--- Lead-lag bivariate regressions ---")
    ll_df = bivariate_lead_lag(pc1, fin_panel, leads=[1, 3, 6])
    print(ll_df)
    save_csv(ll_df, os.path.join(config.TABLES_DIR, "lead_lag_bivariate.csv"))

    # Build correlation matrix for heatmap (include pc1)
    combined = pd.concat([pc1.rename("pc1"), fin_panel], axis=1).dropna()
    corr_matrix = combined.corr()

    return corr_matrix, ll_df


# ---------------------------------------------------------------------------
# Stage 7: Charts
# ---------------------------------------------------------------------------

def stage_charts(pc1, pc1_smooth, loadings_df, variance_df,
                 slow_periods, rec_periods, corr_matrix, ll_df):
    print("\n=== Stage 7: Generating Charts ===")
    generate_all_report_plots(
        pc1=pc1,
        pc1_smooth=pc1_smooth,
        loadings_df=loadings_df,
        variance_df=variance_df,
        regime_periods_list=slow_periods,
        recession_periods=rec_periods,
        corr_matrix=corr_matrix if corr_matrix is not None else pd.DataFrame(),
        bivariate_lead_lag_df=ll_df if ll_df is not None else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PCA Macro Slowdown Indicator Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fetch",    action="store_true", help="Force re-download from FRED")
    group.add_argument("--no-fetch", action="store_true", help="Skip FRED download")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    # Stage 1: Data fetch
    stage_fetch(args)

    # Stage 2: Align & merge
    panel = stage_align()

    # Stage 3: Transform & standardize
    panel_std = stage_transform(panel)

    # Stage 4: PCA
    pc1, pca, loadings_df, variance_df, Z = stage_pca(panel_std)

    # Stage 5: Regime
    pc1_smooth, regime_series, slow_periods, rec_periods = stage_regime(pc1, panel_std)

    # Stage 6: Financial linkage
    corr_matrix, ll_df = stage_financial_linkage(pc1, panel_std)

    # Stage 7: Charts
    stage_charts(pc1, pc1_smooth, loadings_df, variance_df,
                 slow_periods, rec_periods, corr_matrix, ll_df)

    print("\n✅  Pipeline complete. Outputs in:", config.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
