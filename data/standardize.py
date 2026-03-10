"""
data/standardize.py
Z-score standardization and sign alignment for PCA inputs.

Why this matters:
  - PCA is scale-sensitive; variables on different scales dominate.
  - Sign alignment ensures all variables point in the same direction
    before PCA so that PC1 summarizes "macro strength" positively.

Standardization uses an EXPANDING WINDOW (point-in-time) to avoid
look-ahead bias in the backtest. At time t, mu and sigma are estimated
using only data from inception through t-1 (shift(1) ensures no current-
month leakage). A minimum of ZSCORE_MIN_PERIODS observations is required
before a Z-score is emitted; earlier rows are NaN and dropped downstream.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

def zscore_series(series: pd.Series) -> pd.Series:
    """
    DEPRECATED for backtest use — retained only for unit-test compatibility.
    Use expanding_zscore() for all pipeline calls.

    Full-sample z-score: Z = (X - mean) / std.
    NaN values are ignored in the mean/std calculation.
    WARNING: uses future data; introduces look-ahead bias in backtests.
    """
    mu  = series.mean(skipna=True)
    sig = series.std(skipna=True, ddof=1)
    if sig == 0 or np.isnan(sig):
        raise ValueError(f"Zero or NaN std for series '{series.name}' — check data.")
    return (series - mu) / sig


def expanding_zscore(series: pd.Series, min_periods: int = None) -> pd.Series:
    """
    Point-in-time expanding-window Z-score.

    At each time t the mean and std are estimated from [0, t-1] only
    (via .shift(1)), so no future information leaks into the signal.

    Parameters
    ----------
    series      : raw transformed series
    min_periods : minimum history required before emitting a score.
                  Defaults to config.ZSCORE_MIN_PERIODS (36 months).
                  Earlier rows will be NaN and dropped by drop_leading_nans.

    Returns
    -------
    pd.Series of Z-scores, same index as input.
    """
    if min_periods is None:
        min_periods = getattr(config, "ZSCORE_MIN_PERIODS", 36)

    # Compute expanding mu and sigma up to t-1 (shift by 1 = no current-month data)
    mu  = series.expanding(min_periods=min_periods).mean().shift(1)
    sig = series.expanding(min_periods=min_periods).std(ddof=1).shift(1)

    z = (series - mu) / sig
    return z


def standardize_panel(
    panel: pd.DataFrame,
    cols: List[str] = None,
) -> pd.DataFrame:
    """
    Apply point-in-time expanding-window Z-score standardization.

    Each column is standardized using only past data (no look-ahead).
    mu and sigma at time t are estimated from data through t-1 only.

    Parameters
    ----------
    panel : transformed monthly panel
    cols  : list of columns to standardize.
            Defaults to config.MACRO_PCA_COLS + config.FINANCIAL_COLS.

    Returns
    -------
    DataFrame with the same index; standardized columns replace originals.
    Reference columns (nber_recession, cfnai) are passed through unchanged.
    """
    if cols is None:
        cols = config.MACRO_PCA_COLS + config.FINANCIAL_COLS

    df = panel.copy()

    for col in cols:
        if col not in df.columns:
            print(f"  [SKIP] '{col}' not in panel — skipping standardization.")
            continue
        df[col] = expanding_zscore(df[col])
        non_nan = df[col].dropna()
        if len(non_nan) > 0:
            print(f"  Standardized '{col}' (expanding window) -> "
                  f"first valid: {non_nan.index[0].date()}, "
                  f"n={len(non_nan)}")

    return df


# ---------------------------------------------------------------------------
# Sign alignment
# ---------------------------------------------------------------------------

def apply_sign_alignment(
    panel: pd.DataFrame,
    sign_map: Dict[str, int] = None,
    cols: List[str] = None,
) -> pd.DataFrame:
    """
    Flip the sign of variables where sign_map[col] == -1 so that
    all variables are oriented as: HIGH = better macro conditions.

    Must be called AFTER standardization.

    Parameters
    ----------
    panel    : standardized monthly panel
    sign_map : dict mapping col -> +1 or -1.
               Defaults to config.SIGN_MAP.
    cols     : columns to apply alignment to.
               Defaults to config.MACRO_PCA_COLS + config.FINANCIAL_COLS.

    Returns
    -------
    DataFrame with sign-flipped columns where applicable.
    """
    if sign_map is None:
        sign_map = config.SIGN_MAP
    if cols is None:
        cols = config.MACRO_PCA_COLS + config.FINANCIAL_COLS

    df = panel.copy()

    for col in cols:
        if col not in df.columns:
            continue
        s = sign_map.get(col, 1)
        if s == -1:
            df[col] = - df[col]
            print(f"  Sign-flipped '{col}' (was negative orientation)")
        elif s != 1:
            raise ValueError(f"sign_map['{col}'] must be +1 or -1, got {s}")

    return df


# ---------------------------------------------------------------------------
# Convenience: combined pipeline step
# ---------------------------------------------------------------------------

def prepare_pca_matrix(
    panel: pd.DataFrame,
    pca_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Return the standardized + sign-aligned sub-panel ready for PCA.
    Only includes columns in pca_cols (defaults to config.MACRO_PCA_COLS).
    Drops any remaining NaN rows.

    Parameters
    ----------
    panel    : transformed panel (already passed through standardize_panel
               and apply_sign_alignment OR call them inside here)
    pca_cols : columns to include in PCA matrix

    Returns
    -------
    Clean DataFrame with no NaNs, only PCA input columns.
    """
    if pca_cols is None:
        pca_cols = config.MACRO_PCA_COLS

    # Select only PCA input columns that exist
    available = [c for c in pca_cols if c in panel.columns]
    missing   = [c for c in pca_cols if c not in panel.columns]
    if missing:
        print(f"  [WARN] PCA columns not in panel and will be skipped: {missing}")

    Z = panel[available].dropna()
    print(f"  PCA matrix shape: {Z.shape}  "
          f"({Z.index[0].date()} -> {Z.index[-1].date()})")
    return Z


if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range("2000-01-01", periods=50, freq="MS")
    fake_transformed = pd.DataFrame(
        np.random.randn(50, 4),
        index=dates,
        columns=["industrial_production", "unemployment_rate", "pmi_proxy", "vix"]
    )
    fake_transformed["nber_recession"] = 0

    std = standardize_panel(fake_transformed, cols=["industrial_production", "unemployment_rate", "pmi_proxy", "vix"])
    aligned = apply_sign_alignment(
        std,
        sign_map={"industrial_production": 1, "unemployment_rate": -1, "pmi_proxy": 1, "vix": -1},
        cols=["industrial_production", "unemployment_rate", "pmi_proxy", "vix"]
    )
    print(aligned.head())