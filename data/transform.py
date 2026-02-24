"""
data/transform.py
Applies economically sensible transformations to raw panel variables
before standardization and PCA.

Transformation types supported:
  'pct_mom'    → month-over-month percent change
  'pct_yoy'    → year-over-year percent change (12-month)
  'log_return' → log(P_t / P_{t-1}) × 100
  'level'      → no transformation, use raw level
  'change'     → first difference (level_t - level_{t-1})
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# Individual transformation functions
# ---------------------------------------------------------------------------

def pct_change_mom(series: pd.Series) -> pd.Series:
    """Month-over-month percent change (%)."""
    return series.pct_change(periods=1) * 100


def pct_change_yoy(series: pd.Series) -> pd.Series:
    """Year-over-year percent change (12-month, %)."""
    return series.pct_change(periods=12) * 100


def log_return(series: pd.Series) -> pd.Series:
    """Continuously compounded (log) return × 100."""
    return np.log(series / series.shift(1)) * 100


def first_difference(series: pd.Series) -> pd.Series:
    """Simple first difference."""
    return series.diff(1)


def identity(series: pd.Series) -> pd.Series:
    """No transformation — return as-is."""
    return series


# Dispatch table
_TRANSFORM_FUNCS = {
    "pct_mom":    pct_change_mom,
    "pct_yoy":    pct_change_yoy,
    "log_return": log_return,
    "change":     first_difference,
    "level":      identity,
}


# ---------------------------------------------------------------------------
# Apply transformations to the whole panel
# ---------------------------------------------------------------------------

def compute_sp500_return(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute S&P 500 log return from the 'sp500' level column and
    add it as 'sp500_return'. Drops the raw 'sp500' price level column.
    """
    if "sp500" not in panel.columns:
        return panel
    df = panel.copy()
    df["sp500_return"] = log_return(df["sp500"])
    df.drop(columns=["sp500"], inplace=True)
    return df


def apply_transformations(
    panel: pd.DataFrame,
    transform_map: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Apply per-variable transformations specified in transform_map.

    Parameters
    ----------
    panel         : merged monthly panel (raw values)
    transform_map : dict mapping column name → transform type string.
                    Defaults to config.TRANSFORM_MAP.

    Returns
    -------
    DataFrame with transformed columns (NaN rows at the top from lags are
    preserved — drop them later during cleaning).
    """
    if transform_map is None:
        transform_map = config.TRANSFORM_MAP

    # First handle S&P 500 specially (level → return)
    df = compute_sp500_return(panel)

    transformed = {}

    for col in df.columns:
        if col in ("nber_recession", "cfnai"):
            # Reference columns — keep as-is
            transformed[col] = df[col]
            continue

        t_type = transform_map.get(col, "level")
        func = _TRANSFORM_FUNCS.get(t_type)

        if func is None:
            raise ValueError(
                f"Unknown transform type '{t_type}' for column '{col}'. "
                f"Valid types: {list(_TRANSFORM_FUNCS.keys())}"
            )

        transformed[col] = func(df[col])
        if t_type != "level":
            print(f"  Transformed '{col}' → {t_type}")

    result = pd.DataFrame(transformed, index=df.index)
    return result


def drop_leading_nans(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Drop rows with NaN in the specified columns (or all columns if None).
    Typically called after transformations to remove the lag-induced NaN rows.
    """
    if cols is None:
        cols = df.columns.tolist()
    return df.dropna(subset=cols)


if __name__ == "__main__":
    # Smoke-test using a small synthetic series
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    fake = pd.DataFrame({
        "industrial_production": 100 + np.cumsum(np.random.randn(24)),
        "unemployment_rate":     [4.0 + 0.1 * i for i in range(24)],
        "sp500":                 [1400 + 10 * i for i in range(24)],
        "vix":                   [20 + np.random.randn() for _ in range(24)],
    }, index=dates)

    result = apply_transformations(fake)
    print(result.head(5))
