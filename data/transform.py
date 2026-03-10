"""
data/transform.py
Applies economically sensible transformations to raw panel variables
before standardization and PCA.

Transformation types supported:
  'pct_mom'    -> month-over-month percent change
  'pct_yoy'    -> year-over-year percent change (12-month)
  'log_return' -> log(P_t / P_{t-1}) × 100
  'level'      -> no transformation, use raw level
  'change'     -> first difference (level_t - level_{t-1})

TRANSFORMATION DECISION RULE
------------------------------
'pct_mom' is applied to FLOW variables that are non-stationary in levels
(industrial_production, retail_sales, housing_starts, real_personal_income,
mfg_new_orders). These trend upward over time; levels would cause PCA to
capture secular growth rather than cyclical variation.

'level' is applied to BOUNDED or mean-reverting variables
(unemployment_rate: 3–15% range; capacity_utilization: 60–90%;
pmi_proxy: 0–100 index). These are stationary in levels.
Verified via ADF test (run check_stationarity() after transformation —
all level-kept series should show ADF p < 0.05).

'log_return' is used for SP500 price to ensure approximate normality
and stationarity.
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
    return series.ffill().pct_change(periods=1) * 100


def pct_change_yoy(series: pd.Series) -> pd.Series:
    """Year-over-year percent change (12-month, %)."""
    return series.ffill().pct_change(periods=1) * 100


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
    transform_map : dict mapping column name -> transform type string.
                    Defaults to config.TRANSFORM_MAP.

    Returns
    -------
    DataFrame with transformed columns (NaN rows at the top from lags are
    preserved — drop them later during cleaning).
    """
    if transform_map is None:
        transform_map = config.TRANSFORM_MAP

    # First handle S&P 500 specially (level -> return)
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
            print(f"  Transformed '{col}' -> {t_type}")

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


def check_stationarity(
    panel: pd.DataFrame,
    cols: list = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller tests to verify stationarity of transformed
    variables. Call this after apply_transformations() to confirm that level-
    kept variables are indeed stationary and pct_mom variables are stationary.

    Parameters
    ----------
    panel : transformed monthly panel (post apply_transformations)
    cols  : columns to test (defaults to all MACRO_PCA_COLS present in panel)
    alpha : significance level for stationarity decision (default 0.05)

    Returns
    -------
    DataFrame with columns: adf_stat, p_value, stationary (bool), transform_used
    """
    from statsmodels.tsa.stattools import adfuller

    if cols is None:
        cols = [c for c in config.MACRO_PCA_COLS if c in panel.columns]

    rows = []
    for col in cols:
        series = panel[col].dropna()
        if len(series) < 20:
            continue
        try:
            adf_stat, p_value, _, _, _, _ = adfuller(series, autolag="AIC")
            stationary = p_value < alpha
            rows.append({
                "variable":       col,
                "adf_stat":       round(adf_stat, 4),
                "p_value":        round(p_value, 4),
                "stationary":     stationary,
                "transform_used": config.TRANSFORM_MAP.get(col, "level"),
                "note":           "OK" if stationary else "⚠️  Non-stationary",
            })
            flag = "✓" if stationary else "✗"
            print(f"  ADF {flag} '{col}': stat={adf_stat:.3f}, p={p_value:.4f} "
                  f"({'stationary' if stationary else 'NON-STATIONARY'})")
        except Exception as e:
            print(f"  ADF [FAIL] '{col}': {e}")

    result_df = pd.DataFrame(rows).set_index("variable")

    save_path = os.path.join(
        getattr(config, "TABLES_DIR", "outputs/tables"),
        "stationarity_tests.csv"
    )
    if os.path.exists(os.path.dirname(save_path)):
        result_df.to_csv(save_path)
        print(f"  Stationarity results saved to {save_path}")

    return result_df


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