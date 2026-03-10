"""
analysis/financial_linkage.py
Analyzes contemporaneous relationships between the PC1 macro signal
and financial market variables (yield curve, credit spreads, equities, VIX).

Covers:
  - Pairwise correlation table
  - Simple OLS regressions (financial var ~ PC1 and PC1 ~ financial var)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(
    pc1: pd.Series,
    financial_panel: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlations between PC1 and each financial variable.

    Parameters
    ----------
    pc1             : PC1 macro signal series
    financial_panel : DataFrame with columns for each financial variable
    method          : 'pearson' (default), 'spearman', or 'kendall'

    Returns
    -------
    DataFrame with columns: variable, correlation, p_value, n_obs
    """
    from scipy import stats as scipy_stats

    pc1_name = pc1.name or "pc1"
    results = []

    for col in financial_panel.columns:
        combined = pd.DataFrame({pc1_name: pc1, col: financial_panel[col]}).dropna()
        if len(combined) < 10:
            print(f"  [SKIP] Too few observations for '{col}' ({len(combined)})")
            continue

        if method == "pearson":
            r, p = scipy_stats.pearsonr(combined[pc1_name], combined[col])
        elif method == "spearman":
            r, p = scipy_stats.spearmanr(combined[pc1_name], combined[col])
        elif method == "kendall":
            r, p = scipy_stats.kendalltau(combined[pc1_name], combined[col])
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            "variable":    col,
            "correlation": round(r, 4),
            "p_value":     round(p, 4),
            "n_obs":       len(combined),
            "significant": "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")),
        })

    return pd.DataFrame(results).set_index("variable")


# ---------------------------------------------------------------------------
# OLS regression wrapper
# ---------------------------------------------------------------------------

def ols_regression(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    add_const: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression of y on X using statsmodels.

    Parameters
    ----------
    y         : dependent variable
    X         : independent variable(s)
    add_const : if True, add an intercept (default True)

    Returns
    -------
    statsmodels RegressionResults object
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # Align on common date index and drop NaNs
    combined = pd.concat([y.rename("y"), X], axis=1).dropna()
    y_clean = combined["y"]
    X_clean = combined.drop(columns=["y"])

    if add_const:
        X_clean = sm.add_constant(X_clean)

    model = sm.OLS(y_clean, X_clean)
    result = model.fit(cov_type="HC3")  # Heteroskedasticity-robust SE
    return result


def contemporaneous_regressions(
    pc1: pd.Series,
    financial_panel: pd.DataFrame,
    direction: str = "financial_on_pc1",
) -> Dict[str, object]:
    """
    Run simple bivariate OLS regressions for each financial variable vs. PC1.

    Parameters
    ----------
    pc1             : PC1 macro signal
    financial_panel : DataFrame of financial variables
    direction       : 'financial_on_pc1' -> financial_var ~ PC1  (default)
                      'pc1_on_financial' -> PC1 ~ financial_var

    Returns
    -------
    dict mapping column name -> statsmodels results object
    """
    results = {}

    for col in financial_panel.columns:
        if direction == "financial_on_pc1":
            y = financial_panel[col]
            X = pc1
            label = f"{col} ~ PC1"
        elif direction == "pc1_on_financial":
            y = pc1
            X = financial_panel[col]
            label = f"PC1 ~ {col}"
        else:
            raise ValueError(f"Unknown direction: {direction}")

        res = ols_regression(y, X)
        results[col] = res
        print(f"  [{label}]  R²={res.rsquared:.3f}  "
              f"β={res.params.iloc[-1]:.4f}  p={res.pvalues.iloc[-1]:.3f}")

    return results


def regression_summary_table(results: Dict[str, object]) -> pd.DataFrame:
    """
    Compile key stats from a dict of statsmodels results into a summary DataFrame.

    Columns: beta (slope), std_err, t_stat, p_value, R2, n_obs
    """
    rows = []
    for name, res in results.items():
        # Last coefficient (the substantive regressor, not the constant)
        idx = -1
        rows.append({
            "variable":  name,
            "beta":      round(res.params.iloc[idx], 4),
            "std_err":   round(res.bse.iloc[idx], 4),
            "t_stat":    round(res.tvalues.iloc[idx], 3),
            "p_value":   round(res.pvalues.iloc[idx], 4),
            "R2":        round(res.rsquared, 4),
            "n_obs":     int(res.nobs),
            "sig":       "***" if res.pvalues.iloc[idx] < 0.01 else (
                         "**"  if res.pvalues.iloc[idx] < 0.05 else (
                         "*"   if res.pvalues.iloc[idx] < 0.10 else "")),
        })
    return pd.DataFrame(rows).set_index("variable")
