"""
analysis/lead_lag.py
Tests whether financial variables anticipate (lead) changes in macro activity,
as measured by PC1.

Two regression frameworks:
  1. Level prediction:  PC1_{t+h} = α + β₁·FIN₁_t + ... + ε_t
  2. Change prediction: ΔPC1_{t+h} = α + β₁·FIN₁_t + ... + ε_t

Horizons h = 1, 3, 6 months.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Dict

from analysis.financial_linkage import ols_regression, regression_summary_table


# ---------------------------------------------------------------------------
# Lead generator
# ---------------------------------------------------------------------------

def create_lead(series: pd.Series, h: int = 1) -> pd.Series:
    """
    Shift series backward by h months to create a lead variable.
    E.g.: lead_1 at date t holds the value at t+1, so regressing y
    on X_{t} where y = PC1_{t+h} is equivalent.

    Implementation: shift PC1 forward (negative shift) so that
    index t aligns with the value h steps ahead.

    Parameters
    ----------
    series : PC1 or ΔPC1
    h      : forecast horizon in months

    Returns
    -------
    pd.Series shifted by -h (lead)
    """
    return series.shift(-h).rename(f"{series.name}_lead{h}")


# ---------------------------------------------------------------------------
# Lead-lag regression
# ---------------------------------------------------------------------------

def lead_lag_regression(
    pc1: pd.Series,
    financial_panel: pd.DataFrame,
    leads: List[int] = None,
    use_delta: bool = False,
) -> Dict[int, object]:
    """
    For each horizon h in `leads`, regress the h-step-ahead macro signal
    (PC1_{t+h} or ΔPC1_{t+h}) on current financial variables.

    Model:
        PC1_{t+h} = α + β₁·yc_t + β₂·hy_t + β₃·vix_t + β₄·sp_ret_t + ε

    Parameters
    ----------
    pc1             : PC1 series
    financial_panel : DataFrame of financial variables (regressors)
    leads           : list of forecast horizons in months (default [1, 3, 6])
    use_delta       : if True, use ΔPC1 (month-over-month change) as dependent var

    Returns
    -------
    dict mapping horizon h → statsmodels results object
    """
    if leads is None:
        leads = [1, 3, 6]

    results = {}

    y_base = pc1.diff(1).rename("delta_pc1") if use_delta else pc1

    for h in leads:
        y_lead = create_lead(y_base, h)
        dep_label = f"{'ΔPCA1' if use_delta else 'PC1'}_t+{h}"

        # Align: X at t, y at t+h
        X_fin = financial_panel.copy()
        combined = pd.concat([y_lead, X_fin], axis=1).dropna()

        if len(combined) < 20:
            print(f"  [SKIP] h={h}: only {len(combined)} obs after alignment.")
            continue

        y_clean = combined.iloc[:, 0]
        X_clean = sm.add_constant(combined.iloc[:, 1:])
        model   = sm.OLS(y_clean, X_clean)
        res     = model.fit(cov_type="HC3")
        results[h] = res

        sig_vars = [
            f"{var}(p={res.pvalues[var]:.2f})"
            for var in financial_panel.columns
            if var in res.pvalues and res.pvalues[var] < 0.10
        ]
        print(f"  Lead h={h:2d}:  {dep_label} ~ Financial  "
              f"R²={res.rsquared:.3f}  "
              f"Sig vars: {sig_vars if sig_vars else 'none'}")

    return results


def lead_lag_summary_table(
    results: Dict[int, object],
    financial_cols: List[str],
) -> pd.DataFrame:
    """
    Compile lead-lag regression results into a multi-horizon summary table.

    Rows:    financial variables
    Columns: horizon h (β coefficient, with significance stars)
    """
    rows = {}
    for h, res in results.items():
        for col in financial_cols:
            if col in res.params.index:
                beta = res.params[col]
                p    = res.pvalues[col]
                sig  = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                rows.setdefault(col, {})[f"h={h} β"] = f"{beta:.3f}{sig}"

    df = pd.DataFrame(rows).T
    df.index.name = "financial_variable"
    return df


def bivariate_lead_lag(
    pc1: pd.Series,
    financial_panel: pd.DataFrame,
    leads: List[int] = None,
    use_delta: bool = False,
) -> pd.DataFrame:
    """
    Run bivariate (one-at-a-time) lead-lag regressions for each
    financial variable × horizon combination.

    Returns a long-form DataFrame with columns:
        variable, horizon, beta, std_err, p_value, R2
    """
    if leads is None:
        leads = [1, 3, 6]

    y_base = pc1.diff(1).rename("delta_pc1") if use_delta else pc1
    rows = []

    for col in financial_panel.columns:
        for h in leads:
            y_lead = create_lead(y_base, h)
            combined = pd.concat([y_lead, financial_panel[col]], axis=1).dropna()
            if len(combined) < 15:
                continue

            y_c = combined.iloc[:, 0]
            X_c = sm.add_constant(combined.iloc[:, 1:])
            res = sm.OLS(y_c, X_c).fit(cov_type="HC3")

            rows.append({
                "variable": col,
                "horizon":  h,
                "beta":     round(res.params[col], 4),
                "std_err":  round(res.bse[col], 4),
                "t_stat":   round(res.tvalues[col], 3),
                "p_value":  round(res.pvalues[col], 4),
                "R2":       round(res.rsquared, 4),
                "n_obs":    int(res.nobs),
            })

    return pd.DataFrame(rows)
