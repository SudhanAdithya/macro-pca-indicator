"""
analysis/recession_model.py  [OPTIONAL EXTENSION]
Binary recession probability model using logistic regression.

Inputs: PC1 (and optionally financial variables)
Target: NBER recession indicator (0/1)

As emphasized by the professor: do NOT overclaim. Note in-sample limitations
and rare positive class issues explicitly.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import List, Tuple
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_logit_data(
    pc1: pd.Series,
    nber: pd.Series,
    financial_panel: pd.DataFrame = None,
    lag_pc1: int = 0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for the logistic regression.

    Parameters
    ----------
    pc1             : PC1 macro signal
    nber            : NBER recession indicator (0/1)
    financial_panel : optional DataFrame of financial variables to add as regressors
    lag_pc1         : lag PC1 by this many months (for out-of-sample feel)

    Returns
    -------
    X : feature DataFrame
    y : binary target Series
    """
    pc1_lagged = pc1.shift(lag_pc1).rename(f"pc1_lag{lag_pc1}" if lag_pc1 else "pc1")
    parts = [pc1_lagged]

    if financial_panel is not None:
        parts.append(financial_panel)

    X = pd.concat(parts, axis=1)
    y = nber.rename("recession")

    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined.drop(columns=["recession"])
    y_clean = combined["recession"].astype(int)

    pos_rate = y_clean.mean() * 100
    print(f"  Logit data: {len(combined)} obs, "
          f"recession rate = {pos_rate:.1f}%")

    return X_clean, y_clean


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def fit_logit(
    X: pd.DataFrame,
    y: pd.Series,
    add_const: bool = True,
) -> sm.discrete.discrete_model.LogitResults:
    """
    Fit a logistic regression (Logit) via statsmodels.

    Parameters
    ----------
    X         : feature DataFrame
    y         : binary target (0/1)
    add_const : whether to add an intercept

    Returns
    -------
    statsmodels Logit results object
    """
    X_fit = sm.add_constant(X) if add_const else X
    model  = sm.Logit(y, X_fit)
    result = model.fit(method="bfgs", disp=False)
    print(result.summary2())
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    result: sm.discrete.discrete_model.LogitResults,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    add_const: bool = True,
) -> pd.DataFrame:
    """
    Evaluate logistic regression: AUC, confusion matrix, in-sample fit.

    Parameters
    ----------
    result    : fitted Logit result
    X         : feature DataFrame (same as used in fit)
    y         : true binary labels
    threshold : classification threshold for confusion matrix
    add_const : must match what was used in fit_logit()

    Returns
    -------
    Summary DataFrame with key metrics.
    """
    X_pred = sm.add_constant(X) if add_const else X
    proba  = result.predict(X_pred)
    pred   = (proba >= threshold).astype(int)

    auc  = roc_auc_score(y, proba)
    cm   = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "AUC (in-sample)":        round(auc, 4),
        "McFadden Pseudo-R²":     round(result.prsquared, 4),
        "True Positives":         int(tp),
        "False Positives":        int(fp),
        "False Negatives":        int(fn),
        "True Negatives":         int(tn),
        "Recall (sensitivity)":   round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else float("nan"),
        "Precision":              round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else float("nan"),
        "N observations":         int(result.nobs),
    }

    print("\n[NOTE] These are IN-SAMPLE metrics only. "
          "Do not overinterpret due to rare recession episodes and look-ahead bias.")

    return pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])


def train_test_split_time(
    X: pd.DataFrame,
    y: pd.Series,
    split_date: str = "2015-01-01",
) -> Tuple:
    """
    Time-series aware train/test split at a given date.

    Returns (X_train, X_test, y_train, y_test)
    """
    split = pd.Timestamp(split_date)
    X_train = X[X.index <  split]
    X_test  = X[X.index >= split]
    y_train = y[y.index <  split]
    y_test  = y[y.index >= split]

    print(f"  Train: {len(X_train)} obs "
          f"({X_train.index[0].date()} -> {X_train.index[-1].date()})")
    print(f"  Test:  {len(X_test)} obs "
          f"({X_test.index[0].date()} -> {X_test.index[-1].date()})")
    return X_train, X_test, y_train, y_test
