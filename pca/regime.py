"""
pca/regime.py
Converts the PC1 macro activity signal into a binary regime classification
(slowdown vs normal) and provides smoothing utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_pc1(pc1: pd.Series, window: int = None) -> pd.Series:
    """
    Apply a centered rolling average to reduce noise in PC1.
    Uses min_periods=1 so no additional NaNs are introduced at the edges.

    Parameters
    ----------
    pc1    : PC1 time series
    window : rolling window size in months (defaults to config.SMOOTHING_WINDOW)

    Returns
    -------
    Smoothed PC1 series with same index, named 'pc1_smooth'
    """
    if window is None:
        window = config.SMOOTHING_WINDOW
    smoothed = pc1.rolling(window=window, center=False, min_periods=1).mean()
    smoothed.name = "pc1_smooth"
    return smoothed


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(
    pc1: pd.Series,
    threshold: float = None,
    use_smoothed: bool = True,
    smooth_window: int = None,
) -> pd.Series:
    """
    Classify each month as 'slowdown' (1) or 'normal' (0) based on
    whether PC1 is below a given threshold.

    Convention:
        slowdown = 1  (PC1 < threshold  ->  weak macro conditions)
        normal   = 0  (PC1 >= threshold ->  normal/expanding conditions)

    Parameters
    ----------
    pc1           : PC1 series from build_indicator
    threshold     : PC1 level below which a slowdown is flagged
                    (defaults to config.REGIME_THRESHOLD = 0.0)
    use_smoothed  : if True, threshold the smoothed PC1 (recommended)
    smooth_window : rolling window for smoothing (defaults to config value)

    Returns
    -------
    pd.Series of 0/1 regime flags with same DatetimeIndex, named 'slowdown_regime'
    """
    if threshold is None:
        threshold = config.REGIME_THRESHOLD

    signal = smooth_pc1(pc1, smooth_window) if use_smoothed else pc1
    regime = (signal < threshold).astype(int)
    regime.name = "slowdown_regime"

    n_slow = regime.sum()
    pct_slow = n_slow / len(regime) * 100
    print(f"  Regime threshold: {threshold:.2f}")
    print(f"  Slowdown months:  {n_slow} / {len(regime)} ({pct_slow:.1f}%)")

    return regime


# ---------------------------------------------------------------------------
# Regime period extractor (for shading charts)
# ---------------------------------------------------------------------------

def regime_periods(regime_series: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Convert a binary (0/1) regime series into a list of (start, end) tuples
    representing contiguous slowdown episodes. Used for chart shading.

    Parameters
    ----------
    regime_series : 0/1 series with DatetimeIndex (slowdown=1)

    Returns
    -------
    List of (start_date, end_date) Timestamp tuples.
    """
    periods = []
    in_regime = False
    start = None

    for date, val in regime_series.items():
        if val == 1 and not in_regime:
            in_regime = True
            start = date
        elif val == 0 and in_regime:
            in_regime = False
            periods.append((start, date))

    # Close any open period at end of series
    if in_regime and start is not None:
        periods.append((start, regime_series.index[-1]))

    return periods


def nber_recession_periods(nber_series: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Same as regime_periods but for the NBER recession indicator (USREC).
    Convenience wrapper for chart shading.
    """
    return regime_periods(nber_series.fillna(0).astype(int))


# ---------------------------------------------------------------------------
# Performance check: overlap with NBER recessions
# ---------------------------------------------------------------------------

def compare_with_nber(
    regime: pd.Series,
    nber: pd.Series,
) -> pd.DataFrame:
    """
    Compute overlap statistics between the PCA-based slowdown regime
    and NBER official recession dates.

    Returns a summary DataFrame with:
      - true positives (both NBER & regime == 1)
      - false positives (regime=1 but NBER=0)
      - false negatives (NBER=1 but regime=0)
      - coverage (% of NBER months captured)
    """
    # Align
    aligned = pd.DataFrame({"regime": regime, "nber": nber}).dropna()
    aligned["nber"] = aligned["nber"].astype(int)
    aligned["regime"] = aligned["regime"].astype(int)

    tp = int(((aligned["regime"] == 1) & (aligned["nber"] == 1)).sum())
    fp = int(((aligned["regime"] == 1) & (aligned["nber"] == 0)).sum())
    fn = int(((aligned["regime"] == 0) & (aligned["nber"] == 1)).sum())
    tn = int(((aligned["regime"] == 0) & (aligned["nber"] == 0)).sum())

    nber_total = int(aligned["nber"].sum())
    coverage = tp / nber_total * 100 if nber_total > 0 else float("nan")
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else float("nan")

    summary = pd.DataFrame({
        "metric":  ["NBER months total", "True Positives (TP)",
                    "False Positives (FP)", "False Negatives (FN)",
                    "True Negatives (TN)",
                    "Recall / Coverage (%)", "Precision (%)"],
        "value":   [nber_total, tp, fp, fn, tn,
                    round(coverage, 1), round(precision, 1)],
    }).set_index("metric")

    return summary


if __name__ == "__main__":
    np.random.seed(0)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    pc1 = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 240)) + np.random.randn(240) * 0.3,
                    index=dates, name="pc1")

    smooth = smooth_pc1(pc1)
    regime = classify_regime(pc1, threshold=0.0)
    periods = regime_periods(regime)
    print(f"Slowdown episodes: {len(periods)}")
    print(periods[:3])
