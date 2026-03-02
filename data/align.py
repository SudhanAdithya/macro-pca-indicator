"""
data/align.py
Converts raw (possibly daily) series to monthly frequency and merges
all macro, financial, and reference data into one clean panel DataFrame.

Convention used throughout:
  - Macro series from FRED are already monthly → just resample to
    month-start (MS) period to ensure a consistent date index.
  - Financial series (S&P 500, VIX, yield curve, HY spread) are DAILY
    → converted using:
        sp500   : month-end close (last observation of month)
        vix     : monthly average
        yc_10y2y: monthly average
        hy_spread: monthly average
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# Daily → Monthly converters
# ---------------------------------------------------------------------------

def to_monthly_avg(series: pd.Series) -> pd.Series:
    """Resample a daily series to monthly average, period = month-start."""
    return (
        series
        .dropna()
        .resample("MS")
        .mean()
        .rename(series.name)
    )


def to_month_end_close(series: pd.Series) -> pd.Series:
    """
    Resample a daily price series to the last available value each month
    (month-end close convention). Result is period = month-start for
    alignment with macro series.
    """
    monthly = (
        series
        .dropna()
        .resample("MS")
        .last()
        .rename(series.name)
    )
    return monthly


def align_macro_to_monthly(macro_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a macro DataFrame (already monthly or close) has a clean
    month-start DatetimeIndex. FRED monthly series come with dates on
    the 1st of the month, so this is mostly a safety normalisation.
    """
    df = macro_raw.copy()
    df.index = pd.to_datetime(df.index)
    # Resample to MS (month-start) taking last value — handles any
    # irregular dates that may come from FRED
    df = df.resample("MS").last()
    return df


def align_financial_to_monthly(financial_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily financial panel to monthly using the agreed convention:
      - sp500      → month-end close (last of month)
      - vix        → monthly average
      - yc_10y2y   → monthly average
      - hy_spread  → monthly average
    """
    df = financial_raw.copy()
    df.index = pd.to_datetime(df.index)

    monthly_frames = {}

    for col in df.columns:
        s = df[col].dropna()
        if col == "sp500":
            monthly_frames[col] = to_month_end_close(s)
        else:
            # vix, yc_10y2y, hy_spread → average
            monthly_frames[col] = to_monthly_avg(s)

    return pd.DataFrame(monthly_frames)


# ---------------------------------------------------------------------------
# Panel merger
# ---------------------------------------------------------------------------

def merge_panel(
    macro_monthly: pd.DataFrame,
    financial_monthly: pd.DataFrame,
    reference_monthly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Outer-join macro, financial, and optional reference DataFrames on date.
    All inputs must already be at monthly (MS) frequency.

    Returns a combined panel with a DatetimeIndex and all columns.
    """
    # Start with macro
    panel = macro_monthly.copy()

    # Join financial
    panel = panel.join(financial_monthly, how="outer")

    # Join reference (NBER etc.) if provided
    if reference_monthly is not None:
        ref = reference_monthly.copy()
        ref.index = pd.to_datetime(ref.index)
        ref = ref.resample("MS").last()
        panel = panel.join(ref, how="left")

    panel.index.name = "date"
    panel.sort_index(inplace=True)

    return panel


# ---------------------------------------------------------------------------
# Convenience: restrict to clean overlapping period
# ---------------------------------------------------------------------------

def trim_to_overlap(panel: pd.DataFrame, min_coverage: float = 0.7) -> pd.DataFrame:
    """
    Trim the panel to rows where at least `min_coverage` fraction of
    columns are non-NaN. This removes the ragged edges at panel start/end.

    Parameters
    ----------
    panel        : merged monthly panel
    min_coverage : minimum fraction of non-NaN columns required to keep row

    Returns
    -------
    Trimmed DataFrame
    """
    threshold = int(min_coverage * len(panel.columns))
    trimmed = panel.dropna(thresh=threshold)
    print(f"  Panel rows before trim: {len(panel)}  →  after trim: {len(trimmed)}")
    
    if len(trimmed) > 0:
        print(f"  Date range: {trimmed.index[0].date()} → {trimmed.index[-1].date()}")
    else:
        print("  Warning: Trimmed panel is empty. No overlapping data found for the given coverage threshold.")
        
    return trimmed


if __name__ == "__main__":
    # Quick smoke-test: load from raw CSVs and align
    from data.fetch_data import load_raw_series

    print("Loading raw macro series...")
    macro_frames = {}
    for col in config.MACRO_SERIES:
        try:
            macro_frames[col] = load_raw_series(col)
        except FileNotFoundError:
            print(f"  Missing: {col}")
    macro_raw = pd.DataFrame(macro_frames)

    print("Loading raw financial series...")
    fin_frames = {}
    for col in config.FINANCIAL_SERIES:
        try:
            fin_frames[col] = load_raw_series(col)
        except FileNotFoundError:
            print(f"  Missing: {col}")
    fin_raw = pd.DataFrame(fin_frames)

    macro_m = align_macro_to_monthly(macro_raw)
    fin_m   = align_financial_to_monthly(fin_raw)

    panel = merge_panel(macro_m, fin_m)
    panel = trim_to_overlap(panel)
    print(panel.tail())