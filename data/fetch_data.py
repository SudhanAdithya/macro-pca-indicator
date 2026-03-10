"""
data/fetch_data.py
Fetches raw macro, financial, and reference data from FRED using fredapi.
Saves each series as a CSV to data/raw/.
"""

import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import yfinance as yf

import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


def get_fred_client() -> Fred:
    """Load FRED API key from .env and return a Fred client."""
    # Find .env in project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root_dir, ".env")
    load_dotenv(env_path)
    
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY not set. Copy .env.example -> .env and add your key.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=api_key)


def fetch_fred_series(
    fred: Fred,
    series_id: str,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.Series:
    """
    Pull one FRED series and return as a pd.Series with DatetimeIndex.

    Parameters
    ----------
    fred      : authenticated Fred client
    series_id : FRED series identifier string (e.g. 'INDPRO')
    start     : start date string 'YYYY-MM-DD'
    end       : end date string 'YYYY-MM-DD' or None for latest

    Returns
    -------
    pd.Series with datetime index
    """
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.name = series_id
    s.index = pd.to_datetime(s.index)
    return s


def fetch_all_macro(fred: Fred, save: bool = True) -> pd.DataFrame:
    """
    Fetch all macro fundamental series defined in config.MACRO_SERIES.
    Falls back to PMI_FALLBACK_SERIES if a series fetch fails.

    Returns a DataFrame (date index, one column per series).
    Saves individual CSVs to data/raw/.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    frames = {}

    for col_name, series_id in config.MACRO_SERIES.items():
        try:
            s = fetch_fred_series(fred, series_id)
            print(f"  [OK]  {col_name:30s} ({series_id}) — {len(s)} obs, "
                  f"{s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:
            # Try fallback for PMI proxy
            fallback = config.PMI_FALLBACK_SERIES.get(col_name)
            if fallback:
                print(f"  [WARN] {series_id} unavailable ({e}). "
                      f"Using fallback: {fallback}")
                s = fetch_fred_series(fred, fallback)
            else:
                print(f"  [FAIL] {col_name} ({series_id}): {e}")
                continue

        s.name = col_name
        frames[col_name] = s

        if save:
            path = os.path.join(config.RAW_DATA_DIR, f"{col_name}.csv")
            s.to_csv(path, header=True)

    return pd.DataFrame(frames)


def fetch_all_financial(fred: Fred, save: bool = True) -> pd.DataFrame:
    """
    Fetch all financial series defined in config.FINANCIAL_SERIES.
    These are daily and will be converted to monthly in align.py.

    Returns a DataFrame (date index, one column per series).
    Saves individual CSVs to data/raw/.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    frames = {}

    for col_name, series_id in config.FINANCIAL_SERIES.items():
        try:
            if col_name == "sp500":
                print(f"  [WAIT] {col_name} (Hybrid: Yahoo 2000-15, FRED 2016+)...")
                # 1. Fetch from FRED (usually 2016+)
                s_fred = fetch_fred_series(fred, series_id)
                
                # 2. Fetch from Yahoo (full history)
                # Ticker for S&P 500 is ^GSPC
                yf_ticker = yf.Ticker("^GSPC")
                df_yf = yf_ticker.history(start=config.START_DATE, end="2016-01-01")
                s_yahoo = df_yf["Close"]
                s_yahoo.index = pd.to_datetime(s_yahoo.index.date)
                s_yahoo.name = col_name

                # 3. Splice: Yahoo (< 2016) + FRED (>= 2016)
                s_fred_subset = s_fred[s_fred.index >= "2016-01-01"]
                s_yahoo_subset = s_yahoo[s_yahoo.index < "2016-01-01"]
                
                s = pd.concat([s_yahoo_subset, s_fred_subset]).sort_index()
            else:
                s = fetch_fred_series(fred, series_id)
            
            s.name = col_name
            print(f"  [OK]  {col_name:30s} ({series_id}) — {len(s)} obs, "
                  f"{s.index[0].date()} -> {s.index[-1].date()}")
            frames[col_name] = s

            if save:
                path = os.path.join(config.RAW_DATA_DIR, f"{col_name}.csv")
                s.to_csv(path, header=True)

        except Exception as e:
            print(f"  [FAIL] {col_name} ({series_id}): {e}")

    return pd.DataFrame(frames)


def fetch_reference_series(fred: Fred, save: bool = True) -> pd.DataFrame:
    """
    Fetch NBER recession indicator, CFNAI, and 3-Month T-Bill rate (TB3MS)
    for benchmarking.  TB3MS is used as the risk-free rate in Sharpe and
    Jensen's alpha calculations.

    Returns a DataFrame (date index, one column per series).
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    frames = {}

    for col_name, series_id in config.REFERENCE_SERIES.items():
        try:
            s = fetch_fred_series(fred, series_id)
            s.name = col_name
            print(f"  [OK]  {col_name:30s} ({series_id}) — {len(s)} obs")
            frames[col_name] = s

            if save:
                path = os.path.join(config.RAW_DATA_DIR, f"{col_name}.csv")
                s.to_csv(path, header=True)
        except Exception as e:
            print(f"  [FAIL] {col_name} ({series_id}): {e}")

    return pd.DataFrame(frames)


def load_rf_rate(align_index: pd.DatetimeIndex = None) -> pd.Series:
    """
    Load the 3-Month T-Bill rate (TB3MS) from data/raw/ and convert from
    annualised % to a monthly decimal for use in Sharpe / alpha calculations.

    Formula: monthly_rf = annual_pct / 100 / 12

    Parameters
    ----------
    align_index : optional DatetimeIndex; if provided the series is reindexed
                  and forward-filled to match the portfolio return dates.

    Returns
    -------
    pd.Series of monthly risk-free rates as decimals.
    """
    try:
        rf = load_raw_series("tbill_3m")
        rf_monthly = rf / 100 / 12          # annualised % → monthly decimal
        rf_monthly.name = "rf_monthly"
        if align_index is not None:
            rf_monthly = rf_monthly.reindex(align_index, method="ffill").fillna(0)
        return rf_monthly
    except FileNotFoundError:
        print("  [WARN] tbill_3m.csv not found in data/raw/ — "
              "run with --fetch to download TB3MS. Rf defaulting to 0.")
        if align_index is not None:
            return pd.Series(0.0, index=align_index, name="rf_monthly")
        return pd.Series(dtype=float, name="rf_monthly")


def fetch_portfolio_assets(save: bool = True) -> pd.DataFrame:
    """
    Fetch historical daily data for portfolio assets (SPY, AGG) using yfinance.
    Saves results to data/raw/.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    tickers = list(config.PORTFOLIO_ASSETS.values())
    print(f"  [WAIT] Fetching portfolio assets: {tickers}...")
    
    # Fetch data
    df = yf.download(tickers, start=config.START_DATE)
    
    if df.empty:
        print("  [FAIL] No data fetched for portfolio assets.")
        return pd.DataFrame()

    # We want 'Adj Close' or 'Close' for total returns
    if "Adj Close" in df.columns.levels[0]:
        adj_close = df["Adj Close"]
    elif "Close" in df.columns.levels[0]:
        adj_close = df["Close"]
    else:
        # Fallback for older pandas/yfinance or single level
        cols = [c for c in df.columns if "Adj Close" in c or "Close" in c]
        if not cols:
            print(f"  [FAIL] Could not find Close/Adj Close in columns: {df.columns}")
            return pd.DataFrame()
        adj_close = df[cols]
    
    # Rename columns to our config keys
    rename_map = {v: k for k, v in config.PORTFOLIO_ASSETS.items()}
    adj_close = adj_close.rename(columns=rename_map)
    
    if save:
        for col in adj_close.columns:
            path = os.path.join(config.RAW_DATA_DIR, f"asset_{col}.csv")
            adj_close[col].to_csv(path, header=True)
            print(f"  [OK]  portfolio_asset:{col} saved to {path}")

    return adj_close


def load_raw_series(col_name: str) -> pd.Series:
    """Load a previously saved raw CSV from data/raw/."""
    path = os.path.join(config.RAW_DATA_DIR, f"{col_name}.csv")
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.name = col_name
    return s


if __name__ == "__main__":
    print("Fetching FRED data...")
    fred = get_fred_client()

    print("\n--- Macro fundamentals ---")
    macro_raw = fetch_all_macro(fred)

    print("\n--- Financial variables ---")
    financial_raw = fetch_all_financial(fred)

    print("\n--- Reference series ---")
    reference_raw = fetch_reference_series(fred)

    print("\nDone. Raw files saved to data/raw/")