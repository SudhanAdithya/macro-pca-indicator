# PCA-Based Macro Slowdown Indicator

**Team:** Siddhanth Yadav · Kavin Dhanasekar · Sudhan Adithya

A monthly macro activity / slowdown indicator built with Principal Component Analysis on FRED economic data, with lead-lag financial market linkage analysis.

---

## Project Structure

```
macro-pca-indicator/
├── config.py                     # All FRED IDs, transform rules, sign map, parameters
├── main.py                       # End-to-end pipeline runner
├── requirements.txt
├── .env.example                  # Copy to .env and add FRED_API_KEY
│
├── data/
│   ├── fetch_data.py             # FRED API fetcher
│   ├── align.py                  # Daily → monthly conversion, panel merge
│   ├── transform.py              # pct_change, log_return, level transforms
│   ├── standardize.py            # Z-score + sign alignment
│   ├── raw/                      # Downloaded raw CSVs (git-ignored)
│   └── processed/                # Cleaned panel CSVs (git-ignored)
│
├── pca/
│   ├── build_indicator.py        # PCA, PC1 extraction, loadings, variance
│   └── regime.py                 # Slowdown regime classification & NBER comparison
│
├── analysis/
│   ├── financial_linkage.py      # Contemporaneous correlations & regressions
│   ├── lead_lag.py               # Lead-lag predictive regressions
│   └── recession_model.py        # [Optional] Logistic recession probability model
│
├── viz/
│   ├── charts.py                 # Reusable chart functions
│   └── report_plots.py           # Generates & saves all report charts
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pca_signal.ipynb
│   ├── 03_financial_linkage.ipynb
│   └── 04_regime_analysis.ipynb
│
└── outputs/
    ├── pc1_series.csv
    ├── loadings.csv
    ├── regime_series.csv
    ├── tables/                   # Regression & stats tables
    └── charts/                   # PNG charts (01–06)
```

---

## Setup

### 1. Install dependencies

```bash
cd macro-pca-indicator
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a FRED API key

Register at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — it's free.

```bash
cp .env.example .env
# Edit .env and set: FRED_API_KEY=your_actual_key_here
```

---

## Running the Pipeline

### Full pipeline (fetches data + runs all stages)

```bash
python main.py
```

### Re-run without re-fetching FRED data

```bash
python main.py --no-fetch
```

### Force re-download from FRED

```bash
python main.py --fetch
```

---

## Methodology Summary

| Stage | Description |
|-------|-------------|
| 1 | Fetch raw monthly macro + daily financial series from FRED |
| 2 | Align to monthly frequency (month-end for S&P 500, avg for others) |
| 3 | Transform (MoM % change for production/sales, log return for equities, levels for PMI/VIX) |
| 4 | Z-score standardize + sign-align (higher = stronger macro) |
| 5 | Run PCA on macro fundamentals only; extract PC1 as macro signal |
| 6 | Classify slowdown regime (PC1 < 0 on 3-month smooth) |
| 7 | Contemporaneous & lead-lag regressions vs. financial variables |

### Macro PCA Variables

| Variable | FRED ID | Transform | Sign |
|----------|---------|-----------|------|
| Industrial Production | INDPRO | MoM % | + |
| Retail Sales | RSAFS | MoM % | + |
| Unemployment Rate | UNRATE | Level | − (flipped) |
| Housing Starts | HOUST | MoM % | + |
| Capacity Utilization | TCU | Level | + |
| Real Personal Income ex. Transfers | W875RX1 | MoM % | + |
| Mfg New Orders | AMTMNO | MoM % | + |
| PMI Proxy | NAPM / MANEMP | Level | + |

### Financial Variables (Analysis Only)

| Variable | FRED ID | Convention |
|----------|---------|------------|
| Yield Curve (10Y−2Y) | T10Y2Y | Monthly avg |
| HY Credit Spread | BAMLH0A0HYM2 | Monthly avg |
| S&P 500 | SP500 | Log return (month-end) |
| VIX | VIXCLS | Monthly avg |

---

## Key Output Files

| File | Description |
|------|-------------|
| `outputs/pc1_series.csv` | Monthly PC1 macro signal |
| `outputs/loadings.csv` | PC1 loadings per variable |
| `outputs/regime_series.csv` | Slowdown regime (0=normal, 1=slowdown) |
| `outputs/tables/variance_explained.csv` | Scree data |
| `outputs/tables/contemporaneous_correlations.csv` | Financial vs. PC1 correlations |
| `outputs/tables/lead_lag_bivariate.csv` | Lead-lag regression results |
| `outputs/charts/01_pc1_signal.png` | PC1 time series with recession shading |
| `outputs/charts/02_pc1_loadings.png` | Loading bar chart |
| `outputs/charts/03_scree_plot.png` | Variance explained scree |
| `outputs/charts/04_regime_chart.png` | Slowdown regime vs. NBER |
| `outputs/charts/05_correlation_heatmap.png` | Financial correlations |
| `outputs/charts/06_lead_lag_betas.png` | Lead-lag β coefficients |

---

## Notebooks

Run Jupyter:
```bash
jupyter notebook notebooks/
```

| Notebook | Contents |
|----------|----------|
| `01_data_exploration.ipynb` | Raw series plots, missing value audit, correlations |
| `02_pca_signal.ipynb` | PCA walkthrough, loadings, PC1 chart |
| `03_financial_linkage.ipynb` | Contemporaneous + lead-lag analysis |
| `04_regime_analysis.ipynb` | Regime classification vs. NBER |
