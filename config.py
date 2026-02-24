"""
config.py
Central configuration for the PCA Macro Slowdown Indicator project.
All FRED series IDs, transformation rules, sign conventions, and
pipeline parameters live here so nothing is hardcoded in modules.
"""

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = None  # None = pull through latest available date

# ---------------------------------------------------------------------------
# FRED series IDs — Macro fundamentals (PCA inputs)
# ---------------------------------------------------------------------------
MACRO_SERIES = {
    # Series name (key used as column name) : FRED series ID
    "industrial_production":    "INDPRO",       # Industrial Production Index
    "retail_sales":             "RSAFS",        # Advance Retail Sales (monthly)
    "unemployment_rate":        "UNRATE",       # Civilian Unemployment Rate
    "housing_starts":           "HOUST",        # Housing Starts: Total
    "capacity_utilization":     "TCU",          # Total Capacity Utilization
    "real_personal_income":     "W875RX1",      # Real Personal Income ex. Transfers
    "mfg_new_orders":           "AMTMNO",       # Manufacturers' New Orders: Total
    "pmi_proxy":                "NAPM",         # ISM Manufacturing PMI (legacy) — falls back to MANEMP if unavailable
}

# Fallback if NAPM is unavailable on your FRED account (NAPM is sometimes restricted)
PMI_FALLBACK_SERIES = {
    "pmi_proxy": "MANEMP"   # Manufacturing Employees as a cyclical proxy
}

# ---------------------------------------------------------------------------
# FRED series IDs — Financial variables (NOT in core PCA)
# ---------------------------------------------------------------------------
FINANCIAL_SERIES = {
    "yc_10y2y":    "T10Y2Y",    # 10-Year minus 2-Year Treasury Constant Maturity
    "hy_spread":   "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS
    "sp500":       "SP500",     # S&P 500 Index Level (daily → month-end)
    "vix":         "VIXCLS",    # CBOE VIX (daily → monthly avg)
}

# ---------------------------------------------------------------------------
# FRED series IDs — Reference / benchmark
# ---------------------------------------------------------------------------
REFERENCE_SERIES = {
    "nber_recession": "USREC",  # NBER-based Recession Indicator (0/1)
    "cfnai":          "CFNAI",  # Chicago Fed National Activity Index
}

# ---------------------------------------------------------------------------
# Transformation rules
# Key = column name, Value = one of: 'pct_mom', 'pct_yoy', 'log_return', 'level', 'change'
# ---------------------------------------------------------------------------
TRANSFORM_MAP = {
    # Macro
    "industrial_production":  "pct_mom",    # month-over-month % change
    "retail_sales":           "pct_mom",    # month-over-month % change
    "unemployment_rate":      "level",      # level (will be sign-flipped)
    "housing_starts":         "pct_mom",    # month-over-month % change
    "capacity_utilization":   "level",      # level index
    "real_personal_income":   "pct_mom",    # month-over-month % change
    "mfg_new_orders":         "pct_mom",    # month-over-month % change
    "pmi_proxy":              "level",      # PMI-style index — keep as level
    # Financial
    "sp500_return":           "log_return", # computed from sp500 level
    "yc_10y2y":               "level",
    "hy_spread":              "level",
    "vix":                    "level",
}

# ---------------------------------------------------------------------------
# Sign convention map
# +1  → variable is ALREADY positively oriented (high = good)
# -1  → variable needs to be flipped (high = bad)
# Applied AFTER standardization to align all variables so that:
#       High z-score = stronger macro conditions
# ---------------------------------------------------------------------------
SIGN_MAP = {
    # Macro fundamentals
    "industrial_production":  1,
    "retail_sales":           1,
    "unemployment_rate":      -1,   # higher unemployment = worse
    "housing_starts":         1,
    "capacity_utilization":   1,
    "real_personal_income":   1,
    "mfg_new_orders":         1,
    "pmi_proxy":              1,
    # Financial (used in analysis, NOT PCA)
    "sp500_return":           1,
    "yc_10y2y":               1,    # steeper curve = better outlook
    "hy_spread":             -1,    # higher spread = stress
    "vix":                   -1,    # higher fear = stress
}

# Which columns go into the core macro PCA
MACRO_PCA_COLS = [
    "industrial_production",
    "retail_sales",
    "unemployment_rate",
    "housing_starts",
    "capacity_utilization",
    "real_personal_income",
    "mfg_new_orders",
    "pmi_proxy",
]

# Which columns are financial variables (NOT in PCA, used in analysis)
FINANCIAL_COLS = [
    "sp500_return",
    "yc_10y2y",
    "hy_spread",
    "vix",
]

# ---------------------------------------------------------------------------
# Regime classification parameters
# ---------------------------------------------------------------------------
REGIME_THRESHOLD = 0.0          # PC1 < threshold → slowdown regime
SMOOTHING_WINDOW  = 3           # months for rolling average smoothing

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR        = "data/raw"
PROCESSED_DATA_DIR  = "data/processed"
OUTPUTS_DIR         = "outputs"
CHARTS_DIR          = "outputs/charts"
TABLES_DIR          = "outputs/tables"

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
RECESSION_SHADE_COLOR = "#d3d3d3"
SLOWDOWN_SHADE_COLOR  = "#ffe4b5"
FIG_DPI               = 150
FIG_SIZE_WIDE         = (14, 5)
FIG_SIZE_SQUARE       = (8, 6)
