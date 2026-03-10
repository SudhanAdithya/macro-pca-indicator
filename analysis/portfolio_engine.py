"""
analysis/portfolio_engine.py
Engine for backtesting portfolio strategies based on macro regimes.

Key design decisions (addressing professor's feedback):
  1. SIGNAL LAG: All weights are shifted by SIGNAL_LAG_MONTHS (default 1)
     before being applied to returns. Signal observed at end of month t
     drives allocation at end of month t+1 — no same-day execution bias.

  2. BENCHMARK REBALANCING: The 60/40 benchmark is rebalanced monthly back
     to target weights. Turnover at each rebalance is charged at
     TRANSACTION_COST_BPS basis points per unit of one-way turnover.
     The dynamic strategies are charged the same way.

  3. SHARPE RATIO: Computed as (Rp - Rf) / σp × √12, where Rf is the
     3-month T-Bill rate (TB3MS from FRED), converted to monthly decimal.
     Using Rf=0 overstates Sharpe, especially in high-rate periods.

  4. JENSEN'S ALPHA: Computed by regressing (strategy excess return) on
     (benchmark excess return). Reports annualized alpha, beta, t-stat,
     p-value, and information ratio — as shown in class.

  5. PNL BETWEEN REBALANCES: Returns are computed on the actual (drifted)
     weights between monthly rebalances, not on the target weights. This
     correctly reflects the portfolio's actual composition.
"""

import pandas as pd
import numpy as np
import os
import sys
import statsmodels.api as sm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_monthly(daily_series: pd.Series) -> pd.Series:
    """Resample a daily series to month-start using last value of month."""
    return daily_series.resample("MS").last()


def _annualized_return(monthly_rets: pd.Series) -> float:
    """Geometric annualized return from a monthly return series."""
    n_months = len(monthly_rets.dropna())
    total = (1 + monthly_rets.dropna()).prod()
    return total ** (12 / n_months) - 1


def _annualized_vol(monthly_rets: pd.Series) -> float:
    return monthly_rets.dropna().std(ddof=1) * np.sqrt(12)


def _max_drawdown(monthly_rets: pd.Series) -> float:
    cum = (1 + monthly_rets.dropna()).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def _sharpe(monthly_rets: pd.Series, rf_monthly: pd.Series) -> float:
    """
    Annualized Sharpe ratio: (mean excess return / std excess return) × √12.
    rf_monthly must be aligned to monthly_rets.
    """
    aligned = pd.concat([monthly_rets, rf_monthly], axis=1).dropna()
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if excess.std(ddof=1) == 0:
        return 0.0
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(12)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PortfolioEngine:
    def __init__(
        self,
        asset_prices: pd.DataFrame,
        pc1_smooth: pd.Series,
        regimes: pd.Series = None,
        rf_series: pd.Series = None,
    ):
        """
        Parameters
        ----------
        asset_prices : daily prices for 'stocks' and 'bonds'
        pc1_smooth   : monthly smoothed PC1 macro signal
        regimes      : monthly binary regimes (0/1)
        rf_series    : monthly risk-free rate as decimal (e.g. 0.0035 for 3-month T-Bill).
                       If None, defaults to zero (Sharpe will be overstated — warn user).
        """
        self.pc1_smooth  = pc1_smooth
        self.regimes     = regimes

        # Convert daily prices to monthly (month-end close → month-start index)
        monthly_prices = asset_prices.resample("MS").last()
        self.monthly_returns = monthly_prices.pct_change().dropna()

        if rf_series is not None:
            self.rf_monthly = rf_series.reindex(self.monthly_returns.index).ffill()
        else:
            print("  [WARN] No risk-free rate supplied — Sharpe ratios will use Rf=0 (overstated).")
            self.rf_monthly = pd.Series(0.0, index=self.monthly_returns.index)

    # ------------------------------------------------------------------
    # Logistic risk score
    # ------------------------------------------------------------------

    def calculate_risk_score(self, s_t: pd.Series) -> pd.Series:
        """
        R_t = 1 / (1 + exp((S_t - c) / k))
        High PC1 (expansion) → R_t → 0 → max equity
        Low  PC1 (recession) → R_t → 1 → min equity
        """
        c = config.LOGISTIC_PARAMS["c"]
        k = config.LOGISTIC_PARAMS["k"]
        return 1 / (1 + np.exp((s_t - c) / k))

    # ------------------------------------------------------------------
    # Weight computation with signal lag
    # ------------------------------------------------------------------

    def _compute_target_weights(self, mode: str) -> pd.DataFrame:
        """
        Returns a monthly DataFrame with columns ['stocks', 'bonds'] of
        TARGET weights, shifted by SIGNAL_LAG_MONTHS to avoid look-ahead.

        Signal observed at end of month t → trade executes at end of month t+1.
        """
        lag = getattr(config, "SIGNAL_LAG_MONTHS", 1)

        if mode == "benchmark":
            bw = config.BENCHMARK_WEIGHTS
            raw_stocks = pd.Series(bw["stocks"], index=self.monthly_returns.index)

        elif mode == "binary":
            if self.regimes is None:
                raise ValueError("Binary mode requires regimes series.")
            monthly_regimes = self.regimes.reindex(
                self.monthly_returns.index, method="ffill"
            ).fillna(0)
            raw_stocks = monthly_regimes.map(
                lambda r: config.REGIME_WEIGHT_MAP[int(r)]["stocks"]
            )

        elif mode == "continuous":
            pc1_aligned = self.pc1_smooth.reindex(
                self.monthly_returns.index, method="ffill"
            ).fillna(0)
            risk_scores = self.calculate_risk_score(pc1_aligned)
            w_min = config.CONTINUOUS_WEIGHT_LIMITS["min_equity"]
            w_max = config.CONTINUOUS_WEIGHT_LIMITS["max_equity"]
            raw_stocks = w_min + (w_max - w_min) * (1 - risk_scores)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply signal lag — weights at t are based on signal at t-lag
        lagged_stocks = raw_stocks.shift(lag)
        lagged_bonds  = 1.0 - lagged_stocks

        return pd.DataFrame({
            "stocks": lagged_stocks,
            "bonds":  lagged_bonds,
        })

    # ------------------------------------------------------------------
    # Single-strategy backtest with proper rebalancing & transaction costs
    # ------------------------------------------------------------------

    def _backtest_strategy(self, mode: str) -> pd.Series:
        """
        Simulate a monthly-rebalanced portfolio.

        At each month-end:
          1. Compute return on ACTUAL (drifted) weights from prior period.
          2. Rebalance to new target weights, charging transaction costs
             on one-way turnover.

        Returns a monthly net-return pd.Series.
        """
        cost = getattr(config, "TRANSACTION_COST_BPS", 5) / 10_000  # bps → decimal
        target_w = self._compute_target_weights(mode)

        ret_s = self.monthly_returns["stocks"]
        ret_b = self.monthly_returns["bonds"]

        # Align everything to the same index (skip NaN rows from lag)
        idx = target_w.dropna().index.intersection(ret_s.dropna().index)
        tgt = target_w.loc[idx]
        rs  = ret_s.loc[idx]
        rb  = ret_b.loc[idx]

        portfolio_rets = []
        # Start with target weights on first period (no prior drift)
        prev_w_s = float(tgt["stocks"].iloc[0])
        prev_w_b = float(tgt["bonds"].iloc[0])

        for i, date in enumerate(idx):
            # --- Step 1: P&L on actual (drifted) weights ---
            gross = prev_w_s * rs.loc[date] + prev_w_b * rb.loc[date]

            # --- Step 2: Rebalance to new target ---
            # After market moves, actual weights have drifted:
            total = prev_w_s * (1 + rs.loc[date]) + prev_w_b * (1 + rb.loc[date])
            drifted_s = prev_w_s * (1 + rs.loc[date]) / total
            drifted_b = 1.0 - drifted_s

            new_tgt_s = float(tgt["stocks"].loc[date])
            new_tgt_b = 1.0 - new_tgt_s

            # One-way turnover (each side charged separately)
            turnover = abs(new_tgt_s - drifted_s) + abs(new_tgt_b - drifted_b)
            txn_cost = turnover * cost

            net = gross - txn_cost
            portfolio_rets.append(net)

            # Set weights for next period
            prev_w_s = new_tgt_s
            prev_w_b = new_tgt_b

        return pd.Series(portfolio_rets, index=idx, name=mode)

    # ------------------------------------------------------------------
    # Run all strategies
    # ------------------------------------------------------------------

    def run_backtest(self) -> pd.DataFrame:
        """
        Run all three strategies and return a results DataFrame with:
          - monthly returns for each strategy
          - cumulative returns for each strategy
          - risk score and equity weight (continuous strategy)
        """
        strategies = ["benchmark", "continuous"]
        if self.regimes is not None:
            strategies.insert(1, "binary")

        results = {}
        for mode in strategies:
            print(f"  Backtesting '{mode}'...")
            results[mode] = self._backtest_strategy(mode)

        df = pd.DataFrame(results).dropna()

        # Cumulative returns
        for col in results:
            df[f"cum_{col}"] = (1 + df[col]).cumprod()

        # Attach risk score and weight for chart use
        pc1_aligned = self.pc1_smooth.reindex(df.index, method="ffill").fillna(0)
        df["risk_score"]    = self.calculate_risk_score(pc1_aligned)
        df["weight_stocks"] = (config.CONTINUOUS_WEIGHT_LIMITS["min_equity"] +
                               (config.CONTINUOUS_WEIGHT_LIMITS["max_equity"] -
                                config.CONTINUOUS_WEIGHT_LIMITS["min_equity"]) *
                               (1 - df["risk_score"]))
        return df

    # ------------------------------------------------------------------
    # Performance metrics (with proper Sharpe and Jensen's alpha)
    # ------------------------------------------------------------------

    def calculate_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute annualized return, volatility, Sharpe (with Rf), max drawdown,
        and Jensen's alpha vs. benchmark for each strategy.
        """
        strategies = [c for c in ["benchmark", "binary", "continuous"]
                      if c in results.columns]
        rf = self.rf_monthly.reindex(results.index).fillna(0)

        rows = {}
        for col in strategies:
            rets = results[col].dropna()
            ann_ret = _annualized_return(rets)
            ann_vol = _annualized_vol(rets)
            sr      = _sharpe(rets, rf)
            mdd     = _max_drawdown(rets)

            rows[col] = {
                "Annualized Return (%)": round(ann_ret * 100, 2),
                "Annualized Vol (%)":    round(ann_vol * 100, 2),
                "Sharpe Ratio":          round(sr, 3),
                "Max Drawdown (%)":      round(mdd * 100, 2),
                "Alpha vs 60/40 (%)":    None,   # filled below
                "Beta vs 60/40":         None,
                "Alpha t-stat":          None,
                "Alpha p-value":         None,
                "Information Ratio":     None,
            }

        # Jensen's alpha for non-benchmark strategies
        bench_rets = results["benchmark"].dropna()
        for col in strategies:
            if col == "benchmark":
                continue
            alpha_stats = self.compute_alpha(
                strategy_returns=results[col].dropna(),
                benchmark_returns=bench_rets,
                rf_monthly=rf,
            )
            rows[col]["Alpha vs 60/40 (%)"]  = round(alpha_stats["alpha_annualized"] * 100, 3)
            rows[col]["Beta vs 60/40"]        = round(alpha_stats["beta"], 4)
            rows[col]["Alpha t-stat"]         = round(alpha_stats["t_stat_alpha"], 3)
            rows[col]["Alpha p-value"]        = round(alpha_stats["p_value_alpha"], 4)
            rows[col]["Information Ratio"]    = round(alpha_stats["information_ratio"], 3)

        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------
    # Jensen's alpha
    # ------------------------------------------------------------------

    def compute_alpha(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        rf_monthly: pd.Series,
    ) -> dict:
        """
        Jensen's Alpha via OLS regression:
            (Rp - Rf) = α + β(Rb - Rf) + ε

        Parameters
        ----------
        strategy_returns  : monthly returns of the dynamic strategy
        benchmark_returns : monthly returns of the 60/40 benchmark
        rf_monthly        : monthly risk-free rate (decimal)

        Returns
        -------
        dict with: alpha_monthly, alpha_annualized, beta,
                   t_stat_alpha, p_value_alpha, r_squared, information_ratio
        """
        combined = pd.concat(
            [strategy_returns.rename("strat"),
             benchmark_returns.rename("bench"),
             rf_monthly.rename("rf")],
            axis=1
        ).dropna()

        excess_strat = combined["strat"]  - combined["rf"]
        excess_bench = combined["bench"] - combined["rf"]

        X = sm.add_constant(excess_bench.rename("excess_bench"))
        model = sm.OLS(excess_strat, X).fit(cov_type="HC3")

        alpha_m = model.params["const"]
        resid_vol = model.resid.std(ddof=1)
        ir = (alpha_m / resid_vol * np.sqrt(12)) if resid_vol > 0 else 0.0

        return {
            "alpha_monthly":     alpha_m,
            "alpha_annualized":  alpha_m * 12,
            "beta":              model.params["excess_bench"],
            "t_stat_alpha":      model.tvalues["const"],
            "p_value_alpha":     model.pvalues["const"],
            "r_squared":         model.rsquared,
            "information_ratio": ir,
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_portfolio_results(results: pd.DataFrame, metrics: pd.DataFrame):
    os.makedirs(os.path.dirname(config.PORTFOLIO_SERIES_PATH),  exist_ok=True)
    os.makedirs(os.path.dirname(config.PORTFOLIO_METRICS_PATH), exist_ok=True)
    results.to_csv(config.PORTFOLIO_SERIES_PATH)
    metrics.to_csv(config.PORTFOLIO_METRICS_PATH)
    print(f"  [OK]  Portfolio results saved to {config.PORTFOLIO_SERIES_PATH}")
    print(f"  [OK]  Portfolio metrics saved to {config.PORTFOLIO_METRICS_PATH}")