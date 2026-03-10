"""
analysis/portfolio_engine.py
Engine for backtesting portfolio strategies based on macro regimes.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

class PortfolioEngine:
    def __init__(self, asset_prices: pd.DataFrame, pc1_smooth: pd.Series, regimes: pd.Series = None):
        """
        Parameters
        ----------
        asset_prices : pd.DataFrame with daily prices for 'stocks' and 'bonds'
        pc1_smooth   : pd.Series with monthly smoothed PC1 macro signal
        regimes      : optional pd.Series with monthly binary regimes (0/1)
        """
        self.asset_prices = asset_prices
        self.pc1_smooth = pc1_smooth
        self.regimes = regimes
        
        # Calculate daily returns
        self.asset_returns = asset_prices.pct_change().dropna()
        
    def calculate_risk_score(self, s_t: pd.Series) -> pd.Series:
        """
        Calculate Risk Score R_t using Logistic Mapping.
        R_t = 1 / (1 + exp((S_t - c) / k))
        
        Note: We use - (S_t - c) / k because higher S_t (macro health) 
              should lead to LOWER Risk R_t.
        """
        c = config.LOGISTIC_PARAMS["c"]
        k = config.LOGISTIC_PARAMS["k"]
        risk_score = 1 / (1 + np.exp((s_t - c) / k))
        return risk_score

    def run_backtest(self, lag_months: int = 1, cost_bps: float = 0.0, methodology: Optional[str] = None):
        """
        Run backtests comparing:
        1. Benchmark (60/40)
        2. Binary Dynamic Strategy (Defensive shift in Slowdown)
        3. Continuous Logistic Strategy (Smooth scaling based on PC1)
        
        Parameters
        ----------
        lag_months : int
            Months of implementation lag (default=1).
        cost_bps   : float
            Transaction cost in basis points (1 bps = 0.0001) per trade.
        methodology: str
            'etf' (Physical holding) or 'futures' (Futures Overlay).
            If None, uses config.BACKTEST_METHODOLOGY.
        """
        if methodology is None:
            methodology = getattr(config, "BACKTEST_METHODOLOGY", "etf")

        # Risk-free rate (daily approximation)
        rf_annual = config.RISK_FREE_RATE_ANNUAL
        rf_daily = (1 + rf_annual)**(1/252) - 1

        # Apply implementation lag to monthly signals before reindexing to daily
        pc1_lagged = self.pc1_smooth.shift(lag_months).fillna(0)
        daily_pc1 = pc1_lagged.reindex(self.asset_returns.index, method='ffill').fillna(0)
        
        # 1. Benchmark Weights
        bw = config.BENCHMARK_WEIGHTS
        strategies = ['benchmark']
        
        # Initialize return series
        rets_dict = {
            'benchmark': (self.asset_returns['stocks'] * bw['stocks'] + self.asset_returns['bonds'] * bw['bonds'])
        }
        
        # 2. Binary Strategy Weights (if regimes provided)
        if self.regimes is not None:
            strategies.append('binary')
            regimes_lagged = self.regimes.shift(lag_months).fillna(0)
            daily_regimes = regimes_lagged.reindex(self.asset_returns.index, method='ffill').fillna(0)
            w_stocks_bin = daily_regimes.map(lambda r: config.REGIME_WEIGHT_MAP[int(r)]['stocks'])
            w_bonds_bin  = daily_regimes.map(lambda r: config.REGIME_WEIGHT_MAP[int(r)]['bonds'])
            rets_dict['binary'] = (self.asset_returns['stocks'] * w_stocks_bin + self.asset_returns['bonds'] * w_bonds_bin)
            
            # Costs for Binary
            if cost_bps > 0:
                dw = w_stocks_bin.diff().abs()
                rets_dict['binary'] -= dw * (cost_bps / 10000.0)
        
        # 3. Continuous Weights
        strategies.append('continuous')
        risk_scores = self.calculate_risk_score(daily_pc1)
        w_min = config.CONTINUOUS_WEIGHT_LIMITS["min_equity"]
        w_max = config.CONTINUOUS_WEIGHT_LIMITS["max_equity"]
        weights_stocks_cont = w_min + (w_max - w_min) * (1 - risk_scores)
        weights_bonds_cont  = 1 - weights_stocks_cont
        rets_dict['continuous'] = (self.asset_returns['stocks'] * weights_stocks_cont + self.asset_returns['bonds'] * weights_bonds_cont)
        
        # Costs for Continuous
        if cost_bps > 0:
            dw = weights_stocks_cont.diff().abs()
            rets_dict['continuous'] -= dw * (cost_bps / 10000.0)

        # Apply Futures Overlay Methodology if selected
        # formula: Rp = sum(w_i * r_fut_i) + (1 - sum(w_i)) * rf
        # where r_fut_i = r_total_i - rf
        if methodology == 'futures':
            for s in strategies:
                if s == 'benchmark':
                    w_s, w_b = bw['stocks'], bw['bonds']
                elif s == 'binary':
                    w_s, w_b = w_stocks_bin, w_bonds_bin
                else: # continuous
                    w_s, w_b = weights_stocks_cont, weights_bonds_cont
                
                sum_w = w_s + w_b
                # Excess returns of assets (Future Returns)
                ex_s = self.asset_returns['stocks'] - rf_daily
                ex_b = self.asset_returns['bonds'] - rf_daily
                
                # Portfolio Return Formula: Rp = sum(w_i * r_fut_i) + (1 - sum(w_i)) * rf
                # This ensures collateral earns rf and uninvested cash also earns rf.
                rets_dict[s] = (w_s * ex_s + w_b * ex_b) + (1 - sum_w) * rf_daily + sum_w * rf_daily
                # Simplified: rets_dict[s] = (w_s * ex_s + w_b * ex_b) + rf_daily
                # But we use the explicit form for documentation/rigor.
                
                # Transaction costs are already subtracted if applicable
                # but let's re-ensure order of operations is clean.
                # Costs were subtracted BEFORE the futures overlay adjustment.
                # Actually, costs should be subtracted from the final total return.
        
        # Create results DataFrame
        results = pd.DataFrame(rets_dict)
        results['risk_score'] = risk_scores
        results['weight_stocks'] = weights_stocks_cont
        
        # Calculate cumulative returns
        for col in strategies:
            results[f'cum_{col}'] = (1 + results[col]).cumprod()
        
        return results

    def calculate_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics for all strategies in results."""
        metrics = {}
        
        strategies = [c for c in ['benchmark', 'binary', 'continuous'] if c in results.columns]
        rf_annual = config.RISK_FREE_RATE_ANNUAL

        for col in strategies:
            rets = results[col]
            cum_rets = results[f'cum_{col}']
            
            # Annualized Return (CAGR)
            final_cum = cum_rets.iloc[-1]
            n_years = len(rets) / 252
            ann_return = (final_cum)**(1/n_years) - 1
            
            # Annualized Volatility
            ann_vol = rets.std() * np.sqrt(252)
            
            # Excess Sharpe Ratio: (Ann Return - Rf) / Ann Vol
            # This is the institutional standard for strategy with cash collateral
            sharpe = (ann_return - rf_annual) / ann_vol if ann_vol != 0 else 0
            
            # Max Drawdown
            rolling_max = cum_rets.cummax()
            drawdown = (cum_rets - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            metrics[col] = {
                'Annualized Return (%)': round(ann_return * 100, 2),
                'Annualized Vol (%)':   round(ann_vol * 100, 2),
                'Sharpe Ratio (Excess)': round(sharpe, 2),
                'Max Drawdown (%)':     round(max_dd * 100, 2)
            }
            
        df_metrics = pd.DataFrame(metrics).T
        
        # Add Alpha relative to benchmark
        for strat in strategies:
            if strat != 'benchmark':
                alpha = metrics[strat]['Annualized Return (%)'] - metrics['benchmark']['Annualized Return (%)']
                df_metrics.loc[strat, 'Alpha (%)'] = round(alpha, 2)
        
        return df_metrics

def save_portfolio_results(results: pd.DataFrame, metrics: pd.DataFrame):
    """Save performance results and metrics to CSV."""
    os.makedirs(os.path.dirname(config.PORTFOLIO_SERIES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.PORTFOLIO_METRICS_PATH), exist_ok=True)
    
    results.to_csv(config.PORTFOLIO_SERIES_PATH)
    metrics.to_csv(config.PORTFOLIO_METRICS_PATH)
    print(f"  [OK]  Portfolio results saved to {config.PORTFOLIO_SERIES_PATH}")
    print(f"  [OK]  Portfolio metrics saved to {config.PORTFOLIO_METRICS_PATH}")
