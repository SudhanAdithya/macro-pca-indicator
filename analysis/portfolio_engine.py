"""
analysis/portfolio_engine.py
Engine for backtesting portfolio strategies based on macro regimes.
"""

import pandas as pd
import numpy as np
import os
import sys

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

    def run_backtest(self):
        """
        Run backtests comparing:
        1. Benchmark (60/40)
        2. Binary Dynamic Strategy (Defensive shift in Slowdown)
        3. Continuous Logistic Strategy (Smooth scaling based on PC1)
        """
        # Align signals to daily dates
        daily_pc1 = self.pc1_smooth.reindex(self.asset_returns.index, method='ffill').fillna(0)
        
        # 1. Benchmark Returns (Static 60/40)
        bw = config.BENCHMARK_WEIGHTS
        benchmark_returns = (
            self.asset_returns['stocks'] * bw['stocks'] + 
            self.asset_returns['bonds'] * bw['bonds']
        )
        
        # 2. Binary Strategy Returns (if regimes provided)
        binary_returns = None
        if self.regimes is not None:
            daily_regimes = self.regimes.reindex(self.asset_returns.index, method='ffill').fillna(0)
            w_stocks_bin = daily_regimes.map(lambda r: config.REGIME_WEIGHT_MAP[int(r)]['stocks'])
            w_bonds_bin  = daily_regimes.map(lambda r: config.REGIME_WEIGHT_MAP[int(r)]['bonds'])
            binary_returns = (
                self.asset_returns['stocks'] * w_stocks_bin + 
                self.asset_returns['bonds'] * w_bonds_bin
            )
        
        # 3. Continuous Logistic Strategy
        risk_scores = self.calculate_risk_score(daily_pc1)
        w_min = config.CONTINUOUS_WEIGHT_LIMITS["min_equity"]
        w_max = config.CONTINUOUS_WEIGHT_LIMITS["max_equity"]
        
        # w_eq = w_min + (w_max - w_min) * (1 - R_t)
        weights_stocks_cont = w_min + (w_max - w_min) * (1 - risk_scores)
        weights_bonds_cont  = 1 - weights_stocks_cont
        
        continuous_returns = (
            self.asset_returns['stocks'] * weights_stocks_cont + 
            self.asset_returns['bonds'] * weights_bonds_cont
        )
        
        # Create results DataFrame
        results = pd.DataFrame({
            'benchmark': benchmark_returns,
            'continuous': continuous_returns,
            'risk_score': risk_scores,
            'weight_stocks': weights_stocks_cont
        })
        
        if binary_returns is not None:
            results['binary'] = binary_returns
            
        # Calculate cumulative returns
        for col in ['benchmark', 'binary', 'continuous']:
            if col in results.columns:
                results[f'cum_{col}'] = (1 + results[col]).cumprod()
        
        return results

    def calculate_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics for all strategies in results."""
        metrics = {}
        
        strategies = [c for c in ['benchmark', 'binary', 'continuous'] if c in results.columns]
        
        for col in strategies:
            rets = results[col]
            cum_rets = results[f'cum_{col}']
            
            # Annualized Return
            total_return = cum_rets.iloc[-1] - 1
            n_years = len(rets) / 252
            ann_return = (1 + total_return)**(1/n_years) - 1
            
            # Annualized Volatility
            ann_vol = rets.std() * np.sqrt(252)
            
            # Sharpe Ratio
            sharpe = ann_return / ann_vol if ann_vol != 0 else 0
            
            # Max Drawdown
            rolling_max = cum_rets.cummax()
            drawdown = (cum_rets - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            metrics[col] = {
                'Annualized Return (%)': round(ann_return * 100, 2),
                'Annualized Vol (%)':   round(ann_vol * 100, 2),
                'Sharpe Ratio':         round(sharpe, 2),
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
