import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def calculate_performance_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    strategies = ['benchmark', 'binary', 'continuous']
    results = {}

    # Days in the backtest
    num_days = len(df)
    years = num_days / 252.0

    # Annual Risk-free Rate from config
    rf_rate = config.RISK_FREE_RATE_ANNUAL

    for strategy in strategies:
        # returns
        if strategy not in df.columns:
            continue
        rets = df[strategy]
        
        # Cumulative returns (final value)
        cum_col = f'cum_{strategy}'
        final_cum = df[cum_col].iloc[-1]
        
        # CAGR
        cagr = (final_cum) ** (1/years) - 1
        
        # Annualized Volatility
        vol = rets.std() * np.sqrt(252)
        
        # Excess Sharpe Ratio (subtracting Risk-free rate)
        excess_sharpe = (cagr - rf_rate) / vol if vol != 0 else 0
        
        # Max Drawdown
        cum_ret = df[cum_col]
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        results[strategy] = {
            'CAGR': f"{cagr:.2%}",
            'Volatility': f"{vol:.2%}",
            'Sharpe (Excess)': f"{excess_sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.2%}"
        }

    return results

if __name__ == "__main__":
    path = "/Users/sudhan/.gemini/antigravity/scratch/macro-pca-indicator/outputs/portfolio_backtest.csv"
    metrics = calculate_performance_metrics(path)
    
    if metrics:
        metrics_df = pd.DataFrame(metrics).transpose()
        print("\nPerformance Metrics Summary:")
        print(metrics_df)
        
        # Also save to csv for record
        output_path = "/Users/sudhan/.gemini/antigravity/scratch/macro-pca-indicator/outputs/performance_summary.csv"
        metrics_df.to_csv(output_path)
        print(f"\nSummary saved to {output_path}")
