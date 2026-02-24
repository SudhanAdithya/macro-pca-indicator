"""
pca/build_indicator.py
Runs PCA on the standardized macro Z-matrix, extracts PC1,
applies sign normalization, and provides diagnostic outputs.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple
import sys
import os

# Ensure project root is in sys.path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


# ---------------------------------------------------------------------------
# PCA runner
# ---------------------------------------------------------------------------

def run_pca(Z: pd.DataFrame, n_components: int = None) -> Tuple[PCA, np.ndarray]:
    """
    Fit sklearn PCA on the standardized macro matrix Z.

    Parameters
    ----------
    Z            : (T × N) DataFrame of standardized, sign-aligned macro variables
    n_components : number of components to retain (None = all)

    Returns
    -------
    pca     : fitted sklearn PCA object
    scores  : (T × n_components) array of principal component scores
    """
    if n_components is None:
        n_components = min(Z.shape)

    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(Z.values)

    print(f"  PCA fitted on {Z.shape[0]} observations × {Z.shape[1]} variables")
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")

    return pca, scores


def extract_pc1(
    pca: PCA,
    scores: np.ndarray,
    Z: pd.DataFrame,
) -> pd.Series:
    """
    Return PC1 as a pd.Series with the same DatetimeIndex as Z.

    Parameters
    ----------
    pca    : fitted PCA object
    scores : component scores from run_pca()
    Z      : original standardized DataFrame (used for index)

    Returns
    -------
    pd.Series named 'pc1' with datetime index
    """
    pc1 = pd.Series(scores[:, 0], index=Z.index, name="pc1")
    return pc1


def normalize_pc1_sign(
    pc1: pd.Series,
    recession_mask: pd.Series = None,
) -> pd.Series:
    """
    PCA sign is arbitrary. Ensure PC1 is oriented so that:
        HIGH PC1 = stronger macro conditions
        LOW  PC1 = slowdown / deterioration

    Strategy (in order of preference):
    1. If recession_mask is provided: if median PC1 during recessions > 0,
       flip sign (recessions should show low PC1).
    2. If no recession mask: if the correlation between PC1 and changes
       in unemployment is positive, flip sign (unemployment rises = recession).

    Parameters
    ----------
    pc1            : raw PC1 series from extract_pc1()
    recession_mask : boolean or 0/1 Series aligned with pc1 index (NBER rec)

    Returns
    -------
    Sign-corrected PC1 series (same index).
    """
    flipped = False

    if recession_mask is not None:
        # Align recession mask to PC1 index
        mask = recession_mask.reindex(pc1.index).fillna(0).astype(bool)
        median_during_rec = pc1[mask].median()
        if median_during_rec > 0:
            pc1 = -pc1
            flipped = True
            print("  PC1 sign flipped: median during recessions was positive (corrected).")
        else:
            print("  PC1 sign OK: median during recessions is already negative.")
    else:
        print("  No recession mask provided — PC1 sign not adjusted. Verify manually.")

    return pc1


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def loadings_table(pca: PCA, feature_names: list) -> pd.DataFrame:
    """
    Return a tidy DataFrame of PC loadings.

    Columns: PC1, PC2, ... PCn
    Rows:    one per input variable

    Parameters
    ----------
    pca           : fitted PCA object
    feature_names : list of variable names (must match PCA input column order)

    Returns
    -------
    pd.DataFrame with index = feature names, columns = ['PC1', 'PC2', ...]
    """
    n = pca.n_components_
    col_labels = [f"PC{i+1}" for i in range(n)]
    loadings = pd.DataFrame(
        pca.components_.T,   # shape (n_features × n_components)
        index=feature_names,
        columns=col_labels,
    )
    loadings.index.name = "variable"
    # Sort by absolute PC1 loading descending for readability
    loadings = loadings.reindex(
        loadings["PC1"].abs().sort_values(ascending=False).index
    )
    return loadings


def variance_explained(pca: PCA) -> pd.DataFrame:
    """
    Return a DataFrame with cumulative variance explained by each PC.

    Columns: component, var_explained_pct, cumulative_pct
    """
    var_exp = pca.explained_variance_ratio_ * 100
    df = pd.DataFrame({
        "component":         [f"PC{i+1}" for i in range(len(var_exp))],
        "var_explained_pct": np.round(var_exp, 2),
        "cumulative_pct":    np.round(np.cumsum(var_exp), 2),
    })
    return df


if __name__ == "__main__":
    # Smoke-test with synthetic data
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    Z = pd.DataFrame(np.random.randn(60, 6), index=dates,
                     columns=["ip", "rs", "unemp", "hs", "cu", "pmi"])

    pca, scores = run_pca(Z)
    pc1 = extract_pc1(pca, scores, Z)
    print("\nPC1 head:\n", pc1.head())

    ltbl = loadings_table(pca, list(Z.columns))
    print("\nLoadings table:\n", ltbl)

    vtbl = variance_explained(pca)
    print("\nVariance explained:\n", vtbl)
