"""
Phase 4.4 · Cross-method agreement.
See Decision D7. Quantifies how much the "explanation" depends on the arbitrary
choice of tool: Spearman rank-correlation and top-k overlap between the maps
different methods produce on the *same* case.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def rank_correlation(attr_a, attr_b) -> float:
    """Spearman correlation between two flattened |attribution| maps."""
    from scipy.stats import spearmanr

    a = np.abs(np.asarray(attr_a)).ravel()
    b = np.abs(np.asarray(attr_b)).ravel()
    rho = spearmanr(a, b).statistic
    return float(0.0 if np.isnan(rho) else rho)


def topk_overlap(attr_a, attr_b, k_frac: float = 0.1) -> float:
    """Jaccard overlap of the top-``k_frac`` most-attributed elements."""
    a = np.abs(np.asarray(attr_a)).ravel()
    b = np.abs(np.asarray(attr_b)).ravel()
    k = max(1, int(k_frac * a.size))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    return len(top_a & top_b) / len(top_a | top_b)


def agreement_matrix(attr_by_method: dict, metric: str = "spearman") -> pd.DataFrame:
    """Pairwise agreement between methods for one case -> symmetric DataFrame."""
    methods = list(attr_by_method)
    fn = rank_correlation if metric == "spearman" else topk_overlap
    mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for m1, m2 in itertools.combinations(methods, 2):
        v = fn(attr_by_method[m1], attr_by_method[m2])
        mat.loc[m1, m2] = mat.loc[m2, m1] = v
    return mat


def mean_agreement(per_case_matrices) -> pd.DataFrame:
    """Average a list of per-case agreement matrices into one summary matrix."""
    stack = np.stack([m.to_numpy() for m in per_case_matrices])
    ref = per_case_matrices[0]
    return pd.DataFrame(stack.mean(axis=0), index=ref.index, columns=ref.columns)
