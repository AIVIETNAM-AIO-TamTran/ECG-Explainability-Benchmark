"""
Shared, dependency-light helpers: seeding, device selection, and attribution
normalisation. Kept separate from config.py (which stays pure declarations) so
these can be imported anywhere without side effects.
"""

from __future__ import annotations

import os
import random

import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed every RNG we use so runs are reproducible (Decision D15).

    Torch is optional (not installed in every environment), so it is seeded
    only if importable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism at a small speed cost — acceptable for a benchmark.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """Return the best available torch device ('cuda' if present else 'cpu')."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_attribution(attr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardise an attribution map so methods are comparable in Phase 4.

    Takes the absolute value (we care about *importance*, not sign) and scales
    each map to [0, 1] by its own max. Operates on a single (n_leads, n_time)
    map or a batch whose leading axis is the case index.
    """
    a = np.abs(np.asarray(attr, dtype=np.float64))
    axes = tuple(range(a.ndim - 2, a.ndim))  # normalise per-map over (lead, time)
    m = a.max(axis=axes, keepdims=True)
    return a / (m + eps)


def bootstrap_ci(values, statistic=np.mean, n_boot: int = config.N_BOOTSTRAP,
                 alpha: float = 0.05, seed: int = config.SEED):
    """Percentile bootstrap CI for a 1-D array of per-case scores.

    Returns (point_estimate, lo, hi). Used for both model metrics (Phase 2)
    and explanation metrics (Phase 4) so every number ships with an interval.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = statistic(values[rng.integers(0, n, n)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)
