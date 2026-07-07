"""
Phase 4.2 · Sanity — cascading model-parameter randomization (Adebayo et al.).
See Decision D7. This is where Grad-CAM historically collapses onto the QRS
complex regardless of the (now random) weights.

Randomize the frozen network's layers cascadingly from the output backwards,
recomputing the attribution at each stage, and measure Spearman rank-correlation
against the original map. A trustworthy method degrades toward noise (correlation
-> 0); a method that stays highly correlated has FAILED the sanity check.
"""

from __future__ import annotations

import copy

import numpy as np

from .. import config
from ..attribution.generate import attribute_case


def _randomizable_modules(model):
    """Leaf modules carrying learnable weights, in forward order."""
    import torch.nn as nn

    keep = (nn.Conv1d, nn.Linear, nn.BatchNorm1d)
    return [(n, m) for n, m in model.named_modules() if isinstance(m, keep)]


def _reinit(module):
    """Reinitialise a module's parameters to random values."""
    import torch.nn as nn

    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        module.reset_parameters()
        nn.init.normal_(module.weight, 1.0, 0.1)


def sanity_check(model, x, method: str, device=None) -> dict:
    """Cascading-randomization sanity curve for one case + one method.

    ``x`` : tensor (1, n_leads, n_time). Returns per-stage Spearman correlations
    and a summary ``mean_abs_corr`` (low = good) plus ``sanity_score`` = 1 - that
    (high = the method degrades as it should).
    """
    import torch
    from scipy.stats import spearmanr
    from ..utils import get_device

    device = device or get_device()
    original_state = copy.deepcopy(model.state_dict())
    base = attribute_case(model.to(device), x.to(device), method).ravel()

    layers = _randomizable_modules(model)[::-1]     # top (output) first
    correlations = {}
    try:
        for depth in range(1, len(layers) + 1):
            model.load_state_dict(original_state)   # reset, then randomize top `depth`
            for _, m in layers[:depth]:
                _reinit(m)
            attr = attribute_case(model, x, method).ravel()
            rho = spearmanr(base, attr).statistic
            correlations[layers[depth - 1][0]] = float(0.0 if np.isnan(rho) else rho)
    finally:
        model.load_state_dict(original_state)       # always restore the frozen weights

    mean_abs = float(np.mean(np.abs(list(correlations.values())))) if correlations else float("nan")
    return {
        "correlations": correlations,
        "mean_abs_corr": mean_abs,
        "sanity_score": 1.0 - mean_abs,             # higher = degrades properly = trustworthy
    }
