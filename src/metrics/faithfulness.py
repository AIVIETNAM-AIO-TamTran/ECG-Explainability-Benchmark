"""
Phase 4.1 · Faithfulness — deletion/insertion curves + AOPC.
See Decision D7.

Rank time-steps by attribution mass, then progressively remove (deletion) or
reveal (insertion) the top-ranked steps and track the MI probability. A faithful
method makes the probability collapse quickly under deletion and rise quickly
under insertion. Baseline = 0 (the per-lead mean, since signals are z-scored).

Self-contained numpy/torch so it runs without Quantus; the summary statistics
(deletion AUC, insertion AUC, AOPC) match Quantus' definitions.
"""

from __future__ import annotations

import numpy as np

from .. import config


def _prob(model, batch, device):
    """Sigmoid MI probability for a numpy batch (n, n_leads, n_time)."""
    import torch

    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)
        return torch.sigmoid(model(xb).squeeze(1)).cpu().numpy()


def deletion_insertion(model, x, attribution, n_steps: int = 20, device=None):
    """Deletion/insertion faithfulness for one case.

    ``x``           : (n_leads, n_time) preprocessed signal.
    ``attribution`` : (n_leads, n_time) map (any scale; abs value is used).
    Returns dict with deletion/insertion curves + AUCs and AOPC.
    """
    from ..utils import get_device

    device = device or get_device()
    model.to(device).eval()

    x = np.asarray(x, dtype=np.float32)
    importance = np.abs(attribution).sum(axis=0)          # per-time-step (n_time,)
    order = np.argsort(importance)[::-1]                  # most important first
    steps = np.linspace(0, len(order), n_steps + 1, dtype=int)[1:]

    # Build all masked variants for one batched forward per curve.
    del_batch = np.repeat(x[None], n_steps + 1, axis=0)   # index 0 = untouched
    ins_batch = np.zeros_like(del_batch)                  # index 0 = empty baseline
    for s, k in enumerate(steps, start=1):
        idx = order[:k]
        del_batch[s, :, idx] = 0.0                        # remove top-k
        ins_batch[s, :, idx] = x[:, idx]                  # reveal top-k

    del_curve = _prob(model, del_batch, device)
    ins_curve = _prob(model, ins_batch, device)
    frac = np.linspace(0, 1, n_steps + 1)

    return {
        "deletion_curve": del_curve, "insertion_curve": ins_curve, "fraction": frac,
        "deletion_auc": float(np.trapz(del_curve, frac)),     # lower = more faithful
        "insertion_auc": float(np.trapz(ins_curve, frac)),    # higher = more faithful
        # AOPC: mean drop from the original probability as top segments are removed.
        "aopc": float(np.mean(del_curve[0] - del_curve[1:])),
    }


def faithfulness_score(model, x, attribution, **kw) -> float:
    """Single scalar for the master table: insertion_auc - deletion_auc
    (higher = more faithful), bounded roughly in [-1, 1]."""
    r = deletion_insertion(model, x, attribution, **kw)
    return r["insertion_auc"] - r["deletion_auc"]
