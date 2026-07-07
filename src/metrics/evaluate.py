"""
Phase 4 driver · Assemble the master explanation-quality table.
See docs/02_EXECUTION_RUNBOOK.md (Phase 4): "one master table — method x
{faithfulness, sanity, concordance, agreement} with 95% CIs — plus a ranked
verdict on PTB-XL."

Consumes the frozen model, the Phase-2 correct-MI case list, the Phase-1 cohort
(for MI subtypes), and the Phase-3 attribution .npy files. Writes:
  - results/explanation_metrics.csv   (method x metric, mean [lo, hi])
  - results/method_agreement.csv      (mean pairwise Spearman between methods)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..utils import bootstrap_ci
from . import agreement, concordance, faithfulness, sanity


def _load_case_attributions(ecg_id, methods):
    out = {}
    for m in methods:
        path = config.ATTRIB_DIR / m / f"{ecg_id}.npy"
        if path.exists():
            out[m] = np.load(path)
    return out


def build_master_table(model, threshold=None, methods=None, sanity_subsample: int = 30):
    """Compute per-case metrics for every method and aggregate with bootstrap CIs.

    ``sanity_subsample`` caps how many cases the (expensive) cascading-randomization
    sanity check runs on; faithfulness/concordance run on all cases.
    """
    import torch

    from ..data.preprocess import load_arrays

    methods = methods or config.XAI_METHODS
    cohort = pd.read_csv(config.COHORT_CSV, index_col="ecg_id")
    cases = pd.read_csv(config.CORRECT_MI_CASES_CSV)
    x_test, _, ids_test = load_arrays("test")
    signals = {i: x for i, x in zip(ids_test, x_test)}

    per_case = {m: {"faithfulness": [], "concordance_ratio": [],
                    "lead_concordance": [], "sanity": []} for m in methods}
    agree_mats = []

    for n, ecg_id in enumerate(cases["ecg_id"]):
        attrs = _load_case_attributions(ecg_id, methods)
        if len(attrs) < 2:
            continue
        subtype = cohort.loc[ecg_id, "mi_subtype"]
        sig = signals[ecg_id]

        for m, attr in attrs.items():
            per_case[m]["faithfulness"].append(
                faithfulness.faithfulness_score(model, sig, attr))
            c = concordance.concordance(sig, attr, subtype)
            per_case[m]["concordance_ratio"].append(c["concordance_ratio"])
            per_case[m]["lead_concordance"].append(c["lead_concordance"])
            if n < sanity_subsample:
                x = torch.from_numpy(sig[None]).float()
                per_case[m]["sanity"].append(sanity.sanity_check(model, x, m)["sanity_score"])

        agree_mats.append(agreement.agreement_matrix(attrs))

    rows = []
    for m in methods:
        row = {"method": m}
        for metric, vals in per_case[m].items():
            mean, lo, hi = bootstrap_ci(vals) if vals else (np.nan, np.nan, np.nan)
            row[metric] = mean
            row[f"{metric}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.METRICS_TABLE_CSV)
    if agree_mats:
        agreement.mean_agreement(agree_mats).to_csv(
            config.RESULTS_DIR / "method_agreement.csv")
    return table


if __name__ == "__main__":  # python -m src.metrics.evaluate
    from ..models.train import load_frozen

    _model, _thr = load_frozen()
    print(build_master_table(_model, _thr).to_string())
