"""
Unit tests for the dependency-light core logic (no torch / wfdb / captum needed).
Run: `python -m pytest tests` or `python tests/test_pipeline.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data.build_cohort import build_cohort
from src.data.preprocess import preprocess
from src.metrics.agreement import rank_correlation, topk_overlap
from src.metrics.concordance import concordance_ratio, lead_concordance
from src.utils import bootstrap_ci, normalize_attribution


def _fake_meta():
    """Minimal metadata table exercising strict/inclusive + co-morbid + NORM."""
    rows = {
        1: {"scp_codes": {"ASMI": 100.0}, "superclasses": {"MI"}},           # strict anterior MI
        2: {"scp_codes": {"IMI": 50.0}, "superclasses": {"MI"}},             # inclusive-only inferior MI
        3: {"scp_codes": {"AMI": 100.0, "LVH": 100.0},                       # co-morbid MI
            "superclasses": {"MI", "HYP"}},
        4: {"scp_codes": {"NORM": 100.0}, "superclasses": {"NORM"}},         # pure NORM
        5: {"scp_codes": {"NDT": 100.0}, "superclasses": {"STTC"}},          # dropped (other only)
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["strat_fold"] = [10, 10, 10, 10, 10]
    df.index.name = "ecg_id"
    return df


def test_cohort_strict_vs_inclusive():
    meta = _fake_meta()
    strict = build_cohort(meta, "strict")
    inclusive = build_cohort(meta, "inclusive")

    # strict: MI positives = records 1 & 3 (both have an MI code at likelihood 100)
    assert set(strict[strict["label"] == 1].index) == {1, 3}
    # inclusive adds record 2 (IMI at 50)
    assert set(inclusive[inclusive["label"] == 1].index) == {1, 2, 3}
    # record 4 is the only NORM negative in both; record 5 is dropped
    assert set(strict[strict["label"] == 0].index) == {4}
    assert 5 not in strict.index


def test_cohort_comorbid_and_territory():
    strict = build_cohort(_fake_meta(), "strict")
    assert strict.loc[1, "territory"] == "anterior"
    assert strict.loc[1, "concordance_eligible"]           # pure MI, mapped territory
    assert strict.loc[3, "comorbid"]                       # AMI + LVH
    assert not strict.loc[3, "concordance_eligible"]       # excluded from concordance


def test_preprocess_zscore():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 3, size=(4, config.SIGNAL_LENGTH, config.N_LEADS)).astype(np.float32)
    z = preprocess(x)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-4)
    assert np.allclose(z.std(axis=1), 1, atol=1e-3)


def test_preprocess_handles_flat_and_nan():
    x = np.zeros((1, config.SIGNAL_LENGTH, config.N_LEADS), dtype=np.float32)
    x[0, 0, 0] = np.nan
    z = preprocess(x)                                       # must not raise / produce NaN
    assert np.isfinite(z).all()


def test_concordance_ratio_and_leads():
    attr = np.zeros((config.N_LEADS, config.SIGNAL_LENGTH))
    mask = np.zeros_like(attr, dtype=bool)
    attr[6, 100:110] = 1.0          # all mass in V1 (anterior lead), inside mask
    mask[6, 100:110] = True
    assert concordance_ratio(attr, mask) == 1.0
    # V1 is an anterior lead -> full lead-concordance for an anterior subtype
    assert lead_concordance(attr, "ASMI") == 1.0
    # inferior subtype (II/III/aVF) sees none of V1's mass
    assert lead_concordance(attr, "IMI") == 0.0


def test_normalize_attribution_range():
    a = np.array([[[-4.0, 2.0], [0.0, 1.0]]])
    n = normalize_attribution(a)
    assert n.min() >= 0 and np.isclose(n.max(), 1.0)        # abs + scaled to [0, 1]


def test_agreement_self_and_disjoint():
    a = np.random.default_rng(1).normal(size=(12, 1000))
    assert rank_correlation(a, a) > 0.999                   # identical maps
    assert topk_overlap(a, a) == 1.0
    assert 0.0 <= topk_overlap(a, -a) <= 1.0


def test_bootstrap_ci_orders():
    point, lo, hi = bootstrap_ci(np.linspace(0, 1, 200), n_boot=200)
    assert lo <= point <= hi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
