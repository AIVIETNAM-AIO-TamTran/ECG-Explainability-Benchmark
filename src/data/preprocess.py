"""
Phase 1 · Preprocess signals (fix once, apply identically to PTB-XL and MIMIC).
See Decision D10.

  - 100 Hz, 1000 samples x 12 leads, standard LEAD_ORDER (enforced at load time).
  - Per-lead z-score normalisation; NaNs/Infs made finite first.
  - Save X/y arrays + an index mapping every row back to its ecg_id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def preprocess(signals: np.ndarray) -> np.ndarray:
    """Per-lead z-score a batch of signals.

    Input/return shape: (n, time, leads). Each (record, lead) trace is
    standardised to zero mean / unit variance over time. Non-finite samples are
    zero-filled first (rare electrode dropouts), and flat leads (std == 0) are
    left mean-centred rather than divided by zero.
    """
    x = np.asarray(signals, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (config.SIGNAL_LENGTH, config.N_LEADS):
        raise ValueError(
            f"expected (n, {config.SIGNAL_LENGTH}, {config.N_LEADS}), got {x.shape}"
        )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)      # avoid div-by-zero on flat leads
    return ((x - mean) / std).astype(np.float32)


def build_arrays(cohort: pd.DataFrame, signals: np.ndarray):
    """Preprocess + split into per-fold X/y arrays (Decision D10 folds).

    ``cohort`` and ``signals`` must be row-aligned (same order). Returns a dict
    keyed 'train'/'val'/'test', each -> (X, y, ecg_ids).
    """
    x = preprocess(signals)                                # (n, time, leads)
    x = np.transpose(x, (0, 2, 1))                         # -> (n, leads, time) for Conv1d
    y = cohort["label"].to_numpy(dtype=np.int64)
    ids = cohort.index.to_numpy()
    fold = cohort["strat_fold"].to_numpy()

    masks = {
        "train": np.isin(fold, config.TRAIN_FOLDS),
        "val": fold == config.VAL_FOLD,
        "test": fold == config.TEST_FOLD,
    }
    return {name: (x[m], y[m], ids[m]) for name, m in masks.items()}


def save_arrays(splits: dict, out_dir: Path = config.PROCESSED_DIR) -> None:
    """Persist X/y/ecg_id arrays per split (git-ignored; local only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (x, y, ids) in splits.items():
        np.save(out_dir / f"X_{name}.npy", x)
        np.save(out_dir / f"y_{name}.npy", y)
        np.save(out_dir / f"ecg_ids_{name}.npy", ids)


def load_arrays(name: str, out_dir: Path = config.PROCESSED_DIR):
    """Reload a saved split -> (X, y, ecg_ids)."""
    out_dir = Path(out_dir)
    return (
        np.load(out_dir / f"X_{name}.npy"),
        np.load(out_dir / f"y_{name}.npy"),
        np.load(out_dir / f"ecg_ids_{name}.npy", allow_pickle=True),
    )
