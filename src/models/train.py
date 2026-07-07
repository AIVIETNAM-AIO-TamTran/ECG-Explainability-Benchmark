"""
Phase 2 · Train and FREEZE the MI detector.
See Decision D5 and docs/02_EXECUTION_RUNBOOK.md (Phase 2).

  - Model = config.MODEL_NAME (xresnet1d101; fallback xresnet1d50).
  - Binary MI vs NORM; BCE-with-logits + pos_weight for imbalance; AdamW;
    early stop on validation AUROC.
  - Evaluate on TEST_FOLD: AUROC, AUPRC, sensitivity, specificity, F1@Youden,
    with bootstrap 95% CIs. Youden threshold is chosen on validation, not test.
  - FREEZE weights; export correctly-classified MI test cases -> feeds Phase 3.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from ..utils import get_device, set_seed
from ..data.preprocess import load_arrays
from .xresnet1d import get_model


def _loaders():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def ds(split):
        x, y, _ = load_arrays(split)
        return TensorDataset(torch.from_numpy(x).float(),
                             torch.from_numpy(y).float())

    train = DataLoader(ds("train"), batch_size=config.BATCH_SIZE, shuffle=True,
                       num_workers=config.NUM_WORKERS, drop_last=True)
    val = DataLoader(ds("val"), batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    return train, val


def _predict(model, x, device, batch_size=256):
    """Return sigmoid probabilities for a (n, leads, time) numpy array."""
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).float().to(device)
            probs.append(torch.sigmoid(model(xb).squeeze(1)).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def _youden_threshold(y_true, y_prob) -> float:
    """Threshold maximising sensitivity + specificity - 1 (Youden's J)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(y_true, y_prob, threshold: float) -> dict:
    """Full metric panel with bootstrap 95% CIs (Decision D5, runbook Phase 2)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else float("nan")

    # Bootstrap the threshold-free scores over cases.
    rng = np.random.default_rng(config.SEED)
    n = len(y_true)
    aurocs, auprcs = [], []
    for _ in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_prob[idx]))

    def ci(vals):
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2

    return {
        "threshold": threshold,
        "auroc": float(roc_auc_score(y_true, y_prob)), "auroc_ci": ci(aurocs),
        "auprc": float(average_precision_score(y_true, y_prob)), "auprc_ci": ci(auprcs),
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "f1_youden": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n": n, "n_positive": int(y_true.sum()),
    }


def train(model_name: str = config.MODEL_NAME):
    """Train, early-stop on val AUROC, and save the frozen checkpoint."""
    import torch
    from sklearn.metrics import roc_auc_score

    set_seed()
    config.ensure_dirs()
    device = get_device()

    train_loader, val_loader = _loaders()
    _, y_train, _ = load_arrays("train")
    x_val, y_val, _ = load_arrays("val")

    model = get_model(model_name).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.NUM_EPOCHS)

    best_auroc, best_state, epochs_no_improve = -1.0, None, 0
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            optim.step()
        sched.step()

        val_prob = _predict(model, x_val, device)
        val_auroc = roc_auc_score(y_val, val_prob)
        print(f"epoch {epoch:02d}  val_auroc={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc, epochs_no_improve = val_auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch} (best val_auroc={best_auroc:.4f})")
                break

    model.load_state_dict(best_state)
    threshold = _youden_threshold(y_val, _predict(model, x_val, device))

    # ---- FREEZE (Decision D15 guardrail 2) ----
    for p in model.parameters():
        p.requires_grad_(False)
    torch.save({"state_dict": best_state, "model_name": model_name,
                "threshold": threshold, "val_auroc": best_auroc},
               config.CHECKPOINT)
    print(f"frozen checkpoint -> {config.CHECKPOINT}")
    return model, threshold


def load_frozen():
    """Reload the frozen model + its Youden threshold for Phases 3-6."""
    import torch

    ckpt = torch.load(config.CHECKPOINT, map_location="cpu", weights_only=False)
    model = get_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["threshold"]


def export_correct_mi_cases(model=None, threshold=None) -> pd.DataFrame:
    """Evaluate on the test fold, save metrics, and export the exact set of
    correctly-classified MI cases that feeds all attribution work (Decision D11)."""
    if model is None:
        model, threshold = load_frozen()
    device = get_device()
    model.to(device)

    x_test, y_test, ids_test = load_arrays("test")
    prob = _predict(model, x_test, device)

    metrics = evaluate(y_test, prob, threshold)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    pred = (prob >= threshold).astype(int)
    correct_mi = (y_test == 1) & (pred == 1)
    df = pd.DataFrame({"ecg_id": ids_test[correct_mi],
                       "mi_probability": prob[correct_mi]})
    df.to_csv(config.CORRECT_MI_CASES_CSV, index=False)
    print(f"{len(df)} correctly-classified MI cases -> {config.CORRECT_MI_CASES_CSV}")
    return df


if __name__ == "__main__":  # python -m src.models.train
    _model, _thr = train()
    export_correct_mi_cases(_model, _thr)
