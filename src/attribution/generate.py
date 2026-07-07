"""
Phase 3 · Generate attribution maps on the frozen model.
See Decision D6 and docs/02_EXECUTION_RUNBOOK.md (Phase 3).

Panel: Grad-CAM, Grad-CAM++, Saliency, Integrated Gradients, DeepLIFT.
Captum supplies the gradient-based methods; Grad-CAM/++ are implemented here
via forward/backward hooks (Captum has no Grad-CAM++).

Each map is saved as a (n_leads, n_time) array -> attributions/{method}/{ecg_id}.npy.
Note: 1D Grad-CAM localises in *time only* (the conv trunk has mixed the leads
into channels), so its map is broadcast across leads. That limitation is itself
informative for the lead-concordance metric (Phase 4.3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import config
from ..utils import get_device, normalize_attribution


def _cam(model, x, plus_plus: bool) -> np.ndarray:
    """Grad-CAM (or Grad-CAM++) on model.gradcam_layer -> (1, n_time) map."""
    import torch
    import torch.nn.functional as F

    layer = model.gradcam_layer
    activations, gradients = {}, {}

    h1 = layer.register_forward_hook(lambda m, i, o: activations.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logit = model(x)                       # (1, 1)
        logit.sum().backward()
        a = activations["a"]                   # (1, C, T')
        g = gradients["g"]                      # (1, C, T')

        if plus_plus:                          # Grad-CAM++ channel weights
            g2, g3 = g.pow(2), g.pow(3)
            denom = 2 * g2 + (a * g3).sum(dim=2, keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(g)).sum(dim=2, keepdim=True)
        else:                                  # vanilla Grad-CAM: GAP of grads
            weights = g.mean(dim=2, keepdim=True)

        cam = F.relu((weights * a).sum(dim=1, keepdim=True))   # (1, 1, T')
        cam = F.interpolate(cam, size=config.SIGNAL_LENGTH, mode="linear", align_corners=False)
        return cam.squeeze(0).detach().cpu().numpy()           # (1, n_time)
    finally:
        h1.remove()
        h2.remove()


def _captum_attr(model, x, method: str) -> np.ndarray:
    """Saliency / Integrated Gradients / DeepLIFT via Captum -> (n_leads, n_time)."""
    from captum.attr import DeepLift, IntegratedGradients, Saliency

    x = x.clone().requires_grad_(True)
    if method == "saliency":
        attr = Saliency(model).attribute(x, target=0, abs=False)
    elif method == "integrated_gradients":
        attr = IntegratedGradients(model).attribute(x, target=0, baselines=x * 0, n_steps=50)
    elif method == "deep_lift":
        attr = DeepLift(model).attribute(x, target=0, baselines=x * 0)
    else:
        raise ValueError(f"unknown captum method {method!r}")
    return attr.squeeze(0).detach().cpu().numpy()              # (n_leads, n_time)


def attribute_case(model, x, method: str) -> np.ndarray:
    """One case (tensor (1, n_leads, n_time)) + one method -> normalised (12, 1000)."""
    if method == "grad_cam":
        raw = _cam(model, x, plus_plus=False)
    elif method == "grad_cam_pp":
        raw = _cam(model, x, plus_plus=True)
    else:
        raw = _captum_attr(model, x, method)

    if raw.shape[0] == 1:                       # broadcast time-only CAM over leads
        raw = np.repeat(raw, config.N_LEADS, axis=0)
    return normalize_attribution(raw)           # comparable across methods (Phase 4)


def generate_attributions(model, cases, signals, methods=None, out_dir=config.ATTRIB_DIR):
    """Run the panel over every correct-MI case; save one .npy per (method, case).

    ``cases``   : DataFrame/Series/list of ecg_ids (Phase 2's correct_MI_cases).
    ``signals`` : dict ecg_id -> (n_leads, n_time) preprocessed array.
    """
    import torch

    methods = methods or config.XAI_METHODS
    device = get_device()
    model.to(device).eval()

    ecg_ids = cases["ecg_id"] if isinstance(cases, pd.DataFrame) else cases
    out_dir = Path(out_dir)
    for m in methods:
        (out_dir / m).mkdir(parents=True, exist_ok=True)

    for ecg_id in tqdm(list(ecg_ids), desc="attributions"):
        x = torch.from_numpy(np.asarray(signals[ecg_id])[None]).float().to(device)
        for m in methods:
            attr = attribute_case(model, x, m)
            np.save(out_dir / m / f"{ecg_id}.npy", attr.astype(np.float32))


if __name__ == "__main__":  # python -m src.attribution.generate
    from ..models.train import load_frozen
    from ..data.preprocess import load_arrays

    model, _ = load_frozen()
    cases = pd.read_csv(config.CORRECT_MI_CASES_CSV)
    x_test, _, ids_test = load_arrays("test")
    sig = {i: x for i, x in zip(ids_test, x_test)}
    generate_attributions(model, cases, sig)
