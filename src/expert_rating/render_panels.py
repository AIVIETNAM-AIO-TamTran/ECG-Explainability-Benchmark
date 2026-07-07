"""
Phase 5 · Render blinded attribution panels for the cardiologist reader study.
See Decisions D11, D12 and docs/02_EXECUTION_RUNBOOK.md (Phase 5).

IMPORTANT: maps are GENERATED here from Phase 3 outputs, not downloaded.
Cardiologists rate these overlays; they never see the metadata table.

  - Sample ~config.N_RATING_CASES cases x config.RATING_METHODS -> ~60 panels.
  - Overlay each attribution on the 12-lead tracing (uniform style, line coloured
    by attribution intensity).
  - Assign random IDs (MAP_###); the ID->method key is written to a file matching
    the .gitignore blinding-key pattern so it never reaches git.
  - Shuffle presentation order; export panels + a rating-form template.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..utils import normalize_attribution, set_seed


def _plot_panel(signal, attribution, out_path: Path, panel_id: str) -> None:
    """Render one 12-lead panel with the trace coloured by attribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    attr = normalize_attribution(attribution)             # (12, 1000) in [0, 1]
    t = np.arange(config.SIGNAL_LENGTH) / config.SAMPLING_RATE

    fig, axes = plt.subplots(config.N_LEADS, 1, figsize=(10, 14), sharex=True)
    for li, ax in enumerate(axes):
        y = np.asarray(signal[li])
        pts = np.array([t, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="Reds", norm=plt.Normalize(0, 1))
        lc.set_array(attr[li, :-1])
        lc.set_linewidth(1.2)
        ax.add_collection(lc)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
        ax.set_ylabel(config.LEAD_ORDER[li], rotation=0, labelpad=18, va="center")
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(panel_id, fontsize=12)                   # blinded — no method/label shown
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_panels(n_cases: int = config.N_RATING_CASES,
                  methods=None, out_dir=config.FIGURES_DIR / "panels", seed=config.SEED):
    """Sample cases x methods, render blinded panels, and write the key + form.

    Returns the (secret) key DataFrame. Panels + form are shareable with raters;
    the key file is git-ignored (see .gitignore blinding-integrity section).
    """
    from ..data.preprocess import load_arrays

    set_seed(seed)
    methods = methods or config.RATING_METHODS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(config.CORRECT_MI_CASES_CSV)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cases["ecg_id"].to_numpy(),
                        size=min(n_cases, len(cases)), replace=False)

    x_test, _, ids_test = load_arrays("test")
    signals = {i: x for i, x in zip(ids_test, x_test)}

    # Build the full (case, method) list, assign shuffled MAP ids.
    pairs = [(int(c), m) for c in chosen for m in methods]
    rng.shuffle(pairs)
    key_rows = []
    for n, (ecg_id, method) in enumerate(pairs, start=1):
        panel_id = f"MAP_{n:03d}"
        attr = np.load(config.ATTRIB_DIR / method / f"{ecg_id}.npy")
        _plot_panel(signals[ecg_id], attr, out_dir / f"{panel_id}.png", panel_id)
        key_rows.append({"panel_id": panel_id, "ecg_id": ecg_id, "method": method,
                         "presentation_order": n})

    key = pd.DataFrame(key_rows)
    key.to_csv(config.RESULTS_DIR / "expert_rating_key.csv", index=False)   # git-ignored

    # Rating-form template raters DO see (no method column).
    form = key[["panel_id"]].copy()
    for col in ["clinical_plausibility_1to5", "localization_quality_1to5",
                "lead_correctness_1to5", "increases_trust_1to5", "free_text"]:
        form[col] = ""
    form.to_csv(config.RESULTS_DIR / "rating_form_template.csv", index=False)

    print(f"{len(pairs)} panels -> {out_dir}")
    print(f"blinding key -> {config.RESULTS_DIR / 'expert_rating_key.csv'} (git-ignored)")
    return key


if __name__ == "__main__":  # python -m src.expert_rating.render_panels
    render_panels()
