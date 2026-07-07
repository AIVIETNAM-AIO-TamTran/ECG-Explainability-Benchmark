"""
Phase 4.3 · Clinical concordance — the differentiator.
See Decision D8, config.TERRITORY_LEADS / CODE_TO_TERRITORY.

Concordance ratio:
    C = (sum |attribution| inside diagnostic mask) / (sum |attribution| over all leads x time)

The diagnostic mask is the ST-segment / J-point window (where MI injury current
shows) on the leads implicating the labelled MI territory, aggregated over all
beats in the record. NeuroKit2 delineates the beats; a robust fallback keeps the
metric defined even when delineation is imperfect.
"""

from __future__ import annotations

import numpy as np

from .. import config


def _ms_to_samples(ms: float, fs: int = config.SAMPLING_RATE) -> int:
    return int(round(ms / 1000.0 * fs))


def _rpeaks_and_joffsets(signal, fs=config.SAMPLING_RATE):
    """Detect R-peaks (on lead II) and per-beat J-points (QRS offsets).

    Returns (rpeaks, joffsets) in sample units. Falls back to a fixed offset
    after each R-peak if NeuroKit delineation is unavailable/NaN.
    """
    import neurokit2 as nk

    lead_ii = np.asarray(signal[config.LEAD_ORDER.index("II")], dtype=np.float64)
    _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    fallback = _ms_to_samples(config.JPOINT_FALLBACK_MS, fs)

    joffsets = rpeaks + fallback
    try:
        _, waves = nk.ecg_delineate(lead_ii, rpeaks, sampling_rate=fs, method="dwt")
        offs = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
        for i, r in enumerate(rpeaks):
            if i < len(offs) and np.isfinite(offs[i]) and offs[i] > r:
                joffsets[i] = int(offs[i])
    except Exception:
        pass  # keep the fixed-offset fallback
    return rpeaks, joffsets


def build_diagnostic_mask(signal, mi_subtype_code: str) -> np.ndarray:
    """Boolean (n_leads, n_time) mask: ST window x territory leads, all beats.

    ``signal`` : (n_leads, n_time) preprocessed array.
    If the subtype has no mapped territory (sparse codes), the mask spans the ST
    window on *all* leads so the ratio is still defined (lead-concordance then
    carries the territory information separately).
    """
    signal = np.asarray(signal)
    n_leads, n_time = signal.shape
    mask = np.zeros((n_leads, n_time), dtype=bool)

    territory = config.CODE_TO_TERRITORY.get(mi_subtype_code)
    leads = config.TERRITORY_LEADS.get(territory, config.LEAD_ORDER)
    lead_idx = [config.LEAD_ORDER.index(l) for l in leads]

    _, joffsets = _rpeaks_and_joffsets(signal)
    win = _ms_to_samples(config.ST_WINDOW_MS[1]) - _ms_to_samples(config.ST_WINDOW_MS[0])
    start_off = _ms_to_samples(config.ST_WINDOW_MS[0])

    for j in joffsets:
        a, b = max(j + start_off, 0), min(j + start_off + win, n_time)
        if a < b:
            for li in lead_idx:
                mask[li, a:b] = True
    return mask


def concordance_ratio(attribution, mask) -> float:
    """Fraction of |attribution| mass falling inside the diagnostic mask."""
    a = np.abs(np.asarray(attribution))
    total = a.sum()
    return float(a[mask].sum() / total) if total > 0 else float("nan")


def lead_concordance(attribution, mi_subtype_code: str) -> float:
    """Fraction of |attribution| mass on the territory-relevant leads (Decision D8)."""
    a = np.abs(np.asarray(attribution))
    territory = config.CODE_TO_TERRITORY.get(mi_subtype_code)
    leads = config.TERRITORY_LEADS.get(territory)
    if leads is None:
        return float("nan")                      # sparse code: undefined territory
    lead_mass = a.sum(axis=1)
    idx = [config.LEAD_ORDER.index(l) for l in leads]
    total = lead_mass.sum()
    return float(lead_mass[idx].sum() / total) if total > 0 else float("nan")


def concordance(signal, attribution, mi_subtype_code: str) -> dict:
    """Both concordance measures for one case (convenience wrapper)."""
    mask = build_diagnostic_mask(signal, mi_subtype_code)
    return {
        "concordance_ratio": concordance_ratio(attribution, mask),
        "lead_concordance": lead_concordance(attribution, mi_subtype_code),
    }
