"""
Central configuration for the ECG xAI-AMI Benchmark.

All fixed project decisions live here so they are set once and imported everywhere.
See docs/03_DECISIONS_LOG.md for the reasoning behind each value.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths  (data/ is git-ignored; never commit raw ECG data)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PTBXL_DIR = DATA_DIR / "ptbxl"            # unpacked PhysioNet PTB-XL 1.0.3
MIMIC_DIR = DATA_DIR / "mimic_iv_ecg"     # unpacked MIMIC-IV-ECG (credentialed)
PROCESSED_DIR = DATA_DIR / "processed"    # saved arrays + data_card.md
MODELS_DIR = ROOT / "models"
ATTRIB_DIR = ROOT / "attributions"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Canonical artefact locations (populated by the pipeline; most are git-ignored).
DATA_CARD = PROCESSED_DIR / "data_card.md"
COHORT_CSV = PROCESSED_DIR / "cohort.csv"       # ecg_id -> label, mi_subtype, fold
CHECKPOINT = MODELS_DIR / "xresnet1d101_mi_vs_norm.pth"
CORRECT_MI_CASES_CSV = RESULTS_DIR / "correct_MI_cases.csv"
METRICS_TABLE_CSV = RESULTS_DIR / "explanation_metrics.csv"


def ensure_dirs():
    """Create the output directories the pipeline writes to (idempotent)."""
    for d in (PROCESSED_DIR, MODELS_DIR, ATTRIB_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Task + splits  (Decisions D4, D10)
# --------------------------------------------------------------------------- #
TASK = "MI_vs_NORM"          # binary; NOT the 6-experiment structure, NOT 71 classes
SAMPLING_RATE = 100          # Hz -> use filename_lr (1000 samples x 12 leads)
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLD = 9
TEST_FOLD = 10

# Label definition switch (Decision D9). Compute BOTH in Phase 1; lean strict.
LABEL_DEFINITION = "strict"  # "strict" (likelihood==100 & pure-NORM) or "inclusive"
STRICT_LIKELIHOOD = 100.0

# --------------------------------------------------------------------------- #
# MI superclass codes (Decision D9) — union defines the positive class.
# --------------------------------------------------------------------------- #
MI_CODES = [
    "IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI",
    "INJAL", "IPLMI", "IPMI", "INJIN", "PMI", "INJLA", "INJIL",
]
# Too sparse for per-subtype analysis (drop from subtype breakdowns, keep in MI union):
SPARSE_MI_CODES = ["INJIL", "INJLA", "PMI", "INJIN"]
NORM_CODE = "NORM"

# --------------------------------------------------------------------------- #
# Territory -> expected leads, for the concordance metric (Decisions D8, D5)
# 12-lead order assumed: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
# --------------------------------------------------------------------------- #
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

TERRITORY_LEADS = {
    "anterior":    ["V1", "V2", "V3", "V4"],       # AMI, ASMI, ALMI
    "inferior":    ["II", "III", "aVF"],           # IMI, ILMI, IPMI, IPLMI
    "lateral":     ["I", "aVL", "V5", "V6"],       # LMI
}
CODE_TO_TERRITORY = {
    "AMI": "anterior", "ASMI": "anterior", "ALMI": "anterior",
    "IMI": "inferior", "ILMI": "inferior", "IPMI": "inferior", "IPLMI": "inferior",
    "LMI": "lateral",
}

# --------------------------------------------------------------------------- #
# Signal geometry (Decision D10) — fixed once, applied identically everywhere.
# --------------------------------------------------------------------------- #
N_LEADS = 12
SIGNAL_LENGTH = SAMPLING_RATE * 10   # 100 Hz x 10 s = 1000 samples

# --------------------------------------------------------------------------- #
# Model (Decision D5)
# --------------------------------------------------------------------------- #
MODEL_NAME = "xresnet1d101"  # 1D ResNet-101 variant; fallback "xresnet1d50" if compute-limited

# --------------------------------------------------------------------------- #
# Training hyper-parameters (Decision D5, runbook Phase 2)
# Modest, reproducible defaults; the model is a solid detector, not the contribution.
# --------------------------------------------------------------------------- #
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2          # AdamW
EARLY_STOP_PATIENCE = 8      # epochs without val-AUROC improvement
N_BOOTSTRAP = 1000           # bootstrap resamples for 95% CIs
NUM_WORKERS = 0              # DataLoader workers (0 is safe on Windows)

# --------------------------------------------------------------------------- #
# xAI method panel (Decision D6)
# --------------------------------------------------------------------------- #
XAI_METHODS = ["grad_cam", "grad_cam_pp", "saliency", "integrated_gradients", "deep_lift"]

# --------------------------------------------------------------------------- #
# Concordance mask geometry (Decision D8) — ST-segment / J-point window.
# NeuroKit2 gives the R-peaks and T-onsets; the diagnostic window is the
# J-point (QRS offset) out to the ST/T region where MI injury shows.
# --------------------------------------------------------------------------- #
ST_WINDOW_MS = (0, 120)      # ms after the J-point (QRS offset) that defines the ST window
JPOINT_FALLBACK_MS = 40      # if delineation misses the QRS offset, J = R-peak + this offset

# --------------------------------------------------------------------------- #
# Expert rating (Decisions D11, D12)
# --------------------------------------------------------------------------- #
N_RATING_CASES = 15          # cases x methods = panels; e.g. 15 x 4 = 60
RATING_METHODS = ["grad_cam", "saliency", "integrated_gradients", "deep_lift"]
N_RATERS = 3
