# ECG xAI–AMI Benchmark

**Do ECG deep-learning explanations point where cardiologists look?**
A faithfulness-, sanity-, and concordance-benchmarked evaluation of attribution methods for acute myocardial infarction (MI) detection.

---

## What this project is

Deep-learning models detect MI from the 12-lead ECG about as well as cardiologists, but they are black boxes. The field bolts on "explanation" heatmaps (Grad-CAM, saliency, etc.) — yet almost nobody checks whether those heatmaps are trustworthy, and the literature openly **contradicts itself** about the most-used method (Grad-CAM). This project resolves that contradiction with a standardized, cardiologist-anchored, cross-hospital benchmark.

It is a **benchmark of explanations**, not of models: one fixed, competent model; many xAI methods scored against three axes — **faithfulness, sanity, and clinical concordance** — plus a lightweight cardiologist reader study and an external-hospital validation.

---

## Repository map

```
ecg-xai-ami/
├── README.md                     ← you are here
├── docs/
│   ├── 01_PROPOSAL.md/.docx       full research proposal
│   ├── 02_EXECUTION_RUNBOOK.md    phase-by-phase how-to (start here to work)
│   ├── 03_DECISIONS_LOG.md        every locked-in decision + reasoning (source of truth)
│   ├── 04_DATA_LABELS_AND_SAMPLING.md   PTB-XL schema, 71→MI mapping, sample funnel
│   ├── 05_LITERATURE_AND_GAP.md   verified citations + the contradiction
│   ├── Review_Tracker.xlsx        team reading list + assignments
│   └── collaboration/             Dr. Kan Li meeting notes, transcript, slides
├── src/
│   ├── config.py                  central config: paths, label defs, folds, seed, masks
│   ├── utils.py                   seeding · device · attribution norm · bootstrap CIs
│   ├── data/                      load_ptbxl · build_cohort · preprocess
│   ├── models/                    xresnet1d (self-contained) · train (freeze + export)
│   ├── attribution/               generate (Grad-CAM/++ via hooks, IG/DeepLIFT/Saliency via Captum)
│   ├── metrics/                   faithfulness · sanity · concordance · agreement · evaluate (master table)
│   └── expert_rating/             render_panels (blinded maps for cardiologists)
├── tests/                         unit tests for the dependency-light core logic
├── data/                          (git-ignored) raw + processed data; see data/README.md
├── results/  figures/  notebooks/
├── environment.yml / requirements.txt
├── LICENSE                        MIT (code only; datasets keep their PhysioNet licenses)
└── .gitignore
```

Each phase has a runnable entry point:

```bash
python -m src.data.build_cohort        # Phase 1: cohort, data card, preprocessed arrays
python -m src.models.train             # Phase 2: train, freeze, export correct-MI cases
python -m src.attribution.generate     # Phase 3: attribution panel on the frozen model
python -m src.metrics.evaluate         # Phase 4: faithfulness/sanity/concordance/agreement table
python -m src.expert_rating.render_panels   # Phase 5: blinded reader-study panels
python -m pytest tests                 # core-logic tests (no GPU/data required)
```

---

## Quick start

```bash
# 1. environment
conda env create -f environment.yml && conda activate ecg-xai-ami
#   (or: pip install -r requirements.txt)

# 2. data — PTB-XL is open; MIMIC-IV-ECG needs PhysioNet credentialing (start early!)
#    see data/README.md for exact steps. Raw data is NEVER committed.

# 3. follow docs/02_EXECUTION_RUNBOOK.md, phase by phase
```

---

## The plan in one glance

| Phase | What | Aim |
|---|---|---|
| 0 | Setup, MIMIC credentialing, IRB submission, OSF pre-registration | — |
| 1 | Data prep: PTB-XL cohort (MI vs NORM), official folds, data card | 1 |
| 2 | Train + **freeze** xresnet1d101 MI detector | 1 |
| 3 | Generate attributions (Grad-CAM, saliency, IG, DeepLIFT, …) on correct-MI cases | 2 |
| 4 | Score explanations: faithfulness + sanity + concordance + agreement | 2 |
| 5 | Lightweight cardiologist reader study (~60 blinded panels) | 3 |
| 6 | External validation on MIMIC-IV-ECG (do rankings hold across hospitals?) | 4 |
| 7 | Analysis, TRIPOD+AI writeup, code release | — |

Realistic timeline **~4–6 months**; bottlenecks are MIMIC credentialing and IRB/cardiologist scheduling (both started in Phase 0).

---

## Non-negotiable guardrails

1. Official patient-stratified folds only — never a random split.
2. Freeze the model before any attribution work.
3. Pre-register metric definitions before opening the MIMIC set.
4. Keep MIMIC sealed until Phase 6.
5. Every result traces to the data card + a fixed seed.

---

## Team

Vân · Nhân · Dang Nguyen · Hoàng Tâm · Hùng · Tâm (project lead). Role split in `docs/03_DECISIONS_LOG.md`.

## Status

Planning complete; collaboration/IRB outreach to PI or co-PI and senior author. See `docs/collaboration/` and Decision D13.

## Data & ethics

Uses only public/credentialed de-identified ECG data (PTB-XL, MIMIC-IV-ECG). The cardiologist reader study requires IRB review (human subjects = the raters); dataset use does not. See Decision D13. Code is released under the MIT `LICENSE` (code only — the datasets keep their PhysioNet licenses and are never redistributed here).
