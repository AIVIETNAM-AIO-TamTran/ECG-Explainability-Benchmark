# Execution Runbook — ECG xAI–AMI Benchmark

**Companion to:** *"Do ECG Deep-Learning Explanations Point Where Cardiologists Look?"* (proposal)
**Purpose:** a concrete, step-by-step process so the team (Vân, Nhân, Hùng, Tâm) can start working immediately.
**Team:** Vân, Nhân, Hùng, Tâm

---

## 0. Dataset decision (read first)

We use **two datasets**, not all four in the reading list:

| Role | Dataset | Use |
|---|---|---|
| Primary | **PTB-XL** | Train + internal test (all model development happens here) |
| External | **MIMIC-IV-ECG** | Independent validation only — the model is frozen, never trained here |
| Backup only | Chapman-Shaoxing, PhysioNet/CinC-2020 | Used *only* if MIMIC access is delayed, or as an optional 2nd external check |

This mirrors the Strodthoff PTB-XL benchmark (IEEE JBHI 2021, doc 9190034), which anchors on PTB-XL. One clean external set is enough to support the central claim; running on everything adds label-heterogeneity noise without scientific gain.

**Task definition (fixed for the whole project):** binary classification, **MI vs. Normal (NORM)**. Exclude ambiguous / co-morbid records (e.g., MI + hypertrophy) from the concordance analysis.

---

## Overview of phases

| Phase | What | Aim | Lead | Rough time |
|---|---|---|---|---|
| 0 | Setup & access | — | Tâm / Nhân | Week 0 |
| 1 | Data preparation | Aim 1 | Nhân / Hùng | Weeks 1–2 |
| 2 | Model training | Aim 1 | Nhân / Vân | Weeks 2–3 |
| 3 | Attribution generation | Aim 2 | Hùng | Weeks 3–4 |
| 4 | Explanation-quality metrics (novel core) | Aim 2 | Hùng / Vân / Nhân | Weeks 4–6 |
| 5 | Expert rating (lightweight) | Aim 3 | Tâm / Vân | Weeks 5–7 (parallel) |
| 6 | External validation | Aim 4 | Hùng / Nhân | Weeks 6–8 |
| 7 | Analysis, writing, code release | all | Tâm | Weeks 8–10 |

---

## Phase 0 — Project setup (Week 0)

**Goal:** everyone can run code and access data.

1. **Repo & structure.** Create a private GitHub repo `ecg-xai-ami`. Standard layout:
   ```
   data/  src/  models/  attributions/  results/  figures/  notebooks/  docs/
   ```
2. **Environment.** Python 3.10. Install: `torch`, `wfdb`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `captum`, `quantus`, `neurokit2`, `matplotlib`, `tqdm`. Commit an `environment.yml` so results are reproducible.
3. **Data access.**
   - PTB-XL: open — no account needed.
   - MIMIC-IV-ECG: **start credentialing now** (PhysioNet account → CITI "Data or Specimens Only Research" training → sign the data use agreement). This takes ~1–2 weeks, so do it in Week 0. Use the open **demo** set to build the pipeline meanwhile.
4. **Pre-registration.** Draft an OSF pre-registration listing the fixed task, the metric definitions (Phase 4), and the primary hypotheses **before** touching the external set. This is what makes the benchmark credible.
5. **Tracking.** Pick one experiment log (Weights & Biases or a shared CSV). Agree on naming conventions.

**Owners:** Tâm (repo, OSF), Nhân (environment, credentialing), all (accounts).
**Done when:** repo skeleton + environment build on every machine; OSF draft exists; MIMIC credentialing submitted.

---

## Phase 1 — Data preparation (Weeks 1–2) · Aim 1

**Goal:** clean, documented MI-vs-normal arrays with patient-safe splits.

1. **Download PTB-XL** (open access):
   ```
   wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
   ```
2. **Load metadata.** Read `ptbxl_database.csv`; map each record's `scp_codes` to diagnostic **superclasses** using `scp_statements.csv`. Keep the provided `strat_fold` column.
3. **Define the cohort.**
   - Positive = records carrying the **MI** superclass (use the diagnostic-likelihood field to keep high-confidence labels).
   - Negative = **NORM** only.
   - Exclude records that are both MI and another disease superclass **from the concordance subset** (they blur the "where should the model look" reference).
4. **Splits (do NOT invent your own).** Use the official patient-stratified folds: **train = folds 1–8, validation = fold 9, test = fold 10.** This prevents the same patient appearing in train and test.
5. **Preprocessing (fix once, apply everywhere).**
   - Use the **100 Hz** version (1000 samples × 12 leads per record).
   - Per-lead z-score normalization; handle any NaNs; keep leads in the standard order.
   - Save as arrays (`X_train.npy`, `y_train.npy`, …) plus an index mapping back to `ecg_id`.
6. **Data card.** Write `docs/data_card.md`: number of MI vs NORM per split, class balance, exact inclusion/exclusion rules, preprocessing settings. Every later result must be traceable to this.
7. **MIMIC-IV-ECG (once credentialed).** Extract MI-vs-normal cases (via linked diagnoses / cardiologist-report text), apply the **identical** preprocessing. Keep this set sealed until Phase 6.

**Owners:** Nhân (PTB-XL pipeline), Hùng (MIMIC pipeline), Vân (label QC), Tâm (sign off cohort rules).
**Done when:** PTB-XL arrays + data card committed; MIMIC extraction script ready (even if data pending).

---

## Phase 2 — Model training (Weeks 2–3) · Aim 1

**Goal:** one competent, frozen MI detector. The model is *not* the contribution — it just has to be solid.

1. **Model.** Implement a **1D ResNet** (e.g., `resnet1d_wang` / `xresnet1d101`). You can reuse the architecture from the `ecg_ptbxl_benchmarking` repo rather than writing from scratch.
2. **Train** MI-vs-normal (binary). BCE loss with class weighting or focal loss (MI is the minority). AdamW, LR scheduling, early stopping on **validation AUROC**.
3. **Optional baselines (nice to have).** Logistic regression on engineered ECG features, as an interpretable reference point. Skip the tabular deep net unless time allows.
4. **Evaluate on test fold 10:** AUROC, AUPRC, sensitivity, specificity, F1 at the Youden-optimal threshold, each with **bootstrap 95% CIs**. Check calibration.
5. **Freeze.** Save the final weights. Export the list of **correctly-classified MI test cases** — this exact set is the input to all attribution work in Phase 3–4.

**Owners:** Nhân (model), Vân (baselines + metric reporting), Tâm (review).
**Done when:** frozen checkpoint + metrics table + `correct_MI_cases.csv` committed.

---

## Phase 3 — Attribution generation (Weeks 3–4) · Aim 2

**Goal:** produce explanation maps from every method, on the same cases, from the same frozen model.

1. **Wire up Captum** on the frozen model.
2. For each correctly-classified MI case, compute attributions with the panel:
   **Grad-CAM, Grad-CAM++, Saliency (input gradients), Integrated Gradients, DeepLIFT** (add **attention** only if you used an attention model).
3. Save each as a per-lead × per-timestep array (12 × 1000) with the case id and method name.
4. Normalize/standardize attributions so methods are comparable in Phase 4.

**Owners:** Hùng (attribution pipeline), Nhân (model hooks).
**Done when:** `attributions/{method}/{case_id}.npy` exist for all methods and cases.

---

## Phase 4 — Explanation-quality metrics (Weeks 4–6) · Aim 2 — the novel core

**Goal:** quantify *how good each explanation is*, not just how the classifier performs. Three axes + agreement. Use **Quantus** to standardize where possible.

### 4.1 Faithfulness — "does the highlighted evidence drive the prediction?"
- Rank signal segments by attribution; progressively **mask top-ranked** segments (deletion) and **reveal** them (insertion); track the change in predicted MI probability.
- Summaries: area under the **deletion / insertion** curves (RISE-style) and the **AOPC** perturbation curve. A faithful method causes a steep confidence drop when its top segments are removed.

### 4.2 Sanity — "does the explanation break when it should?"
- **Model-parameter randomization** (cascading randomization of network weights) and **label/data randomization** (Adebayo et al.).
- Measure rank-correlation between the original and the randomized attribution. A trustworthy method degrades toward noise. **Watch for Grad-CAM collapsing onto the QRS complex regardless** — that is the known failure this project adjudicates.

### 4.3 Clinical concordance — "does it look where MI lives?" (differentiator)
- Use **NeuroKit2** to delineate **P wave / QRS complex / ST segment** per beat, per lead.
- Build the **MI diagnostic-region mask**: the ST-segment / J-point window and, where derivable, the leads implicating the labeled MI territory.
- **Concordance ratio** for method *m* on case *i*:

  `C(m,i) = (Σ |attribution| inside diagnostic mask) / (Σ |attribution| over all leads × timesteps)`

  Higher C = the explanation concentrates on the ECG evidence a cardiologist actually uses. Also compute a **lead-concordance** (fraction of attribution mass in the territory-relevant leads). Aggregate over beats to reduce single-beat gradient noise.

### 4.4 Cross-method agreement
- Rank-correlation and top-k overlap between methods on the same cases — quantifies how much the "explanation" depends on the arbitrary choice of tool.

**Owners:** Hùng (4.1 faithfulness + 4.2 sanity via Quantus), Vân (4.3 concordance + NeuroKit2), Nhân (4.4 agreement), Tâm (metric definitions, statistics).
**Done when:** one master table — method × {faithfulness, sanity, concordance, agreement} with 95% CIs — plus a ranked verdict on PTB-XL.

> **Note on the Grad-CAM contradiction this design resolves.** The two prior papers disagreeing on Grad-CAM (Strodthoff et al., sanity checks on 12-lead diagnostic ECG; arXiv:2211.12702, localization/pointing-game/degradation on arrhythmia beats) differ in *both* task and metric family, not just in their conclusion. Axes 4.1–4.3 above are deliberately built to reproduce the logic of both prior approaches — sanity-style (4.2) and localization/degradation-style (4.1, 4.3) — under one fixed task (MI vs. Normal) and one fixed dataset (PTB-XL). This is what lets the project claim it's controlling for the confound rather than adding a third, incomparable data point. State this explicitly in the manuscript's Discussion (see Phase 7).

---

## Phase 5 — Expert rating (Weeks 5–7, runs in parallel) · Aim 3 (lightweight)

**Goal:** anchor the automated metrics to cardiologist judgment, cheaply.

1. **Ethics first.** Obtain an IRB/ethics determination (rating de-identified model outputs is usually exempt/non-human-subjects, but confirm before collecting).
2. **Sample.** Draw ~40–60 cases × selected methods; **blind and shuffle** (raters must not know which method produced which map).
3. **Render.** Overlay each attribution on the 12-lead tracing as a clean figure.
4. **Form.** Short Likert items: clinical plausibility; localization quality; "would this increase my trust?" (Google Forms or REDCap).
5. **Collect.** 2–3 cardiologists / EPs rate independently.
6. **Analyze.** Inter-rater agreement (Krippendorff α / weighted κ); correlate mean expert score with the automated concordance and faithfulness from Phase 4. Positive correlation = concordance is a valid scalable proxy.

**Owners:** Tâm (cardiologist recruitment + IRB), Vân (rendering + form), Nhân (stats).
**Done when:** ratings dataset + agreement stats + correlation with automated metrics.

---

## Phase 6 — External validation (Weeks 6–8) · Aim 4

**Goal:** test whether the explanation *rankings* transfer to a different hospital — the headline generalization result.

1. **Apply the frozen PTB-XL model** to the MIMIC-IV-ECG MI/normal test set; report external AUROC (context, not the headline).
2. **Re-run the entire attribution + metric pipeline** (Phases 3–4) on the MIMIC cases.
3. **Compare method rankings** PTB-XL vs MIMIC (rank-correlation of the method orderings). Faithful, high-concordance methods should keep their rank; unstable ones should not.

**Owners:** Hùng (external run), Nhân (ranking comparison), Tâm (interpretation).
**Done when:** cross-site ranking comparison table/figure.

---

## Phase 7 — Analysis, writing, release (Weeks 8–10)

1. **Figures:** deletion/insertion curves; sanity-degradation plots; concordance heatmaps on example ECGs; the master ranking table; cross-site ranking comparison.
2. **Statistics:** paired method comparisons with multiplicity correction (Holm–Bonferroni); all CIs by bootstrap.
3. **Write** per **TRIPOD+AI**; the Discussion resolves the Grad-CAM-vs-saliency contradiction and states which method(s) the field should use for ECG-MI. Frame this explicitly as a **controlled re-test**: prior disagreement (Strodthoff et al. vs. arXiv:2211.12702) arose partly from different tasks/datasets/metrics; this project fixes all three and reports whether the disagreement persists.
4. **Release:** clean code, frozen weights, concordance masks, split indices; finalize OSF. Target venue per proposal (Comput Biol Med / CMPB / JAMIA / EHJ-Digital Health).

**Owners:** all; Tâm leads writing.
**Done when:** manuscript draft + public reproducible repo.

---

## Quick per-person starting point

- **Nhân** — Phase 0 environment + MIMIC credentialing; then PTB-XL loading + model training.
- **Hùng** — MIMIC extraction script; then attribution pipeline (Captum) + faithfulness/sanity (Quantus).
- **Vân** — label QC; then the concordance metric (NeuroKit2 delineation + mask) and the expert-rating figures.
- **Tâm** — cohort rules, OSF pre-registration, IRB + cardiologist recruitment, statistics, writing lead.

## Guardrails (do these or the benchmark is not credible)
1. Use the **official patient-stratified folds** — never a random split.
2. **Freeze the model** before any attribution work; never retrain to "improve" an explanation.
3. **Pre-register metric definitions** before opening the MIMIC set.
4. Keep the MIMIC external set **sealed** until Phase 6.
5. Every result traces back to the **data card** and a fixed random seed.
