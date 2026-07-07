# Project Decisions Log

**Project:** ECG xAI–AMI Benchmark — *"Do ECG Deep-Learning Explanations Point Where Cardiologists Look?"*
**Purpose of this file:** the single source of truth for every design decision that has been locked in during planning. If a choice was debated and settled, it is recorded here with the reasoning, so the team never re-litigates it and reviewers can see it was deliberate. Update this file whenever a decision changes.

---

## D1 — Research question and framing

**Decision:** The project asks *which post-hoc explanation (xAI) method is most trustworthy for deep-learning detection of myocardial infarction (MI) from the 12-lead ECG* — and whether that answer holds across hospitals. It is a **benchmark of explanations**, not a benchmark of model architectures and not a new-model paper.

**Framing correction (important):** The "deep learning beats old ML via dense layers" claim is only defensible for **unstructured** cardiac data (ECG waveforms, imaging), NOT for tabular clinical data (where gradient-boosted trees still win). Because we work on raw ECG signals, deep learning genuinely dominates AND a CNN on a waveform is a true black box — so the trust problem is real here, not borrowed. Keep the framing scoped to signals.

---

## D2 — The gap we exploit: the Grad-CAM contradiction (verified)

**Decision:** The headline motivation is a genuine, verified contradiction in the ECG-xAI literature about Grad-CAM.

- **Paper A** — Strodthoff et al., *Explaining Deep Learning for ECG Analysis* (arXiv:2305.17043; Comput Biol Med 2024): Grad-CAM, the most widely used method, **fails** the sanity check — it collapses onto the high-amplitude QRS complex regardless of diagnosis; saliency maps were the only method to pass.
- **Paper B** — *Evaluating Feature Attribution Methods for Electrocardiogram* (arXiv:2211.12702): Grad-CAM **outperforms** the second-best method by a large margin.

**Verified nuance:** the two papers differ in *task* and *metric family* (Paper A: sanity checks on 12-lead diagnostic ECG; Paper B: localization/pointing-game/degradation on arrhythmia beats, Icentia11K, ResNet-18). So part of the disagreement may be a task/metric artifact — which is *why* our controlled re-test (fixed task, fixed dataset, unified metrics) is a real contribution, not a me-too.

---

## D3 — Datasets: two, not four

**Decision:**
| Role | Dataset | Use |
|---|---|---|
| Primary | **PTB-XL** | Train + internal test (all model development) |
| External | **MIMIC-IV-ECG** | Independent validation only; model frozen, never trained here |
| Backup only | Chapman-Shaoxing, PhysioNet/CinC-2020 | Only if MIMIC access is delayed, or as an optional 2nd external check |

**Reasoning:** One clean external set is enough to support the cross-hospital claim. Running on all datasets adds label-heterogeneity noise without scientific gain. Mirrors the Strodthoff PTB-XL benchmark (IEEE JBHI 2021, doc 9190034), which anchors on PTB-XL.

**Links:** PTB-XL https://physionet.org/content/ptb-xl/1.0.3/ · MIMIC-IV-ECG https://physionet.org/content/mimic-iv-ecg/1.0/ (credentialed) · demo https://physionet.org/content/mimic-iv-ecg-demo/0.1/

---

## D4 — Task definition: binary MI vs. NORM

**Decision:** Fixed, single task = **binary classification, MI (superclass) vs. NORM**.

- We do **NOT** use the six-experiment structure from the Strodthoff benchmark (`all`, `diagnostic`, `subdiagnostic`, `superdiagnostic`, `form`, `rhythm`). We borrow only the **superclass label-mapping** utility from the `superdiagnostic` experiment, then reduce to binary.
- We do **NOT** model the 71 raw SCP codes as 71 classes.
- Exclude ambiguous / co-morbid records (MI co-occurring with another superclass) from the concordance analysis.

**Writeup sentence:** "We adopt PTB-XL's diagnostic superclass mapping to define a binary MI-vs-NORM task, rather than the full multi-task benchmark structure of Strodthoff et al., since our objective is to evaluate explanation methods on a single, clinically well-defined target."

---

## D5 — Model: xresnet1d101 (one model, many xAI methods)

**Decision:** Primary model = **`xresnet1d101`** — a 101-layer ResNet adapted for **1D** ECG signals (NOT the 2D image ResNet-101).

- **Lighter fallback:** `xresnet1d50` if GPU/compute is tight — the literature treats its performance as compatible with xresnet1d101 within error bars, so this substitution is defensible.
- **One frozen model, multiple xAI methods** applied to it — NOT multiple models. Reasoning: our question is "which xAI method," so varying the model would confound it. This matches how attribution-method papers (DeepLIFT, Grad-CAM, IG) are evaluated.
- **Why not a Transformer?** Grad-CAM requires convolutional feature maps; a CNN keeps the full method panel available and stays aligned with the two contradiction papers (both signal-based CNNs). Have this answer ready for reviewers/collaborators.

**Sources:** Strodthoff PTB-XL benchmark (arXiv:2004.13701 / IEEE JBHI 2021) — xresnet1d101 was the best/reference model. Still in active use 2024–2025 (medRxiv replication 2025.01.27.25321112; PMC12088039 transfer-learning study). Precedent for xAI-on-xresnet family: arXiv:2310.07463 (saliency on XResNet50 for ECG aging). No prior work does a multi-method attribution benchmark on xresnet1d101 for MI — that gap is our novelty.

---

## D6 — xAI method panel

**Decision:** Apply, via Captum, to the same frozen model on the same cases:
**Grad-CAM, Grad-CAM++, Saliency (input gradients), Integrated Gradients, DeepLIFT** (add attention only if an attention model is used). Optionally KernelSHAP/LIME on windows for cross-family comparison.

---

## D7 — Three evaluation axes + agreement (the novel core)

**Decision:** Score the *explanations*, not just the classifier. Use Quantus to standardize where possible.

1. **Faithfulness** — deletion/insertion curves (RISE-style) + AOPC. Does removing top-attributed segments collapse the prediction?
2. **Sanity** — Adebayo-style model-parameter and data randomization. Does the attribution degrade toward noise? (Where Grad-CAM historically fails.)
3. **Clinical concordance** — does attribution land on the ECG evidence a cardiologist uses for MI? (See D8.)
4. **Cross-method agreement** — rank-correlation / top-k overlap across methods.

**Confound-control note:** axes 1–3 deliberately reproduce the logic of *both* contradiction papers (sanity-style + localization/degradation-style) under one fixed task and dataset. State this explicitly in the Discussion.

---

## D8 — Clinical concordance metric

**Decision:**
```
C(m,i) = ( Σ |attribution| inside diagnostic mask ) / ( Σ |attribution| over all leads × timesteps )
```
- Use **NeuroKit2** to delineate P / QRS / ST per beat, per lead.
- Diagnostic mask = ST-segment / J-point window + the leads implicating the labeled MI territory.
- Also compute **lead-concordance** (attribution mass in territory-relevant leads).
- Aggregate over beats to reduce single-beat gradient noise.

**Territory → lead mapping** (from the specific MI subtype code, retained even though the task is binary):
| MI subtype code | Territory | Expected leads |
|---|---|---|
| AMI, ASMI, ALMI | Anterior / anteroseptal | V1–V4 |
| IMI, ILMI, IPMI, IPLMI | Inferior | II, III, aVF |
| LMI | Lateral | I, aVL, V5–V6 |

---

## D9 — Labeling: collapse 71 → MI superclass, retain subtype

**Decision:**
- Parse per-record `scp_codes` (stringified dict → `ast.literal_eval`), join to `scp_statements.csv` to get `diagnostic_class` (superclass).
- **Positive** = any code mapping to superclass **MI**; **Negative** = **NORM** only.
- **Retain the specific MI code** per positive case (needed for D8 lead-concordance).
- **MI superclass codes:** IMI, ASMI, ILMI, AMI, ALMI, INJAS, LMI, INJAL, IPLMI, IPMI, INJIN, PMI, INJLA, INJIL.
- Drop tiny/borderline injury codes (INJIL 15, INJLA 17, PMI 17) from per-subtype analysis.

**Open decision — strict vs. inclusive (LEAN: strict):**
- **Strict:** MI likelihood = 100 and pure-NORM. Cleaner "dog," more defensible for a trustworthiness benchmark, smaller n. **Preferred**, unless the strict test-fold pool drops too low.
- **Inclusive:** any MI code at any likelihood. Larger n, noisier labels. Fallback.
- **Action:** compute BOTH counts in the Phase 1 script and record in the data card before committing.

---

## D10 — Splits and preprocessing

**Decision:**
- Use PTB-XL's **official patient-stratified folds**: train = folds 1–8, validation = fold 9, test = fold 10. Never invent a random split.
- Preprocessing: **100 Hz** version (1000 samples × 12 leads), per-lead z-score, handle NaNs, standard lead order. Fix once, apply identically everywhere (incl. MIMIC).

---

## D11 — Sample funnel (estimates until real parse)

**Decision / expectation** (these are *estimates* — real numbers come from the Phase 1 parse):
| Stage | Approx. count |
|---|---|
| MI records (superclass, whole dataset) | ~5,469 (vs. ~9,514 NORM) |
| Strict, clean MI (likelihood=100, non-co-morbid) | ~3,500 |
| MI in test fold (fold 10 ≈ 10%) | ~355 |
| **Correctly-classified MI (sensitivity ~0.80–0.88)** | **~300–350** |

**Two distinct "how many" numbers — do not confuse:**
- **Automated metrics** (faithfulness/sanity/concordance) run on ALL ~300–350 cases × each method (~1,500–1,750 maps). Statistical backbone.
- **Cardiologist rating** uses a SMALL blinded sample: **~40–60 panels** (e.g. 15 cases × 4 methods). NOT all cases, and NOT × the 6 experiments.

**Fallbacks if strict test pool < ~150:** widen to inclusive definition, or evaluate on validation+test folds (9–10) — never touch training folds.

---

## D12 — Expert rating (Phase 5) design

**Decision:**
- **The maps are self-generated in Phase 3**, not downloaded. Each = one PTB-XL ECG + one method's attribution overlay. Experts never see the metadata table.
- Sample ~40–60 panels (cases × methods), **blinded and shuffled**; ID→method key kept in a separate file raters never see.
- Short Likert form: clinical plausibility; localization quality; lead correctness; "would this increase my trust?"; optional free text. Google Forms or REDCap.
- 2–3 cardiologists/EPs rate **independently**. Pilot 3–5 panels first.
- Analyze: inter-rater agreement (Krippendorff α / ICC) + correlate expert scores with automated concordance/faithfulness. Positive correlation ⇒ automated metric is a valid scalable proxy.

---

## D13 — Ethics / IRB

**Decision:**
- **Datasets need NO new IRB.** PTB-XL was ethics-approved for open release; MIMIC-IV-ECG collection was IRB-reviewed at BIDMC with a consent waiver — downstream use only needs PhysioNet credentialing (CITI "Data or Specimens Only Research" + DUA). Still obtain a written **exempt / non-human-subjects determination** letter from an institution for the manuscript's ethics statement.
- **The expert-rating step DOES need IRB review** — the cardiologists are human subjects (their judgments are the data), even though the ECGs are public/de-identified. Almost certainly **exempt/expedited**. Do NOT self-declare; the IRB must issue the determination.
- **Path (chosen):** Option 2 — a WashU faculty collaborator serves as PI/co-PI (since submitting as PI at a prior institution isn't available).
- **Collaboration status:** Outreach to **Dr. Kan Li** (kanl@wustl.edu — Professor of Medicine/Cardiology, Director of Echocardiography Lab; CNN-based cardiac imaging researcher). Meeting scheduled. Backup channel: AIHealth Institute (aihealth@wustl.edu). See `docs/collaboration/`.
- **Submit IRB in Phase 0, in parallel with MIMIC credentialing** — both are calendar-bound (2–6 weeks) and outside our control.

---

## D14 — Timeline

**Decision:** ~**4–6 months** realistic part-time (not the optimistic 10 weeks). Two calendar bottlenecks, both external: (1) MIMIC credentialing, (2) IRB determination + cardiologist scheduling. Mitigation: start both in Phase 0; run the expert-rating track in parallel with Phases 2–4.

---

## D15 — Guardrails (non-negotiable for credibility)

1. Use the official patient-stratified folds — never a random split.
2. Freeze the model before any attribution work; never retrain to "improve" an explanation.
3. Pre-register the task, metric definitions, and hypotheses (OSF) **before** opening the MIMIC set.
4. Keep the MIMIC external set **sealed** until Phase 6.
5. Every result traces back to the data card and a fixed random seed.

---

## D16 — Reporting and venues

**Decision:** Write per **TRIPOD+AI**. Discussion resolves the Grad-CAM contradiction as a *controlled re-test* and states which method(s) the field should use for ECG-MI. Release code, frozen weights, concordance masks, split indices; finalize OSF.
**Target venues:** Computers in Biology and Medicine, Computer Methods and Programs in Biomedicine, JAMIA, European Heart Journal – Digital Health, npj Digital Medicine.

---

## Team & role split

- **Nhân** — environment + MIMIC credentialing; PTB-XL loading; model training.
- **Hùng** — MIMIC extraction; attribution pipeline (Captum); faithfulness/sanity (Quantus).
- **Vân** — label QC; concordance metric (NeuroKit2 + masks); expert-rating figures.
- **Tâm** — cohort rules; OSF pre-registration; IRB + cardiologist recruitment; statistics; writing lead.

---

*Last updated: during planning phase. Revise entries here (not in scattered notes) whenever a decision changes.*
