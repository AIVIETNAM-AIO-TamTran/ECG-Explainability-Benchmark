# Research Proposal

## Do ECG Deep-Learning Explanations Point Where Cardiologists Look? A Faithfulness-, Sanity-, and Concordance-Benchmarked Evaluation of Attribution Methods for Acute Myocardial Infarction Detection

**Principal investigator:** Tam Tran
**Draft version:** 1.0 · Working proposal
**Study type:** Retrospective, secondary analysis of public de-identified data + lightweight structured expert rating
**Data:** Fully public / credentialed public ECG repositories (no new patient recruitment)

---

## 1. Summary

Deep neural networks now match or exceed cardiologist-level performance in interpreting the 12-lead electrocardiogram (ECG), particularly for myocardial infarction (MI). Because these models are opaque, the field routinely bolts on a post-hoc "explanation" — most often Grad-CAM or a saliency map — to justify the prediction. The unexamined assumption is that these heatmaps show *where the model actually looked* and, ideally, that this coincides with where a cardiologist would look (the ST segment, pathological Q waves, the relevant anatomical leads).

That assumption is shaky, and the ECG literature is openly contradictory about it: at least one rigorous study found that Grad-CAM — the single most-used attribution method in ECG analysis — **fails basic sanity checks**, collapsing onto the high-amplitude QRS complex regardless of the diagnosis, while another benchmark reported Grad-CAM as the clear best performer. When two careful studies reach opposite conclusions, the field needs a standardized, disease-specific, externally validated adjudication rather than another one-off application.

This project trains a standard deep-learning MI detector on the gold-standard public PTB-XL dataset and then, holding the model fixed, benchmarks a panel of attribution methods along three axes that the current literature almost never combines: **(i) faithfulness** (does the highlighted evidence actually drive the prediction), **(ii) sanity** (does the explanation degrade appropriately when the model or data is randomized), and **(iii) clinical concordance** (does the explanation localize to the ECG segments and leads that define MI for a cardiologist). We add a **lightweight expert-rating layer** (2–3 cardiologists blindly scoring a sample of maps) and test whether the resulting method rankings **transfer to a second, independent hospital cohort** (MIMIC-IV-ECG). The deliverable is an evidence-based, reproducible, open-source answer to a practical question: *which explanation method should ECG-AI actually use for MI, and does it hold up across sites?*

---

## 2. Background and significance

### 2.1 Where deep learning genuinely wins — and where the black box is real
On structured/tabular clinical data, gradient-boosted decision trees remain the state of the art and typically outperform deep neural networks; deep learning's advantage does not reliably materialize there. The picture is different for **unstructured cardiac signals**. On raw 12-lead ECG waveforms, convolutional and convolutional-recurrent networks reach cardiologist-comparable accuracy — recent multi-centre AMI models report AUROC ≈ 0.85 under cross-hospital external validation. Critically, a CNN operating on a raw waveform is a *true* black box (unlike a tree ensemble, whose splits are at least inspectable), so the interpretability problem here is genuine rather than rhetorical. This is the correct setting in which to motivate explainable AI (xAI).

### 2.2 The state of xAI for ECG
Across ECG deep-learning studies, explainability is dominated by a small set of attribution methods — SHAP, saliency maps, Grad-CAM, and LIME — applied *post hoc* to check whether the model's attention aligns with clinically plausible morphology. Two problems recur throughout this literature:

1. **Explanation quality is rarely measured.** Papers report classifier metrics (AUROC, sensitivity, specificity) but seldom report quantitative faithfulness or stability of the *explanations themselves*. There is no accepted benchmark, and reviews explicitly call for standardized faithfulness/stability metrics, cardiologist validation of explanation maps, and public code release.
2. **The methods disagree, and some fail outright.** Saliency maps are visually immediate but unstable; Grad-CAM is intuitive but has been shown to attribute relevance to the QRS complex (the highest-amplitude region) rather than the diagnostically relevant segment. The ECG is unusually well-suited to exposing this because its periodic structure and well-defined features (P wave, QRS complex, ST segment) allow precise temporal and spatial sanity checks — an advantage this proposal exploits directly.

### 2.3 Why MI, and why now
MI is the highest-yield target for a concordance-based evaluation because, unlike diffuse or rhythm diagnoses, its ECG evidence is *spatially and temporally localizable by rule*: ST-segment deviation, pathological Q waves, and T-wave changes, appearing in the specific leads that map to a coronary territory. That gives us a defensible, physiology-grounded "where should the model look" reference against which any attribution map can be scored — something generic localization benchmarks lack. MI is also of direct translational importance: timely detection changes management, and a trustworthy explanation is a precondition for bedside and wearable-triage deployment.

---

## 3. Gap statement

The unfilled gap is not "nobody explains ECG models." It is that **the field has not standardized how to tell a trustworthy ECG explanation from a misleading one**, has not tied that judgment to *disease-specific* clinical evidence, and has not shown that explanation quality (not just accuracy) *generalizes across hospitals*. The published contradiction over Grad-CAM's reliability is a symptom of this gap. This proposal targets it directly.

---

## 4. Objectives and hypotheses

**Aim 1 — Build a credible, standard MI detector (not the contribution).**
Train and validate a deep-learning classifier for MI on PTB-XL, reporting standard discrimination metrics, so that all downstream explanation analysis rests on a competent model.

**Aim 2 — Benchmark attribution methods on three axes (the core contribution).**
For a fixed set of correctly classified MI cases, quantify each attribution method's **faithfulness**, **sanity**, and **clinical concordance**, and quantify **cross-method agreement**.
- *H2a:* Attribution methods will diverge substantially in faithfulness and concordance despite similar visual plausibility.
- *H2b:* At least one widely used method (Grad-CAM anticipated) will fail one or more sanity checks, replicating and extending prior findings in a disease-specific setting.

**Aim 3 — Anchor the benchmark to cardiologist judgment (lightweight).**
Have 2–3 cardiologists/electrophysiologists blindly rate a stratified sample of attribution maps for clinical plausibility and localization quality; correlate expert ratings with the automated concordance and faithfulness metrics.
- *H3:* Automated concordance will correlate positively with expert-rated plausibility, supporting concordance as a scalable proxy.

**Aim 4 — Test cross-site generalization of the explanations.**
Re-run the full benchmark on an independent hospital cohort (MIMIC-IV-ECG) and test whether the *method rankings* — not merely classifier accuracy — are preserved.
- *H4:* Faithful, high-concordance methods will retain their ranking across cohorts; unstable methods will not.

---

## 5. Methods

### 5.1 Data sources
- **Primary (training + internal test): PTB-XL v1.0.3.** 21,837 clinical 12-lead ECGs (10 s) from 18,885 patients, cardiologist-annotated with hierarchical SCP-ECG labels including a Myocardial Infarction (MI) superclass, provided at 100 Hz and 500 Hz with recommended patient-stratified 10-fold splits.
- **External validation: MIMIC-IV-ECG v1.0.** ~800,000 diagnostic 12-lead ECGs (10 s, 500 Hz) from ~160,000 patients at BIDMC, linkable to the MIMIC-IV clinical database and to cardiologist report text, enabling an independent MI-labeled test set from a different country, era, and acquisition system.
- **Optional secondary check: Chapman–Shaoxing–Ningbo (PhysioNet "ecg-arrhythmia" v1.0.0).** ~10,000+ patients, 12-lead, 500 Hz. Note: this cohort is *rhythm/arrhythmia-focused* rather than MI-labeled, so it is suitable only for robustness spot-checks of attribution behavior, not as a primary MI external set. (The PhysioNet/CinC Challenge 2020–2021 aggregated databases are an alternative source of additional MI-labeled ECGs if a second external cohort is desired.)

Links and access notes are in Section 11.

### 5.2 Task definition and labeling
Binary MI vs. normal, following the common high-label-quality convention: positive = records carrying MI (optionally MI + ST/T-change) superclass labels; negative = clearly normal (NORM) records; exclude ambiguous or co-morbid overlaps (e.g., MI with hypertrophy) and retain only cardiologist-validated, high-confidence labels. A secondary analysis may narrow to *acute* MI where subtype/timing metadata permit, to sharpen the concordance reference. Use the dataset's recommended patient-stratified splits (folds 1–8 train, 9 validation, 10 test) to prevent patient leakage.

### 5.3 Model (Aim 1)
Primary model: a 1D residual CNN (ResNet-style) on 100 Hz raw 12-lead input, a standard and reproducible baseline for PTB-XL MI detection. Secondary/optional: a convolutional-recurrent-attention model (e.g., Conv-BiLSTM-Attention) to permit an attention-based attribution comparison. Handle class imbalance with class weighting or focal loss. Report AUROC, AUPRC, sensitivity, specificity, and F1 at a Youden-optimized threshold with 95% CIs (bootstrap). The model is deliberately conventional; the novelty is downstream.

### 5.4 Attribution panel (Aim 2)
Applied to the *same* fixed model and the same set of correctly classified MI cases, implemented through a single open-source library (Captum) for reproducibility:
- Grad-CAM and Grad-CAM++
- Vanilla saliency (input gradients)
- Integrated Gradients
- DeepLIFT
- (If an attention model is used) attention weights, treated as a candidate explanation rather than assumed valid
- Optional model-agnostic references: KernelSHAP and/or LIME on a windowed representation, for cross-family comparison

### 5.5 Evaluation axes (Aim 2 core)

**(a) Faithfulness — does the highlighted evidence drive the prediction?**
Deletion/insertion (perturbation) curves: rank signal segments by attribution, progressively mask (deletion) or reveal (insertion) the top-ranked segments, and track the change in predicted MI probability. Summarize as area under the deletion/insertion curve. A faithful method causes a steep confidence drop when its top-ranked segments are removed.

**(b) Sanity — does the explanation know when it should break?**
Adebayo-style sanity checks adapted to ECG: (i) *model-parameter randomization* (cascading randomization of network weights) and (ii) *label/data randomization*. A trustworthy attribution should degrade toward noise as the model is randomized. Quantify degradation with rank correlation between original and randomized attributions. This is where prior work caught Grad-CAM's QRS-amplitude bias; we test it explicitly for MI.

**(c) Clinical concordance — does it look where MI lives?**
Define a physiology-grounded reference mask per case: the ST segment / J-point region and, where derivable, the anatomically relevant leads for the labeled MI territory. Delineate cardiac intervals using an open QRS/wave delineator (e.g., NeuroKit2) to locate P/QRS/ST windows. Score each attribution map by the fraction of its mass falling within the diagnostic window/leads versus elsewhere (a localization/concordance ratio), plus a beat-aggregated ("glocal") version to reduce single-beat gradient noise.

**(d) Cross-method agreement.**
Rank-correlation and top-k overlap between methods on the same cases, to quantify how much the "explanation" depends on the arbitrary choice of tool.

### 5.6 Expert rating (Aim 3, lightweight)
Draw a stratified random sample (~40–60 cases × selected methods, blinded and shuffled). Two to three cardiologists/EPs independently rate each map on short Likert items (e.g., clinical plausibility; localization quality; would-this-increase-my-trust). Report inter-rater agreement (Krippendorff's α / weighted κ) and correlate mean expert scores with the automated concordance and faithfulness metrics. This satisfies the field's stated need for cardiologist validation without the cost of a full reader/decision study.

### 5.7 External generalization (Aim 4)
Freeze the PTB-XL-trained model, apply it to the MIMIC-IV-ECG MI/normal test set, and recompute the entire attribution benchmark. The key readout is whether the *ordering* of methods by faithfulness/sanity/concordance is preserved across cohorts (rank-correlation of method rankings). Report the classifier's external AUROC as context, but the headline is explanation-ranking stability.

### 5.8 Statistical analysis
Bootstrap 95% CIs for all classifier and attribution metrics; paired comparisons across methods with appropriate correction for multiplicity (e.g., Holm–Bonferroni). Pre-register the metric definitions and primary hypotheses (e.g., on OSF) before external-cohort analysis to guard against post-hoc tuning.

### 5.9 Reproducibility
Public data only; single attribution library; fixed seeds; released code, trained weights, split indices, and the concordance-mask definitions. Report following TRIPOD-AI / applicable xAI reporting guidance.

---

## 6. Expected outcomes and contribution
1. A **standardized, disease-specific benchmark** for ECG-MI explanations combining faithfulness + sanity + clinical concordance — currently missing from the literature.
2. A **clear, adjudicated answer** to the Grad-CAM-vs-saliency contradiction for MI, with evidence rather than assertion.
3. Evidence on whether **automated concordance can proxy cardiologist judgment**, enabling scalable evaluation.
4. First-line evidence on whether **explanation quality generalizes across hospitals**, not just accuracy.
5. An **open, reusable toolkit** (code + masks + protocol) other groups can apply to new ECG tasks.

**Framing discipline (novelty guard):** the honest positioning is *"standardize and adjudicate explanation reliability for MI, with disease-specific concordance and cross-site transfer,"* not *"first to explain ECG."* Sanity-checking ECG attributions has partial prior art; the differentiating combination here is MI-specific diagnostic-region concordance + cardiologist rating + cross-dataset explanation-ranking stability, packaged as a benchmark.

---

## 7. Indicative timeline (part-time)
- **Months 1–2:** Data access/credentialing (MIMIC), pipeline, model training, internal metrics (Aim 1).
- **Months 2–4:** Attribution panel + faithfulness/sanity/concordance implementation (Aim 2).
- **Month 4:** Expert-rating design, IRB/ethics determination, cardiologist recruitment (Aim 3).
- **Months 4–5:** External cohort benchmark (Aim 4).
- **Months 5–6:** Analysis, expert rating collection, manuscript, code release.

---

## 8. Ethics and data governance
All datasets are de-identified and publicly released for research; no new patient contact. PTB-XL is open access; MIMIC-IV-ECG requires PhysioNet credentialing (CITI "Data or Specimens Only Research" training + a signed data use agreement). The expert-rating component uses clinicians rating de-identified model outputs and typically qualifies as non-human-subjects or exempt research, but a formal IRB determination will be obtained before that phase.

---

## 9. Limitations and mitigations
- *Concordance reference is a simplification.* The "correct" region for MI is rule-based, not ground truth about the model. Mitigation: treat concordance as one axis among three; corroborate with faithfulness and expert rating.
- *Label noise / heterogeneous MI definitions across cohorts.* Mitigation: high-confidence labels, explicit inclusion rules, sensitivity analyses.
- *Small expert panel.* Mitigation: keep expert rating as a supporting, not primary, endpoint; report agreement transparently.
- *Attention ≠ explanation.* Mitigation: attention treated as a candidate, evaluated by the same metrics.

---

## 10. Candidate venues
Methods/informatics: *Computers in Biology and Medicine*, *Computer Methods and Programs in Biomedicine*, *JAMIA*. Clinical-digital: *European Heart Journal – Digital Health*, *npj Digital Medicine*.

---

## 11. Data availability — links

**Primary training / internal test**
- **PTB-XL v1.0.3 (PhysioNet):** https://physionet.org/content/ptb-xl/1.0.3/
  DOI: https://doi.org/10.13026/kfzx-aw45 · Dataset paper: https://www.nature.com/articles/s41597-020-0495-6 · Open access (direct download / `wget`).

**External validation**
- **MIMIC-IV-ECG v1.0 (PhysioNet):** https://physionet.org/content/mimic-iv-ecg/1.0/
  Requires PhysioNet credentialed access + data use agreement.
- **MIMIC-IV-ECG Demo v0.1 (open access, for pipeline testing before credentialing):** https://physionet.org/content/mimic-iv-ecg-demo/0.1/

**Optional secondary / robustness check**
- **Chapman–Shaoxing–Ningbo, "A large scale 12-lead ECG database for arrhythmia study" v1.0.0 (PhysioNet):** https://physionet.org/content/ecg-arrhythmia/1.0.0/
  (Rhythm-focused; not MI-labeled — use for attribution-behavior spot-checks only.)
- **PhysioNet/CinC Challenge 2020 (aggregated 12-lead databases, incl. MI labels):** https://physionet.org/content/challenge-2020/1.0.2/

**Core tooling (open source)**
- Captum (attribution methods): https://captum.ai/
- NeuroKit2 (ECG delineation for concordance masks): https://github.com/neuropsychology/NeuroKit

---

## 12. Selected references
1. Wagner P, Strodthoff N, Bousseljot R-D, et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data. 2020;7:154. doi:10.1038/s41597-020-0495-6.
2. Gow B, Pollard T, Nathanson LA, et al. *MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset* (v1.0). PhysioNet. 2023.
3. Zheng J, Zhang J, Danioko S, et al. *A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients.* Scientific Data / PhysioNet. 2020.
4. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. *Sanity Checks for Saliency Maps.* NeurIPS. 2018.
5. Selvaraju RR, Cogswell M, Das A, et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV. 2017.
6. Lundberg SM, Lee S-I. *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS. 2017.
7. Ribeiro MT, Singh S, Guestrin C. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME). KDD. 2016.
8. Strodthoff N, et al. *Explaining deep learning for ECG analysis: Building blocks for auditing and knowledge discovery.* Computers in Biology and Medicine. 2024. (Sanity checks reveal Grad-CAM's temporal-attribution failure; saliency passes.)
9. *Evaluating Feature Attribution Methods for Electrocardiogram.* 2022. arXiv:2211.12702.
10. *Visual interpretation of deep learning model in ECG classification: a comprehensive evaluation of feature attribution methods.* Computers in Biology and Medicine. 2024.
11. *Explainable deep-learning-based techniques for ECG-based heart disease classification: a systematic literature review and future directions.* Computers in Biology and Medicine. 2025.
12. Mothilal RK, Sharma A, Tan C. *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations* (DiCE). FAccT. 2020. (For optional counterfactual extension.)

*Reference details should be verified and completed against the source records before submission.*
