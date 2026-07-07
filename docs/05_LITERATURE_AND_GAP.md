# Literature and Gap Reference

Verified sources underpinning the project. Every claim here was checked against the source during planning. See `docs/Review_Tracker.xlsx` for the full team reading list and assignments.

---

## The core contradiction (project motivation)

**Paper A — Grad-CAM FAILS on ECG.**
Strodthoff N., et al. *Explaining Deep Learning for ECG Analysis: Building Blocks for Auditing and Knowledge Discovery.* arXiv:2305.17043; Computers in Biology and Medicine, 2024.
→ Verified quote of finding: GradCAM, the most widely used attribution method in ECG analysis, fails to satisfy the sanity check; saliency maps were the only method to pass, showing high temporal specificity. Three of four methods show a strong bias toward the QRS complex (highest amplitude).
- https://arxiv.org/abs/2305.17043 · https://www.sciencedirect.com/science/article/pii/S0010482524006097

**Paper B — Grad-CAM is BEST on ECG.**
*Evaluating Feature Attribution Methods for Electrocardiogram.* arXiv:2211.12702.
→ Verified quote of finding: using localization / pointing-game / degradation metrics on 11 methods, Grad-CAM outperforms the second-best method by a large margin.
- https://arxiv.org/abs/2211.12702

**Verified nuance (why this isn't already resolved):** Paper A uses sanity/parameter-randomization checks on 12-lead diagnostic ECG; Paper B uses localization/degradation metrics on arrhythmia beats (Icentia11K, ResNet-18). Different task + different metric family → the disagreement may be a study-design artifact. Our controlled re-test (fixed task = MI vs NORM, fixed dataset = PTB-XL, unified metric suite) is what adjudicates it.

---

## Dataset and model provenance

**PTB-XL dataset.**
Wagner P., et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 2020;7:154. — 21,837 records, 18,885 patients, cardiologist-annotated, hierarchical SCP superclasses incl. MI.
- https://doi.org/10.1038/s41597-020-0495-6 · https://physionet.org/content/ptb-xl/1.0.3/

**Model + benchmark (the paper you linked, doc 9190034).**
Strodthoff N., et al. *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.* arXiv:2004.13701; IEEE JBHI 2021.
→ Verified: resnet/inception 1D-CNNs strongest; a newly proposed variant **xresnet1d101** showed the best performance; authors used xresnet1d101 as their reference model for deeper analysis.
- https://arxiv.org/abs/2004.13701 · code: https://github.com/helme/ecg_ptbxl_benchmarking

**xresnet1d101 still current (2024–2025).**
- Replication: *[Re] Deep Learning for ECG Analysis.* medRxiv 2025.01.27.25321112 — re-ran the benchmark incl. `fastai_xresnet1d101` under added-noise robustness. (Preprint; confirm peer-review status before citing as peer-reviewed.) https://www.medrxiv.org/content/10.1101/2025.01.27.25321112v1.full
- Transfer learning: *Transfer learning in ECG diagnosis: Is it effective?* (accepted Mar 2025) — builds on pretraining xresnet1d101 on PTB-XL. https://pmc.ncbi.nlm.nih.gov/articles/PMC12088039/
- 50-vs-101 interchangeability: self-supervised ECG paper notes xresnet1d50 performance is compatible with best-performing xresnet1d101 within error bars, while more parameter-efficient. (Basis for the lighter-fallback decision.)

**Precedent for xAI on the xresnet family (closest to our method).**
*Using explainable AI to investigate ECG changes during healthy aging.* arXiv:2310.07463 — saliency analysis on **XResNet50** localizing to P/QRS/T regions. Same lineage/approach; uses depth-50, not 101. No prior work does a multi-method attribution *benchmark* on xresnet1d101 for MI → our novelty.

---

## Evaluation methodology (metric design)

- **Sanity checks:** Adebayo J., et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018. arXiv:1810.03292.
- **Faithfulness — deletion/insertion:** Petsiuk V., et al. *RISE.* BMVC 2018. arXiv:1806.07421. https://github.com/eclique/RISE
- **Faithfulness — remove-and-retrain:** Hooker S., et al. *ROAR.* NeurIPS 2019. arXiv:1806.10758.
- **Faithfulness — AOPC/perturbation:** Samek W., et al. IEEE TNNLS 2017. arXiv:1509.06321.
- **Disagreement framing:** Krishna S., et al. *The Disagreement Problem in Explainable ML.* arXiv:2202.01602.
- **Eval toolkit (main engine):** Hedström A., et al. *Quantus.* JMLR 2023. arXiv:2202.06861. https://github.com/understandable-machine-intelligence-lab/Quantus
- **Benchmark scaffold:** Agarwal C., et al. *OpenXAI.* NeurIPS D&B 2022. arXiv:2206.11104. https://github.com/AI4LIFE-GROUP/OpenXAI
- **ECG delineation (concordance masks):** NeuroKit2. https://github.com/neuropsychology/NeuroKit
- **Attribution library:** Captum. https://captum.ai/

## Reporting / clinical framing
- **TRIPOD+AI:** Collins GS., et al. BMJ 2024. https://doi.org/10.1136/bmj-2023-078378
- **Interpretable-vs-explain counter-argument (for Discussion):** Rudin C. *Stop explaining black box ML models…* Nature Machine Intelligence 2019. arXiv:1811.10154.

---

## The gap in one sentence

The field explains ECG models but almost never *evaluates whether the explanations are trustworthy*; no standardized, disease-specific, externally-validated benchmark exists — and the one method everyone uses (Grad-CAM) has contradictory reliability evidence. This project fills exactly that gap for MI.
