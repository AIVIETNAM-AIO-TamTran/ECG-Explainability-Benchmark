# Zoom Meeting Notes — Dr. Kan Li
## ECG xAI–AMI Benchmark Project · Collaboration Discussion

**Goal for this meeting:** (1) get Dr. Li genuinely interested in the science, (2) confirm whether he'll serve as PI/co-PI for the IRB submission, (3) clarify his desired level of involvement (PI only vs. also a rater/co-author), (4) leave with a concrete next step and timeline.

---

## 0. Opening (1–2 min)

- Thank him for making time, acknowledge his echo/CNN imaging work as the reason you reached out.
- One-line framing so he knows where this is going before the details:
  > "This is a benchmark study asking whether the 'explanations' deep learning models give for ECG diagnoses actually point to the same evidence a cardiologist would use — and I'm hoping to find a WashU collaborator who could help sponsor this through IRB and maybe weigh in on the clinical side."
- Ask permission to walk through a short overview, then leave plenty of room for his questions/reactions.

---

## 1. The problem, in plain terms (2–3 min)

Say this roughly as prose, not read verbatim:

- Deep learning models now read ECGs about as well as cardiologists for things like MI — but they're black boxes.
- To make them trustworthy, people bolt on "explanation" tools (Grad-CAM, saliency maps, SHAP) that produce a heatmap showing what the model supposedly focused on.
- The problem: **nobody checks whether these heatmaps are actually correct.** Two papers in this space have looked at the same question and reached opposite conclusions about whether Grad-CAM is reliable on ECG — one found it fails basic sanity checks (it just lights up the QRS complex regardless of diagnosis), another found it's the best method.
- That contradiction is the gap. This project resolves it with a standardized benchmark, tied specifically to what a cardiologist would consider correct evidence for MI (ST-segment, Q-waves, territory-specific leads).

**Anticipate this question:** *"Why does this matter clinically?"*
> Answer: if hospitals or vendors deploy ECG-AI with an "explanation" feature to build clinician trust, and the explanation is systematically pointing at the wrong thing, that's worse than no explanation — it's false reassurance. This work is about establishing which explanation methods can actually be trusted before that kind of deployment.

---

## 2. What the project actually does (3–4 min)

Frame it as four stages, matching the proposal:

1. **Train a standard model** — a 1D CNN on the public PTB-XL dataset (21,837 ECGs) to detect MI vs. normal. Nothing novel here — it just needs to be competent.
2. **Generate explanations** — apply the standard toolkit of attribution methods (Grad-CAM, saliency, Integrated Gradients, etc.) to the model's correct MI predictions.
3. **Score the explanations, not just the model** — this is the actual contribution:
   - *Faithfulness*: does removing the highlighted ECG region actually change the prediction?
   - *Sanity*: does the explanation degrade appropriately when the model is randomized? (This is where Grad-CAM has failed in prior work.)
   - *Clinical concordance*: does the highlighted region land on the ST-segment / Q-wave / correct leads — the actual diagnostic evidence for MI?
4. **Validate with cardiologists and a second hospital** — a small panel (2–3 cardiologists) blindly rates a sample of ~40–60 maps, and the whole benchmark is re-run on an independent dataset (MIMIC-IV-ECG) to see if the rankings hold up across hospitals.

---

## 3. The specific ask (2–3 min) — be direct here

Lay out clearly what you need, in this order:

1. **IRB sponsorship as PI or co-PI** — specifically for the lightweight expert-rating component (Phase 5). This involves 2–3 cardiologists rating de-identified, public ECG images; no patient contact, no PHI. Likely qualifies for exempt/expedited review.
2. **(Optional, gauge his interest)** Whether he'd like to be one of the 2–3 rating cardiologists himself, or help recruit colleagues who would.
3. **(Optional)** Whether he'd want a co-author role shaping the clinical framing / concordance-metric definition (which ECG regions count as "correct" evidence for MI) — this is a natural fit given his imaging/CNN background.

**Be explicit about time cost**, since this is what most people actually want to know before saying yes:
- PI/IRB sponsorship: mostly front-loaded (protocol review + signature), light ongoing oversight.
- Rating 40–60 maps: roughly 1–1.5 hours, one sitting, once the maps are ready (months from now).
- Co-authorship input on the concordance metric: a couple of short conversations/reviews, whenever convenient.

---

## 4. Step-by-step project plan to present

Use this as the "here's the plan" walk-through — condensed from the full runbook, phase by phase:

| Phase | What happens | Timing | His involvement |
|---|---|---|---|
| **0. Setup** | Repo, environment, MIMIC-IV-ECG credentialing request, OSF pre-registration | Week 0–2 | None yet |
| **1. Data prep** | Download PTB-XL (public), define MI vs. Normal cohort, use official patient-stratified splits | Weeks 1–3 | None yet |
| **2. Model training** | Train and freeze a 1D-ResNet MI detector; report AUROC/sensitivity/specificity | Weeks 3–5 | None yet |
| **3. Attribution generation** | Run Grad-CAM, saliency, Integrated Gradients, DeepLIFT on correctly-classified MI cases | Weeks 5–6 | None yet |
| **4. Explanation-quality metrics (core)** | Faithfulness, sanity checks, and the clinical-concordance metric (needs a cardiologist's input on what counts as correct MI evidence) | Weeks 6–9 | **Light input here** — reviewing/sanity-checking the concordance-region definitions |
| **5. Expert rating** | IRB submission → recruit 2–3 cardiologists → blinded rating of ~40–60 maps → analyze agreement | Runs in parallel, weeks 5–10 (mostly a calendar dependency) | **His main role** — IRB sponsorship, possibly rating |
| **6. External validation** | Re-run the full pipeline on MIMIC-IV-ECG; check if explanation rankings hold across hospitals | Weeks 9–11 | None |
| **7. Writing & release** | Manuscript, open-source code release, target venue (Comput Biol Med / JAMIA / EHJ-Digital Health) | Weeks 11–14 | **Co-author review** of clinical framing/discussion |

**Overall realistic timeline: ~4–6 months**, with the IRB determination and cardiologist scheduling being the two calendar-driven bottlenecks — which is exactly why getting his sponsorship locked in early matters.

---

## 5. Anticipated questions and how to answer them

- **"What's already been done in this space?"**
  Be upfront: SHAP-on-ECG and Grad-CAM-on-ECG are not new individually. The novelty is combining disease-specific concordance (tied to MI diagnostic criteria) + cross-hospital ranking stability + lightweight clinician validation into one standardized benchmark — nobody has adjudicated the Grad-CAM contradiction this way.

- **"Why MI specifically?"**
  Because MI has spatially/temporally well-defined diagnostic evidence (ST-segment, Q-waves, specific leads per territory) — which gives you a concrete, physiology-grounded reference to score explanations against. Diffuse or rhythm diagnoses don't offer that.

- **"Do you have funding?"**
  Be honest — this is currently a team/self-funded project using public data and open-source tools; no grant yet. If he asks about resourcing, be ready to say you're open to discussing whether a small institutional or center-level support (e.g., AIHealth Institute) might be worth pursuing later, but the study is designed to run without it.

- **"Who else is on the team?"**
  Vân, Nhân, Hùng, and you (Tâm) — mention their divided responsibilities briefly (model training, attribution pipeline, concordance metric, project/statistics lead) so he sees it's a real, organized team, not a one-person ask.

- **"What do you need from me right now, today?"**
  Be ready with a clear, small first ask: *"A yes/no on whether you're willing to serve as PI or co-PI on the IRB submission, and if yes, I'll send over a short protocol draft for your review this week."*

---

## 6. Closing (1–2 min)

- Summarize what you heard him say he's willing to do (mirror it back so there's no ambiguity).
- Confirm concrete next step and date, e.g.:
  > "I'll send the protocol draft and the full proposal by [date]. Could we check in again in [X weeks] once MIMIC credentialing and the IRB draft are further along?"
- Thank him again, and offer to loop him in on the shared repo/tracker if he wants visibility into progress.

---

## Quick-reference cheat sheet (keep visible during the call)

- **One-liner:** "We're benchmarking whether ECG-AI explanation methods actually point to real diagnostic evidence for MI, and whether that holds across hospitals."
- **The ask:** IRB PI/co-PI sponsorship for a minimal-risk, 2–3-cardiologist rating study.
- **Data:** 100% public (PTB-XL + MIMIC-IV-ECG), no new patient data.
- **His time cost:** front-loaded IRB signature + light ongoing input; ~1–1.5 hrs if he also rates.
- **Timeline:** ~4–6 months total; his part matters most in Phase 5 (weeks 5–10).
- **Team:** Vân, Nhân, Hùng, Tâm — already assigned and working.
