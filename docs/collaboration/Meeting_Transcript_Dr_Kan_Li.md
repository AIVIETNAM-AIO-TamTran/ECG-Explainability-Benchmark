# Full Meeting Transcript / Script — Zoom with Dr. Kan Li
## ECG xAI–AMI Benchmark Project

This is a word-for-word script you can read from or closely paraphrase. Sections in *[brackets]* are stage directions (when to share screen, when to pause). It's written to sound natural if read aloud slowly — don't rush.

---

### [Opening — before sharing screen]

"Hi Dr. Li, thank you so much for making time today — I know you're busy, so I'll be efficient with your time.

Quick context on me: I'm Tam Tran, a clinical research coordinator affiliated with WashU, and I work across cardiovascular AI research, global health, and manuscript coordination. I reached out to you specifically because your work on deep learning image analysis for cardiomyopathy is really close to a project I'm building right now, and I'm hoping we might be a good fit to collaborate.

Would it be alright if I shared my screen for about two minutes to walk through the idea, and then we can talk through it together?"

*[Wait for yes, then share screen — Slide 1]*

---

### [Slide 1 — Title]

"So this project is called the ECG Explainability Benchmark. The one-line version is: we're asking whether the AI explanations for myocardial infarction actually point to where a cardiologist would look — and whether that holds up across hospitals."

*[Advance to Slide 2]*

---

### [Slide 2 — The Problem]

"Here's the problem in plain terms.

Deep learning models can now read a 12-lead ECG and detect things like MI about as well as a cardiologist. But they're black boxes — you get a prediction, but not a reason.

So the field has started adding 'explanation' tools on top — things like Grad-CAM or saliency maps — that generate a heatmap over the ECG, supposedly showing what the model was looking at.

*[Pause briefly]*

Here's the part that's actually the interesting gap: nobody has really checked whether these heatmaps are telling the truth. And there's a real contradiction in the literature about this — one careful study found that Grad-CAM, which is the most commonly used method, actually fails a basic sanity check on ECG data. It just lights up the QRS complex — the tallest spike — no matter what the diagnosis is. But a different benchmark paper looked at the same question and said Grad-CAM was the best method.

So you have two serious papers reaching opposite conclusions about the same tool. That's the gap this project is built to resolve — with a standardized, disease-specific benchmark instead of another one-off application."

*[If he asks "why does this matter clinically" — answer:]*

"If a hospital or a vendor deploys an ECG-AI system with an explanation feature to build clinician trust, and that explanation is systematically pointing at the wrong evidence, that's arguably worse than having no explanation at all — it's false reassurance. So before any of this gets near deployment, we need to know which explanation methods can actually be trusted."

*[Advance to Slide 3]*

---

### [Slide 3 — The Four Steps]

"So here's how we approach it, in four steps.

**Step one** — we train a completely standard model. A 1D convolutional neural network on PTB-XL, which is a large public ECG dataset, to detect MI versus normal. Nothing novel here — it just needs to be a competent, credible model.

**Step two** — we generate explanations for that model's predictions, using the standard toolkit: Grad-CAM, saliency maps, Integrated Gradients, DeepLIFT.

**Step three, and this is the actual contribution** — we score the explanations themselves, not just the model. Three things: does removing the highlighted region actually change the prediction — that's faithfulness. Does the explanation degrade properly when we randomize the model — that's the sanity check, the same kind of test that caught Grad-CAM's problem. And does the highlighted region land on the ST-segment, the Q-waves, the leads that actually define MI for a cardiologist — we call that clinical concordance.

**Step four** — we validate all of this with actual cardiologists, a small panel rating a blinded sample of these maps, and we re-run the whole thing on a second, independent hospital dataset — MIMIC-IV-ECG — to see if our conclusions about which method is trustworthy actually hold up across hospitals, not just on one dataset."

*[Advance to Slide 4]*

---

### [Slide 4 — The Ask]

"So, here's specifically what I'm hoping to ask of you — and I want to be upfront that this is a defined, scoped ask, not an open-ended commitment.

The main thing is **IRB sponsorship** — serving as PI or co-PI specifically for that expert-rating piece I just mentioned. It's minimal risk: 2 to 3 cardiologists blindly scoring about 40 to 60 already-de-identified, publicly-sourced ECG images. No patient contact, no protected health information. It would likely qualify for exempt or expedited review. This is mostly front-loaded — reviewing a short protocol and signing off — with light ongoing oversight after that.

Two optional things, only if you're interested: whether you'd like to be one of the rating cardiologists yourself, which would take about an hour to an hour and a half, one sitting, once the maps are ready — probably a couple of months out. And whether you'd want to weigh in as a co-author on the clinical side — specifically helping define what counts as 'correct' ECG evidence for MI, which is exactly the kind of judgment your imaging and diagnostic background would strengthen.

And just to be clear on the data side — everything here uses public, de-identified datasets. We're not collecting any new patient data at any point."

*[Advance to Slide 5]*

---

### [Slide 5 — Plan, Team, Next Step]

"Just to give you the full picture — here's the rough plan. Setup and data prep, then model training, then the attribution and scoring work, then the expert rating — which is where your involvement matters most — running in parallel with an external validation on that second hospital dataset, and finally writing it up. Altogether, we're estimating about four to six months, realistically, with your piece and the MIMIC data-access approval being the two things that are more about calendar time than actual effort.

On the team side — I'm not doing this alone. I have three teammates, Vân, Nhân, and Hùng, who are already assigned to the model training, the attribution pipeline, and statistics respectively, and I'm leading the overall project and the clinical/IRB side.

So, very concretely — what I'd love to know today is just: would you be willing to serve as PI or co-PI on the IRB submission for this? If so, I can have a short protocol draft in your inbox this week for you to look over."

*[Stop sharing screen here, or leave slide 5 up while you talk]*

---

### [Open discussion — let him respond]

*[Give him real space here. Don't fill silence. Let him ask questions.]*

**If he asks "what's already been done in this space?":**
"Fair question — individually, SHAP-on-ECG and Grad-CAM-on-ECG aren't new. What's missing is combining a disease-specific concordance check tied to actual MI diagnostic criteria, a cross-hospital stability test, and a lightweight clinician validation — into one standardized benchmark. Nobody's adjudicated that Grad-CAM contradiction this way yet."

**If he asks "why MI specifically, why not other conditions?":**
"Because MI has really well-defined diagnostic evidence — the ST-segment, Q-waves, specific leads depending on the territory. That gives us something concrete to score explanations against. A rhythm diagnosis or something more diffuse wouldn't give us that same clean reference point."

**If he asks about funding:**
"Right now this is a self-directed, team-run project using public data and open-source tools — no grant funding yet. If it'd be helpful, I'm open to discussing whether something like the AIHealth Institute might be worth approaching later, but the study is designed to be able to run without external funding."

**If he seems hesitant about time commitment:**
"Totally understand — the ask today is really just the IRB sponsorship piece. Everything else is optional, and I'm happy to keep you looped in at whatever level makes sense for you."

**If he says yes:**
"That's wonderful, thank you so much. I'll put together the short protocol and send it over along with the full proposal — would [specific date, e.g. 'by Friday'] work, and should I check back in with you in a couple of weeks once I've heard back on the MIMIC data access?"

**If he says he needs to think about it:**
"Of course, totally understand. I'll send over the full written proposal so you have everything in front of you, and I'll follow up in [a week / two weeks] — whichever is easier for you."

**If he says no / not the right fit:**
"No problem at all, I really appreciate you taking the time to hear me out. If anyone comes to mind who might be a better fit, I'd be grateful for an introduction."

---

### [Closing]

"Thank you again for your time today, Dr. Li — this was really helpful. I'll follow up by email with [the protocol draft / the full proposal / whatever was agreed], and I look forward to hopefully working together on this."

*[End call]*

---

## Quick nerves-check reminders

- You don't need to say everything on this page — the slides carry the structure, you're just narrating them.
- It's fine to pause and think; better than rushing through a memorized line.
- If you forget where you are, glance at the current slide title — it tells you which section you're in.
- The single most important sentence to land, if nothing else: *"Would you be willing to serve as PI or co-PI on the IRB submission for this?"* — everything else supports getting to that ask clearly.
