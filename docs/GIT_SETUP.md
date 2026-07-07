# Committing and pushing this repo

Step-by-step to get this into your GitHub repo. Run these from **inside** the `ecg-xai-ami/` folder.

## 1. Initialize (first time only)

```bash
cd ecg-xai-ami
git init
git branch -M main
```

## 2. Sanity-check what will be committed

The `.gitignore` already excludes raw data, model checkpoints, credentials, and the
blinding key. Confirm nothing large or sensitive is staged:

```bash
git add -A
git status            # review the list — should be docs, src, configs; NO data/*.npy, NO .env
```

If you see anything under `data/`, a `*.pth`, or a credentials file, stop and fix
`.gitignore` before committing.

## 3. First commit

```bash
git commit -m "Initial commit: ECG xAI-AMI benchmark — planning, decisions, and pipeline scaffold"
```

## 4. Connect your GitHub repo and push

Create an empty repo on GitHub first (no README/license, so it doesn't conflict), then:

```bash
git remote add origin https://github.com/<your-username>/ecg-xai-ami.git
git push -u origin main
```

## 5. Recommended before making it public

- Add a `LICENSE` file (MIT is common for research code; check with collaborators/institution first).
- Keep the repo **private** until the IRB determination and collaboration with Dr. Kan Li are settled.
- Never commit the expert-rating ID→method key, PhysioNet credentials, or raw ECG data
  (all already git-ignored).

## Suggested commit hygiene going forward

- Commit the **data card** (`data/processed/data_card.md` is git-ignored by folder, so
  copy the final card into `docs/` if you want it version-controlled) once Phase 1 numbers are real.
- Update `docs/03_DECISIONS_LOG.md` in the same commit whenever a decision changes — that
  file is the project's memory; keep it current.
- Tag milestones: `git tag phase1-complete`, etc.
