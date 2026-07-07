# data/

**Nothing in this folder is committed to git** (see `.gitignore`). Raw ECG data is
excluded for two reasons: PhysioNet licensing terms, and file size.

## Expected layout (populated locally, not in the repo)

```
data/
├── ptbxl/            # PTB-XL 1.0.3, unpacked from PhysioNet (open access)
├── mimic_iv_ecg/     # MIMIC-IV-ECG, unpacked (requires credentialing)
└── processed/        # X/y arrays + data_card.md produced by Phase 1
```

## Getting the data

**PTB-XL (open access):**
```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/ptbxl/
```

**MIMIC-IV-ECG (credentialed — start early, ~1–2 weeks):**
1. Create a PhysioNet account.
2. Complete CITI "Data or Specimens Only Research" training.
3. Sign the credentialed-data DUA.
4. Download from https://physionet.org/content/mimic-iv-ecg/1.0/
5. Meanwhile, prototype on the open demo set:
   https://physionet.org/content/mimic-iv-ecg-demo/0.1/

## Data card

Phase 1 must write `data/processed/data_card.md` with MI/NORM counts (strict + inclusive,
per fold), exclusion counts, quality-flag distribution, preprocessing settings, and the
random seed. Every downstream result must trace back to it. See docs/04.
