# Voice Cloning and Fake Audio Detection (VCFAD)

## Project Overview

This project focuses on the development of two AI systems within the audio domain:

1. **Voice Cloning (VC):** Converts a speaker’s voice into another speaker’s voice using parallel text-audio datasets.
2. **Fake Audio Detection (FAD):** Detects whether a given spoken audio is authentic or synthetically generated.

This dual system serves the broader goal of improving speech-based authentication by identifying whether audio content is real or fake.

---

## Datasets

- **[TIMIT Dataset](https://github.com/philipperemy/timit):** Used for voice cloning tasks. Includes aligned text-audio pairs from 630 speakers across 8 US dialects.
- **[CommonVoice Dataset](https://commonvoice.mozilla.org/en/datasets):** Used for fake audio detection tasks. Contains natural speech from volunteers and serves as both positive (real) and negative (fake) label sources.

---

## Goals

- Build a **voice cloning model** that converts a source speaker's voice to match a target speaker.
- Train a **fake audio detection model** to distinguish natural vs. machine-generated speech.

---

## Success Metrics

- **Voice Cloning:** Evaluated using **Word Error Rate (WER)** on transcribed cloned speech.
- **Fake Audio Detection:** Evaluated using **F1-score** and **accuracy** on labeled audio examples.

---

## Tools & Techniques

- Python, Librosa, NumPy, Pandas
- Feature extraction: MFCCs, spectrograms
- WER calculation and ASR evaluation
- Classifier models for detection (likely CNNs or tree-based models)
- Preprocessing: Dataset splitting, label generation

---

## Data Access

Due to licensing constraints, the datasets are not included in this repo. However, you can access them publicly:

- TIMIT: [GitHub mirror](https://github.com/philipperemy/timit)
- CommonVoice: [Mozilla Datasets](https://commonvoice.mozilla.org/en/datasets)

---

## Author

**Nadeem Bishtawi**  
*AI Resident — Apziva | Audio Intelligence, ML, and Speech Analysis*
