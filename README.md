# NeuroDyads GSoC 2026 — Brain-to-Brain Decoder Pre-Task
**Candidate:** Vennela Varshini Anasoori

**Email:** vennelavarshini07@gmail.com

**GitHub:** [vennelavarshini18](https://github.com/vennelavarshini18)

---

## Overview
This is the pre-task submission for the **NeuroDyads project under ML4SCI GSoC 2026**.
It processes two simultaneous EEG recordings from a single conversational dyad,
applies CEBRA contrastive representation learning to learn joint neural embeddings,
and interprets the resulting embedding geometry across two affective conditions.

---

## What's Inside

### `NeuroDyad_Screening.ipynb`
A single end-to-end Jupyter notebook covering all four required parts.

---

**Part 1 — Preprocessing**
- Loaded two raw 64-channel EEG files (.EDF format, 250Hz, EGI HydroCel nets)
- Identified and correctly interpreted marker structure:
  - Participant A: VBeg/DIN1/VEnd markers
  - Participant B: DIN1-only markers
- Segmented both recordings into two conditions:
  - **Positive affect**: marker 1 → marker 2 (~148 seconds)
  - **Negative affect**: marker 3 → end of file (~153 seconds)
- Removed Channel 65 (VREF vertex reference) from both participants
- Applied ICA (FastICA, 20 components) for artifact removal
- Identified and excluded artifact components based on variance analysis
- Plotted power spectrum before and after ICA on the same axes for Participant A

**Part 2 — CEBRA Embedding**
- Concatenated preprocessed segments (positive then negative affect) for both participants
- Built a T×128 joint matrix (first 64 columns = Participant A, last 64 = Participant B)
- Z-normalized each channel independently across time
- Applied CEBRA with 3-dimensional output embeddings and affect condition labels
- Ran a shuffled-data control to validate learned structure
- Reported KNN decoding accuracy (5-fold CV) for both main and control models

**Part 3 — Interpreting the Embedding**
- Analyzed 3D embedding geometry for cluster separation, transitions, and outliers
- Interpreted dense regions as stable neural synchrony states during conversation
- Interpreted transitional regions as behavioral shifts between affective conditions
- Explained what the shuffled control result reveals about CEBRA's learning

**Part 4 — Critical Reflection**
- Identified temporal misalignment between participants as the single biggest
  specific limitation of this analysis
- Proposed concrete improvements: synchronization verification, ICLabel-based
  artifact rejection, multiple CEBRA seeds, and cross-entropy distance metrics

---

## Key Results
- KNN decoding accuracy: **1.0000** on main model — embedding perfectly separates
  positive and negative affect conditions
- Control (shuffled labels): accuracy drops toward chance — confirms CEBRA learned
  genuine structure from the joint neural data, not temporal autocorrelation
- ICA successfully reduced low-frequency artifact power visible in the pre-ICA spectrum

---

## Data
Two raw EEG files (.EDF) from a single conversational dyad.
Available at: https://alabama.box.com/s/gjdws83el45i1f99wt5qvmcmaspazpfe

---

## Setup
```bash
pip install mne cebra scikit-learn matplotlib numpy jupyter
jupyter notebook NeuroDyad_Screening.ipynb
```
