# NeuroDyads GSoC 2026: Brain-to-Brain Decoder Pre-Task

**Candidate:** Vennela Varshini Anasoori  
**Email:** vennelavarshini07@gmail.com  
**GitHub:** [vennelavarshini18](https://github.com/vennelavarshini18)


## Overview

Pre-task submission for the NeuroDyads project under ML4SCI GSoC 2026.
Processes two simultaneous EEG recordings from a conversational dyad,
applies CEBRA to learn joint neural embeddings, and interprets the
embedding geometry across two affective conditions.


## What is Inside

**Part 1: Preprocessing**  
Loaded both EDF files, removed VREF channel, segmented using DIN1 markers
with a length check to verify time-locking. Ran ICA on each segment and
rejected artifact components based on time series shape inspection, with a
specific reason documented for every excluded component.

**Part 2: CEBRA Embedding**  
Built T x 128 joint matrix, z-normalized, trained CEBRA with affect labels.
Ran shuffled label control. Reported KNN accuracy and goodness-of-fit score
for both main and control models.

**Part 3: Interpretation**  
Analyzed 3D embedding geometry and connected observations to KNN and GOF numbers.

**Part 4: Reflection**  
Identified timing mismatch as the main specific limitation. Proposed improvements
including synchronization verification, ICLabel, multiple seeds, and cross-entropy
distance metrics (Roca et al. 2023).


## Key Results

| Metric | Main Model | Shuffled Control |
|---|---|---|
| KNN accuracy | 1.0000 | 0.5009 |
| Goodness of fit | 0.8172 | 0.0 |


## Data

Available at: https://alabama.box.com/s/gjdws83el45i1f99wt5qvmcmaspazpfe

## Setup
```bash
pip install mne cebra scikit-learn matplotlib numpy jupyter
jupyter notebook NeuroDyad_Screening.ipynb
```