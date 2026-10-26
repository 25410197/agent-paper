---
name: project-fotca
description: FOTCA (21_AFNO, Hu et al. 2023) reproduction decisions — complex-conv FFT patch embed, partial timm ViT weight copy, 30 epochs + early stopping
metadata:
  type: project
---

FOTCA notebook built 2026-10-20 (papers/21_AFNO/outputs). User approved all 20 suggested defaults and added early stopping (patience 7) on val_accuracy.

**Why:** paper (Group C) omits AFNO/classifier details; no GitHub; 100 epochs too slow on Kaggle.
**How to apply:** for FFT->Conv->IFFT embeddings, use a complex conv (W_r + iW_i) — a real-valued spectrum into irfft2 makes tokens point-symmetric. Reproduced params 78.78M vs paper 59.14M (classifier width 1024 assumption). Users in this workspace may request early stopping alongside best-checkpoint saving.
