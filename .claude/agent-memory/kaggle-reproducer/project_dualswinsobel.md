---
name: project-dualswinsobel
description: Reproduction 2026-10-20 of Han & Guo 2024 (fpls) two-channel Swin-T + trainable Sobel edge branch, PyTorch/timm, early stopping user-requested
metadata:
  type: project
---
Paper folder papers/16_fpls; outputs DualSwinSobel_Experiment.ipynb + DualSwinSobel_ReproduceSummary.md.
User OK'd all brainstorm defaults (heads 3/6/12/24, drop_path 0.2, Adam 1e-4 constant, ImageNet norm inside model, channel concat 1536).
User additionally requested EARLY STOPPING (patience 15, min_delta 1e-4, monitor val_acc, EPOCHS as max) — a deviation from paper.
**Why:** 2x Swin-T x 100 epochs x 2 scenarios risks Kaggle 12h limit.
**How to apply:** For heavy transformer reproductions, expect user to want early stopping; ask/offer it up front. Dataset path: /kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300 (train/val/test).
