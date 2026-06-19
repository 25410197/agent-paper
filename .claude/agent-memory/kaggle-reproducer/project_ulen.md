---
name: project-ulen
description: ULEN reproduction (Precision Agriculture 2023) — PyTorch, ShuffleNet-v2 RDWConv + SPP, ~111,758 param constraint, channels reverse-engineered
metadata:
  type: project
---

ULEN — "An ultra-lightweight efficient network for image-based plant disease and pest infection detection" (Wang et al., Precision Agriculture 2023, 24:1836-1861, DOI 10.1007/s11119-023-10020-0). Reproduced 2026-06-18 into `papers/08 - An ultra-lightweight efficient network/outputs/` (ULEN_Experiment.ipynb + ULEN_ReproduceSummary.md).

**Why:** User reproducing this lightweight CNN onto their chili dataset. Group C paper — topology in figures (Fig.5-8) but per-block channel counts NOT published; only total params 111,758 (Table 5) as constraint.

**How to apply:**
- Framework **PyTorch** (paper-confirmed). Adam, CrossEntropy, batch 16, epochs 20, input 256x256, all paper-confirmed. LR schedule 1e-3 (ep1-5) -> 1e-4 (ep6-20) via MultiStepLR(milestones=[5], gamma=0.1) — paper-confirmed step schedule.
- Split 70/20/10 stratified per class (paper) — implemented with sklearn train_test_split stratify on ImageFolder targets, but make_splits() auto-detects pre-split train/val/test folders first (user dataset is pre-split).
- **Channels reverse-engineered, NOT from paper:** C=64 (stem + 3 RDWConv), C2=128 (final 1x1). Analytic count = 116,806 at num_classes=38 vs paper 111,758 (+5,048). Discrepancy dominated by final Linear (21*C2*num_classes). Documented as estimate; notebook compares torchinfo total vs 111,758.
- **Architecture:** ConvBlock3x3(full conv 3->64, +BN+ReLU+MaxPool) -> 2x MaxPool -> 3x RDWConv(stride=1) -> ConvBlock1x1(64->128, full conv, +BN+ReLU) -> SPP{4,2,1}=21*C2 -> Linear. First+last conv are STANDARD (full) conv, middle blocks depthwise (paper: "except for the first and last conv layer").
- **RDWConv = ShuffleNet-v2 unit (Fig.7a):** channel split half/half; skip passes C/2; main = DWConv3x3->BN->ReLU->Conv1x1->BN; concat -> channel_shuffle(groups=2). Channels UNCHANGED. NOT element-wise residual add despite "residual" name.
- **SPP** uses AdaptiveAvgPool {4x4,2x2,1x1} -> flatten -> concat = 21*C features. Linear in_features fixed regardless of spatial size.
- **Class weighting = DEVIATION from paper** (paper uses only stratified split). Notebook auto-enables via core rule #11 (max/min>=1.5) and prints warning. Same pattern noted across all reproductions.
- Aug Scenario 2 (4 techniques, workspace req — paper has NONE): RandomRotation(30), RandomResizedCrop(256,scale=(0.8,1.0)), RandomHorizontalFlip, ColorJitter(b/c/s=0.2).
- Built via throwaway python script (_build_ulen_nb.py) using Write tool (NOT bash heredoc — heredoc broke on embedded single-quotes like param_groups[0]['lr']). Script json.dump's .ipynb, then validated with json.load + ast.parse on every code cell, then deleted. See [[project-litemdc]] for the same build method. Dataset path convention: chili at /kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300.
