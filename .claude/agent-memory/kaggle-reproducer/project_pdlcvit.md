---
name: project-pdlcvit
description: PDLC-ViT reproduction (2026-06-19) — MTL ViT paper adapted to classification-only branch, ViT-Small specs user-confirmed, co-scale/co-attention interpreted
metadata:
  type: project
---

# PDLC-ViT Reproduction (2026-06-19)

Paper: Hemalatha & Jayachandran, "A Multitask Learning-Based Vision Transformer for Plant Disease
Localization and Classification", Int. J. Computational Intelligence Systems (2024) 17:188.
DOI 10.1007/s44196-024-00597-3. Original framework MATLAB 2023a → reproduced in PyTorch.

**Why:** User reproducing SOTA plant-disease models onto own Kaggle dataset
(`/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300`). See [[user-profile]].

**How to apply (if revisited):**
- Scope reduced to **classification branch only** — segmentation head (PlantDoc/localization) dropped,
  MTL loss simplified to L_cls (CrossEntropyLoss). User-confirmed in implement_plan.md.
- Paper publishes ZERO quantitative ViT specs (patch/dim/depth/heads/MLP/input/norm). User confirmed
  ViT-Small config: 224 input, patch 16 (196 tokens), embed 384, depth 6, heads 6, MLP 1536, ImageNet norm,
  100 epochs + early stop patience=7 on val_loss, 70/15/15 stratified. These are NOT from paper — disclaimer required.
- Paper-confirmed (Table 2): Adam, LR 0.001, StepLR(step=15,gamma=0.5), batch 16, dropout 0.2, weight_decay 0.0005.
- Co-scale + Co-attention have NO layer spec in paper → implemented by interpretation, flagged
  `# [INTERPRETED - not in paper]`. Co-scale = 2 conv branches (3x3 + 5x5) concat+Linear; Co-attention = inter-patch MHSA.
- Cross-attention (Fig.3): branch_CS→Z1, branch_CA→Z2, MHCA(Z1,Z2)→Z. Encoder x6 on patch tokens + CLS + pos.
  Classification head (Fig.4 adapted): MHCA(query=CLS, key/value=Z) → MLP. Paper used Sigmoid → replaced with
  logits for CrossEntropyLoss (documented).
- Notebook built via a throwaway Python generator script (json module) then validated with ast.parse per code
  cell + a 27-item rules checklist; generator deleted after. Gotcha: inside `code(r"...")` raw strings, use
  single-quote `'''docstrings'''` NOT escaped `\"\"\"` (raw string keeps backslashes literal → syntax error).

Outputs: `papers/10 - PDLCViT_Experiment/outputs/PDLCViT_Experiment.ipynb` (45 cells, 32 code) +
`PDLCViT_ReproduceSummary.md`. All 14 core rules satisfied.
