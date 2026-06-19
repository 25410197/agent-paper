---
name: project-maizeformerx
description: MaizeFormerX reproduction — architecture confirmed from GitHub 2026-05-27, notebook updated
metadata:
  type: project
---

**Reproduced:** MaizeFormerX (Scientific Reports 2026) on user's maize leaf disease dataset.

**Paper folder:** `papers/MaizeFormerX a lightweight/`
**Output notebook:** `papers/MaizeFormerX a lightweight/outputs/MaizeFormerX_Experiment.ipynb`
**Summary:** `papers/MaizeFormerX a lightweight/outputs/MaizeFormerX_ReproduceSummary.md`

**Architecture CONFIRMED from GitHub (https://github.com/rezaul-h/MaizeFormerX/) — 2026-05-27:**
- MultiScalePatchEmbed: patch_sizes=[8,16,32], embed_dims=[64,96,128] (per-scale, NOT shared D=256)
- CSAF: fusion_dim=192, mechanism=self-attention over concat of all 1029 tokens (NOT cross-attention), num_heads=4
- Transformer Encoder: depth=6, num_heads=6 (NOT 8), dim=192, mlp_ratio=4.0, drop_path linear 0->0.1
- Classifier: CLS token [:, 0] (NOT GlobalAvgPool), Dropout(0.3) -> Linear(192, num_classes)
- Full flow: [B,784,64]+[B,196,96]+[B,49,128] -> CSAF -> [B,1029,192] -> +CLS -> [B,1030,192] -> Encoder -> cls_out

**Why:** Initial reverse-engineering from 9.6M param budget gave wrong architecture. User confirmed from GitHub source code. Notebook fully updated 2026-05-27.

**How to apply:** Architecture is now fully confirmed. Only remaining assumptions: warmup type (linear) and T_max for CosineAnnealingLR (= MAX_EPOCHS).

**Paper hyperparams (confirmed):** LR=5e-5, WD=5e-5, batch=32, dropout=0.3, epochs=30, patience=4, warmup=300 steps, CosineAnnealingLR.
**Augmentation (Dataverse config):** HFlip(p=0.5), Rotation(±30°), Zoom(0.9-1.1), Brightness(±20%).
**Split used in notebook:** 75/15/10 (paper uses 80/5/15 on their datasets).
