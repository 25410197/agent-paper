---
name: project-tlmvit
description: TLMViT-VGG19 reproduction 2026-10-20 — Keras 3 frozen VGG19 + 8-layer ViT on 8x8x512 feature map, head input = 4 patch tokens flattened (256)
metadata:
  type: project
---

TLMViT (Tabbakh & Barpanda, IEEE Access 2023) reproduced as VGG19-only (user choice) on 2026-10-20.
Outputs: papers/20_TLMViT/outputs/TLMViT_VGG19_Experiment.ipynb (40 cells) + ReproduceSummary.md.

Key user-confirmed assumptions: head input = LN + drop class token + flatten 4x64=256; GELU everywhere; no dropout; key_dim 16; LN eps 1e-6; vgg19.preprocess_input in tf.data (no Lambda); brightness 0.2. Table 5 "RMSprop adam" conflict -> Adam.

**Why:** paper ambiguous on head input; 256 only matches 4 patch tokens.
**How to apply:** For Keras 3 hybrid models, keep preprocessing in tf.data and use registered custom layers so `.keras` full-model checkpoints reload (avoid Lambda → weights.h5 workaround used in [[project-litemdc]]). Dataset path pattern same as [[project-ldinet]].
