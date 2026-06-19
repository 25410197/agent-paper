---
name: project-ldinet
description: LDI-NET reproduction 2026-06-19 — PyTorch hybrid CNN+Transformer, multi-label paper adapted to single-label dataset
metadata:
  type: project
---

**Reproduced:** LDI-NET (Yang et al., Scientific Reports 2024, DOI 10.1038/s41598-024-62452-x) on user's single-label plant disease dataset.

**Paper folder:** `papers/09- LDI-NET — Plant Disease Experiment/`
**Outputs:** `outputs/LDINET_Experiment.ipynb` + `outputs/LDINET_ReproduceSummary.md`

**Key adaptation:** Paper is MULTI-LABEL (12 plant + 19 disease + 3 severity = 34 DETR-style query embeddings, Module 3 = Multi-Label Decoder with self+cross attention). User dataset is SINGLE-LABEL → Module 3 replaced with **GAP + LayerNorm + Linear**, loss = single CrossEntropyLoss. Module 1 (Feature Tokenizer) and Module 2 (Token Encoder) kept as-is.

**Architecture (all confirmed by user 2026-06-19, NOT in paper):** embed_dim=384, n_heads=6, FTM blocks N=2, encoder blocks=2, mlp_ratio=4 (hidden=1536), dropout=0.1, conv_out=384, patch=16, img=224 → 196 tokens. Pre-norm transformer, nn.MultiheadAttention(batch_first=True). Fused token = compact_spatial + context_rich (residual add).

**Hyperparams (from paper):** AdamW lr=6.25e-6, StepLR(step=4, gamma=0.5), batch=16, epochs=60. **Confirmed by user:** weight_decay=0.01, ImageNet norm, seed=42.

**Dataset:** `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` — pre-split train/val/test, ImageFolder per split, num_classes auto-detect.

**Why:** Plan had all 11 Vietnamese questions pre-answered in "THÔNG SỐ ĐÃ XÁC NHẬN BỞI NGƯỜI DÙNG" section — no blocking confirmation needed, built directly.

**How to apply:** If user revisits LDI-NET or wants the multi-label version restored, Module 3 decoder needs DETR-style learnable queries + cross-attention to fused tokens (see implement_plan.md Module 3 / Fig 3b). Notebook smoke-tested for shapes manually (local torch DLL broken, couldn't run). See [[feedback-vit-assumptions]].
