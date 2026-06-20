# MEMORY INDEX — kaggle-reproducer agent

- [User Profile](user_profile.md) — Maize/plant disease domain, custom Kaggle dataset, student/researcher reproducing SOTA
- [ViT Assumption Defaults](feedback_vit_assumptions.md) — Reverse-engineer embed_dim/num_layers/num_heads from total params when not stated
- [MaizeFormerX Project](project_maizeformerx.md) — Reproduction 2026-05-27: architecture assumptions (D=256,L=6,H=8), confirmed hyperparams, output files
- [MRW-CNN Project](project_mrwcnn.md) — Reproduction 2026-06-16: TF/Keras, SGD, conv config assumed (Fig.9 image), chili substitute dataset
- [Custom CNN Project](project_customcnn_cropdesign.md) — Reproduction 2026-06-16: TF/Keras, Adam, fully self-proposed 6-conv/5-pool layer config, chili substitute dataset
- [Lite-MDC Project](project_litemdc.md) — Reproduction 2026-06-18: TF/Keras, MDsConv multi-kernel depthwise module, ~2.29M params verified, chili substitute dataset
- [ULEN Project](project_ulen.md) — Reproduction 2026-06-18: PyTorch, ShuffleNet-v2 RDWConv + SPP, channels reverse-engineered to ~111,758 params, chili dataset
- [LDI-NET Project](project_ldinet.md) — Reproduction 2026-06-19: PyTorch hybrid CNN+Transformer, multi-label paper adapted single-label (decoder→GAP+Linear), D=384/H=6 user-confirmed
- [PDLC-ViT Project](project_pdlcvit.md) — Reproduction 2026-06-19: PyTorch MTL ViT adapted classification-only, co-scale/co-attention interpreted, ViT-Small specs user-confirmed
- [Meta-Ensemble Project](project_metaensemble.md) — Reproduction 2026-06-19: TF/Keras 2-level stacking (MLP-Mixer+LSTM+SVM), patch Dense(1728→128) inflates params vs paper, chili dataset
