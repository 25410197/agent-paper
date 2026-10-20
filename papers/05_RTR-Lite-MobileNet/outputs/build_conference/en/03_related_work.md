### 2.1. Baseline Selection Criteria

**Selection criteria for the baseline pool.** Table 3 comprises 17 baselines from 15 studies, selected by:

- peer-reviewed work published 2023–2026 on image-based plant-disease classification;
- a clear architectural focus on deployability (lightweight CNN / ViT / hybrid) or a broad CNN-vs-ViT comparative study;
- architectural detail sufficient (layer configuration, embedding widths, head counts) to reproduce or adapt for the target task without contacting the authors;
- coverage of the four directions below rather than only the strongest family.

Studies were excluded when they targeted detection/segmentation rather than classification, or when no corresponding classification output could be constructed. The baseline pool therefore represents the deployment-oriented literature of the last three years, not an exhaustive survey; the models were reimplemented and evaluated by us under a common protocol described in Section 4.1, so Table 3 does not use the figures originally published by the authors.

Studies that perform only detection or segmentation with no image-level classification output were excluded from the baseline pool. Multitask models are kept if the original architecture has an explicit classification branch that can be adapted to the target task; Table 2 details the specific adaptation applied to the one such case in the pool, PDLC-ViT (Hemalatha and Jayachandran, 2024).

### 2.2. Related Architectural Directions

- **Lightweight CNNs** keep parameters low through efficient attention or compact blocks. RTR_Lite_MobileNetV2 (Duhan et al., 2025), our base architecture, adds triplet attention (Misra et al., 2021) and a spatial-attention residual block to MobileNetV2 (Sandler et al., 2018). ULEN (Wang et al., 2023) pushes ultra-lightweight design down to 28 thousand parameters; MRW-CNN (Joseph et al., 2024) and Lite-MDC (Bhagat et al., 2024) target real-time field deployment; CustomMNetV3L (Rahaman et al., 2026) adapts MobileNetV3-Large.
- **Vision Transformers** directly exploit relations between tokens, unlike deep CNNs, which can extend the receptive field to aggregate broad context but do not directly model pairwise interactions between every position. ViTs typically carry higher cost: ViT (Ali et al., 2025), TrIncNet (Gole et al., 2023) with a Trans-Inception block, PMVT (Li et al., 2023) for mobile deployment, and PDLC-ViT (Hemalatha and Jayachandran, 2024) for multitask localization plus classification.
- **Hybrid CNN-Transformer architectures** closest to this line of work include GrapeLeafNet (Karthik et al., 2024), MaizeFormerX (Rahman et al., 2026), and LDI-NET (Yang et al., 2024), typically at 3–9 million parameters.
- **Broad comparative studies** include Mishra et al. (2026), Sujatha et al. (2025), and the lightweight IoT meta-ensemble of Maurya et al. (2024).

These works use different datasets, splits, and training protocols, so they do not yet form a single shared comparison across architectural contribution, degradation robustness, and domain gap.
