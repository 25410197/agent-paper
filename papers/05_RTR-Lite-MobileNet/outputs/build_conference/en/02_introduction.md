## 1. Introduction

Chili pepper diseases substantially reduce crop yield, and farmers need a diagnostic tool that runs directly on an edge device: simultaneously lightweight and accurate, two goals that usually conflict.

Lightweight convolutional networks address the size constraint but, being purely local, struggle with relationships between distant leaf regions; vision transformers capture such global relations but are typically heavier. Hybrid CNN-Transformer designs compromise, yet most prior work reports single-seed accuracy without variance-controlled ablation, multi-technique interpretability, or a comparison against a multimodal foundation model. More fundamentally, such work reports the *assembled* model without measuring whether each component earns its parameters.

This work starts from RTR_Lite_MobileNet [5], which augments MobileNetV2 [16] with attention blocks, and proposes **ChiliEdgeNet**, a sub-700-thousand-parameter hybrid integrating three parameter-efficient LLM techniques. The contributions are:

- **ChiliEdgeNet**: fully offline on CPU at 19.28 ms/image (2.7 MB), 99.20% on both augmented and baseline with **671,173 parameters** — higher than the base architecture [5] (98.12%) while using **35% fewer** parameters — leading the pool of a 27-configuration SOTA reproduction.
- **ChiliEdge-DS**, a field-collected chili disease dataset from southern Vietnam (5 classes, 2,461 images, fixed split; Section 3.1), released on Kaggle [23].
- A **component-level audit**: a test-time degradation benchmark plus **retrain-from-scratch ablation** at *block* level (Tiny Transformer) and *individual technique* level (2D-RoPE, GQA, RMSNorm), every comparison tested per-image on the same 373 images. This yields the central claim: **all three techniques are close to free — 41,856 parameters saved at no measurable accuracy cost — while the Tiny Transformer block contributes significantly under blur (*p* = 0.0001).**
- Two XAI techniques from maximally distinct mechanism families (Occlusion Sensitivity, black-box perturbation; Attention Rollout, reading the Transformer's internals) plus a zero-shot Gemini comparison validate the model.
