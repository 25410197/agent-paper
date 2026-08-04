## 5. Model Interpretability

### 5.1. Explainable AI

High accuracy alone is not enough to trust a model, which could exploit spurious cues such as background — a real risk here, where every image is shot against a white backdrop. We ran eight interpretability methods on the trained model (Grad-CAM [21] and Grad-CAM++, Score-CAM, Integrated Gradients, SHAP, LIME, Occlusion Sensitivity, Attention Rollout) and present the **two whose mechanisms are furthest apart**, since they are unlikely to fail in the same way. **Occlusion Sensitivity** [24] is a **perturbation** method: it slides a grey patch over the image and measures the drop in true-class probability, **never looking inside the model** — no gradients, no weights, no activations — so it is evidence independent of any architectural assumption. **Attention Rollout** [22] does the opposite: it reads the attention matrices **inside** the Transformer block and traces attention flow from the output token back to the input grid.

One method treats the model as a black box; the other reads the internals of the very block we propose. Both localise disease regions and ignore the background (Figures 3-4): the Occlusion map lights up on diseased tissue — rot lesions for anthracnose, spot clusters for leaf spot, chlorotic areas for yellow leaf — while Attention Rollout concentrates on the leaf/fruit centre. Agreement between two mechanisms sharing no assumptions is evidence that the accuracy reflects genuine disease features rather than an artefact of one technique or a background cue. The cut point of Section 3.2 pays a direct dividend: with a 14x14 grid the Rollout map has **196 cells instead of 49** — fine enough to delineate a symptom region rather than merely indicate a quadrant.

<!-- IMG2 -->
![Figure 3. Occlusion Sensitivity for the five disease classes (top: input; bottom: sensitivity overlay). Bright areas are regions whose occlusion costs the model the most confidence; the method treats the model as a black box.](figs/xai_occlusion.png)

![Figure 4. Attention Rollout of the Tiny Transformer block on the 14x14 grid; attention concentrates on the object centre (leaf/fruit), ignoring the background.](figs/xai_attn_rollout.png)
<!-- /IMG2 -->

### 5.2. Comparison with a Vision LLM

As an independent cross-check, a general-purpose multimodal LLM (Gemini 3.1 Flash Lite, accessed 8 July 2026) classifies the raw images zero-shot; this is a feasibility contrast, not a peer comparison (the LLM is untrained here, our model specialized). Over the **full 373-image test set** (all calls returned successfully) the LLM reaches **78.55%** against **99.20%**, with errors concentrated on the `leaf_curl`/`yellow_leaf` boundary our model also finds hardest. The specialized model additionally runs offline at near-zero marginal cost with structured five-class outputs, which a cloud LLM cannot offer on-device. Both systems are scored on the **same 373 images**.

**On latency.** The ~110x figure contrasts a **local 19.28 ms inference** with a **2,117 ms median cloud API round-trip**. These are not like-for-like: the round-trip includes network latency, server-side queueing and provider-side batching, none of which are computation by the LLM. It is a **system-level latency comparison** — what a user of each deployment would actually wait — not a claim that one architecture computes ~110x faster. The accuracy gap is a fair comparison; the latency gap is a deployment-mode comparison. A commercial endpoint is also a moving target: versions update continuously, so this is reproducible only against the stated version and access date.

**Table 10. Proposed model versus Gemini 3.1 Flash Lite on the full 373-image test set (accessed 8 July 2026).**

| System | Accuracy | Latency/image |
|---|---|---|
| **ChiliEdgeNet (local CPU)** | **99.20%** | **19.28 ms** (local inference) |
| Gemini 3.1 Flash Lite | 78.55% | 2,117 ms (median API round-trip) |
