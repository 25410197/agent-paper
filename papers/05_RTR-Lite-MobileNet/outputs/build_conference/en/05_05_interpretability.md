### 4.4. Model Interpretability

The analyses in this section use ChiliEdgeNet's original checkpoint, trained only on ChiliEdge-DS and used to produce the in-domain results in Table 3. The augmented checkpoint used for the domain-transfer experiment in Section 4.5 is a separate adapted model and is not used to produce the interpretability results below.

#### 4.4.1. Explainable AI (XAI)

High accuracy alone is not sufficient to interpret model behaviour, particularly since ChiliEdge-DS predominantly uses a controlled white background. This work applies three complementary XAI methods on the ChiliEdgeNet augmented-scenario checkpoint (seed 42), each using one representative image per class. Gradient SHAP (Erion et al., 2021) estimates pixel contribution from gradients relative to a baseline distribution; Occlusion Sensitivity (Zeiler and Fergus, 2014) occludes 32×32-pixel regions in turn and measures the change in output; Attention Rollout (Abnar and Zuidema, 2020) aggregates attention flow through the Transformer block across the full 14×14 grid. Because the architecture uses mean pooling and has no *CLS* token, the Rollout map is produced by averaging attention flow across the 196 tokens, rather than tracing from a single output token.

The three maps offer distinct perspectives (Figures 4–6): Gradient SHAP gives pixel-level attribution distributed over the object surface; Occlusion shows region-level sensitivity; Attention Rollout reflects the Transformer's internal attention propagation. Across the five illustrative images, high-response regions generally intersect with the leaf/fruit or the diseased area, though the degree of concentration differs across methods and classes. Since the evaluation uses only one image per class and no ground-truth symptom segmentation exists, this result is an exploratory qualitative survey only; it does not prove the model has learned correct pathological features, fully ignored the background, or captured a genuine causal relationship. The 14×14 grid provides 196 spatial positions, four times more than a 7×7 grid, giving the Rollout map finer visual resolution.

<!-- IMG2 -->
![Figure 4. Occlusion Sensitivity for the five disease classes (top: original image; bottom: sensitivity map overlay). Bright regions are where occlusion costs the model the most confidence; this technique treats the model as a black box.](../../results/notebook_c2/xai_occlusion.png)

![Figure 5. Gradient SHAP for the five classes (top: original image; bottom: pixel-attribution magnitude overlay), computed with 20 samples and two constant baseline tensors (0 and 1) in the normalized input space.](../../results/notebook_c2/xai_shap.png)

![Figure 6. Attention Rollout of the Tiny Transformer block on the 14×14 grid, aggregated by averaging attention flow across the 196 tokens since the model uses mean pooling with no *CLS* token.](../../results/notebook_c2/xai_attn_rollout.png)
<!-- /IMG2 -->

#### 4.4.2. Comparison with a General-Purpose Multimodal Model

This work uses the general-purpose multimodal model Gemini 3.1 Flash Lite, API model ID *gemini-3.1-flash-lite*, accessed on 8 July 2026 (Google DeepMind, 2026) with default generation settings to classify the raw images directly in a zero-shot setting. The model was prompted, in Vietnamese, to classify each image into exactly one of the five ChiliEdge-DS labels; the *predicted_class* field is scored directly against these five strings, with no synonym translation or post-hoc mapping. All 373 API calls returned valid, parseable output, with no exclusions. The exact prompt text is preserved in the study's accompanying materials rather than reproduced here, since this is a secondary reference comparison, not the paper's main contribution: Gemini was not fine-tuned on ChiliEdge-DS, whereas ChiliEdgeNet is a specialized classifier trained with supervision for exactly the five classes.

Gemini reaches 78.55% accuracy, while ChiliEdgeNet reaches 98.93% on the same 373-image test set, matching Table 3. Of the three *yellow_leaf* images misclassified by ChiliEdgeNet as *leaf_curl* (Section 4.2), Gemini correctly classified all three as *yellow_leaf*.

**On latency.** The approximately 105x figure contrasts 20.24 ms of local inference with a 2,123 ms median cloud API round-trip. The two figures are not equivalent: the round-trip includes network latency, server-side queueing, and provider-side batching—none of which is computation intrinsic to the model. This is a comparison between two deployment modes, not a claim that one architecture computes roughly 105x faster than the other. Similarly, although accuracy is computed on the same 373 images, the accuracy gap does not constitute a peer comparison. The commercial endpoint may also be updated over time, so this comparison is reproducible only against the stated model ID, prompt, and access date.

**Table 9.** Comparison of ChiliEdgeNet with Gemini 3.1 Flash Lite on the full 373-image test set (accessed 8 July 2026).

| System | Accuracy | Latency/image |
|---|---|---|
| **ChiliEdgeNet (local CPU)** | **98.93%** | **20.24 ms** (local inference) |
| Gemini 3.1 Flash Lite (*gemini-3.1-flash-lite*) | 78.55% | 2,123 ms (median API round-trip) |
