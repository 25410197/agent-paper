### 4.4. Model Interpretability

The analyses in this section use **ChiliEdgeNet's original checkpoint**, trained only on ChiliEdge-DS and used to produce the in-domain results in Table 3. The augmented checkpoint used for the domain-transfer experiment in Section 4.5 is a separate adapted model and is not used to produce the interpretability results below.

#### 4.4.1. Explainable AI (XAI)

High accuracy alone is not sufficient to interpret model behaviour, particularly since ChiliEdge-DS predominantly uses a controlled white background. This work applies **three complementary XAI methods** on the ChiliEdgeNet augmented-scenario checkpoint (seed 42), each using one representative image per class. **Gradient SHAP** (Erion et al., 2021) estimates pixel contribution from gradients relative to a baseline distribution; **Occlusion Sensitivity** (Zeiler and Fergus, 2014) occludes 32×32-pixel regions in turn and measures the change in output; **Attention Rollout** (Abnar and Zuidema, 2020) aggregates attention flow through the Transformer block across the full 14×14 grid. Because the architecture uses mean pooling and has no `CLS` token, the Rollout map is produced by averaging attention flow across the 196 tokens, rather than tracing from a single output token.

The three maps offer distinct perspectives (Figures 4–6): Gradient SHAP gives pixel-level attribution distributed over the object surface; Occlusion shows region-level sensitivity; Attention Rollout reflects the Transformer's internal attention propagation. Across the five illustrative images, high-response regions generally intersect with the leaf/fruit or the diseased area, though the degree of concentration differs across methods and classes. Since the evaluation uses only one image per class and no ground-truth symptom segmentation exists, this result is an exploratory qualitative survey only; it does not prove the model has learned correct pathological features, fully ignored the background, or captured a genuine causal relationship. The 14×14 grid provides 196 spatial positions, four times more than a 7×7 grid, giving the Rollout map finer visual resolution.

<!-- IMG2 -->
![Figure 4. Occlusion Sensitivity for the five disease classes (top: original image; bottom: sensitivity map overlay). Bright regions are where occlusion costs the model the most confidence; this technique treats the model as a black box.](../../results/notebook_c2/xai_occlusion.png)

![Figure 5. Gradient SHAP for the five classes (top: original image; bottom: pixel-attribution magnitude overlay), computed with 20 samples and two constant baseline tensors (0 and 1) in the normalized input space.](../../results/notebook_c2/xai_shap.png)

![Figure 6. Attention Rollout of the Tiny Transformer block on the 14×14 grid, aggregated by averaging attention flow across the 196 tokens since the model uses mean pooling with no `CLS` token.](../../results/notebook_c2/xai_attn_rollout.png)
<!-- /IMG2 -->

#### 4.4.2. Comparison with a General-Purpose Multimodal Model

This work uses the general-purpose multimodal model **Gemini 3.1 Flash Lite**, API model ID `gemini-3.1-flash-lite`, accessed through the Gemini API with an API key created and managed via Google AI Studio. The model was accessed on **8 July 2026** (Google DeepMind, 2026) to classify the raw images directly in a zero-shot setting. No temperature, top-p, or token limit was set separately; the run used the endpoint's default generation configuration at the access date. This is a reference comparison, not a fully matched baseline: Gemini was not fine-tuned on ChiliEdge-DS, whereas ChiliEdgeNet is a specialized classifier trained with supervision for exactly the five classes.

**Prompt used.** The prompt below is an English translation, provided for readability; the exact text actually sent to the API was in unaccented Vietnamese (reproduced verbatim in the Vietnamese-language version of this paper, Section 4.4.2, for exact reproducibility). The translation is faithful sentence-for-sentence, including the instruction that the `label` field itself be written in Vietnamese — that field's values are not used for scoring (see below) and were left in Vietnamese rather than re-run, since translating them would not change any reported number:

```text
You are a plant pathology expert. The image shows a chili pepper leaf/fruit that may be diseased.
Identify ONE single rectangular region bounding the clearest abnormality/disease symptom
in the image (disease spots, rot lesions, yellowing areas, curling/wrinkling areas...).
If the leaf is completely healthy, bound the entire leaf/fruit.

Then classify the image into EXACTLY ONE of the following 5 labels (you must choose exactly 1 of the 5,
you may not create another label): anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf.

Return ONLY a single JSON array in the following format, with nothing else besides the JSON:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "short_symptom_name_in_vietnamese", "predicted_class": "one_of_the_5_labels_above", "reason": "one short sentence explaining the VISUAL BASIS for choosing that label"}]
box_2d coordinates on a 0-1000 scale (normalized to image height/width).
```

**Label mapping and invalid outputs.** The `predicted_class` field is scored directly against exactly one of the five strings `anthracnose`, `healthy`, `leaf_curl`, `leaf_spot`, and `yellow_leaf`; no synonym translation, manual relabeling, or post-hoc mapping is applied. The `label` field and the `reason` sentence are stored for inspection only and do not enter the accuracy computation. The parser prefers JSON; if the JSON is malformed, a regular expression is used to extract the fields. API errors, missing required fields, unparseable output, or a `predicted_class` outside the five labels are retried up to three times; if still unsuccessful, the record is tagged `gemini_error`, excluded from Gemini's accuracy denominator, and the error count must be reported separately alongside coverage.

The artifact currently stores all **373 images**, run on **8 July 2026**, with **100%** of calls valid and no labels outside the allowed set. Gemini reaches **78.55%** accuracy, while ChiliEdgeNet reaches **98.93%** on the same 373-image test set — matching Table 3. Of the three `yellow_leaf` images in the test set, two are also two of the three errors where ChiliEdgeNet mispredicted `leaf_curl`, as noted in Section 4.2.2, while ChiliEdgeNet correctly classified the third; Gemini correctly classified all three as `yellow_leaf`.

**On latency.** The approximately 105x figure contrasts **20.24 ms of local inference** with a **2,123 ms median cloud API round-trip**. The two figures are not equivalent: the round-trip includes network latency, server-side queueing, and provider-side batching — none of which is computation intrinsic to the model. This is a comparison between two deployment modes, not a claim that one architecture computes roughly 105x faster than the other. Similarly, although accuracy is computed on the same 373 images, the accuracy gap does not constitute a peer comparison, since the two systems differ in training regime and degree of specialization. The commercial endpoint may also be updated over time, so this comparison is reproducible only against the stated model ID, prompt, and access date.

**Table 9. Comparison of ChiliEdgeNet with Gemini 3.1 Flash Lite on the full 373-image test set (accessed 8 July 2026).**

| System | Accuracy | Latency/image |
|---|---|---|
| **ChiliEdgeNet (local CPU)** | **98.93%** | **20.24 ms** (local inference) |
| Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) | 78.55% | 2,123 ms (median API round-trip) |
