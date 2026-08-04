## 4. Experiments and Results

### 4.1. Setup

Experiments run on a Kaggle Notebook (NVIDIA Tesla T4) in PyTorch 2.10, training each model under two scenarios — **baseline** and **augmented** — per the configuration in Table 2; validation/test are never augmented. The best-validation checkpoint is reloaded before test evaluation. Every variant — proposed model and all ablations — uses **seed 123** and is re-initialised from it before each run, so variants differ only in the component under study.

**Table 2. Training configuration.**

| Item | Value |
|---|---|
| Input size | 224x224 |
| Baseline transform | Resize, ToTensor, Normalize |
| Augmented transform | 30-degree rotation, ColorJitter, horizontal flip, affine |
| Epochs | 30 |
| Batch size | 32 |
| Loss | Cross-entropy, label smoothing 0.05 |
| LR schedule | 10% linear warm-up, then cosine decay |
| Learning rate (backbone / new blocks) | 1e-4 / 1e-3 |
| Seed | 123 (every variant) |

#### 4.1.1. Run Count and Statistical Testing

Results for the proposed model and its ablations come from a **single run (seed 123)**, a deliberate choice compensated by a stricter test protocol. With 373 test images each image is worth 0.27 pp, so a mean ± SD over a few seeds still cannot separate models in the 98–99% band (Section 6). Instead every model-to-model comparison **among our own variants** uses a **per-image paired McNemar test** on **exactly the same 373 images**, which answers *"is this difference within random noise?"* directly without estimating run-to-run variance. Each McNemar entry reports **b / c**: images the first model gets right and the second wrong, and vice versa. Every proposed-model and ablation variant in this paper is evaluated at the same seed 123 under an identical protocol, so the comparisons in Section 4.4 are like-for-like. The base architecture [5] compared against in Section 4.2 is a separate single run at **seed 42** (not reproduced at seed 123) — that comparison is therefore between two single runs under the same training protocol, not a seed-matched pair; Section 6 states this limitation.

#### 4.1.2. Inference Latency

The CPU figure used throughout (**19.28 ms/image**, 51.9 images/s) is measured on the CPU of the same Kaggle VM used for training — **Intel(R) Xeon(R) @ 2.00 GHz, 4 logical cores** — with PyTorch (FP32), batch 1, 10 warm-up then 50 timed iterations. On that machine a Tesla T4 gives **5.21 ms/image** at batch 1 and **0.82 ms/image** at batch 32. Two caveats: this is a **server-class x86 CPU, not edge hardware** (Section 6), and parameter count is **not** a proxy for speed — holding the grid at 14x14 rather than 7x7 makes the final convolutional stages work over four times as many spatial positions. Parameters govern memory footprint; feature-map resolution governs runtime. We report both.

### 4.2. Comparison with SOTA Models

The proposed model reaches **99.20%** (macro-F1 0.9923) baseline and **99.20%** (macro-F1 0.9922) augmented — the two scenarios agree on exactly the same 370/373 clean images. Table 3 places it against all 27 configurations reproduced from 15 SOTA studies under the same protocol: it sits among the accuracy leaders while being among the lightest. Because the baselines are single-run reproductions not individually tuned for this dataset, Table 3 reads as competitiveness under a common protocol, not a definitive ranking of the architectures.

#### 4.2.1. Comparison with the Base Architecture

This is the comparison the paper's claim rests on. ChiliEdgeNet reaches **99.20%** augmented, higher than RTR_Lite_MobileNet [5] (**98.12%**, macro-F1 0.9821) by **+1.07 pp — 4 images out of 373** — while using only **671,173 against 1,032,568 parameters, 35% fewer**. This is the headline result: more accurate and lighter at the same time, with no trade-off. A protocol note: ChiliEdgeNet here runs at **seed 123**, while the RTR_Lite_MobileNet [5] figure is a separate single reproduction at **seed 42** (Section 4.1) — same training/evaluation protocol, but not the same seed, so this is a comparison between two single runs rather than a seed-matched pair.

The 4-image gap is small and deserves to be read at its true confidence: the two 95% Wilson intervals ([96.18, 99.09] and [97.66, 99.73]) overlap substantially — the ceiling of a 373-image test set cannot resolve two models both near 99% — and both come from a single run each (Section 6). The contribution that does not depend on this noise, and is therefore the one the paper emphasizes most, remains **parameter efficiency — the parameter count is an exact figure that needs no significance test.** Clean accuracy is also not the only axis that matters for a field tool; Section 4.3 evaluates degraded conditions, where the two *scenarios* separate sharply even though the *architectures* do not.

**Table 3. Two-scenario accuracy, macro-F1, and parameters on the chili dataset.** *(Except ChiliEdgeNet, which runs at seed 123, rows are single seed-42 runs under the same protocol; only the most accurate variant of each model family is kept. Differences below ~1 pp correspond to fewer than 4 images out of 373, fall inside the Wilson interval, and should **not** be read as a reliable ranking — see Section 6.)*

| Model | Acc Baseline (%) | Acc Augmented (%) | Macro-F1 (Aug) | Parameters |
|---|---|---|---|---|
| **ChiliEdgeNet (ours)** | **99.20** | **99.20** | **0.9922** | **671,173** |
| RTR_Lite_MobileNet [5] | 97.32 | 98.12 | 0.9821 | 1,032,568 |
| GrapeLeafNet [13] | 91.15 | 97.59 | 0.9769 | 5,308,035 |
| PMVT-XS [6] | 82.31 | 96.25 | 0.9644 | 2,367,515 |
| ResNet-50 [11] | 95.98 | 95.71 | 0.9586 | 23,597,957 |
| EfficientNetB0 [11] | 94.37 | 94.37 | 0.9463 | 4,055,976 |
| TrIncNet [3] | 80.70 | 90.62 | 0.9092 | 7,186,661 |
| InceptionV3 + SVM [14] | 93.83 | 90.35 | 0.9069 | 25,112,264 |
| Meta-Ensemble [1] | 88.74 | 88.74 | 0.8908 | 1,252,638 |
| CustomMNetV3L [12] | 87.67 | 86.86 | 0.8639 | 3,275,909 |
| ULEN [8] | 93.03 | 85.52 | 0.8603 | 28,069 |
| MaizeFormerX [2] | 79.89 | 82.04 | 0.8185 | 3,553,061 |
| AlexNet [11] | 85.52 | 77.75 | 0.7747 | 46,767,493 |
| Lite-MDC [7] | 68.10 | 76.68 | 0.7658 | 2,288,965 |
| ViT [15] | 72.39 | 73.99 | 0.7472 | 50,013,061 |
| MRW-CNN [4] | 74.26 | 73.73 | 0.7377 | 797,413 |
| PDLC-ViT [10] | 74.26 | 70.78 | 0.7054 | 17,547,269 |
| LDI-NET [9] | 70.78 | 68.63 | 0.6839 | 8,798,597 |

<!-- SBS -->
#### 4.2.2. Error Analysis

On the 373 augmented test images the model misclassifies only 3 (Fig. 1). All three involve `yellow_leaf`: one `yellow_leaf` image read as `leaf_curl`, and one `healthy` and one `leaf_spot` absorbed into `yellow_leaf` — the ambiguous chlorosis/curling boundary where a human eye also hesitates. With so few errors the test set has limited resolving power: 99.20% carries a 95% Wilson interval of [97.66%, 99.73%], overlapping every model within ~1 pp in Table 3 (Section 6).

![Figure 1. Confusion matrix of the proposed model on the test set (augmented scenario).](figs/c14_cm_s2.png)
<!-- /SBS -->

### 4.3. Robustness to Degraded Capture Conditions

The decisive question for a field tool, as Section 4.2 already posed it, is which model retains accuracy once the image is no longer perfect — not who leads on a clean-image axis that has already saturated. We add degradations applied at **test time** in the spirit of ImageNet-C: neither transform appears during training, and the corrupted sets are generated **once with fixed seeds**, so every model in Sections 4.3–4.4 is scored on pixel-identical inputs.

**Table 4. The two degradation conditions and what they model.** *(Severity measured over all 373 images.)*

| Condition | Real-world factor | Parameter | PSNR | Pixels changed |
|---|---|---|---|---|
| Gaussian blur | camera shake, defocus | kernel k = 5 | 33.4 dB | 21.41% |
| Brightness reduction | shooting under foliage or at dusk | factor x0.4 | 9.9 dB | 99.97% |

**Table 5. The proposed model across the three conditions, with 95% Wilson intervals (n = 373).**

| Condition | Baseline (%) | Augmented (%) | Wilson 95% (Augmented) |
|---|---|---|---|
| Clean | 99.20 | **99.20** | [97.66 – 99.73] |
| Blur (k=5) | 91.42 | **95.98** | [93.47 – 97.55] |
| Low light (x0.4) | 86.06 | **99.20** | [97.66 – 99.73] |
| **Mean** | 92.23 | **98.12** | – |

Table 5 shows what augmentation actually buys, entirely hidden if one looks only at clean images. On clean data the two scenarios are **exact ties** (both 370/373). Under **blur** they differ by **4.56 pp** (91.42% against 95.98%, 17 images), and under **low light** by **13.14 pp** — 86.06% against 99.20%, i.e. **49 images** — with their Wilson intervals ([82.17, 89.21] against [97.66, 99.73]) not coming close to overlapping. The ColorJitter/RandomAffine components teach tolerance to intensity and geometric variation, real capture conditions when photographing under foliage or at dusk. **The proposed model is therefore the augmented one**; all ablations use it.

Worth stating plainly: on this dataset **most of the robustness to blur and low light is bought by the training recipe, not the architecture**. Section 4.4 measures what the architecture adds on top of that — smaller than the training recipe, but no longer negligible: the Transformer block's clearest contribution lands exactly on the blur axis.

![Figure 2. The test conditions: clean, Gaussian blur (k = 5), and brightness reduction (x0.4), one sample per class.](figs/c14_bench_samples.png)

### 4.4. Ablation Study

Every variant is **retrained from scratch** with the component under study removed or replaced, using the same seed 123, recipe and 30 epochs; the baseline row is the proposed model trained in the same pipeline. We deliberately do **not** use inference-time knockout: that measures how much a trained network *depends* on a block, not how much the block *contributes*, and conflating the two is a known methodological error.

#### 4.4.1. Block-Level Ablation

**Table 6. Removing each block and retraining (augmented scenario, seed 123).**

| Variant | Parameters | Clean | Blur | Low light | Mean |
|---|---|---|---|---|---|
| **Full (proposed)** | 671,173 | **99.20** | **95.98** | **99.20** | **98.12** |
| w/o Tiny Transformer | 555,845 | 99.20 | 90.08 | 97.86 | 95.71 |

**Table 7. Per-image paired McNemar tests, full model against each variant.** *(Positive delta = the full model is better, i.e. the block contributes. b / c = images the full model gets right and the variant wrong, and vice versa.)*

| Block removed | Condition | Delta (pp) | b / c | *p* |
|---|---|---|---|---|
| Tiny Transformer | **Blur** | **+5.90** | 26 / 4 | **0.0001** |
| Tiny Transformer | Low light | +1.34 | 5 / 0 | 0.0625 |
| Tiny Transformer | Clean | 0.00 | 0 / 0 | 1.000 |

**The Tiny Transformer's clearest measurable contribution is under blur, and it reaches statistical significance.** Removing it costs **5.90 pp** under blur — 26 images broken against just 4 improved, *p* = 0.0001 — while on clean images the two models are **identical** (0/0, *p* = 1.000). It adds a further 1.34 pp under low light (*p* = 0.0625, at the edge of *p* < 0.05). The block is worth 115,328 parameters. The wider point survives regardless: whatever the Transformer contributes, it is **not** clean accuracy — it only shows up once the input is already degraded.

#### 4.4.2. Technique-Level Ablation

The technical contribution named in the title is the integration of **three** parameter-efficient LLM techniques. Block-level ablation cannot separate them: with a single Transformer layer, removing "the Transformer block" removes all three at once. Each variant here replaces **exactly one** technique with its standard counterpart, holds everything else fixed, and retrains from scratch. This section answers whether each technique earns its parameter saving.

**Table 8. Replacing each LLM technique with its standard counterpart (augmented scenario, seed 123).**

| Variant | Replaces | Δ params | Clean | Blur | Low light | Mean |
|---|---|---|---|---|---|---|
| **Proposed** | 2D-RoPE + GQA + RMSNorm | — | 99.20 | 95.98 | **99.20** | **98.12** |
| `learned_pe` | 2D-RoPE swapped for learned PE | **+25,088** | 99.20 | 95.44 | 98.66 | 97.77 |
| `mha` | GQA swapped for Multi-Head Attention | **+16,384** | 99.20 | **96.25** | 98.12 | 97.86 |
| `layernorm` | RMSNorm swapped for LayerNorm | +384 | 98.66 | **96.25** | 97.86 | 97.59 |

**Table 9. Per-image paired McNemar tests, proposed block against each single-technique replacement.** *(Positive delta = the proposed block is better.)*

| Technique replaced | Condition | Delta (pp) | b / c | *p* |
|---|---|---|---|---|
| 2D-RoPE swapped for learned PE | Low light | +0.54 | 2 / 0 | 0.500 |
| 2D-RoPE swapped for learned PE | Blur | +0.54 | 4 / 2 | 0.688 |
| 2D-RoPE swapped for learned PE | Clean | 0.00 | 1 / 1 | 1.000 |
| GQA swapped for MHA | Low light | +1.07 | 5 / 1 | 0.219 |
| GQA swapped for MHA | Blur | −0.27 | 6 / 7 | 1.000 |
| GQA swapped for MHA | Clean | 0.00 | 2 / 2 | 1.000 |
| RMSNorm swapped for LayerNorm | Low light | +1.34 | 5 / 0 | 0.063 |
| RMSNorm swapped for LayerNorm | Blur | −0.27 | 4 / 5 | 1.000 |
| RMSNorm swapped for LayerNorm | Clean | +0.54 | 2 / 0 | 0.500 |

**All three techniques are close to free — none loses with statistical significance.** This is the clearest departure from our previous architecture version (with RES_Block, seed 42), where RMSNorm lost to LayerNorm significantly under blur; at the current architecture and seed, the comparison closest to *p* < 0.05 anywhere in Table 9 is RMSNorm/low light (*p* = 0.063), and there RMSNorm (proposed) actually **leads** LayerNorm rather than losing.

**2D-RoPE is close to free.** It encodes position with **zero learned parameters**; the learned alternative needs a 196x128 = **25,088-parameter** table, 3.7% of the model. No axis separates them (*p* between 0.500 and 1.000); the proposed block edges out the learned alternative under both degraded axes and ties on clean. 2D-RoPE therefore buys a 25,088-parameter saving at no measurable accuracy cost — and unlike a learned table, carries no dependence on the token count fixed at training time.

**GQA is close to free.** Sharing two key/value heads across four query heads saves **16,384 parameters**. The proposed block ties MHA on clean, trails insignificantly under blur (−0.27 pp, *p* = 1.000), and leads under low light (+1.07 pp, *p* = 0.219) — none of it significant. The memory advantage that motivated GQA for large LLMs is not material at this scale, but the parameter saving is real and does not carry a measurable cost here.

**RMSNorm is also close to free — no longer the "losing" technique it was in our previous architecture version.** LayerNorm is only insignificantly better under blur (−0.27 pp, *p* = 1.000); under low light and on clean, RMSNorm (proposed) actually leads LayerNorm (+1.34 pp *p* = 0.063 and +0.54 pp *p* = 0.500), though neither reaches significance. Dropping LayerNorm's mean-subtraction and shift terms saves 0.06% of the model with no measurable accuracy cost at the current architecture and seed. This is a notable reversal from our own previous architecture version (with RES_Block, seed 42), where the same comparison gave *p* = 0.019 in the opposite direction — further reason Section 6 frames these ablation findings as single-seed observations, not established effects.

**Statistical scope.** All results here are **single-seed**. Table 9 comprises 9 tests (3 variants x 3 conditions), so the Bonferroni threshold is α = 0.05/9 ≈ 0.006; **no comparison in Table 9 reaches even *p* < 0.05** (the closest is RMSNorm/low light at *p* = 0.063). We therefore present this section as **observations that inform design**, not established effects — and specifically do not conclude that any of the three techniques "loses." What depends on no test is the arithmetic: 2D-RoPE and GQA together remove **41,472 parameters** and no measurement in Tables 8–9 shows an accuracy cost for any of the three techniques.

### 4.5. Realtime Laptop-Camera Demo

Beyond the static benchmark on the fixed test set (Sections 4.2–4.4), we additionally exercise the proposed model in a **live, real-time** inference pipeline: opening the laptop webcam, running the exact ChiliEdgeNet checkpoint (augmented, 671,173 parameters) inside a camera loop, classifying only a fixed centre ROI, and measuring end-to-end latency (preprocessing plus inference) on the laptop CPU for each prediction. The input is a diseased-leaf image displayed on a phone screen and held up to the ROI; ground truth is manually tagged at the moment of capture so that correctness can be scored.

**Table 11. Realtime laptop-camera demo — 5 measurements, one per disease class (n = 5).**

| Class (ground truth) | Prediction | Correct | Confidence (%) | Latency (ms) |
|---|---|---|---|---|
| anthracnose | anthracnose | Yes | 93.98 | 23.62 |
| healthy | healthy | Yes | 95.24 | 15.51 |
| leaf_curl | leaf_curl | Yes | 94.33 | 18.20 |
| leaf_spot | leaf_spot | Yes | 95.94 | 13.73 |
| yellow_leaf | yellow_leaf | Yes | 95.07 | 19.34 |
| **Mean** | | **5/5 (100%)** | **94.91** | **18.08** |

All 5/5 measurements were classified correctly, with mean top-1 confidence 94.91% and mean latency 18.08 ms/image (median 18.20; min 13.73; max 23.62 ms) — measured on the laptop CPU, distinct from the machine behind the 19.28 ms/image figure (Kaggle VM CPU) in Section 4.1, since it also includes webcam capture and preprocessing overhead rather than inference alone.

This result must be read within its true scope: **n = 5 is far too small to be a statistical benchmark** — unlike the 373-image test set used throughout the paper — so no Wilson interval or McNemar test is computed for this table. The laptop CPU here remains a consumer-class x86 chip, not a genuine embedded or mobile chip, and the input images are a secondary capture of a phone screen rather than real field photographs. The result should therefore be read only as **qualitative proof of feasibility**: the camera-to-preprocessing-to-inference-to-display pipeline works correctly end to end, and latency remains in the low tens of milliseconds on an ordinary consumer CPU with no GPU required. It complements — and does not substitute for — the still-missing measurement on physical edge hardware (Section 6).
