# ChiliEdgeNet: A Lightweight CNN-Transformer Hybrid Toward On-Device Chili Pepper Disease Recognition

**Keywords:** chili pepper disease classification, lightweight deep learning, CNN–Transformer, edge AI, few-shot domain adaptation, explainable AI.

## Abstract

Plant-disease diagnosis aimed at resource-constrained devices needs a model that is both accurate and lightweight. This work proposes ChiliEdgeNet, a CNN-Transformer hybrid architecture built by truncating MobileNetV2 and integrating a Tiny Transformer block. We also introduce ChiliEdge-DS, a publicly released dataset of 2,461 chili pepper leaf and fruit images across five classes. ChiliEdgeNet reaches 98.93% accuracy with 671,173 parameters, about 35% fewer than the base architecture. The model is 2.7 MB and has an inference latency of 20.24 ms/image on a server-class CPU. In a synthetic brightness-reduction stress test at factor 0.2, the full model outperforms the variant without the Transformer by 3.75 percentage points. On a clean cross-dataset query set of 1,463 images across four classes shared with Mendeley Data, accuracy rises from 30.96% at the original checkpoint to 89.75% after adapting with 42 support images; both checkpoints use seed 42. A paired McNemar test records 870 images turning from incorrect to correct and 10 in the opposite direction ($p = 1.83\times10^{-242}$). These results position ChiliEdgeNet as a candidate for offline inference on resource-constrained hardware, but direct evaluation on edge hardware and domain-transfer confirmation on more independent datasets remain necessary.

## 1. Introduction

Early detection of plant disease from images can support monitoring and treatment. However, deployment at the point of image capture requires a model with low computational cost, offline operation, and sufficiently high accuracy (Joseph et al., 2024; Maurya et al., 2024; Duhan et al., 2025) — requirements that matter most for resource-constrained devices.

Lightweight CNNs offer an advantage in parameter count and local spatial inductive bias; a sufficiently deep network can still reach a large receptive field, but it does not directly model pairwise interactions between every position the way self-attention does. Vision Transformers support exploiting global relations, but standard configurations typically carry higher computational cost (Gole et al., 2023; Li et al., 2023). Hybrid CNN-Transformer architectures can therefore reconcile the two properties.

Within the body of work surveyed in Section 2, results are reported on different datasets and protocols, so they do not yet form a shared comparison that simultaneously analyzes each component's contribution, response to synthetic degradation, and background domain gap. The gap considered here is not the absolute absence of each individual analysis, but the absence of a protocol that combines these evaluation axes on the same data split and training procedure. In particular, high accuracy on a controlled-background dataset does not guarantee performance on natural-background images. XAI is used to probe prediction behaviour, while a general-purpose multimodal model provides a reference comparison.

Building on the RTR_Lite_MobileNet architecture (Duhan et al., 2025), based on MobileNetV2 (Sandler et al., 2018), this work proposes ChiliEdgeNet, a CNN-Transformer hybrid under 700 thousand parameters. The work focuses on four main contributions:

- **Lightweight architecture:** ChiliEdgeNet reaches 98.93% accuracy with 671,173 parameters, about 35% fewer parameters than the base architecture under the same seed-42 protocol. The model is 2.7 MB and has an inference time of 20.24 ms/image on a Kaggle virtual-machine CPU.
- **The ChiliEdge-DS dataset:** 2,461 chili pepper leaf and fruit images across five classes, collected in southern Vietnam against a controlled white background, split into fixed train/validation/test sets and released on Kaggle (Hải, 2026c).
- **Controlled evaluation and domain transfer:** the work runs a degraded-image benchmark and a retrain-from-scratch ablation with seed 42. On a clean query set of 1,463 images across four classes shared with Mendeley Data, the original checkpoint reaches 30.96% and the checkpoint after adding 42 support images reaches 89.75%. Both checkpoints use seed 42, the same architecture and base hyperparameters; the SHA-256 intersection between support and query is zero. A paired comparison records 870 images turning from incorrect to correct and 10 in the opposite direction. The adapted checkpoint serves only the domain-transfer analysis and does not replace the main model.
- **Interpretation and reference comparison:** three XAI methods are used for qualitative analysis. A general-purpose multimodal model is evaluated zero-shot on the full 373 test images as a reference comparison, not a peer baseline.
- **Laptop-camera integration:** over 62 predictions at batch size 1 on an Intel Core i7-13650HX CPU, the median preprocessing-inference latency is 29.70 ms. The measurement excludes frame capture, ROI cropping, format conversion before the ROI, UI rendering, or file writes; this figure is therefore not a camera FPS or full-pipeline latency.

The remainder of this paper is organized as follows. Section 2 reviews the theoretical background and related architectural directions. Section 3 describes ChiliEdge-DS and the proposed ChiliEdgeNet architecture. Section 4 reports the experimental setup, results, ablations, interpretability analysis, and cross-dataset evaluation. Section 5 discusses the limitations of the current evidence, and Section 6 concludes.

## 2. Theoretical Background and Related Work

### 2.1. Lightweight CNNs and MobileNetV2

With an input feature map of size $(H \times W \times C_{in})$, kernel $(k \times k)$ and $C_{out}$ output channels, a standard convolution requires approximately

$$HWk^2C_{in}C_{out}$$

multiply-accumulate operations. Depthwise separable convolution decomposes this computation into a per-channel depthwise convolution and a $1\times1$ pointwise convolution that mixes information across channels. The cost is then approximately

$$HW\left(k^2C_{in}+C_{in}C_{out}\right),$$

giving a ratio relative to standard convolution of

$$\frac{1}{C_{out}}+\frac{1}{k^2}.$$

This is the basis for how the MobileNet family reduces computation on resource-constrained devices. MobileNetV2 organizes these convolutions inside the *inverted residual with linear bottleneck* block (Sandler et al., 2018). Each block first uses a pointwise convolution to expand the channel count, then a depthwise convolution to extract spatial features, and finally a linear projection back to a low-dimensional space. When the stride equals 1 and the input/output sizes are compatible, a shortcut connects the two thin bottleneck representations. The final projection uses no activation function to limit information loss in the low-dimensional space. ChiliEdgeNet inherits this local inductive bias and computational efficiency from the retained portion of MobileNetV2.

### 2.2. Image Tokens and Self-Attention

Let the CNN backbone produce a feature map $F\in\mathbb{R}^{C\times H\times W}$. After a channel projection, the feature map is flattened into a sequence

$$X\in\mathbb{R}^{N\times d},\qquad N=HW,$$

where each row is a token representing one spatial location. From $X$, the query, key, and value matrices are formed by

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

Scaled dot-product attention (Vaswani et al., 2017) is defined as

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)V,$$

where $d_h$ is the dimensionality of each attention head. Unlike convolution, which aggregates only a local neighbourhood directly at each layer, self-attention lets every token interact directly with every other token. In exchange, the attention matrix has $O(N^2)$ complexity in the number of tokens. ChiliEdgeNet uses a $14\times14$ grid, i.e. $N=196$, to retain more spatial positions than a $7\times7$ grid while still bounding the cost of a small Transformer block.

The Transformer block in this work uses a pre-norm structure with two residual branches:

$$X' = X + \operatorname{GQA}(\operatorname{RMSNorm}(X)),$$

$$Y = X' + \operatorname{MLP}(\operatorname{RMSNorm}(X')).$$

The MLP consists of two linear layers, GELU, and dropout. The residual connection provides a direct path for representations and gradients; pre-norm normalizes the input of each transformation rather than normalizing after the addition.

### 2.3. Grouped-Query Attention

Standard multi-head attention creates separate keys and values for every query head. Grouped-query attention (GQA) sets the number of query heads $h_q$ larger than the number of key/value heads $h_{kv}$, then lets a group of $h_q/h_{kv}$ query heads share one key/value pair (Ainslie et al., 2023). GQA sits between multi-head attention, where $h_q=h_{kv}$, and multi-query attention, where every query head shares a single key/value head.

If the embedding has dimension $d$ and the projections use no bias, the query and output projections of GQA still require $2d^2$ parameters, while the key and value require

$$2d^2\frac{h_{kv}}{h_q}.$$

The total parameter count of the four attention projections is therefore

$$P_{GQA}=2d^2+2d^2\frac{h_{kv}}{h_q}.$$

With $d=128$, $h_q=4$, and $h_{kv}=2$, each key/value head serves two query heads and the module avoids 16,384 parameters compared with four-head multi-head attention. In this work's implementation, the key and value are projected with two heads and then repeated by group before computing attention. The benefit asserted here is therefore a reduction in projection parameters; we do not extrapolate this into a conclusion about inference-speed gains.

### 2.4. 2D-RoPE Positional Encoding and RMSNorm

Self-attention carries no inherent information about token order or coordinates. Rotary position embedding (RoPE) encodes position by rotating each pair of components of the query and key by a position-dependent angle (Su et al., 2024). For a two-dimensional pair $z=(z_1,z_2)$, the rotation at position $p$ has the form

$$R(p\theta)z=\begin{bmatrix}\cos(p\theta) & -\sin(p\theta)\\\sin(p\theta) & \cos(p\theta)\end{bmatrix}z.$$

The original RoPE formulation is presented for one-dimensional sequences (Su et al., 2024). For an image grid, this work uses axial 2D-RoPE: each query/key head is split into two halves; one half is rotated by the row coordinate ($y$) and the other by the column coordinate ($x$), following the RoPE-Axial extension for Vision Transformers of Heo et al. (2025). This injects 2D coordinates into the query and key without learning a positional-embedding table. With 196 tokens and embedding dimension 128, a learned positional table would require $196\times128=25{,}088$ parameters.

RMSNorm normalizes by the root-mean-square rather than subtracting the mean as LayerNorm does (Zhang and Sennrich, 2019). For a token $x\in\mathbb{R}^{d}$, the normalization used in the model is

$$\operatorname{RMSNorm}(x)_i=g_i\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon}},$$

where $g_i$ is a learned scale coefficient and $\varepsilon=10^{-6}$. The implementation has no shift bias. ChiliEdgeNet has three normalization points of dimension 128, so RMSNorm avoids $3\times128=384$ bias parameters relative to three equivalent LayerNorms. This is an exact structural difference; RMSNorm's effect on accuracy still must be confirmed by ablation rather than inferred directly from the formula.

### 2.5. Transfer Learning and Few-Shot Domain Adaptation

Transfer learning leverages knowledge from a source domain or task to support a target domain or task (Pan and Yang, 2010). This work involves two levels of transfer. First, the MobileNetV2 backbone uses ImageNet-pretrained weights before being fine-tuned on ChiliEdge-DS. Second, the cross-dataset experiment treats ChiliEdge-DS as the source domain and the Mendeley Data image set as the target domain, which differs in acquisition conditions, background, and symptom distribution.

Let the source training set be $\mathcal{D}_s$, the labelled support set of the target domain be $S_t$, and the target-domain query set be $Q_t$. The current adaptation procedure forms

$$\mathcal{D}_{adapt}=\mathcal{D}_s\cup S_t$$

and optimizes the average cross-entropy over this set:

$$\mathcal{L}_{CE}=-\frac{1}{|\mathcal{D}_{adapt}|}\sum_{(x,y)\in\mathcal{D}_{adapt}}\log p_{\theta}(y\mid x).$$

After adaptation, $Q_t$ is used only for evaluation. The minimal independence condition is

$$S_t\cap Q_t=\varnothing,$$

verified in this work by SHA-256 hashes of the image files. If a query image were present in the support set, the result would no longer reflect genuine generalization to unseen target images.

In episodic few-shot learning, an episode typically comprises an $N$-way $K$-shot problem with a controlled number of support examples per class; Prototypical Networks is one example of metric learning under this setup (Snell et al., 2017). The experiments in this work are not trained episodically, do not learn prototypes, and have an uneven number of support examples across the four classes. The precise term used in this paper is therefore supervised domain adaptation with a few-shot support set, not meta-learning or balanced $N$-way $K$-shot.

### 2.6. Baseline Selection Criteria

**Selection criteria for the baseline pool.** Table 3 comprises 17 baselines from 15 studies, selected by: (i) peer-reviewed work published 2023–2026 on image-based plant-disease classification; (ii) a clear architectural focus on deployability (lightweight CNN / ViT / hybrid) or a broad CNN-vs-ViT comparative study; (iii) architectural detail sufficient (layer configuration, embedding widths, head counts) to reproduce or adapt for the target task without contacting the authors; (iv) coverage of the four directions below rather than only the strongest family. Studies were excluded when they targeted detection/segmentation rather than classification, or when no corresponding classification output could be constructed. The baseline pool therefore represents the deployment-oriented literature of the last three years, not an exhaustive survey; the models were reimplemented and evaluated by us under a common protocol described in Section 4.1, so Table 3 does not use the figures originally published by the authors.

Studies that perform only detection or segmentation with no image-level classification output were excluded from the baseline pool. Multitask models are kept if the original architecture has an explicit classification branch that can be adapted to the target task. Under this criterion, PDLC-ViT (Hemalatha and Jayachandran, 2024) is kept because it has a classification head alongside its segmentation branch. Since ChiliEdge-DS provides no masks, the experiment implements only the classification branch and removes the segmentation head and segmentation loss; the compared version is therefore denoted PDLC-ViT (classification-only), not the original multitask model.

### 2.7. Related Architectural Directions

Lightweight CNNs keep parameters low through efficient attention or compact blocks. RTR_Lite_MobileNetV2 (Duhan et al., 2025) — our base architecture — adds triplet attention (Misra et al., 2021) and a spatial-attention residual block to MobileNetV2 (Sandler et al., 2018). ULEN (Wang et al., 2023) pushes ultra-lightweight design down to 28 thousand parameters; MRW-CNN (Joseph et al., 2024) and Lite-MDC (Bhagat et al., 2024) target real-time field deployment; CustomMNetV3L (Rahaman et al., 2026) adapts MobileNetV3-Large. Deep CNNs can extend the receptive field to aggregate broad context, but do not directly model pairwise interactions between every position the way self-attention does. Vision Transformers directly exploit relations between tokens but typically carry higher cost: ViT (Ali et al., 2025), TrIncNet (Gole et al., 2023) with a Trans-Inception block, PMVT (Li et al., 2023) for mobile deployment, and PDLC-ViT (Hemalatha and Jayachandran, 2024) for multitask localization plus classification. Hybrid CNN-Transformer architectures closest to this line of work include GrapeLeafNet (Karthik et al., 2024), MaizeFormerX (Rahman et al., 2026), and LDI-NET (Yang et al., 2024), typically at 3–9 million parameters. Broad comparative studies include Mishra et al. (2026), Sujatha et al. (2025), and the lightweight IoT meta-ensemble of Maurya et al. (2024).

### 2.8. Evaluation Gap

The works above use different datasets, splits, and training protocols; consequently, the figures each author reports do not form a single shared comparison that can simultaneously assess architectural contribution, response to synthetic image degradation, and background domain gap. This observation does not mean that every individual study lacks all such analyses, but rather points to the need for a protocol that combines these evaluation axes on the same data and training procedure. In this direction, ChiliEdgeNet stays under 700 thousand parameters by truncating MobileNetV2 and using three parameter-efficient Transformer techniques — 2D-RoPE, GQA, and RMSNorm. The experimental section retrains ablation variants from scratch with seed 42, tests on synthetically degraded images, uses paired McNemar tests for controlled ablation comparisons, and additionally examines domain transfer, XAI, and a reference comparison against a general-purpose multimodal model.

## 3. Proposed Method

### 3.1. Dataset

ChiliEdge-DS comprises 2,461 chili pepper leaf and fruit images across five classes: `anthracnose`, `healthy`, `leaf_curl`, `leaf_spot`, and `yellow_leaf`. The images were collected directly by the author in the Củ Chi area, Ho Chi Minh City, Vietnam. The data was split once, at an approximate 65:20:15 ratio, into `train`, `val`, and `test` sets of 1,597, 491, and 373 images respectively. This fixed split is kept unchanged throughout all main experiments.

All original images in ChiliEdge-DS were photographed directly by the author, who holds rights to them. Images were taken against a white background using a smartphone camera; the device model was not recorded. Images have a native resolution of approximately 12,000 × 9,000 pixels, saved at 300 × 300 pixels, and resized to 224 × 224 for model input. The dataset is publicly released on Kaggle (Hải, 2026c).

Labels were assigned manually by the author based on visual presentation: `anthracnose` shows dark or necrotic lesions consistent with anthracnose symptoms; `healthy` shows no clearly observable symptoms; `leaf_curl` shows leaf curling or twisting; `leaf_spot` shows discrete lesion spots on the leaf surface; and `yellow_leaf` shows predominant leaf yellowing. These are phenotype-based labels from visual observation, not pathogen identification confirmed by laboratory testing.

ChiliEdge-DS is publicly released on Kaggle (Hải, 2026c) without an open license attached (the license field at the source remains "Unknown"). This is a deliberate choice by the author to retain full control over reuse of the original images collected by the author, not an editorial oversight.

The ratio between the largest and smallest class is 1.29, so no class weighting was applied. Macro-F1 is reported alongside accuracy so that each class contributes equally to the aggregate result. The white background is a controlled acquisition condition; the gap with other-source images and natural backgrounds is examined in Section 4.5 and discussed in Section 6.

**Table 1.** Per-class distribution of ChiliEdge-DS across the fixed split.

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| anthracnose | 264 | 81 | 62 | 407 |
| healthy | 334 | 102 | 78 | 514 |
| leaf_curl | 328 | 104 | 82 | 514 |
| leaf_spot | 326 | 101 | 76 | 503 |
| yellow_leaf | 345 | 103 | 75 | 523 |
| **Total** | **1,597** | **491** | **373** | **2,461** |

### 3.2. Architecture: ChiliEdgeNet

CNNs exploit local features well but remain limited in directly modelling relations between distant regions. ChiliEdgeNet therefore combines a truncated MobileNetV2 with a Tiny Transformer block.

#### 3.2.1. Backbone Cut Point

ChiliEdgeNet uses MobileNetV2 pretrained on ImageNet and keeps the layers from `features[0]` to `features[13]`. With a 224 × 224 input, the backbone produces a feature map of size 96 × 14 × 14, corresponding to 196 spatial positions.

The backbone is cut before `features[14]`, because this layer reduces the feature map to 7 × 7 and the token count from 196 to 49. The cut point `cut = 14` was chosen through a preliminary sweep and also gives the Attention Rollout map four times as many spatial positions as a 7 × 7 grid.

#### 3.2.2. Feature Processing Path

The image is processed through the following pipeline:

1. MobileNetV2 `features[0:14]`;
2. Conv 1 × 1 mapping 96 to 128 channels;
3. BatchNorm, GELU, and dropout 0.1;
4. Flatten the feature map into 196 image tokens, each with 128 dimensions;
5. Tiny Transformer block;
6. RMSNorm and mean pooling over the token dimension;
7. Dropout 0.2 and a linear layer predicting five classes.

#### 3.2.3. Tiny Transformer Block

ChiliEdgeNet is a purely vision-based model, using no text encoder, tokenizer, or pretrained language weights. The Tiny Transformer uses a pre-norm structure consisting of:

* `RMSNorm → GQA combined with 2D-RoPE → residual connection`;
* `RMSNorm → MLP with GELU → residual connection`.

The model uses embedding dimension 128, four query heads, two key/value heads, an MLP ratio of 2, and dropout of 0.1.

Three parameter-efficient techniques are used:

* 2D-RoPE: uses an axial variant for the image grid, replacing a learned positional embedding and saving 25,088 parameters (Heo et al., 2025).
* GQA: lets four query heads share two key/value heads, saving 16,384 parameters (Ainslie et al., 2023).
* RMSNorm: replaces LayerNorm, saving a total of 384 parameters (Zhang and Sennrich, 2019).

Together these three techniques save 41,856 parameters, roughly 6.2% of ChiliEdgeNet's parameter count. The model has 671,173 parameters, about 35.0% fewer than the base architecture (Duhan et al., 2025), which has 1,032,568 parameters.

#### 3.2.4. Removed Blocks

The base architecture (Duhan et al., 2025) additionally uses Triplet Attention (Misra et al., 2021) and a RES_Block combining ECA with SE. Neither block appears in ChiliEdgeNet.

The necessity of Triplet Attention and `RES_Block` was examined with a 2×2 add-back design comprising four configurations: both blocks, only Triplet Attention, only `RES_Block`, and neither. Each configuration was retrained from scratch with seed 42 under the same protocol; results and removal contrasts are presented in Section 4.3. In this seed-42 run, adding either block did not improve accuracy in the evaluated contrasts, so the final model does not use them; this result should not be read as evidence of stability across multiple runs or of equivalence between the two architectures.

![Figure 1. ChiliEdgeNet architecture. Panel A: overall pipeline from the 224x224x3 input through the truncated MobileNetV2 backbone (features[0:14]), a 1x1 convolution reducing channels, 196 tokens, the Tiny Transformer block, mean pooling and the classifier, with tensor shapes at each step. Panel B: detail of one Tiny Transformer block, showing RMSNorm, Grouped Query Attention with axial 2D-RoPE (4 query heads, 2 key/value heads) with a residual connection, and a GELU MLP (expansion ratio 2.0) with a residual connection.](../../results/architecture/chiliedgenet_architecture_en.png)

### 3.3. Evaluation Metrics

This work reports top-1 accuracy, macro-F1, and total parameter count. Every test-set accuracy value carries a 95% Wilson confidence interval at $N = 373$.

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i = y_i)$$

$$F1_c = 2\cdot\frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C}F1_c,\qquad C=5$$

## 4. Experiments and Results

### 4.1. Setup

Experiments were run on Kaggle with an NVIDIA Tesla T4 GPU; the reproduction code uses PyTorch or TensorFlow depending on the architecture. Each baseline is evaluated under two scenarios:

- **Baseline:** no data augmentation applied.
- **Augmented:** the data-augmentation techniques in Table 2 applied.

The `val` and `test` sets are never augmented. For epoch-trained models, the checkpoint with the best validation criterion for each configuration is reloaded before test evaluation.

**Common evaluation framework.** Seventeen baselines use the same ChiliEdge-DS dataset, the same fixed `train`/`val`/`test` split, seed 42, and the metrics accuracy, macro-F1, and parameter count. Both scenarios are evaluated on the same non-augmented test set. Architecture-dependent settings such as input size, batch size, epoch count, optimizer, and augmentation intensity follow each model's own reproduction design. The comparison is therefore controlled for data, split, and metric, but does not use a fully uniform training recipe across every architecture. Two cases where test results influenced selection — RTR_Lite_MobileNet (Duhan et al., 2025) as the primary baseline and InceptionV3 + SVM as the representative for Sujatha et al. (2025) — are disclosed in Section 4.2 and interpreted only as exploratory analysis.

The common 17-baseline evaluation framework is limited to in-domain results on the ChiliEdge-DS test set: accuracy, macro-F1, and parameter count. Additional analyses of blur, low light, architectural ablation, XAI, and domain transfer were carried out only for ChiliEdgeNet and its corresponding controlled variants. This work therefore does not present these experiments as a uniform robustness or domain-transfer benchmark for all 17 baselines.

The baselines were reproduced or adapted from the descriptions in the original works to fit the five-class classification task on ChiliEdge-DS. Significant changes are summarized below; these versions are therefore not treated as faithful reconstructions of the original models.

| Baseline | Component retained | Component adapted |
|---|---|---|
| PDLC-ViT (classification-only; Hemalatha and Jayachandran, 2024) | Co-scale, co-attention, cross-attention, Transformer encoder, and classification head | Segmentation/localization branch removed; converted from multitask to five-class classification using only classification loss; ViT hyperparameters not fully disclosed were set during reproduction. |
| LDI-NET (Yang et al., 2024) | Feature Tokenizer and Token Encoder | The multi-label decoder predicting plant type, disease, and severity was replaced with global average pooling and a five-class linear classifier. |
| InceptionV3 + SVM (Sujatha et al., 2025) | Frozen backbone for feature extraction and SVM for classification | Adapted to ChiliEdge-DS; StandardScaler, PCA retaining 95% of variance, and the exploratory selection described in Section 4.2 were applied. |
| Meta-Ensemble (Maurya et al., 2024) | MLP-Mixer, CNN–LSTM branch, and SVM meta-classifier | Output changed to five classes; patch-embedding configuration implemented as set here, since the original description did not provide full detail. |

The proposed model and all ablation variants use seed 42 and are re-initialized before each run. In the ablation experiments, configurations therefore differ only in the component removed, added back, or replaced.

**Table 2.** Training configuration for ChiliEdgeNet and its ablation variants.

| Item | Value |
|---|---|
| Input image size | 224 × 224 |
| Baseline transform | Resize, ToTensor, Normalize |
| Augmented transform | 30° rotation, ColorJitter, horizontal flip, affine |
| Epochs | 30 |
| Batch size | 32 |
| Loss function | Cross-entropy, label smoothing 0.05 |
| Learning-rate schedule | 10% linear warm-up → cosine decay |
| Learning rate (backbone / new blocks) | 1e-4 / 1e-3 |
| Seed | **42** |

#### 4.1.1. Run Count and Statistical Testing

Results for the proposed model and its ablation variants are reported from a single run with seed 42. This work does not report a mean or standard deviation across multiple seeds and does not interpret these results as evidence of stability across initializations. The McNemar test is used to assess the difference between two models on a per-image basis over the same 373-image test set.

For each comparison, the notation $b/c$ denotes:

- $b$: images the first model predicts correctly while the second predicts incorrectly;
- $c$: images the first model predicts incorrectly while the second predicts correctly.

McNemar's test checks whether the number of discordant cases in the two directions differs significantly. This test does not substitute for assessing variance across multiple seeds; that limitation is discussed in Section 6.

In this work, McNemar is applied to comparisons between ChiliEdgeNet and its ablation variants on the same 373 test images, corresponding to Tables 5–7, and to the paired comparison of the two checkpoints on the same 1,463 cross-dataset query images in Table 10b. The test is not used to rank all 17 baselines in Table 3.

The base architecture RTR_Lite_MobileNet (Duhan et al., 2025) in Section 4.2 was also reproduced with seed 42. The comparison is therefore seed-matched and follows the same procedure, but each architecture was still run only once.

#### 4.1.2. Inference Latency Measurement

ChiliEdgeNet's CPU latency is 20.24 ms/image, equivalent to 49.41 images/second, measured on an Intel Xeon 2.00 GHz CPU with 4 logical cores on a Kaggle virtual machine. The measurement setup uses PyTorch FP32, batch size 1, with 10 warm-up iterations and 50 timed iterations. The 49.41 images/second figure is inference throughput converted from a single-image server benchmark, not the FPS of a camera application.

On the same environment, the Tesla T4 GPU reaches:

- 5.46 ms/image at batch size 1;
- 0.82 ms/image at batch size 32.

The CPU used is a server-class x86 CPU, not edge hardware. These results therefore reflect only the inference cost in the current experimental environment.

### 4.2. Comparison with State-of-the-Art Models

At seed 42, ChiliEdgeNet reaches 97.86% in the baseline scenario and 98.93% in the augmented scenario. The corresponding macro-F1 values are 0.9790 and 0.9897.

A total of 17 baselines from 15 recent studies were reproduced and evaluated under the same framework, presented in full in Table 3. Because the baselines were each run once and not individually tuned for ChiliEdge-DS, this table reflects results under shared data, split, and metrics, not an absolute ranking between architectures.

RTR_Lite_MobileNet (Duhan et al., 2025) was chosen as the primary baseline because it is the direct base architecture of ChiliEdgeNet and has the highest observed accuracy among the evaluated baselines on the ChiliEdge-DS test set. Since the test result contributed to this baseline selection, the direct comparison with RTR_Lite_MobileNet is treated as exploratory analysis on the current split, not an independent test evaluation establishing statistical ranking or a general superiority claim. The parameter reduction remains a structural comparison and does not depend on test-based selection.

For the study by Sujatha et al. (2025), ten combinations of two backbones (VGG19 and InceptionV3) with five classifiers (SVM, kNN, Random Forest, AdaBoost, and Decision Tree) were evaluated on the same data split. The InceptionV3 + SVM configuration, which had the highest observed test accuracy, was chosen as the representative in Table 3. Because this selection used test results, it is likewise treated as exploratory analysis on the current split and is not used to claim statistical superiority or establish an absolute ranking between architectures.

#### 4.2.1. Comparison with the Base Architecture

In the seed-42 run, ChiliEdgeNet reaches 98.93% augmented accuracy, higher than RTR_Lite_MobileNet (Duhan et al., 2025) at 98.12%, a difference of 0.80 percentage points, or 3 images, on the 373-image test set.

ChiliEdgeNet uses 671,173 parameters, while RTR_Lite_MobileNet uses 1,032,568 parameters, a reduction of about 35%. Both models use seed 42 and the same procedure, but each has been run only once; the 95% Wilson confidence intervals overlap, so the accuracy difference should be interpreted cautiously.

**Table 3.** Two-scenario accuracy, macro-F1, and parameter count on the chili dataset. *(Each model is a single seed-42 run under the same evaluation framework; differences below about 1 pp correspond to only 1–4 images out of 373.)*

| Model | Acc Baseline (%) | Acc Augmented (%) | F1-macro (Aug) | Parameters |
|---|---|---|---|---|
| **ChiliEdgeNet (ours)** | **97.86** | **98.93** | **0.9897** | **671,173** |
| RTR_Lite_MobileNet (Duhan et al., 2025) | 97.32 | 98.12 | 0.9821 | 1,032,568 |
| GrapeLeafNet (Karthik et al., 2024) | 91.15 | 97.59 | 0.9769 | 5,308,035 |
| PMVT-XS (Li et al., 2023) | 82.31 | 96.25 | 0.9644 | 2,367,515 |
| ResNet-50 (Mishra et al., 2026) | 95.98 | 95.71 | 0.9586 | 23,597,957 |
| EfficientNetB0 (Mishra et al., 2026) | 94.37 | 94.37 | 0.9463 | 4,055,976 |
| TrIncNet (Gole et al., 2023) | 80.70 | 90.62 | 0.9092 | 7,186,661 |
| InceptionV3 + SVM (Sujatha et al., 2025) | 93.83 | 90.35 | 0.9069 | 25,112,264 |
| Meta-Ensemble (Maurya et al., 2024) | 88.74 | 88.74 | 0.8908 | 1,252,638 |
| CustomMNetV3L (Rahaman et al., 2026) | 87.67 | 86.86 | 0.8639 | 3,275,909 |
| ULEN (Wang et al., 2023) | 93.03 | 85.52 | 0.8603 | 28,069 |
| MaizeFormerX (Rahman et al., 2026) | 79.89 | 82.04 | 0.8185 | 3,553,061 |
| AlexNet (Mishra et al., 2026) | 85.52 | 77.75 | 0.7747 | 46,767,493 |
| Lite-MDC (Bhagat et al., 2024) | 68.10 | 76.68 | 0.7658 | 2,288,965 |
| ViT (Ali et al., 2025) | 72.39 | 73.99 | 0.7472 | 50,013,061 |
| MRW-CNN (Joseph et al., 2024) | 74.26 | 73.73 | 0.7377 | 797,413 |
| PDLC-ViT (classification-only; Hemalatha and Jayachandran, 2024) | 74.26 | 70.78 | 0.7054 | 17,547,269 |
| LDI-NET (Yang et al., 2024) | 70.78 | 68.63 | 0.6839 | 8,798,597 |

#### 4.2.2. Error Analysis

In the augmented scenario, ChiliEdgeNet mispredicts the following images:

- one `leaf_spot` image predicted as `healthy`;
- three `yellow_leaf` images predicted as `leaf_curl`.

Accuracy reaches 98.93% with a 95% Wilson confidence interval of [97.28%, 99.58%]. The remaining errors cluster at the boundary between yellowing and curling symptoms.

![Figure 2. Confusion matrix of ChiliEdgeNet on the test set (augmented scenario).](../../results/notebook_c1/c14_cm_s2.png)

### 4.3. Ablation Analysis

Every variant is retrained from scratch with seed 42, using the same data, training recipe, and 30 epochs. The current model contains neither Triplet Attention nor `RES_Block`. Since clean-image accuracy is already near saturation, the role of the Tiny Transformer is further examined with a multi-level stress test on synthetically degraded images.

#### 4.3.1. Block-Level Ablation

Two direct cuts respectively remove the Tiny Transformer and disable every skip connection of the `InvertedResidual` blocks in the backbone. For Triplet Attention and `RES_Block`, this work uses a 2×2 add-back design: both present, only Triplet, only `RES_Block`, and neither (the current model).

**Table 4.** Architectural ablation results on clean images, seed 42.

| Configuration | Parameters | Accuracy (%) |
|---|---:|---:|
| **Current: no Triplet/RES_Block** | **671,173** | **98.93** |
| Remove Tiny Transformer | 555,845 | 99.20 |
| Disable MobileNetV2 residuals | 671,173 | 96.51 |
| Add Triplet + RES_Block | 671,732 | 97.59 |
| Triplet only | 671,473 | 98.12 |
| RES_Block only | 671,432 | 98.93 |

**Table 5.** McNemar test for the two direct cuts on clean images. Positive delta means the current model outperforms the variant; $p_{Holm}$ is corrected for two predefined contrasts.

| Cut | Delta (pp) | b / c | $p$ | $p_{Holm}$ |
|---|---:|---:|---:|---:|
| Remove Tiny Transformer | −0.27 | 0 / 1 | 1.0000 | 1.0000 |
| Disable MobileNetV2 residuals | +2.41 | 10 / 1 | 0.0117 | **0.0234** |

On clean images, the variant without the Transformer scores one image higher, so this run does not show the block's benefit. In contrast, disabling the residual connections reduces net correct predictions by nine, and the difference remains significant after Holm correction. This result supports keeping MobileNetV2's active skip branches.

**Table 6.** Three contrasts removing Triplet Attention/RES_Block from the both-present configuration. Holm-corrected $p$-values for three contrasts on clean images.

| Contrast | Delta (pp) | $p_{Holm}$ |
|---|---:|---:|
| Remove `RES_Block` | +0.54 | 0.625 |
| Remove Triplet Attention | +1.34 | 0.375 |
| Remove both, back to the current model | +1.34 | 0.375 |

No contrast reaches significance after Holm correction. In this run, `RES_Block` alone gives the same accuracy as the current model, while Triplet Attention shows a trend toward lower performance. The results support removing both blocks to avoid unnecessary complexity, but do not prove the architectures are equivalent.

#### 4.3.2. Stress Test on the Role of the Tiny Transformer

The stress test uses the same 373 test images and two seed-42 checkpoints: full ChiliEdgeNet and a variant retrained after removing the Tiny Transformer. Gaussian blur was generated with `cv2.GaussianBlur` (`sigmaX=0`) at kernel sizes 3, 5, 7, and 9; brightness reduction was generated by multiplying all pixel intensities by 0.8, 0.6, 0.4, and 0.2. Eight conditions were fixed before scoring. The clean image is only a reference point and is not part of the family of eight tests; Holm-corrected exact McNemar tests are applied across the eight degradation conditions.

![Figure 3. Accuracy of ChiliEdgeNet with and without the Tiny Transformer across Gaussian blur levels and synthetic brightness-reduction factors.](../../results/stress/c14_tf_stress_curves.png)

**Table 7.** Stress test of the Tiny Transformer under synthetic degradation levels. Positive delta means the model with the Tiny Transformer performs better.

| Condition | With Transformer (%) | Without Transformer (%) | Delta (pp) | b / c | $p_{Holm}$ |
|---|---:|---:|---:|---:|---:|
| Clean | 98.93 | 99.20 | −0.27 | 0 / 1 | – |
| Blur, kernel 3 | 97.86 | 98.12 | −0.27 | 2 / 3 | 1.0000 |
| Blur, kernel 5 | 97.32 | 96.25 | +1.07 | 6 / 2 | 1.0000 |
| Blur, kernel 7 | 94.91 | 92.76 | +2.14 | 15 / 7 | 0.8028 |
| Blur, kernel 9 | 92.23 | 89.28 | +2.95 | 22 / 11 | 0.5610 |
| Brightness × 0.8 | 98.66 | 99.20 | −0.54 | 0 / 2 | 1.0000 |
| Brightness × 0.6 | 98.66 | 98.93 | −0.27 | 1 / 2 | 1.0000 |
| Brightness × 0.4 | 98.66 | 98.12 | +0.54 | 3 / 1 | 1.0000 |
| Brightness × 0.2 | **97.05** | 93.30 | **+3.75** | 16 / 2 | **0.0105** |

On clean images and mild degradation levels, the Tiny Transformer does not yet produce a clear difference. From kernel 5 onward, the advantage under blur grows with the degradation level, but individual contrasts do not reach significance after Holm correction. At brightness factor 0.2, the full model reaches 97.05%, 3.75 pp higher, reducing errors from 25 to 11; this is the only contrast that remains significant after correction. These results suggest the Tiny Transformer helps maintain representations under severe visual degradation, rather than providing a stable advantage on clean images. This benefit comes with 115,328 additional parameters relative to the variant without the Transformer.

#### 4.3.3. Technique-Level Ablation

Each variant replaces exactly one technique in the Tiny Transformer with its standard counterpart while keeping the remaining components unchanged.

**Table 8.** Ablation of individual techniques in the Tiny Transformer on clean images, seed 42.

| Variant | Replaced component | Parameters | Δ params | Accuracy (%) |
|---|---|---:|---:|---:|
| **ChiliEdgeNet** | 2D-RoPE + GQA + RMSNorm | **671,173** | – | **98.93** |
| `learned_pe` | 2D-RoPE → learned PE | 696,261 | +25,088 | 98.93 |
| `mha` | GQA → MHA | 687,557 | +16,384 | 98.93 |
| `layernorm` | RMSNorm → LayerNorm | 671,557 | +384 | 98.39 |

On clean images, learned PE and MHA give the same correct/incorrect predictions as ChiliEdgeNet; LayerNorm is lower by two images, but McNemar shows no significance ($p=0.5$). The three choices — 2D-RoPE, GQA, and RMSNorm — are not shown by this single-seed data to be superior in accuracy. In terms of parameter structure, they respectively avoid 25,088, 16,384, and 384 parameters relative to the standard alternatives, totalling 41,856 parameters if all three standard alternatives were used simultaneously; the combined three-replacement variant has not been separately trained.

### 4.4. Model Interpretability

The analyses in this section use ChiliEdgeNet's original checkpoint, trained only on ChiliEdge-DS and used to produce the in-domain results in Table 3. The augmented checkpoint used for the domain-transfer experiment in Section 4.5 is a separate adapted model and is not used to produce the interpretability results below.

#### 4.4.1. Explainable AI (XAI)

High accuracy alone is not sufficient to interpret model behaviour, particularly since ChiliEdge-DS predominantly uses a controlled white background. This work applies three complementary XAI methods on the ChiliEdgeNet augmented-scenario checkpoint (seed 42), each using one representative image per class. Gradient SHAP (Erion et al., 2021) estimates pixel contribution from gradients relative to a baseline distribution; Occlusion Sensitivity (Zeiler and Fergus, 2014) occludes 32×32-pixel regions in turn and measures the change in output; Attention Rollout (Abnar and Zuidema, 2020) aggregates attention flow through the Transformer block across the full 14×14 grid. Because the architecture uses mean pooling and has no `CLS` token, the Rollout map is produced by averaging attention flow across the 196 tokens, rather than tracing from a single output token.

The three maps offer distinct perspectives (Figures 4–6): Gradient SHAP gives pixel-level attribution distributed over the object surface; Occlusion shows region-level sensitivity; Attention Rollout reflects the Transformer's internal attention propagation. Across the five illustrative images, high-response regions generally intersect with the leaf/fruit or the diseased area, though the degree of concentration differs across methods and classes. Since the evaluation uses only one image per class and no ground-truth symptom segmentation exists, this result is an exploratory qualitative survey only; it does not prove the model has learned correct pathological features, fully ignored the background, or captured a genuine causal relationship. The 14×14 grid provides 196 spatial positions, four times more than a 7×7 grid, giving the Rollout map finer visual resolution.

<!-- IMG2 -->
![Figure 4. Occlusion Sensitivity for the five disease classes (top: original image; bottom: sensitivity map overlay). Bright regions are where occlusion costs the model the most confidence; this technique treats the model as a black box.](../../results/notebook_c2/xai_occlusion.png)

![Figure 5. Gradient SHAP for the five classes (top: original image; bottom: pixel-attribution magnitude overlay), computed with 20 samples and two constant baseline tensors (0 and 1) in the normalized input space.](../../results/notebook_c2/xai_shap.png)

![Figure 6. Attention Rollout of the Tiny Transformer block on the 14×14 grid, aggregated by averaging attention flow across the 196 tokens since the model uses mean pooling with no `CLS` token.](../../results/notebook_c2/xai_attn_rollout.png)
<!-- /IMG2 -->

#### 4.4.2. Comparison with a General-Purpose Multimodal Model

This work uses the general-purpose multimodal model Gemini 3.1 Flash Lite, API model ID `gemini-3.1-flash-lite`, accessed through the Gemini API with an API key created and managed via Google AI Studio. The model was accessed on 8 July 2026 (Google DeepMind, 2026) to classify the raw images directly in a zero-shot setting. No temperature, top-p, or token limit was set separately; the run used the endpoint's default generation configuration at the access date. This is a reference comparison, not a fully matched baseline: Gemini was not fine-tuned on ChiliEdge-DS, whereas ChiliEdgeNet is a specialized classifier trained with supervision for exactly the five classes.

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

The artifact currently stores all 373 images, run on 8 July 2026, with 100% of calls valid and no labels outside the allowed set. Gemini reaches 78.55% accuracy, while ChiliEdgeNet reaches 98.93% on the same 373-image test set — matching Table 3. Of the three `yellow_leaf` images in the test set, two are also two of the three errors where ChiliEdgeNet mispredicted `leaf_curl`, as noted in Section 4.2.2, while ChiliEdgeNet correctly classified the third; Gemini correctly classified all three as `yellow_leaf`.

**On latency.** The approximately 105x figure contrasts 20.24 ms of local inference with a 2,123 ms median cloud API round-trip. The two figures are not equivalent: the round-trip includes network latency, server-side queueing, and provider-side batching — none of which is computation intrinsic to the model. This is a comparison between two deployment modes, not a claim that one architecture computes roughly 105x faster than the other. Similarly, although accuracy is computed on the same 373 images, the accuracy gap does not constitute a peer comparison, since the two systems differ in training regime and degree of specialization. The commercial endpoint may also be updated over time, so this comparison is reproducible only against the stated model ID, prompt, and access date.

**Table 9.** Comparison of ChiliEdgeNet with Gemini 3.1 Flash Lite on the full 373-image test set (accessed 8 July 2026).

| System | Accuracy | Latency/image |
|---|---|---|
| **ChiliEdgeNet (local CPU)** | **98.93%** | **20.24 ms** (local inference) |
| Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) | 78.55% | 2,123 ms (median API round-trip) |

### 4.5. Four-Class Cross-Dataset Evaluation After Few-Shot Adaptation

The stress tests in Section 4.3 only synthesize blur and brightness reduction from ChiliEdge-DS images, so they do not fully capture the acquisition domain gap. This work carries out an additional evaluation on the Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh, version 1, released on Mendeley Data under a CC BY 4.0 licence (Nirob et al., 2025). The source images were collected at a garden in Charpolisha, Jamalpur (Bangladesh) between October 2024 and February 2025; the disease-leaf portion comprises 1,856 images across six source labels.

Only four labels with a direct correspondence to the ChiliEdgeNet label space are used. Two labels, `Bacterial Spot` (156 images) and `White spot` (195 images), are excluded because the model has no corresponding target class. The source dataset has no anthracnose class, so there is no `anthracnose` ground-truth row. However, the checkpoint still retains all five outputs used during training; any prediction toward `anthracnose` is still counted as incorrect, and this prediction column is kept in the confusion matrix.

**Table 10a.** Verified support/query split from the four mapped classes.

| Source label (Nirob et al., 2025) | ChiliEdgeNet label | Original images | Few-shot support | Clean query | Corrected test folder |
|---|---|---:|---:|---:|---:|
| `Healthy Leaf` | `healthy` | 458 | 8 | 450 | 450 |
| `Curl Virus` | `leaf_curl` | 423 | 9 | 414 | 414 |
| `Cercospora Leaf Spot` | `leaf_spot` | 180 | 16 | 164 | 164 |
| `Nutrition Deficiency` | `yellow_leaf` | 444 | 9 | 435 | 435 |
| **Total** |  | **1,505** | **42** | **1,463** | **1,463** |

**Few-shot setup.** The support set comprises 42 images taken as-is from Nirob et al. (2025): 8 `healthy`, 9 `leaf_curl`, 16 `leaf_spot`, and 9 `yellow_leaf`. The support images were added to ChiliEdge-DS's training set to create a separate adapted checkpoint; ChiliEdge-DS's own in-domain validation and test sets are unchanged. This is supervised domain adaptation with a small support set, not episodic meta-learning. The number of shots differs across classes, so this is not a balanced *N*-way *K*-shot setup.

To control for data leakage, the support and query sets were tracked with SHA-256 hashes rather than filenames alone. The query set is defined as the complement of the 42 support images within the 1,505 images across the four classes, and an audit confirmed the hash intersection between the two is zero. The few-shot checkpoint was trained with seed 42.

To support independent reproduction, the clean 1,463-image query set is released as the Chili Leaf Disease Cross-Dataset Query Set, Kaggle version 2 (Hải, 2026a), and the 42 support images are released as Chili Leaf Disease Few-Shot, Kaggle version 1 (Hải, 2026b). Both packages are derived splits of the original Mendeley Data dataset and do not replace the requirement to cite the source dataset (Nirob et al., 2025).

**Results on the clean query set.** Both checkpoints were evaluated on the same 1,463-image clean query set. The original checkpoint uses seed 42 (SHA-256 verified); the checkpoint after few-shot adaptation also uses seed 42 (SHA-256 verified). Images were resized to 224×224 pixels and normalized with ImageNet statistics. Macro precision, recall, and F1 are averaged only over the four classes with ground truth; predictions toward `anthracnose` still reduce accuracy.

**Table 10b.** Results before and after few-shot adaptation on the 1,463-image clean query set.

| Configuration | Seed | Accuracy | Precision-macro | Recall-macro | F1-macro | Correct images |
|---|---:|---:|---:|---:|---:|---:|
| Original checkpoint, no few-shot | 42 | 30.96% | 0.4337 | 0.2843 | 0.1766 | 453/1,463 |
| Checkpoint after few-shot | 42 | 89.75% | 0.8837 | 0.8683 | 0.8744 | 1,313/1,463 |
| **Difference** |  | **+58.78 pp** | **+0.4500** | **+0.5840** | **+0.6978** | **+860** |

At the original checkpoint, `leaf_curl` has recall 1.0000 but precision only 0.3013, while `yellow_leaf` has recall zero. After few-shot adaptation, F1 for `healthy`, `leaf_curl`, `leaf_spot`, and `yellow_leaf` reaches 0.9168, 0.9619, 0.7557, and 0.8633 respectively. A per-image prediction comparison shows 443 images correctly predicted by both checkpoints, 140 images incorrectly predicted by both, 870 images turning from incorrect at the original checkpoint to correct after few-shot, and 10 images in the opposite direction. A two-sided exact McNemar test gives $p=1.83\times10^{-242}$. This is a single paired contrast on an out-of-domain dataset and does not belong to the ablation's Holm test family. Predictions for all 1,463 images and the paired statistics are stored in the metadata package accompanying this study.

![Figure S1. Confusion matrix of the original checkpoint on the 1,463-image clean query set.](<../../results/Cross check default/cross_dataset_local_cm.png>)

![Figure S2. Confusion matrix of the checkpoint after few-shot adaptation on the same 1,463-image clean query set.](<../../results/cross check them data/cross_dataset_local_cm.png>)

### 4.6. Preprocessing and Inference Latency Evaluation in a Laptop-Camera Application

ChiliEdgeNet was integrated into a laptop camera application, where the user places a leaf inside a region of interest (ROI) and triggers a prediction on each frame. This experiment evaluates only preprocessing and inference latency as a proof of integration feasibility; it does not use camera frames to estimate real-field accuracy.

The test session comprises 62 predictions, run at batch size 1 on a 13th Gen Intel Core i7-13650HX CPU. The measured interval starts from converting the ROI's RGB image to PIL, continuing through resize, normalization, model inference, softmax, and top-1 prediction. The measurement excludes time for reading frames from the camera, cropping the ROI, converting BGR to RGB, rendering the UI, or writing files.

**Table 11.** Preprocessing and inference latency of ChiliEdgeNet on a laptop CPU.

| Metric | Latency (ms) |
|---|---:|
| Number of measurements | 62 |
| Mean | 32.18 |
| Median | 29.70 |
| 95th percentile | 46.49 |
| Minimum | 17.33 |
| Maximum | 122.76 |

A median of 29.70 ms and a 95th percentile of 46.49 ms show that the preprocessing-and-inference stage can respond quickly on a laptop CPU without a GPU. However, the results come from a single measurement session on a single device and do not include the application's full capture-to-display cost; the test device is also not dedicated edge hardware. This experiment therefore provides only preliminary evidence of ChiliEdgeNet's feasibility for integration into a camera application, not a representative measurement of the full pipeline or of dedicated edge hardware. The labelled data in the demo session is unbalanced and contains many repeated frames of the same subject, so accuracy and a confusion matrix are not reported in this section.

## 5. Limitations

**Statistical resolution of the test set.** With 373 images, one image is worth 0.27 pp, and the proposed model errs on 4 images. The 0.80 pp (3-image) difference from the base architecture (Duhan et al., 2025) falls within the noise band of a single split: ChiliEdgeNet's 95% Wilson interval [97.28, 99.58] overlaps with the [96.18, 99.09] interval of Duhan et al. (2025). Both models are seed-42-matched but each has only been run once. Furthermore, RTR_Lite_MobileNet (Duhan et al., 2025) was chosen as the primary baseline after reviewing the baseline pool's test results, so this comparison is exploratory and does not constitute an independent test evaluation. Establishing a definitive ranking would require selection by validation followed by a single test evaluation, multiple seeds, and/or a direct McNemar test between the two prediction sets. The 35% parameter reduction does not depend on training noise or baseline-selection criteria.

**Run count and statistical power.** The proposed model and all ablations come from a single seed-42 run. At the architectural level, disabling MobileNetV2 residuals reduces accuracy by 2.41 pp on clean images, and the difference remains significant after Holm correction for the two direct cuts ($p_{Holm}=0.0234$). Across the eight stress-test conditions, the Tiny Transformer's advantage reaches significance after Holm correction only at brightness factor 0.2 ($p_{Holm}=0.0105$); the remaining levels reflect a severity-dependent trend rather than a broadly established effect. All three Triplet/`RES_Block` contrasts are non-significant after Holm correction. These conclusions are single-seed observations and need confirmation across multiple runs.

**Scope of the stress test.** The experiment examines four Gaussian blur levels and four brightness-reduction factors synthesized from the same test set. The design does not yet cover sensor noise, JPEG compression, motion blur, or real images captured under genuine low-light conditions; results should not be extrapolated to every capture condition.

**Domain gap and scope of the few-shot evaluation.** The evaluation uses only the four Mendeley Data labels that map directly to `healthy`, `leaf_curl`, `leaf_spot`, and `yellow_leaf`; it does not evaluate `anthracnose` and excludes two source labels the model does not support. On the clean 1,463-image query set, with zero hash intersection with the support set, accuracy reaches 30.96% at the original checkpoint and 89.75% after few-shot adaptation. Both checkpoints were trained with seed 42, the same architecture, and the same base hyperparameters; training records should still be strengthened with the executed logs or configuration. Since only one out-of-domain dataset and one fixed support set were used, this improvement does not demonstrate that the few-shot protocol generalizes to other data sources or support-selection methods.

**Hardware.** The 20.24 ms/image figure is measured on a server-class Kaggle virtual-machine CPU, not edge hardware. INT8 and FP16 in the current run reduce storage size but have not yet produced a latency speed-up; benchmarking on a phone or embedded computer with a real deployment backend is still needed.

## 6. Conclusion

At seed 42, ChiliEdgeNet reaches 98.93% accuracy, macro-F1 0.9897, using 671,173 parameters, with CPU inference at 20.24 ms/image. Compared with RTR_Lite_MobileNet (Duhan et al., 2025) at the same seed, the model is 0.80 pp (3 images) higher and uses about 35% fewer parameters; since each architecture was run only once and the confidence intervals overlap, the accuracy advantage should be read as a trend rather than an established statistical ranking.

The ablation results show:

1. Disabling MobileNetV2's residual connections reduces accuracy by 2.41 pp on clean images; the difference remains significant after Holm correction for the two direct cuts.
2. The Tiny Transformer shows no advantage on clean images or mild degradation, but the gap grows under severe degradation. At brightness factor 0.2, the full model is 3.75 pp higher, and the difference is significant after Holm correction.
3. Triplet Attention and `RES_Block` do not provide a stable benefit. The current model, which omits both, matches or exceeds the add-back configurations, but the contrasts are not significant after Holm correction.
4. 2D-RoPE, GQA, and RMSNorm together avoid 41,856 parameters relative to using all three standard alternatives simultaneously; single-seed results do not show an accuracy advantage on clean images.
5. On the clean 1,463-image cross-dataset query set, adaptation with 42 support images raises accuracy from 30.96% to 89.75% (+58.78 pp); a paired McNemar test records 870 improvements and 10 regressions ($p=1.83\times10^{-242}$). This result applies only to the four mapped classes.

Future directions include repeating the experiments across multiple seeds, checking for duplicate/near-duplicate images across splits, extending the stress test to real acquisition degradations, and measuring latency and memory on physical edge hardware with a real quantization backend.

## References

Abnar, S., Zuidema, W., 2020. Quantifying attention flow in transformers. In: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pp. 4190–4197. https://doi.org/10.18653/v1/2020.acl-main.385.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., Sanghai, S., 2023. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4895–4901. https://doi.org/10.18653/v1/2023.emnlp-main.298.

Ali, M., Salma, M., El Haji, M., Jamal, B., 2025. Plant disease detection using vision transformers. International Journal of Electrical and Computer Engineering 15 (2), 2334–2344. https://doi.org/10.11591/ijece.v15i2.pp2334-2344.

Bhagat, S., Kokare, M., Haswani, V., Hambarde, P., Taori, T., Ghante, P.H., Patil, D.K., 2024. Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop. Smart Agricultural Technology 7, 100408. https://doi.org/10.1016/j.atech.2024.100408.

Duhan, S., Gulia, P., Gill, N.S., Narwal, E., 2025. RTR_Lite_MobileNetV2: A lightweight and efficient model for plant disease detection and classification. Current Plant Biology 42, 100459. https://doi.org/10.1016/j.cpb.2025.100459.

Erion, G., Janizek, J.D., Sturmfels, P., Lundberg, S.M., Lee, S.-I., 2021. Improving performance of deep learning models with axiomatic attribution priors and expected gradients. Nature Machine Intelligence 3 (7), 620–631. https://doi.org/10.1038/s42256-021-00343-w.

Gole, P., Bedi, P., Marwaha, S., Haque, M.A., Deb, C.K., 2023. TrIncNet: A lightweight vision transformer network for identification of plant diseases. Frontiers in Plant Science 14, 1221557. https://doi.org/10.3389/fpls.2023.1221557.

Google DeepMind, 2026. Gemini 3.1 Flash-Lite: Model Card. https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/ (accessed 8 July 2026).

[dataset] Hải, T.M., 2026a. Chili Leaf Disease Cross-Dataset Query Set, Kaggle dataset version 2. Kaggle. https://www.kaggle.com/datasets/tminhhi/chili-leaf-disease-cross-dataset-query-set/version/2 (accessed 4 August 2026). Derived from Nirob et al. (2025); license: CC BY 4.0.

[dataset] Hải, T.M., 2026b. Chili Leaf Disease Few-Shot, Kaggle dataset version 1. Kaggle. https://www.kaggle.com/datasets/tminhhi/chili-leaf-disease-few-shot/version/1 (accessed 4 August 2026). Derived from Nirob et al. (2025); license: CC BY 4.0.

[dataset] Hải, T.M., 2026c. ChiliEdge-DS: Chili Pepper Leaf Disease Dataset, version 1.0 (Kaggle dataset version 2). Kaggle. https://www.kaggle.com/datasets/tminhhi/ot-21-06/versions/2 (accessed 4 August 2026); license: not publicly licensed (all rights reserved by the author).

Hemalatha, S., Jayachandran, J.J.B., 2024. A multitask learning-based vision transformer for plant disease localization and classification. International Journal of Computational Intelligence Systems 17 (1), 188. https://doi.org/10.1007/s44196-024-00597-3.

Heo, B., Park, S., Han, D., Yun, S., 2025. Rotary position embedding for Vision Transformer. In: Computer Vision – ECCV 2024, Lecture Notes in Computer Science, vol. 15068, pp. 289–305. https://doi.org/10.1007/978-3-031-72684-2_17.

Joseph, D.S., Pawar, P.M., Chakradeo, K., 2024. Real-time plant disease dataset development and detection of plant disease using deep learning. IEEE Access 12, 16310–16333. https://doi.org/10.1109/ACCESS.2024.3358333.

Karthik, R., Menaka, R., Ompirakash, S., Bala Murugan, P., Meenakashi, M., Lingaswamy, S., Won, D., 2024. GrapeLeafNet: A dual-track feature fusion network with Inception-ResNet and Shuffle-Transformer for accurate grape leaf disease identification. IEEE Access 12, 19612–19624. https://doi.org/10.1109/ACCESS.2024.3361044.

Li, G., Wang, Y., Zhao, Q., Yuan, P., Chang, B., 2023. PMVT: A lightweight vision transformer for plant disease identification on mobile devices. Frontiers in Plant Science 14, 1256773. https://doi.org/10.3389/fpls.2023.1256773.

Maurya, R., Mahapatra, S., Rajput, L., 2024. A lightweight meta-ensemble approach for plant disease detection suitable for IoT-based environments. IEEE Access 12, 28096–28108. https://doi.org/10.1109/ACCESS.2024.3367443.

Mishra, U., Pandey, A., Logeswari, G., Tamilarasi, K., 2026. Deep learning-based disease detection in potato and mango leaves: A comparative study of CNN, AlexNet, ResNet, and EfficientNet. Scientific Reports 16, 2788. https://doi.org/10.1038/s41598-025-32607-5.

Misra, D., Nalamada, T., Arasanipalai, A.U., Hou, Q., 2021. Rotate to attend: Convolutional triplet attention module. In: Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 3139–3148. https://doi.org/10.1109/WACV48630.2021.00318.

[dataset] Nirob, M.A.S., Siam, A.K.M.F.K., Bishshash, P., Assaduzzaman, M., 2025. Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh, version 1. Mendeley Data. https://doi.org/10.17632/w9mr3vf56s.1. Published 19 February 2025; license: CC BY 4.0.

Pan, S.J., Yang, Q., 2010. A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering 22 (10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191.

Rahaman, J., Paul, P., Chowdhury, A., Quamruzzaman, M., 2026. A customized MobileNetV3Large-based deep learning framework for plant disease detection. Discover Artificial Intelligence 6, 110. https://doi.org/10.1007/s44163-025-00733-8.

Rahman, M.M., Gony, M.N., Ullah, M.S., Shuvra, S.M.K., Haque, R., Ahmed, M.R., et al., 2026. MaizeFormerX: A lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis. Scientific Reports 16, 15160. https://doi.org/10.1038/s41598-026-44550-0.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C., 2018. MobileNetV2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510–4520. https://doi.org/10.1109/CVPR.2018.00474.

Snell, J., Swersky, K., Zemel, R.S., 2017. Prototypical networks for few-shot learning. In: Advances in Neural Information Processing Systems 30, pp. 4077–4087.

Su, J., Ahmed, M., Lu, Y., Pan, S., Wen, B., Liu, Y., 2024. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063. https://doi.org/10.1016/j.neucom.2023.127063.

Sujatha, R., Krishnan, S., Chatterjee, J.M., Gandomi, A.H., 2025. Advancing plant leaf disease detection integrating machine learning and deep learning. Scientific Reports 15, 11552. https://doi.org/10.1038/s41598-024-72197-2.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., et al., 2017. Attention is all you need. In: Advances in Neural Information Processing Systems 30, pp. 5998–6008.

Wang, B., Zhang, C., Li, Y., Cao, C., Huang, D., Gong, Y., 2023. An ultra-lightweight efficient network for image-based plant disease and pest infection detection. Precision Agriculture 24 (5), 1836–1861. https://doi.org/10.1007/s11119-023-10020-0.

Yang, B., Li, M., Li, F., Wang, Y., Liang, Q., Zhao, R., et al., 2024. A novel plant type, leaf disease and severity identification framework using CNN and transformer with multi-label method. Scientific Reports 14, 11664. https://doi.org/10.1038/s41598-024-62452-x.

Zeiler, M.D., Fergus, R., 2014. Visualizing and understanding convolutional networks. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 818–833. https://doi.org/10.1007/978-3-319-10590-1_53.

Zhang, B., Sennrich, R., 2019. Root mean square layer normalization. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 12360–12371.
