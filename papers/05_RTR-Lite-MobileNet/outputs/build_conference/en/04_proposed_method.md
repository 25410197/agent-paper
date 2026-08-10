## 3. Proposed Method

### 3.1. Dataset

**ChiliEdge-DS** comprises 2,461 chili pepper leaf and fruit images across five classes: `anthracnose`, `healthy`, `leaf_curl`, `leaf_spot`, and `yellow_leaf`. The images were collected directly by the author in the Củ Chi area, Ho Chi Minh City, Vietnam. The data was split once, at an approximate 65:20:15 ratio, into `train`, `val`, and `test` sets of 1,597, 491, and 373 images respectively. This fixed split is kept unchanged throughout all main experiments.

All original images in ChiliEdge-DS were photographed directly by the author, who holds rights to them. Images were taken against a white background using a smartphone camera; the device model was not recorded. Images have a native resolution of approximately 12,000 × 9,000 pixels, saved at 300 × 300 pixels, and resized to 224 × 224 for model input. The dataset is publicly released on Kaggle (Hải, 2026c).

Labels were assigned manually by the author based on visual presentation: `anthracnose` shows dark or necrotic lesions consistent with anthracnose symptoms; `healthy` shows no clearly observable symptoms; `leaf_curl` shows leaf curling or twisting; `leaf_spot` shows discrete lesion spots on the leaf surface; and `yellow_leaf` shows predominant leaf yellowing. These are phenotype-based labels from visual observation, not pathogen identification confirmed by laboratory testing.

ChiliEdge-DS is publicly released on Kaggle (Hải, 2026c) without an open license attached (the license field at the source remains "Unknown"). This is a deliberate choice by the author to retain full control over reuse of the original images collected by the author, not an editorial oversight.

The ratio between the largest and smallest class is 1.29, so no class weighting was applied. Macro-F1 is reported alongside accuracy so that each class contributes equally to the aggregate result. The white background is a controlled acquisition condition; the gap with other-source images and natural backgrounds is examined in Section 4.5 and discussed in Section 6.

**Table 1. Per-class distribution of ChiliEdge-DS across the fixed split.**

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

ChiliEdgeNet uses MobileNetV2 pretrained on ImageNet and keeps the layers from `features[0]` to `features[13]`. With a 224 × 224 input, the backbone produces a feature map of size **96 × 14 × 14**, corresponding to 196 spatial positions.

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

* **2D-RoPE:** uses an axial variant for the image grid, replacing a learned positional embedding and saving **25,088 parameters** (Heo et al., 2025).
* **GQA:** lets four query heads share two key/value heads, saving **16,384 parameters** (Ainslie et al., 2023).
* **RMSNorm:** replaces LayerNorm, saving a total of **384 parameters** (Zhang and Sennrich, 2019).

Together these three techniques save **41,856 parameters**, roughly **6.2%** of ChiliEdgeNet's parameter count. The model has **671,173 parameters**, about **35.0% fewer** than the base architecture (Duhan et al., 2025), which has 1,032,568 parameters.

#### 3.2.4. Removed Blocks

The base architecture (Duhan et al., 2025) additionally uses Triplet Attention (Misra et al., 2021) and a RES_Block combining ECA with SE. Neither block appears in ChiliEdgeNet.

The necessity of Triplet Attention and `RES_Block` was examined with a 2×2 add-back design comprising four configurations: both blocks, only Triplet Attention, only `RES_Block`, and neither. Each configuration was retrained from scratch with seed 42 under the same protocol; results and removal contrasts are presented in Section 4.3. In this seed-42 run, adding either block did not improve accuracy in the evaluated contrasts, so the final model does not use them; this result should not be read as evidence of stability across multiple runs or of equivalence between the two architectures.

![Figure 1. ChiliEdgeNet architecture. Panel A: overall pipeline from the 224x224x3 input through the truncated MobileNetV2 backbone (features[0:14]), a 1x1 convolution reducing channels, 196 tokens, the Tiny Transformer block, mean pooling and the classifier, with tensor shapes at each step. Panel B: detail of one Tiny Transformer block, showing RMSNorm, Grouped Query Attention with axial 2D-RoPE (4 query heads, 2 key/value heads) with a residual connection, and a GELU MLP (expansion ratio 2.0) with a residual connection.](../../results/architecture/chiliedgenet_architecture_en.png)

### 3.3. Evaluation Metrics

This work reports top-1 accuracy, macro-F1, and total parameter count. Every test-set accuracy value carries a 95% Wilson confidence interval at $N = 373$.

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i = y_i)$$

$$F1_c = 2\cdot\frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C}F1_c,\qquad C=5$$
