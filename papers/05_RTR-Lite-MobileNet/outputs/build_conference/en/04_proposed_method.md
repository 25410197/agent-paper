## 3. Proposed Method

### 3.1. Dataset

ChiliEdge-DS comprises 2,461 chili pepper leaf and fruit images across five classes: *anthracnose*, *healthy*, *leaf_curl*, *leaf_spot*, and *yellow_leaf*. The images were collected by the author in the Củ Chi area, Ho Chi Minh City, Vietnam. The data was split once, at an approximate 65:20:15 ratio, into *train*, *val*, and *test* sets of 1,597, 491, and 373 images respectively. This fixed split is kept unchanged throughout all main experiments.

Images were taken against a white background using a smartphone camera; the device model was not recorded. Images have a native resolution of approximately 12,000 × 9,000 pixels, saved at 300 × 300 pixels, and resized to 224 × 224 for model input. The dataset is publicly released on Kaggle (Hải, 2026c).

Labels were assigned manually by the author based on visual presentation: *anthracnose* shows dark, necrotic lesions on the leaf or fruit surface; *healthy* shows no clearly observable symptoms; *leaf_curl* shows leaf curling or twisting; *leaf_spot* shows discrete lesion spots on the leaf surface; and *yellow_leaf* shows predominant leaf yellowing. These are phenotype-based labels from visual observation, not pathogen identification confirmed by laboratory testing.

ChiliEdge-DS is publicly released on Kaggle (Hải, 2026c) without an open license attached (the license field at the source remains "Unknown"). This is a deliberate choice by the author to retain full control over image reuse, not an oversight.

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

CNNs exploit local features well but remain limited in directly modelling relations between distant regions. ChiliEdgeNet therefore combines a truncated MobileNetV2 with a Tiny Transformer block. Relative to the base architecture (Duhan et al., 2025), it also omits Triplet Attention (Misra et al., 2021) and a *RES_Block* combining ECA with SE; Section 4.3 reports the ablation evidence behind this choice.

#### 3.2.1. Backbone Cut Point

ChiliEdgeNet uses MobileNetV2 pretrained on ImageNet and keeps the layers from *features[0]* to *features[13]*. With a 224 × 224 input, the backbone produces a feature map of size 96 × 14 × 14, corresponding to 196 spatial positions.

The backbone is cut before *features[14]*, because this layer reduces the feature map to 7 × 7 and the token count from 196 to 49. The cut point *cut = 14* was chosen through a preliminary sweep and also gives the Attention Rollout map four times as many spatial positions as a 7 × 7 grid.

#### 3.2.2. Feature Processing Path

The image is processed through the following pipeline:

1. MobileNetV2 *features[0:14]*;
2. Conv 1 × 1 mapping 96 to 128 channels;
3. BatchNorm, GELU, and dropout 0.1;
4. Flatten the feature map into 196 image tokens, each with 128 dimensions;
5. Tiny Transformer block;
6. RMSNorm and mean pooling over the token dimension;
7. Dropout 0.2 and a linear layer predicting five classes.

#### 3.2.3. Tiny Transformer Block

ChiliEdgeNet is a purely vision-based model, using no text encoder, tokenizer, or pretrained language weights. The Tiny Transformer uses a pre-norm structure consisting of:

* RMSNorm, followed by GQA combined with 2D-RoPE, followed by a residual connection;
* RMSNorm, followed by an MLP with GELU, followed by a residual connection.

The model uses embedding dimension 128, four query heads, two key/value heads, an MLP ratio of 2, and dropout of 0.1.

Three parameter-efficient techniques are used:

* **2D-RoPE.** Rotary position embedding rotates each pair of query/key components by a position-dependent angle (Su et al., 2024), using an axial variant for the image grid that rotates one half of each head by the row coordinate and the other half by the column coordinate, following the RoPE-Axial extension for Vision Transformers of Heo et al. (2025). This replaces a learned positional-embedding table, saving 25,088 parameters.
* **GQA.** Grouped-query attention lets four query heads share two key/value heads (Ainslie et al., 2023), saving 16,384 parameters compared with four-head multi-head attention. The benefit quantified here is a reduction in projection parameters; it is not extrapolated into a claim about inference-speed gains.
* **RMSNorm.** Replaces LayerNorm by normalizing with the root-mean-square rather than subtracting the mean, with no shift bias (Zhang and Sennrich, 2019), saving a total of 384 parameters. This is an exact structural difference; RMSNorm's effect on accuracy still must be confirmed by ablation rather than inferred directly from the parameter count.

Together these three techniques save 41,856 parameters, roughly 6.2% of ChiliEdgeNet's parameter count. The model has 671,173 parameters, about 35.0% fewer than the base architecture (Duhan et al., 2025), which has 1,032,568 parameters.

![Figure 1. ChiliEdgeNet architecture. Panel A: overall pipeline from the 224x224x3 input through the truncated MobileNetV2 backbone (features[0:14]), a 1x1 convolution reducing channels, 196 tokens, the Tiny Transformer block, mean pooling and the classifier, with tensor shapes at each step. Panel B: detail of one Tiny Transformer block, showing RMSNorm, Grouped Query Attention with axial 2D-RoPE (4 query heads, 2 key/value heads) with a residual connection, and a GELU MLP (expansion ratio 2.0) with a residual connection.](../../results/architecture/chiliedgenet_architecture_en.png)

### 3.3. Evaluation Metrics

This work reports top-1 accuracy, macro-F1, and total parameter count. Every test-set accuracy value carries a 95% Wilson confidence interval at $N = 373$. Accuracy is the proportion of correctly predicted images out of $N$. For each class, precision and recall are computed from true positives (TP), false positives (FP), and false negatives (FN):

$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \frac{TP}{TP+FN}$$

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision}+\text{Recall}}$$

Macro-F1 is the unweighted average of the per-class F1 score across the five classes, so that each class contributes equally regardless of its size.
