## 3. Proposed Method

### 3.1. Dataset

**ChiliEdge-DS** comprises leaf and fruit images of five classes — `anthracnose`, `healthy`, `leaf_curl`, `leaf_spot`, `yellow_leaf` — field-collected in southern Vietnam, pre-split into `train`/`val`/`test` (1,597 / 491 / 373; 2,461 total) and preserved throughout so results reflect generalization to unseen data. Images were photographed against a white background to make the diseased region stand out, at ~12,000×9,000 native resolution later resized to 300×300; the dataset is released on Kaggle [23]. The split is close to balanced (largest/smallest class = 1.29 < 1.5), so no class weighting was applied and macro-F1 is not distorted by a dominant class. The white background is a **controlled** acquisition setting unlike a phone camera in the field; Section 4.3 probes part of this gap and Section 6 states what it does not cover.

**Table 1. Per-class distribution of ChiliEdge-DS across the fixed split.**

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| anthracnose | 264 | 81 | 62 | 407 |
| healthy | 334 | 102 | 78 | 514 |
| leaf_curl | 328 | 104 | 82 | 514 |
| leaf_spot | 326 | 101 | 76 | 503 |
| yellow_leaf | 345 | 103 | 75 | 523 |
| **Total** | **1,597** | **491** | **373** | **2,461** |

### 3.2. Architecture: ChiliEdgeNet

Convolution's limited receptive field keeps MobileNetV2 local, whereas symptoms like leaf curl and yellowing span the leaf and need relations between distant regions. The backbone is therefore truncated and its tail replaced by a Transformer block operating on feature-grid cells as image tokens.

#### 3.2.1. Backbone Cut Point

We truncate at `features[1:14]`, yielding a **96-channel 14x14** map from a 224x224 input, i.e. **196 tokens**. The choice is mechanistic: in MobileNetV2 `features[14]` is the stride-32 stage, so cutting *after* it collapses the grid to **7x7 (49 tokens)**, at which resolution the convolutional stack has already integrated information across nearly the whole image, leaving the Transformer almost no spatial relation to learn. Every cut from `features[1:8]` to `features[1:14]` preserves the 14x14 grid; a preliminary sweep selected cut 14, the deepest of them. The grid also pays off for interpretability: the Attention Rollout map (Section 5.1) is **14x14 rather than 7x7** — four times finer.

#### 3.2.2. Feature Path

A 1x1 convolution after the stem compresses 96 channels to the 128-dimensional embedding; the 196 grid cells enter the Transformer block as image tokens; a token-wise mean and a linear layer produce five class scores.

#### 3.2.3. Tiny Transformer Block

Despite the name, ChiliEdgeNet **is purely a vision model and contains no language component** — no text encoder, no tokenizer, no pretrained language weights. It is trained end-to-end on images only with a cross-entropy loss, like an ordinary classifier. Architecturally the block is a standard Vision Transformer except for **three parameter-efficient techniques borrowed from LLMs**:

- **2D-RoPE** rotates query/key vectors by a coordinate-dependent angle per grid axis, encoding 2D position **without a single learned parameter** [18] — against a learned positional embedding needing a 196x128 = **25,088-parameter** table.
- **GQA** lets four query heads share two key/value heads, saving **16,384** attention parameters [19].
- **RMSNorm** drops LayerNorm's mean-subtraction and shift terms, saving **384** parameters [20].

Together these save **41,856 parameters (5.9% of the model)**; Section 4.4.2 measures, one technique at a time, whether that saving costs accuracy. At embedding dimension 128, one Transformer layer, MLP ratio 2 and dropout 0.1, the model has **671,173 parameters** — **35% fewer** than [5] (1,032,568).

#### 3.2.4. Removed Blocks

The base architecture [5] also places a Triplet Attention gate [17] and a residual block with a channel-attention gate (RES_Block, combining ECA and SE) after the stem. Neither is part of ChiliEdgeNet: a preliminary ablation on an otherwise identical model, replicated across two independent runs, showed that removing Triplet Attention makes the network *better* under both blur and low light — a negative result we acted on rather than reported around. RES_Block was subsequently dropped as well during a later architecture simplification pass; unlike Triplet Attention, we do not have a direct same-seed with/without ablation quantifying its individual contribution, so we make no quantitative claim about that decision.

### 3.3. Evaluation Metrics

We report top-1 accuracy, macro-averaged F1 (so minority classes weigh equally), and total parameter count. Every accuracy figure carries a 95% Wilson confidence interval at n = 373. For reference,

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad \text{F1-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
