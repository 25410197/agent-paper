# PAPER CONTENT: 09-LDI-NET — Plant Disease Experiment

**Source PDF:** `papers/09- LDI-NET — Plant Disease Experiment/inputs/s41598-024-62452-x.pdf`
**Extracted:** 2026-06-19

---

                                                                                    www.nature.com/scientificreports

OPEN A novel plant type, leaf disease

       and severity identification

       framework using CNN

       and transformer with multilabel

       method

                  Bin Yang1,2, Mingwei Li1,2, Fei Li3, Yongbo Wang3, Qiaokang Liang1,2, Ruiyuan Zhao3,
                  Caihong Li3 & Jianwu Wang4*

                    The growth of plants is threatened by numerous diseases. Accurate and timely identification of
                    these diseases is crucial to prevent disease spreading. Many deep learning-based methods have been
                    proposed for identifying leaf diseases. However, these methods often combine plant, leaf disease,
                    and severity into one category or treat them separately, resulting in a large number of categories
                    or complex network structures. Given this, this paper proposes a novel leaf disease identification
                    network (LDI-NET) using a multi-label method. It is quite special because it can identify plant type,
                    leaf disease and severity simultaneously using a single straightforward branch model without
                    increasing the number of categories and avoiding extra branches. It consists of three modules, i.e.,
                    a feature tokenizer module, a token encoder module and a multi-label decoder module. The LDI-
                    NET works as follows: Firstly, the feature tokenizer module is designed to enhance the capability
                    of extracting local and long-range global contextual features by leveraging the strengths of
                    convolutional neural networks and transformers. Secondly, the token encoder module is utilized
                    to obtain context-rich tokens that can establish relationships among the plant, leaf disease and
                    severity. Thirdly, the multi-label decoder module combined with a residual structure is utilized to fuse
                    shallow and deep contextual features for better utilization of different-level features. This allows the
                    identification of plant type, leaf disease, and severity simultaneously. Experiments show that the
                    proposed LDI-NET outperforms the prevalent methods using the publicly available AI challenger 2018
                    dataset.

                    Plant diseases pose a threat to the global food production. Accurate and timely identification of plant diseases
                    is crucial to prevent disease spread and ensure food security. Therefore, plant disease identification has thus
                    become one of the most urgent problems that needs to be solved in the agricultural intelligence community.

                        Usually, plant disease identification is typically done through manual observation, which is known to be
                    laborious and time-consuming. With the rapid development of computer vision, machine learning has become one
                    of the most popular technologies for identifying plant diseases.

                    Method
                    Overview

                    Figure 1 shows the architecture of the LDI-NET, which comprises a feature tokenizer module, a token encoder
                    module, and a multi-label decoder module. The feature tokenizer module is designed for extracting feature by
                    combining the strengths of CNN and transformer. The token encoder module further establishes relationships
                    among the features of plant type, leaf disease, and severity by facilitating information exchange. Moreover, the
                    multi-label decoder module integrates fused tokens and adaptive feature embeddings to selectively extract
                    features based on identifying plant, disease or severity, thus the accurate identification results for plant, leaf disease
                    and severity can be obtained. The overall constitution of the LDI-NET is as follows:

                    (1) The feature tokenizer module consists of four parts: Plant disease images are divided into patches. These
                          patches are then flattened and fed into the transformer blocks to capture long-range global information.
                          Next, a convolutional layer is connected after the transformer blocks to capture local information. In the
                          end, the patches are flattened to compact spatial feature tokens, where each token corresponds to a specific
                          aspect of the image such as color, texture or shape.

                    (2) The token encoder module consists of two parts: Extracted tokens are fed into multi-head self-attention
                          to facilitates information flowing that establish relationships among the features for plant, leaf disease
                          and severity identification. Then the Multilayer Perceptron (MLP) structure further enhances the feature
                          representation ability of model. As a result, the module generates tokens that are context-rich.

                    (3) The multi-label decoder module consists of three parts: The self-attention takes adaptive feature embed-
                          dings as input for Q, K, and V. The cross-attention takes the outputs of the self-attention as input for Q,
                          and the fused tokens as input for K and V. It selectively extracts leaf or disease region features based on
                          plant, disease or severity identification. Extracted features are then fed into an MLP. The decoder learns
                          the correlations between the labels and features to predict identification results.

                    Feature tokenizer module
                    CNN and transformer are commonly utilized as backbone networks for feature extraction. While CNN can
                    capture rich local information and provide inductive bias to improve the generalization ability of models, it
                    has limitations in capturing long-range global information due to the limited size of its receptive fields. On the
                    other hand, the transformer has shown great potential in capturing long-range global contextual visual information
                    by utilizing the global relationship among spatial pixels. However, the transformer blocks have a problem
                    in capturing local information, and the lack of inductive bias leads to a low generalization ability of models.

                        The feature tokenizer module adopts a sliding convolution window of fixed size (convolution size = 16 x 16 or
                    32 x 32) to divide the image into patches (patch size=16 x 16 or 32 x 32), which results in a disorderly arrangement
                    of spatial position information. Therefore, position embeddings are added to each patch to preserve the
                    original spatial position relationship. Then, the flattened patches are fed into the transformer blocks, which can
                    extract long-range global features. The transformer blocks lack the inductive bias and ability to capture local
                    information. To address these issues, a convolutional layer is connected. This module can express the multi-label
                    information among pixels as compact spatial feature tokens. The feature tokenizer module can be formulated
                    as follows:

                                            Y = Conv2d(TB ... TB(Conv2d(I)))    (1)

                    where Y denotes tokens, Conv2d denotes convolution, TB denotes the transformer block, and I denotes plant
                    disease image.

                    Token encoder module
                    The token encoder module consists of two multi-label encoder blocks.
                    Figure 3a illustrates the structure of the token encoder block, which consists of a multi-head self-attention (MSA)
                    and an MLP. Additionally, necessary residual connections and norms are added to prevent model degradation
                    and accelerate convergence during training.

                        (1) Multi-head Self-Attention (MSA): The SA can be formulated by calculating the similarity between Q and K:

                                            SA(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V    (5)

                    MSA can be introduced as follows:

                                            hi = SA(Q * WiQ, K * WiK, V * Wiv)                (6)
                                            MSA(q, k, v) = Concat(h1, h2, ..., hn) * WO      (7)

                        (2) Multilayer Perceptron (MLP): The MLP consists of linear layers, a GELU activation function and dropout layers.

                    Multilabel decoder module with residual structure
                    The multi-label decoder module consists of two multi-label decoder blocks. Figure 3b illustrates that the
                    multi-label decoder block with two inputs includes a multi-head self-attention, a multi-head cross-attention and
                    an MLP. The first input is discriminative adaptive feature embeddings. The second input is the fused tokens
                    (compact spatial feature tokens + context-rich tokens via residual addition).

                    Experiments and results
                    Dataset description

                    The dataset employed in this paper is the AI challenger 2018 dataset. It comprises 61 categories with a total of
                    31,718 plant disease images for training, 4540 for validation and 4514 for testing. This paper utilized 30,210
                    images from the training set and 4324 from the validation set, containing 56 categories after eliminating 5
                    categories that have only a few images. The training set is split into 80% training and 20% validation. The
                    validation set is used for testing.

                    Number of images:
                    - Training: 24,159
                    - Validation: 6,054
                    - Testing: 4,324

                    Multi-label encoding: 34 label categories total
                    - 12 plant categories (Table 1)
                    - 19 disease categories (Table 2)
                    - 3 severity categories: Healthy, General, Serious (Table 3)

                    Table 1. Plant statistics:
                    Name         | Training set | Validation set | Test set
                    Apple        | 1382         | 348            | 247
                    Cherry       | 658          | 166            | 115
                    Citrus       | 3194         | 800            | 583
                    Corn         | 1824         | 454            | 325
                    Grape        | 2202         | 548            | 394
                    Peach        | 1500         | 378            | 268
                    Pepper       | 1353         | 336            | 241
                    Potato       | 1130         | 280            | 202
                    Pumpkin      | 1025         | 260            | 184
                    Soybean      | 1142         | 288            | 204
                    Strawberry   | 817          | 200            | 144
                    Tomato       | 7932         | 1996           | 1417

                    Table 2. Disease statistics:
                    Name                       | Training | Validation | Test
                    Black Blight Fungus        | 551      | 140        | 99
                    Black Measles Fungus       | 738      | 184        | 133
                    Black Rot Fungus           | 675      | 168        | 120
                    Black Spot                 | 287      | 76         | 52
                    Cedar Rust                 | 146      | 36         | 26
                    CZTD                       | 286      | 72         | 51
                    Early Blight               | 1126     | 280        | 201
                    Greening June              | 2903     | 724        | 531
                    Late Blight                | 1654     | 416        | 296
                    Leaf Blight                | 623      | 152        | 109
                    Leaf Mold                  | 529      | 132        | 94
                    Leaf Spot                  | 564      | 142        | 100
                    Powdery Mildew             | 1205     | 306        | 214
                    Puccinia Polysora          | 670      | 168        | 120
                    Scab                       | 1831     | 460        | 326
                    Septoria Leaf Spot Fungus  | 980      | 248        | 175
                    Spider Mite Damage         | 649      | 164        | 116
                    Target Spot Bacteria       | 53       | 12         | 9
                    YLCV Virus                 | 3107     | 780        | 555

                    Table 3. Severity statistics:
                    Name      | Training | Validation | Test
                    General   | 7600     | 1893       | 1356
                    Serious   | 10,977   | 2767       | 1971
                    Healthy   | 5582     | 1394       | 997

                    Data augmentation (training set only): randomly cropped and resized to 224x224 pixels, then randomly
                    horizontally flipped.

                    Baseline models
                    Three types: (a) Single nets — single-branch model for three separate identifications (plant, disease, severity
                    each independently using ViT backbone); (b) Joint net — single-branch model identifying all 56
                    plant-disease-severity combinations; (c) Multi-task — shared ViT backbone with three separate FC branches
                    for plant, disease, severity.

                    Implementation details
                    - Hardware: Intel i9-12900K CPU, NVIDIA 3090ti GPU
                    - OS: Ubuntu
                    - Framework: PyTorch
                    - Epochs: 60
                    - Batch size: 16
                    - Optimizer: AdamW
                    - LR Scheduler: StepLR with initial learning rate = 6.25e-6, step size = 4, gamma = 0.5
                    - Runs: 5 times, results averaged

                    Experimental results (Table 4):
                    Models                        | Single nets | Joint net | Multi-task | LDI-NET
                    Plant accuracy (%)            | 99.28       | 99.35     | 99.28      | 99.42
                    Disease accuracy (%)          | 98.71       | 98.44     | 98.38      | 98.68
                    Severity accuracy (%)         | 87.86       | 87.47     | 87.51      | 88.55
                    Plant-disease-severity (PDSA%)| 86.52       | 86.26     | 85.82      | 87.40

                    Table 6. Per-plant accuracy (%):
                    Apple:99.42/99.35/99.28/97.57 (SN/JN/MT/LDI)
                    Apple: SN=97.17, JN=96.76, MT=96.76, LDI=97.57
                    Cherry: all=100
                    Citrus: all=100
                    Corn: SN=99.69, JN=100, MT=100, LDI=100
                    Grape: all=100
                    Peach: SN=97.39, JN=97.76, MT=97.39, LDI=97.76
                    Pepper: SN=99.59, JN=99.59, MT=100, LDI=99.59
                    Potato: SN=97.52, JN=96.53, MT=97.03, LDI=99.01
                    Pumpkin: all=100
                    Soybean: SN=100, JN=100, MT=99.51, LDI=99.51
                    Strawberry: SN=100, JN=100, MT=100, LDI=99.31
                    Tomato: SN=99.17, JN=99.58, MT=99.36, LDI=99.44

                    Table 7. Per-disease accuracy (%):
                    Black Blight Fungus: all=100
                    Black Measles Fungus: SN=100, JN=99.25, MT=99.25, LDI=100
                    Black Rot Fungus: all=99.17
                    Black Spot: SN=92.31, JN=90.38, MT=92.31, LDI=92.31
                    Cedar Rust: all=100
                    CZTD: SN=90.20, JN=80.39, MT=86.27, LDI=86.27
                    Early Blight: SN=98.00, JN=96.52, MT=96.02, LDI=96.52
                    Greening June: all=100
                    Late Blight: SN=98.31, JN=99.32, MT=97.64, LDI=97.97
                    Leaf Blight: SN=100, JN=100, MT=100, LDI=99.08
                    Leaf Mold: SN=97.87, JN=95.74, MT=96.81, LDI=97.87
                    Leaf Spot: SN=92.00, JN=96.00, MT=94.00, LDI=96.00
                    Powdery Mildew: SN=99.53, JN=100, MT=100, LDI=100
                    Puccinia Polysora: SN=100, JN=100, MT=99.25, LDI=100
                    Scab: SN=99.39, JN=98.77, MT=98.77, LDI=99.39
                    Septoria Leaf Spot Fungus: SN=97.71, JN=97.71, MT=97.14, LDI=99.43
                    Spider Mite Damage: all=95.69
                    Target Spot Bacteria: SN=100, JN=88.89, MT=77.78, LDI=77.78
                    YLCV Virus: SN=99.64, JN=99.28, MT=98.92, LDI=99.64

                    Table 8. Per-severity accuracy (%):
                    Healthy:  SN=98.40, JN=99.40, MT=98.80, LDI=99.50
                    General:  SN=78.91, JN=77.80, MT=79.79, LDI=81.49
                    Serious:  SN=88.19, JN=88.08, MT=87.16, LDI=87.87

                    Table 9. Backbone comparison (Precision/Recall/F1score/PDSA %):
                    VGG11:    93.72 / 93.65 / 93.64 / 83.21
                    VGG16:    94.12 / 94.08 / 94.08 / 84.46
                    ResNet18: 93.43 / 93.34 / 93.35 / 82.74
                    ResNet50: 94.49 / 94.46 / 94.46 / 85.50
                    SwinT:    94.78 / 94.65 / 94.67 / 85.68
                    SwinB:    94.92 / 94.91 / 94.91 / 86.52
                    FTM(ps=32): 95.00 / 94.99 / 94.99 / 86.68
                    FTM(ps=16): 95.28 / 95.29 / 95.28 / 87.40  <- BEST

                    Table 10. Ablation: number of encoder and decoder blocks (Precision/Recall/F1/PDSA %):
                    Row 1: enc=0, dec=1: 95.03 / 95.04 / 95.03 / 86.70
                    Row 2: enc=0, dec=2: 95.27 / 95.29 / 95.27 / 87.37
                    Row 3: enc=0, dec=3: 95.16 / 95.18 / 95.16 / 87.07
                    Row 4: enc=1, dec=1: 95.20 / 95.22 / 95.20 / 87.19
                    Row 5: enc=1, dec=2: 95.26 / 95.27 / 95.25 / 87.39
                    Row 6: enc=1, dec=3: 94.93 / 94.93 / 94.93 / 86.36
                    Row 7: enc=2, dec=1: 95.11 / 95.13 / 95.11 / 86.96
                    Row 8: enc=2, dec=2: 95.28 / 95.29 / 95.28 / 87.40  <- BEST (optimal)
                    Row 9: enc=2, dec=3: 94.94 / 94.96 / 94.93 / 86.53

                    Conclusion
                    The LDI-NET outperforms the other three baseline models in the AI challenger 2018 dataset.
                    Plant accuracy: 99.42%, Severity accuracy: 88.55%, PDSA: 87.40%.
                    Optimal architecture: 2 token encoder blocks + 2 multi-label decoder blocks.

---

## EXTRACTED FIGURES (visual analysis)

### Figure 1 — Architecture of the proposed LDI-NET (page 3, `papers/09- LDI-NET — Plant Disease Experiment/inputs/figures/page3_fig1_zoom.png`)

The LDI-NET has three main modules arranged left to right:

**Feature Tokenizer Module (left block):**
- Input: plant disease image
- Step 1: Conv2d (first convolutional layer to divide image into patches via sliding window, patch size 16x16)
- Step 2: Flatten
- Step 3: Position Embedding (added to flattened patches)
- Step 4: Transformer Block (x2, stacked, with skip connections) — repeated N times
- Step 5: Convolution (final Conv2d to capture local information)
- Output: "compact spatial token" (sequence of compact spatial feature tokens)

**Token Encoder Module (middle block):**
- Input: compact spatial tokens
- Contains: 2 x Token Encoder Block (stacked vertically)
  - Each block: Multi-head Self-Attention -> residual add -> Norm -> MLP -> residual add -> Norm
- Also takes compact spatial tokens and adds them (residual/skip connection marked with "+" symbol) to the output of the token encoder blocks
- Output: "context-rich token" (+ fused token via residual)
- The fused token = context-rich token + compact spatial token (residual connection shown with circle-plus symbol)

**Multi-Label Decoder Module (right block):**
- Input: fused token (K, V inputs) + Query (adaptive feature embeddings, one per label)
- Contains: 2 x Multi-Label Decoder Block (stacked vertically)
  - Each block: Multi-head Self-Attention (on queries) -> Multi-head Cross-Attention (Q from self-attn, K/V from fused token) -> MLP
- Output heads: multiple classification outputs shown at right
  - Labels visible: "Serious", "Tomato", "Late Blight" (example outputs)
  - Query labels visible at bottom: "General", "Serious", "Tomato", "Cherry", "Scab", "Late Blight" (sample adaptive embeddings shown)
- Each query embedding corresponds to one of the 34 labels (12 plants + 19 diseases + 3 severities)

### Figure 2 — Architecture of the feature tokenizer module (page 3, `papers/09- LDI-NET — Plant Disease Experiment/inputs/figures/page3_fig2_zoom.png`)

Sequential pipeline (left to right):
1. **Conv2d** (leftmost orange box) — initial convolution, divides image into patches
2. **Flatten** (blue box)
3. **Position Embedding** (yellow-green box, adds positional info via "+" circle)
4. **Transformer Block** repeated x N (shown as dashed group with two identical blocks inside):
   - Each block: Norm -> Multi-head Self-Attention -> residual add (+) -> Norm -> MLP -> residual add (+)
   - Two such transformer blocks shown inside the repeated group, repeated "x N" times
5. **"......"** (ellipsis indicating repetition)
6. Another set of: Norm -> Multi-head Self-Attention (+) -> Norm -> MLP (+)
7. **Conv2d** (rightmost orange box) — final convolutional layer for local feature capture

The "x N" label is shown below the repeated transformer block group, indicating N repetitions of the transformer block sub-sequence.

### Figure 3 — Architecture of the token encoder and multi-label decoder block (page 4, `papers/09- LDI-NET — Plant Disease Experiment/inputs/figures/page4_fig3_zoom.png`)

Four sub-diagrams (a), (b), (c), (d):

**(a) Token Encoder Block:**
- Input (bottom)
- Norm
- Multi-head Self-Attention (blue box)
- residual add (+)
- Norm
- MLP (purple box)
- residual add (+)
- Output (top)

**(b) Multi-label Decoder Block:**
- Input (bottom)
- Norm
- Multi-head Self-Attention (pink/red box) — takes Input as Q, K, V
- residual add (+)
- Norm
- Multi-head Cross-Attention (red box) — takes self-attn output as Q; K and V come from fused tokens
- residual add (+)
- Norm
- MLP (purple box)
- residual add (+)
- Output (top)

**(c) Multi-head Self-Attention structure:**
- Input (bottom)
- Linear projection
- Feature Split (splits into heads: K1, K2,...Kn for K; Q1, Q2,...Qn for Q; V1, V2,...Vn for V)
- Expansion of Q and K representations
- Scaled Dot-Product Attention (per head): MatMul(Q,K) -> Scale -> Softmax -> MatMul with V -> outputs O1, O2,...On
- Concat (concatenate all head outputs)
- Output (top)

**(d) MLP structure:**
- Linear (bottom)
- GELU activation
- Dropout
- Linear
- Dropout
- Output (top, via residual)
- Scale (shown between MatMul and Softmax in attention path)

### Figure 5 — Illustration of the baseline models (page 8, `papers/09- LDI-NET — Plant Disease Experiment/inputs/figures/page8.png`)

Three baseline architectures shown:

**(a) Single nets:**
- Architecture: Patch Embedding -> Transformer Encoder (x3, shown as three repeated blocks) -> "..." -> Transformer Encoder -> three separate output heads: [Crop | Disease | Severity]
- Uses ViT (Visual Transformer) as backbone
- Three separate runs/branches for three tasks

**(b) Joint net:**
- Architecture: Patch Embedding -> Transformer Encoder (x3, shown) -> "..." -> Transformer Encoder -> single output head: [Crop-Disease-Severity]
- Single branch that outputs combined 56-class prediction

**(c) Multi-task:**
- Architecture: Patch Embedding -> Transformer Encoder (x3, shared) -> "..." -> Transformer Encoder -> three output heads: [Crop], [Disease], [Severity]
- Shared backbone with three separate FC output branches

### Table 9 — Backbone comparison (page 12, `papers/09- LDI-NET — Plant Disease Experiment/inputs/figures/page12_table9_zoom.png`)

Confirmed values (Precision / Recall / F1score / PDSA %):
| Backbone   | VGG11 | VGG16 | ResNet18 | ResNet50 | SwinT | SwinB | FTM(ps=32) | FTM(ps=16) |
|------------|-------|-------|----------|----------|-------|-------|------------|------------|
| Precision  | 93.72 | 94.12 | 93.43    | 94.49    | 94.78 | 94.92 | 95.00      | **95.28**  |
| Recall     | 93.65 | 94.08 | 93.34    | 94.46    | 94.65 | 94.91 | 94.99      | **95.29**  |
| F1score    | 93.64 | 94.08 | 93.35    | 94.46    | 94.67 | 94.91 | 94.99      | **95.28**  |
| PDSA       | 83.21 | 84.46 | 82.74    | 85.50    | 85.68 | 86.52 | 86.68      | **87.40**  |

### Table 10 — Ablation: encoder/decoder block count (page 12, `papers/09- LDI-NET — Plant Disease Experiment/inputs/figures/page12_table10_zoom.png`)

Columns: Num of encoder (0/1/2) + Num of decoder (1/2/3) -> Precision / Recall / F1score / PDSA

| Row | Enc | Dec | Precision | Recall | F1score | PDSA  |
|-----|-----|-----|-----------|--------|---------|-------|
| 1   | 0   | 1   | 95.03     | 95.04  | 95.03   | 86.70 |
| 2   | 0   | 2   | 95.27     | 95.29  | 95.27   | 87.37 |
| 3   | 0   | 3   | 95.16     | 95.18  | 95.16   | 87.07 |
| 4   | 1   | 1   | 95.20     | 95.22  | 95.20   | 87.19 |
| 5   | 1   | 2   | 95.26     | 95.27  | 95.25   | 87.39 |
| 6   | 1   | 3   | 94.93     | 94.93  | 94.93   | 86.36 |
| 7   | 2   | 1   | 95.11     | 95.13  | 95.11   | 86.96 |
| 8   | 2   | 2   | **95.28** | **95.29** | **95.28** | **87.40** |
| 9   | 2   | 3   | 94.94     | 94.96  | 94.93   | 86.53 |

Best configuration: **2 Token Encoder Blocks + 2 Multi-Label Decoder Blocks**
