### 4.1. Setup

Experiments were run on Kaggle with an NVIDIA Tesla T4 GPU; the reproduction code uses PyTorch or TensorFlow depending on the architecture. Each baseline is evaluated under two scenarios:

- **Baseline:** no data augmentation applied.
- **Augmented:** the data-augmentation techniques in Table 2 applied.

The *val* and *test* sets are never augmented. For epoch-trained models, the checkpoint with the best validation criterion for each configuration is reloaded before test evaluation.

**Common evaluation framework.** Seventeen baselines use the same ChiliEdge-DS dataset, the same fixed *train*/*val*/*test* split, seed 42, and the metrics accuracy, macro-F1, and parameter count. Both scenarios are evaluated on the same non-augmented test set. Architecture-dependent settings such as input size, batch size, epoch count, optimizer, and augmentation intensity follow each model's own reproduction design. The comparison is therefore controlled for data, split, and metric, but does not use a fully uniform training recipe across every architecture. Two cases where test results influenced selection—RTR_Lite_MobileNet (Duhan et al., 2025) as the primary baseline and InceptionV3 + SVM as the representative for Sujatha et al. (2025)—are disclosed in Section 4.2 and interpreted only as exploratory analysis.

The common 17-baseline evaluation framework is limited to in-domain results on the ChiliEdge-DS test set: accuracy, macro-F1, and parameter count. Additional analyses of blur, low light, architectural ablation, XAI, and domain transfer were carried out only for ChiliEdgeNet and its corresponding controlled variants. This work therefore does not present these experiments as a uniform robustness or domain-transfer benchmark for all 17 baselines.

The table below summarizes the significant changes applied when adapting each baseline architecture to the five-class task.

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

**Run count and statistical testing.** Results for the proposed model and its ablation variants are reported from a single run with seed 42. This work does not report a mean or standard deviation across multiple seeds and does not interpret these results as evidence of stability across initializations. The McNemar test is used to assess the difference between two models on a per-image basis over the same 373-image test set.

For each comparison, the tables report two raw counts, written as b / c, alongside the McNemar $p$-value:

- $b$: images the first model predicts correctly while the second predicts incorrectly;
- $c$: images the first model predicts incorrectly while the second predicts correctly.

McNemar's test checks whether the number of discordant cases in the two directions differs significantly. This test does not substitute for assessing variance across multiple seeds; that limitation is discussed in Section 6.

In this work, McNemar is applied to comparisons between ChiliEdgeNet and its ablation variants on the same 373 test images, corresponding to Tables 5–7, and to the paired comparison of the two checkpoints on the same 1,463 cross-dataset query images in Table 10b. The test is not used to rank all 17 baselines in Table 3.

The base architecture RTR_Lite_MobileNet (Duhan et al., 2025) in Section 4.2 was also reproduced with seed 42. The comparison is therefore seed-matched and follows the same procedure, but each architecture was still run only once.

**Inference latency measurement.** ChiliEdgeNet's CPU latency is 20.24 ms/image, equivalent to 49.41 images/second, measured on an Intel Xeon 2.00 GHz CPU with 4 logical cores on a Kaggle virtual machine. The measurement setup uses PyTorch FP32, batch size 1, with 10 warm-up iterations and 50 timed iterations. The 49.41 images/second figure is inference throughput converted from a single-image server benchmark, not the FPS of a camera application.

On the same environment, the Tesla T4 GPU reaches:

- 5.46 ms/image at batch size 1;
- 0.82 ms/image at batch size 32.

The CPU used is a server-class x86 CPU, not edge hardware. These results therefore reflect only the inference cost in the current experimental environment.
