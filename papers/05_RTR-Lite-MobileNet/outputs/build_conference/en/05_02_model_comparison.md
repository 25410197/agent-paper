### 4.2. Comparison with State-of-the-Art Models

At seed 42, ChiliEdgeNet reaches 97.86% in the baseline scenario and 98.93% in the augmented scenario. The corresponding macro-F1 values are 0.9790 and 0.9897.

A total of 17 baselines from 15 recent studies were reproduced and evaluated under the same framework, presented in full in Table 3. Because the baselines were each run once and not individually tuned for ChiliEdge-DS, this table reflects results under shared data, split, and metrics, not an absolute ranking between architectures.

RTR_Lite_MobileNet (Duhan et al., 2025) was chosen as the primary baseline because it is the direct base architecture of ChiliEdgeNet and has the highest observed accuracy among the evaluated baselines on the ChiliEdge-DS test set. Since the test result contributed to this baseline selection, the direct comparison with RTR_Lite_MobileNet is treated as exploratory analysis on the current split, not an independent test evaluation establishing statistical ranking or a general superiority claim. The parameter reduction remains a structural comparison and does not depend on test-based selection.

For the study by Sujatha et al. (2025), ten combinations of two backbones (VGG19 and InceptionV3) with five classifiers (SVM, kNN, Random Forest, AdaBoost, and Decision Tree) were evaluated on the same data split. The InceptionV3 + SVM configuration, which had the highest observed test accuracy, was chosen as the representative in Table 3. Because this selection used test results, it is likewise treated as exploratory analysis on the current split and is not used to claim statistical superiority or establish an absolute ranking between architectures.

**Comparison with the base architecture.** In the seed-42 run, ChiliEdgeNet reaches 98.93% augmented accuracy, higher than RTR_Lite_MobileNet (Duhan et al., 2025) at 98.12%, a difference of 0.80 percentage points, or 3 images, on the 373-image test set.

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

**Error analysis.** In the augmented scenario, ChiliEdgeNet mispredicts the following images:

- one *leaf_spot* image predicted as *healthy*;
- three *yellow_leaf* images predicted as *leaf_curl*.

Accuracy reaches 98.93% with a 95% Wilson confidence interval of [97.28%, 99.58%]. The remaining errors cluster at the boundary between yellowing and curling symptoms.

![Figure 2. Confusion matrix of ChiliEdgeNet on the test set (augmented scenario).](../../results/notebook_c1/c14_cm_s2.png)
