### 4.5. Four-Class Cross-Dataset Evaluation After Few-Shot Adaptation

The stress tests in Section 4.3 only synthesize blur and brightness reduction from ChiliEdge-DS images, so they do not fully capture the acquisition domain gap. This work carries out an additional evaluation on the Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh, version 1, released on Mendeley Data under a CC BY 4.0 licence (Nirob et al., 2025). The source images were collected at a garden in Charpolisha, Jamalpur (Bangladesh) between October 2024 and February 2025; the disease-leaf portion comprises 1,856 images across six source labels.

Only four labels with a direct correspondence to the ChiliEdgeNet label space are used. Two labels, *Bacterial Spot* (156 images) and *White spot* (195 images), are excluded because the model has no corresponding target class. The source dataset has no anthracnose class, so there is no *anthracnose* ground-truth row. However, the checkpoint still retains all five outputs used during training; any prediction toward *anthracnose* is still counted as incorrect, and this prediction column is kept in the confusion matrix.

**Table 10a.** Verified support/query split from the four mapped classes.

| Source label (Nirob et al., 2025) | ChiliEdgeNet label | Original images | Few-shot support | Clean query | Corrected test folder |
|---|---|---:|---:|---:|---:|
| *Healthy Leaf* | *healthy* | 458 | 8 | 450 | 450 |
| *Curl Virus* | *leaf_curl* | 423 | 9 | 414 | 414 |
| *Cercospora Leaf Spot* | *leaf_spot* | 180 | 16 | 164 | 164 |
| *Nutrition Deficiency* | *yellow_leaf* | 444 | 9 | 435 | 435 |
| **Total** |  | **1,505** | **42** | **1,463** | **1,463** |

**Few-shot setup.** The support set comprises 42 images taken as-is from Nirob et al. (2025): 8 *healthy*, 9 *leaf_curl*, 16 *leaf_spot*, and 9 *yellow_leaf*. The support images were added to ChiliEdge-DS's training set to create a separate adapted checkpoint; ChiliEdge-DS's own in-domain validation and test sets are unchanged. This is supervised domain adaptation with a small support set, not episodic meta-learning. The number of shots differs across classes, so this is not a balanced *N*-way *K*-shot setup.

To control for data leakage, the support and query sets were tracked with SHA-256 hashes rather than filenames alone. The query set is defined as the complement of the 42 support images within the 1,505 images across the four classes, and an audit confirmed the hash intersection between the two is zero. The few-shot checkpoint was trained with seed 42.

To support independent reproduction, the clean 1,463-image query set is released as the Chili Leaf Disease Cross-Dataset Query Set, Kaggle version 2 (Hải, 2026a), and the 42 support images are released as Chili Leaf Disease Few-Shot, Kaggle version 1 (Hải, 2026b). Both packages are derived splits of the original Mendeley Data dataset and do not replace the requirement to cite the source dataset (Nirob et al., 2025).

**Results on the clean query set.** Both checkpoints were evaluated on the same 1,463-image clean query set. The original checkpoint uses seed 42 (SHA-256 verified); the checkpoint after few-shot adaptation also uses seed 42 (SHA-256 verified). Images were resized to 224×224 pixels and normalized with ImageNet statistics. Macro precision, recall, and F1 are averaged only over the four classes with ground truth; predictions toward *anthracnose* still reduce accuracy.

**Table 10b.** Results before and after few-shot adaptation on the 1,463-image clean query set.

| Configuration | Seed | Accuracy | Precision-macro | Recall-macro | F1-macro | Correct images |
|---|---:|---:|---:|---:|---:|---:|
| Original checkpoint, no few-shot | 42 | 30.96% | 0.4337 | 0.2843 | 0.1766 | 453/1,463 |
| Checkpoint after few-shot | 42 | 89.75% | 0.8837 | 0.8683 | 0.8744 | 1,313/1,463 |
| **Difference** |  | **+58.78 pp** | **+0.4500** | **+0.5840** | **+0.6978** | **+860** |

At the original checkpoint, *leaf_curl* has recall 1.0000 but precision only 0.3013, while *yellow_leaf* has recall zero. After few-shot adaptation, F1 for *healthy*, *leaf_curl*, *leaf_spot*, and *yellow_leaf* reaches 0.9168, 0.9619, 0.7557, and 0.8633 respectively. A per-image prediction comparison shows 443 images correctly predicted by both checkpoints, 140 images incorrectly predicted by both, 870 images turning from incorrect at the original checkpoint to correct after few-shot, and 10 images in the opposite direction. A two-sided exact McNemar test gives $p=1.83\times10^{-242}$. This is a single paired contrast on an out-of-domain dataset and does not belong to the ablation's Holm test family. Predictions for all 1,463 images and the paired statistics are stored in the metadata package accompanying this study.

![Figure S1. Confusion matrix of the original checkpoint on the 1,463-image clean query set.](<../../results/Cross check default/cross_dataset_local_cm.png>)

![Figure S2. Confusion matrix of the checkpoint after few-shot adaptation on the same 1,463-image clean query set.](<../../results/cross check them data/cross_dataset_local_cm.png>)
