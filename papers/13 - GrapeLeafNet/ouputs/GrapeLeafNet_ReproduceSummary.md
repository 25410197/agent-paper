# GrapeLeafNet — Reproduce Summary

**Paper:** GrapeLeafNet: A Dual-Track Feature Fusion Network With Inception-ResNet and Shuffle-Transformer for Accurate Grape Leaf Disease Identification  
**Journal:** IEEE Access, 2024 — DOI: 10.1109/ACCESS.2024.3361044  
**Authors:** R. Karthik et al.  
**Framework:** PyTorch (confirmed in paper Section IV.C)  
**Notebook:** `papers/13 - GrapeLeafNet/ouputs/grapeleafnet.ipynb`

---

## Architecture Overview

GrapeLeafNet là mạng dual-track fusion:
- **CNN Track:** Conv init → 4× CNNStage (Conv + InceptionResNet Block + MaxPool + Dropout) → Conv proj → CBAM attention
- **Shuffle Transformer Track:** PatchEmbed → 4 stages ShuffleTransformerBlock (WMSA + NWC + Shuffle-WMSA) với PatchMerging
- **Fusion:** Concatenate hai track → Conv fusion → Coordinate Attention → Classifier

## Confirmed Parameters (from paper)

| Parameter | Value | Source |
|---|---|---|
| Optimizer | Adam | Section IV.C |
| Learning rate | 1e-4 | Section IV.C |
| Weight decay | 1e-4 | Section IV.C |
| Epochs | 120 | Section IV.E.1 |
| Input size | 256×256 | Section IV.A |
| Framework | PyTorch | Section IV.C |
| Total params (paper) | 8,836,697 | Table 4 |

## Assumed Parameters (paper không cung cấp)

| Parameter | Assumed Value | Risk |
|---|---|---|
| CNN Track filter sizes | 8→16→32→64→128 per stage | HIGH — gây lệch ~40% params |
| embed_dim Shuffle Transformer | 32 | HIGH |
| ST depths / num_heads | [1,1,1,1] / [1,2,4,8] | HIGH |
| Dropout rate | 0.1 (stages), 0.5 (classifier) | MEDIUM |
| Loss function | CrossEntropyLoss | LOW |
| Batch size | 32 | LOW |
| Normalization | ImageNet mean/std | LOW |

> **Lưu ý:** Paper (Table 4) báo cáo 8,836,697 params. Notebook hiện có ~5.3M params do thiếu thông số chi tiết về filter size và embed_dim. Đây là giới hạn của paper, không phải lỗi implement.

## Augmentation (Scenario 2) — from Section IV.B

- Rotation: 0–40°
- Shear: 0–0.2
- Width shift: 0–0.2
- Horizontal Flip
- Vertical Flip
- Brightness: 0.5–1.5
- Gaussian Blur: sigma 0–0.8

## Dataset

- **Paper:** PlantVillage Grape Leaf — 4 classes (Healthy, Black Rot, Black Measles, Isariopsis Leaf Spot)
- **Notebook:** Custom plant disease dataset — 5 classes (auto-detected)
- **Split:** Pre-split train/val/test (không dùng train_test_split)
