# SKILL: DEEP LEARNING PAPER EXTRACTION

Khi đọc một file PDF, trích xuất đầy đủ các mục sau. Với mỗi mục: ghi rõ **giá trị** và **vị trí trong paper** (Mục số / Table số / trang). Nếu không tìm thấy → ghi `MISSING`.

---

## 1. FRAMEWORK & ENVIRONMENT
- Framework: PyTorch / TensorFlow / Keras / Caffe / MATLAB / Khác?
- Phiên bản (nếu đề cập): ví dụ PyTorch 1.13, TF 2.x
- GPU sử dụng (nếu đề cập): NVIDIA V100, RTX 3090...

---

## 2. ARCHITECTURE OVERVIEW
- Tên model (chính xác như trong paper): ví dụ `MobileNetV3-Small`, `PlantDiseaseNet`, `ResNet50`
- Loại kiến trúc: Custom CNN / Pretrained backbone + head / ViT / Hybrid / Ensemble
- **Pretrained backbone:**
  - Có dùng pretrained không? (Yes / No / Không nêu)
  - Nếu có: tên backbone + nguồn weights (ImageNet / COCO / custom pretrain)
  - Có freeze backbone không? Unfreeze sau bao nhiêu epochs?
- Input size: H × W × C (ví dụ: 224 × 224 × 3)

---

## 3. LAYER-BY-LAYER DETAILS
Trích xuất từng block/layer theo thứ tự, điền vào bảng:

| Layer / Block | Loại | Filters / Units | Kernel | Stride | Padding | Activation | BN? | Dropout? |
|--------------|------|----------------|--------|--------|---------|-----------|-----|---------|
| ... | | | | | | | | |

Chú ý đặc biệt:
- **Skip connections / Residual:** Có không? Loại (additive / concatenation)?
- **Attention modules:** CBAM / SE Block / ECA / Spatial Attention / Channel Attention?
- **Custom blocks:** Mô tả chi tiết block mới (nếu paper đề xuất)
- **Pooling layers:** GlobalAveragePooling / MaxPooling / AdaptiveAvgPool?
- **Classifier head:** số FC layers, units, activation cuối (Softmax / không cần nếu dùng CrossEntropy)

---

## 4. HYPERPARAMETERS TRAINING
| Thông số | Giá trị | Vị trí |
|---------|--------|--------|
| Optimizer | | |
| Learning rate khởi đầu | | |
| LR Scheduler (loại) | | |
| LR Scheduler (tham số) | ví dụ: step_size=10, gamma=0.1 | |
| Warmup epochs | | |
| Loss function | | |
| Batch size | | |
| Số epochs | | |
| Weight decay / L2 regularization | | |
| Momentum (nếu SGD) | | |
| Gradient clipping | | |
| Mixed precision (FP16) | | |

---

## 5. DATA & PREPROCESSING
- Dataset: tên + nguồn (GitHub / Kaggle / tự thu thập)
- Số classes + danh sách class names
- Tổng số ảnh + Train/Val/Test split (tỷ lệ hoặc số lượng cụ thể)
- Normalization: mean + std (ImageNet default hay custom?)
- Class imbalance: Có không? Xử lý bằng cách nào? (weighted loss, oversampling, undersampling)

---

## 6. DATA AUGMENTATION
Với mỗi kỹ thuật: ghi rõ tên + tham số + có trong paper không?

| Kỹ thuật | Tham số | Paper đề cập? |
|---------|--------|--------------|
| Rotation | range = ? | |
| Zoom / RandomResizedCrop | scale = ? | |
| Horizontal Flip | — | |
| Vertical Flip | — | |
| ColorJitter / Brightness | range = ? | |
| Cutout / Random Erasing | — | |
| MixUp / CutMix | alpha = ? | |
| Gaussian Blur | — | |
| Elastic Transform | — | |

---

## 7. EVALUATION & RESULTS
- Metrics báo cáo trong paper: Accuracy / F1 / Precision / Recall / AUC?
- Averaging method: Macro / Micro / Weighted?
- Kết quả trên dataset chính (điền vào bảng để đối chiếu sau)
- Có Ablation study không?
- Có so sánh với SOTA models khác không?
- Có Grad-CAM / visualization không?
- Inference time / FPS (nếu có)?

---

## 8. WEIGHT INITIALIZATION (nếu train from scratch)
- Xavier / He / Kaiming / Default PyTorch?
- Có đề cập không?

---

## QUY TẮC TRÍCH XUẤT
1. **Không suy diễn** — nếu paper không nêu rõ, ghi `MISSING`, không tự điền.
2. **Ghi vị trí** — mọi giá trị phải kèm "Mục X" hoặc "Table Y" hoặc "trang Z".
3. **Ưu tiên bảng và hình** — thông số thường nằm trong Table hoặc Figure caption, không chỉ trong text.
4. **Phân biệt claim vs actual** — một số paper nêu accuracy trong abstract khác với Table kết quả chi tiết — dùng Table.
