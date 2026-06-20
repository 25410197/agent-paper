# IMPLEMENT PLAN: MRW-CNN (Maize, Rice, Wheat CNN)

**Paper:** Real-Time Plant Disease Dataset Development and Detection of Plant Disease Using Deep Learning
**Tác giả:** Diana Susan Joseph, Pranav M. Pawar, Kaustubh Chakradeo
**Journal/Conference:** IEEE Access, Vol. 12, 2024
**DOI:** 10.1109/ACCESS.2024.3358333
**Ngày phân tích:** 2026-06-16 (cập nhật sau khi đọc trực tiếp Figure 9 bằng hình ảnh)

---

## PHÂN LOẠI
**Nhóm:** A (kiến trúc nay đã rõ ràng — đọc trực tiếp Figure 9 bằng vision, chỉ còn thiếu stride/padding)
**Lý do:** Bài báo đề xuất CNN mới (MRW-CNN), nêu đầy đủ optimizer/lr/batch/epoch/input size, và Figure 9 (đọc bằng hình ảnh, không phải suy diễn) cho biết chính xác thứ tự, số filters và kernel size của cả 5 conv layer.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc — Model chính: MRW-CNN (train from scratch)
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | MRW-CNN (Maize, Rice, Wheat CNN) | Tiêu đề Section V, Contribution 6 |
| Loại | CNN train from scratch | Section V |
| Framework | TensorFlow / Keras (CUDNN) | Section VI.A "Experiment Setup" — **người dùng chọn giữ nguyên TF/Keras** |
| Input size | 224 × 224 | Section III.B (ảnh resize về 224×224) |
| Pretrained backbone | Không (train from scratch) | Section V, Abstract |
| Số output classes | 5 (mỗi dataset: 4 bệnh + 1 healthy) | Section IV, V |

### Cấu trúc Layer MRW-CNN — ĐỌC TRỰC TIẾP TỪ FIGURE 9 (vision, không suy diễn)

> Xem transcription đầy đủ trong `paper_content.md` mục `## EXTRACTED FIGURES (visual analysis)`. Render gốc: `papers/04- Real-Time Plant Disease Dataset Development/inputs/figures/fig9_crop.png` và `fig9_left.png`.

| Thứ tự | Layer | Kernel | Filters | Stride | Padding | Activation | Nguồn |
|--------|-------|--------|---------|--------|---------|-----------|-------|
| 1 | Conv Block 1 | 3×3 | **32** | ⚠️ KHÔNG ghi trong hình | ⚠️ KHÔNG ghi | ReLU | Fig. 9 (CONFIRMED filters/kernel) |
| 2 | Conv Block 2 | 3×3 | **64** | ⚠️ KHÔNG ghi | ⚠️ KHÔNG ghi | ReLU | Fig. 9 |
| 3 | Conv Block 3 | 3×3 | **64** | ⚠️ KHÔNG ghi | ⚠️ KHÔNG ghi | ReLU | Fig. 9 |
| 4 | Conv Block 4 | 3×3 | **256** | ⚠️ KHÔNG ghi | ⚠️ KHÔNG ghi | ReLU | Fig. 9 |
| 5 | Conv Block 5 | 3×3 | **256** | ⚠️ KHÔNG ghi | ⚠️ KHÔNG ghi | ReLU | Fig. 9 |
| — | MaxPooling | **KHÔNG CÓ** — hình không vẽ pooling layer nào giữa hay sau các conv block | — | — | — | — | Fig. 9 + text Section V đều khẳng định chỉ có GAP ở cuối |
| — | Global Average Pooling | — | — | — | — | — | ✅ CONFIRMED (Fig. 9 + text) |
| — | Dense (output) | — | 5 (num_classes) | — | — | Softmax | ✅ CONFIRMED (Fig. 9 + text) |

**Mỗi block = Conv2D → BatchNormalization → ReLU** (mỗi box trong hình đều theo đúng thứ tự này).

> ⚠️ **CHỈ CÒN 1 ĐIỂM CHƯA RÕ:** stride và padding không xuất hiện ở BẤT KỲ ĐÂU trong paper (đã grep toàn văn, không có từ "stride"/"padding" nào liên quan đến conv layer). Đây là quyết định quan trọng vì ảnh hưởng trực tiếp đến output shape mỗi block và tổng số tham số — KHÔNG giống quyết định "MaxPool2×2 sau mỗi block" mà người dùng đã duyệt trước đó (đã sai, bị loại bỏ vì Figure 9 không vẽ pooling nào cả). Xem mục `CẦN XÁC NHẬN` bên dưới.

### Kiến trúc — 8 Model fine-tuned (transfer learning, để tham khảo/so sánh — KHÔNG build trong notebook này)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Danh sách 8 model | VGG16, ResNet50, ResNet101, InceptionV3, InceptionResNetV2, MobileNetV2, MobileNet, Xception | Section IV, VI.A |
| Pretrained | ImageNet weights, tải từ Keras | Section IV, VI.A |

### Hyperparameters (CONFIRMED)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | SGD | Section VI.B + Section V |
| Learning rate | 0.001 | Section VI.B |
| Batch size | 32 | Section VI.B |
| Epochs | 100 | Section VIII.A "All the models were trained for 100 epochs" |
| Loss function | CategoricalCrossentropy | Người dùng xác nhận (paper không nêu rõ, chuẩn cho multi-class+softmax) |
| Normalization | Scale /255 (KHÔNG ImageNet mean/std) | Người dùng xác nhận (train from scratch) |
| Early stopping | Có cho 8 model fine-tuned; MRW-CNN không nêu rõ → dùng best-checkpoint theo rule dự án #10 | Section IV |

### Data Split
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Train / Val / Test (paper gốc) | 80% / 10% / 10% | Section IV |
| Train / Val / Test (dataset thực tế dùng) | Pre-split sẵn trên Kaggle, dùng trực tiếp | Xem mục Dataset |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Horizontal flip | có | Section III.C |
| Rotation | 90 độ | Section III.C |
| Zoom range | 0.2 | Section III.C |
| Brightness range | 0.2 đến 0.8 | Section III.C |

> Theo rule dự án: chỉ 4 augmentation chuẩn (Rotation, Zoom, Horizontal Flip, Brightness) — tham số hóa đúng theo paper.

### Dataset Thực Tế (substitute — paper gốc không công khai)
| Thông số | Giá trị |
|---------|--------|
| Đường dẫn | `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300` |
| Cấu trúc | Pre-split `train/`, `val/`, `test/`, mỗi folder 5 class: anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf |
| Lý do thay thế | Dataset maize/rice/wheat gốc của paper không công khai |

### Kết quả trong paper (tham khảo — testing accuracy)
| Dataset | MRW-CNN (đề xuất) | Best fine-tuned |
|---------|-------------------|------------------|
| Maize | 0.9704 | Xception (0.9580) |
| Rice | 0.9706 | Xception (0.9728) |
| Wheat | 0.9808 | MobileNetV2 (0.9632) |

---

## ĐÃ XÁC NHẬN (người dùng quyết định ngày 2026-06-16, sau khi đọc Figure 9)

1. **Stride = 2** cho cả 5 conv layer (paper không nêu ở đâu cả — đây là giá trị giả định duy nhất còn lại). Padding = 'same'. Lý do: không có MaxPooling/pooling layer nào khác trong network (đã xác nhận qua Figure 9 + text), nên stride=2 đóng vai trò downsampling — pattern CNN thông thường, nhẹ compute hơn stride=1. Spatial size giảm dần: 224→112→56→28→14→7 trước Global Average Pooling.
   - ⚠️ Phải ghi rõ trong notebook: "stride=2, padding='same' — ASSUMED (paper không nêu); filters/kernel/order/no-pooling — CONFIRMED từ Figure 9".

---

## GHI CHÚ IMPLEMENTATION

- **Deliverable bắt buộc:** MRW-CNN (model đề xuất). 8 model fine-tuned không build trong notebook này.
- **Đã sửa lỗi:** Bản plan trước đó (trước khi đọc Figure 9 bằng hình) giả định filters tăng dần 32→64→128→256→512 kèm MaxPool2×2 sau mỗi block — SAI. Figure 9 cho thấy rõ filters thực tế là 32→64→64→256→256 và KHÔNG có MaxPooling nào. Notebook cũ (`MRW-CNN_Experiment.ipynb` viết trước khi sửa) cần được viết lại theo cấu hình đúng này.
- **Workflow đã cập nhật:** `pdf-extractor` giờ tự động render + đọc hình bằng vision khi text không đủ thông tin kiến trúc (xem `.claude/agents/pdf_extractor.md` Bước 2, script `.claude/scripts/render_pdf_page.py`). Áp dụng cho mọi bài báo từ nay.
- **Class imbalance:** Paper nói dataset gốc cân bằng; dataset ớt thực tế vẫn auto-detect theo rule #11.
