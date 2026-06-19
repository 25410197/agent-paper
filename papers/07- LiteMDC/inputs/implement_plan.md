# IMPLEMENT PLAN: Lite-MDC

**Paper:** Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop
**Tác giả:** Sandesh Bhagat, Manesh Kokare, Vineet Haswani, Praful Hambarde, Trupti Taori, P.H. Ghante, D.K. Patil
**Journal/Conference:** Smart Agricultural Technology 7 (2024) 100408, Elsevier
**DOI:** 10.1016/j.atech.2024.100408
**Ngày phân tích:** 2026-06-18

---

## PHÂN LOẠI
**Nhóm:** A
**Lý do:** Paper đề xuất kiến trúc mạng mới có tên cụ thể (Lite-MDC) với bảng layer đầy đủ (Table 5 — 26 layer, authoritative), block MDsConv mô tả chi tiết kèm pseudocode (Algorithm 1) và sơ đồ (Fig. 7, 8a, 8b đã extract), đầy đủ hyperparameters training (optimizer, batch size, loss, LR schedule, input size) và kết quả thực nghiệm trên 4 dataset. Đủ thông số để phục dựng hoàn toàn.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | Lite-MDC | Abstract / Tiêu đề / Mục 4.3 |
| Loại | CNN nhẹ — Multi-kernel Depthwise Separable Convolution (lấy VGG16 làm baseline) | Mục 4.3 |
| Framework | Keras với TensorFlow backend | Mục 5.2 |
| Input size | 224 x 224 x 3 | Mục 4.3 / Table 5 / Mục 5.2 |
| Pretrained backbone | Không (train from scratch — kiến trúc tự thiết kế, không nêu transfer learning cho Lite-MDC) | Mục 4.3 |
| Tổng params | ~2.2 M | Abstract / Table 9 / Mục 4.3 |

### Module MDsConv (Multi-kernel Depthwise Separable Convolution)
Nguồn: Mục 4.2, Algorithm 1, Fig. 7c, Fig. 8b. Luồng dữ liệu chính xác:
1. Input H x W x C → **Channel Split** thành 2 nhóm bằng nhau: [C/2, C/2]
2. Nhánh trái (C/2 kênh): **DepthwiseConv2D 3x3** (padding='same') → BatchNorm → (LeakyReLU)
3. Nhánh phải (C/2 kênh): **DepthwiseConv2D 5x5** (padding='same') → BatchNorm → (LeakyReLU)
4. **Concatenate** 2 nhánh → H x W x C
5. **Pointwise Conv2D 1x1** (N filters) → BatchNorm → (LeakyReLU)
6. Output H x W x N

**Lưu ý quan trọng về stride:** Trong Table 5, MDConv và PConv được liệt kê thành 2 dòng riêng, nhưng cột "Output shape" cho thấy stride giảm spatial xảy ra ở **bước depthwise của MDConv** (ví dụ MDConv_1 stride=2 đưa 224→112). PConv luôn stride=1. Khi implement, đặt `strides` của DepthwiseConv2D theo cột Stride của dòng MDConv tương ứng.

### Cấu trúc Layer đầy đủ (Table 5 — AUTHORITATIVE, theo thứ tự Input → Output)
| # | Layer Name | Kernel / Filters | Stride | Output shape | Ghi chú |
|---|-----------|------------------|--------|--------------|---------|
| 1 | Input Image | - | - | 224 x 224 x 3 | |
| 2 | Conv0 | 3x3, 64 filters | 1 | 224 x 224 x 64 | Conv2D thường + BN + LeakyReLU |
| 3 | MDConv_1 | dw 3x3 x32 + dw 5x5 x32 | 2 | 112 x 112 x 64 | split 64→[32,32] |
| 4 | PConv_1 | 1x1, in64→out64 | 1 | 112 x 112 x 64 | pointwise |
| 5 | MDConv_2 | dw 3x3 x32 + dw 5x5 x32 | 1 | 112 x 112 x 64 | split 64→[32,32] |
| 6 | PConv_2 | 1x1, in128→out128 | 1 | 112 x 112 x 128 | xem ghi chú PConv bên dưới |
| 7 | MDConv_3 | dw 3x3 x64 + dw 5x5 x64 | 2 | 56 x 56 x 128 | split 128→[64,64] |
| 8 | PConv_3 | 1x1, in128→out128 | 1 | 56 x 56 x 128 | |
| 9 | MDConv_4 | dw 3x3 x64 + dw 5x5 x64 | 1 | 56 x 56 x 128 | split 128→[64,64] |
| 10 | PConv_4 | 1x1, in128→out256 | 1 | 56 x 56 x 256 | |
| 11 | MDConv_5 | dw 3x3 x128 + dw 5x5 x128 | 1 | 56 x 56 x 256 | split 256→[128,128] |
| 12 | PConv_5 | 1x1, in256→out256 | 1 | 56 x 56 x 256 | |
| 13 | MDConv_6 | dw 3x3 x128 + dw 5x5 x128 | 2 | 28 x 28 x 256 | split 256→[128,128] |
| 14 | PConv_6 | 1x1, in256→out256 | 1 | 28 x 28 x 256 | |
| 15 | MDConv_7 | dw 3x3 x128 + dw 5x5 x128 | 1 | 28 x 28 x 256 | split 256→[128,128] |
| 16 | PConv_7 | 1x1, in256→out512 | 1 | 28 x 28 x 512 | |
| 17 | MDConv_8 | dw 3x3 x256 + dw 5x5 x256 | 1 | 28 x 28 x 512 | split 512→[256,256] |
| 18 | PConv_8 | 1x1, in512→out512 | 1 | 28 x 28 x 512 | |
| 19 | MDConv_9 | dw 3x3 x256 + dw 5x5 x256 | 2 | 14 x 14 x 512 | split 512→[256,256] |
| 20 | PConv_9 | 1x1, in512→out512 | 1 | 14 x 14 x 512 | |
| 21 | MDConv_10 | dw 3x3 x256 + dw 5x5 x256 | 1 | 14 x 14 x 512 | split 512→[256,256] |
| 22 | PConv_10 | 1x1, in512→out512 | 1 | 14 x 14 x 512 | |
| 23 | MDConv_11 | dw 3x3 x256 + dw 5x5 x256 | 1 | 14 x 14 x 512 | split 512→[256,256] |
| 24 | PConv_11 | 1x1, in512→out512 | 1 | 14 x 14 x 512 | |
| 25 | MDConv_12 | dw 3x3 x256 + dw 5x5 x256 | 2 | 7 x 7 x 512 | split 512→[256,256] |
| 26 | PConv_12 | 1x1, in512→out512 | 1 | 7 x 7 x 512 | |
| 27 | DwConv_13 | depthwise 7x7 x512 | 1 | 1 x 1 x 512 | depthwise toàn cục (valid padding, kernel=7 trên feature map 7x7) |
| 28 | PConv_13 | 1x1, in512→out512 | 1 | 1 x 1 x 512 | |
| 29 | Conv_14 | 1x1, in512→out512 | 1 | 1 x 1 x 512 | thay cho FC layer (pointwise 512 filters) |
| 30 | Global Average Pooling | - | - | 512 | |
| 31 | Softmax | classifier | - | 5 (= số class dataset) | |

### Ghi chú diễn giải Table 5 (từ paper)
- MDConv "3x3x32, 5x5x32" = 32 filter depthwise kernel 3x3 + 32 filter depthwise kernel 5x5 → tổng 64 kênh output trước pointwise.
- PConv "1x1 x A x B" = pointwise conv 1x1, A kênh vào → B kênh ra.
- **Mâu thuẫn nội tại cần lưu ý:** cột "Filters size" của một số PConv (vd PConv_2 ghi `in128→out128`) không khớp liền mạch với output channels của MDConv liền trước (MDConv_2 ra 64). Cột **Output shape** mới là nguồn đáng tin để xác định số kênh thực tế tại mỗi tầng (xem cột Output shape trong bảng trên). Khi implement, ưu tiên khớp theo **Output shape** — đây là chuỗi kênh: 3→64→64→64→128→128→256→256→256→256→512→512→512→512→512→512→512→512. Pointwise tại mỗi block đặt số filter = output channels mong muốn theo cột Output shape.

### Quy ước chung toàn mạng (Mục 4.3)
- Tất cả layer theo sau bởi **Batch Normalization**.
- **LeakyReLU** sau mỗi BN, NGOẠI TRỪ layer cuối cùng của mạng (không activation ở tầng cuối trước softmax).
- Stride 2 dùng để giảm chiều không gian (thay cho max pooling).
- FC thay bằng pointwise 1x1 (512 filters) — chính là Conv_14.
- Kết thúc: Global Average Pooling → Softmax → 5 class.
- **Dropout:** tăng dần từ 0.05 đến 0.4 ở vài layer cuối (paper không liệt kê vị trí/giá trị chính xác từng layer — xem CẦN XÁC NHẬN).

### Hyperparameters
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | Adam | Mục 5.2 |
| Learning rate (khởi đầu) | Không nêu giá trị cụ thể | (xem CẦN XÁC NHẬN) |
| LR Scheduler | ReduceLROnPlateau — nếu val_loss không đổi 5 epoch liên tiếp → giảm LR theo factor 0.3 | Mục 5.2 |
| Warmup epochs | Không đề cập | - |
| Batch size | 16 | Mục 5.2 |
| Epochs | Không đề cập | (xem CẦN XÁC NHẬN) |
| Loss function | Categorical cross-entropy | Mục 5.2 |
| Weight decay | Không đề cập | - |
| Activation (hidden) | LeakyReLU (alpha không nêu — dùng default Keras 0.3, hoặc xem CẦN XÁC NHẬN) | Mục 4.3 |
| Dataset split | 60% train / 20% val / 20% test | Mục 5.2 |
| Normalization | Trừ mean từng kênh RGB khỏi mỗi pixel (per-channel mean subtraction) | Mục 3.1.3 |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | range = 360 | Mục 5.2 |
| Width shift | 0.2 | Mục 5.2 |
| Height shift | 0.2 | Mục 5.2 |
| Shear | 0.2 | Mục 5.2 |
| Zoom | 0.2 | Mục 5.2 |
| Brightness | range 0.1–1.0 | Mục 5.2 |
| Horizontal flip | có | Mục 5.2 |
| Vertical flip | có | Mục 5.2 |
| Fill mode | nearest | Mục 5.2 |
| Lưu ý | Số ảnh tăng 2x nhờ augmentation | Mục 5.2 |

> **Lưu ý ràng buộc dự án (CLAUDE.md rule 3):** Notebook PHẢI có 2 kịch bản.
> - **Scenario 1 (Baseline):** training transform CHỈ Resize + ToTensor/rescale + Normalize. KHÔNG augmentation.
> - **Scenario 2 (Augmented):** dùng 04 thuật toán: Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness (lấy tham số tham khảo từ bảng trên). Validation/Test KHÔNG aug.

### GitHub Source
**Link (dataset):** https://github.com/sandesh001/First-Pigeon-pea-datset-for-diseases-detection (Mục 3.1.3 — repo dataset, không phải code model)
**Link (code model):** Không tìm thấy link source code của Lite-MDC trong paper.
**link.txt trong inputs:** orchestrator nên kiểm tra file `papers/07- Advancing real-time plant disease detection/inputs/link.txt` (có thể chứa link dataset Kaggle để chạy notebook).

### Kết quả trong paper (tham khảo)
| Dataset | Accuracy | Precision | Recall | F1 | Params |
|---------|---------|-----------|--------|-----|--------|
| Pigeon-pea (proposed) | 94.14% (Table 9) / 95.01% (Table 7) | 94 | 94 | 95 | 2.2 M |
| PlantVillage | 99.78% | 99 | 99 | 99 | 2.2 M |
| Cassava | 86.4% | 86 | 90 | 91 | 2.2 M |
| Apple leaf | 97.2% (Table 10) / 97.36% (Table 7) | 86 | 90 | 91 | 2.2 M |

FPS: 33–34 trên NVIDIA P100. FLOPS Lite-MDC ~0.61 G. Computations ~1167 M.

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG có (hoặc không đủ rõ) trong bài báo. Vui lòng trả lời trước khi viết notebook.

1. **Learning rate khởi đầu:** Paper chỉ nêu dùng Adam + ReduceLROnPlateau (factor 0.3, patience 5 trên val_loss) nhưng KHÔNG cho giá trị LR ban đầu. Bạn muốn dùng giá trị nào? (gợi ý: 1e-3 — default Adam).
2. **Số epochs:** Paper không đề cập số epoch training. Bạn muốn train bao nhiêu epochs? (gợi ý: 50–100; trên Kaggle GPU nên cân nhắc thời gian).
3. **Dropout chi tiết:** Paper nói "dropout tăng dần từ 0.05 đến 0.4 ở vài layer cuối" nhưng KHÔNG nêu chính xác đặt ở những layer nào và giá trị từng layer. Bạn muốn:
   (a) Tôi tự đặt dropout tăng dần ở các block cuối (vd 0.05 ở MDConv_9-block → ... → 0.4 trước GAP) theo diễn giải hợp lý, hay
   (b) Bạn chỉ định cụ thể vị trí/giá trị?
4. **LeakyReLU alpha (negative slope):** Paper không nêu giá trị alpha. Dùng default Keras (0.3) hay giá trị khác (gợi ý phổ biến: 0.1 hoặc 0.01)?
5. **Dataset sẽ chạy trên Kaggle:** Notebook sẽ dùng dataset nào? (pigeon-pea / PlantVillage / Cassava / Apple, hoặc dataset Kaggle khác). Số class output ở tầng softmax phải khớp dataset thực tế (paper: pigeon-pea=5). Vui lòng cung cấp Kaggle dataset slug để wiring đường dẫn.
6. **Normalization cụ thể:** Paper dùng "per-channel mean subtraction" (không chia std, không nêu mean cụ thể). Bạn muốn:
   (a) Tính mean per-channel trực tiếp từ tập train, hay
   (b) Dùng chuẩn ImageNet mean/std (phổ biến khi reproduce trên Keras), hay
   (c) Đơn giản rescale 1/255?

---

## GHI CHÚ IMPLEMENTATION

1. **Framework:** Theo paper và yêu cầu người dùng → dùng **Keras / TensorFlow**. (Lưu ý: các ràng buộc CLAUDE.md viết theo PyTorch — khi dùng TF cần ánh xạ tương đương: `model.summary()` thay torchinfo, `tf.random.set_seed(42)`, `ModelCheckpoint` lưu best theo `val_accuracy`, callback ReduceLROnPlateau, `sklearn.metrics.classification_report` vẫn dùng được trên numpy predictions).

2. **Build MDsConv block (TF/Keras):** dùng `tf.split` hoặc Lambda split kênh thành 2 nửa → `DepthwiseConv2D(kernel_size=3, strides=s, padding='same')` và `DepthwiseConv2D(kernel_size=5, strides=s, padding='same')` → `Concatenate()` → `BatchNormalization` → `Conv2D(N, 1)` (pointwise) → `BatchNormalization`. LeakyReLU sau mỗi BN trừ tầng cuối mạng. Lưu ý DepthwiseConv2D trong Keras có `depth_multiplier=1` → giữ nguyên số kênh nhánh (C/2), khớp paper.

3. **Stride placement:** Áp stride (2 hoặc 1) vào DepthwiseConv2D của MDConv theo cột Stride Table 5. Pointwise luôn stride=1. Kiểm tra output shape khớp đúng cột "Output shape" của Table 5 — đây là cách verify kiến trúc đúng.

4. **DwConv_13 (7x7 trên feature map 7x7):** đây là depthwise global, dùng `DepthwiseConv2D(7, padding='valid')` để ra 1x1x512. Có thể thay bằng GAP nhưng paper liệt kê rõ là DwConv 7x7 → giữ đúng để khớp param count (~2.2M).

5. **Số kênh — nguồn chân lý:** Khi có mâu thuẫn giữa cột "Filters size" và "Output shape" trong Table 5 (đặc biệt PConv_2), LẤY THEO cột Output shape. Chuỗi kênh mục tiêu: 64→64→64→128→128→256→256→256→256→512→512→512→512→512→512→512→512. Sau khi build xong, `model.summary()` total params nên xấp xỉ 2.2M — nếu lệch nhiều cần rà lại số filter pointwise.

6. **Tầng cuối không activation:** Conv_14 (pointwise 512) là layer cuối trước GAP — theo paper "LeakyReLU except the last one", nên KHÔNG đặt LeakyReLU sau BN của Conv_14. Softmax chỉ ở Dense/output cuối (5 class).

7. **Class imbalance:** Dataset pigeon-pea mất cân bằng (Sterility Mosaic 3652 vs Phytophthora 1691 → ratio ~2.16 ≥ 1.5). Theo rule 11, bật auto class-weighting nếu max/min ≥ 1.5 (đối với dataset thực tế chạy trên Kaggle, tính lại trên tập train).

8. **Rủi ro mismatch:** Block MDsConv split kênh giả định số kênh chia hết cho 2 — tất cả tầng trong Table 5 đều thỏa (64,128,256,512). An toàn.

9. **Metrics bắt buộc (rule 4):** Total params, Accuracy, F1 macro, Recall macro, classification_report per class, Confusion Matrix, Learning Curve (dpi=300, cell riêng). Áp dụng cho cả 2 scenario.
