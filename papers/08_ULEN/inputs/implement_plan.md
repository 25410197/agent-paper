# IMPLEMENT PLAN: ULEN (Ultra-Lightweight Efficient Network)

**Paper:** An ultra-lightweight efficient network for image-based plant disease and pest infection detection
**Tác giả:** Beibei Wang, Chenxiao Zhang, Yanyan Li, Chunxia Cao, Daye Huang, Yan Gong
**Journal/Conference:** Precision Agriculture (2023) 24:1836–1861
**DOI:** 10.1007/s11119-023-10020-0
**Ngày phân tích:** 2026-06-18

---

## PHÂN LOẠI
**Nhóm:** C (có model rõ ràng về topology + đầy đủ hyperparameters training, NHƯNG thiếu một số thông số kiến trúc cốt lõi: số filters/channels của từng block, cấu hình chính xác của conv block đầu, số node FC head).
**Lý do:** Paper đề xuất kiến trúc mới có tên (ULEN), nêu rõ framework, optimizer, LR schedule, batch size, epochs, loss, input size, data split — đủ để train. Tuy nhiên các hình (Fig. 5–8) chỉ vẽ topology, KHÔNG ghi số kênh/filters cho từng block, và text cũng không liệt kê. Tổng params = 111,758 là ràng buộc duy nhất để dò ngược số kênh. Vì vậy cần xác nhận với người dùng trước khi build chính xác.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | ULEN | Abstract / Tiêu đề |
| Loại | CNN lightweight (depth-wise separable + residual + channel shuffle + SPP) | Mục "ULEN architecture" |
| Framework | PyTorch | Mục "Model training" ("implemented in PyTorch library") |
| Input size | 256 × 256 × 3 | Mục "The Cassava dataset" ("resized into 256 x 256 pixels") |
| Pretrained backbone | Không — train from scratch | Toàn bài (không đề cập transfer learning cho ULEN) |
| Tổng số params (ràng buộc) | 111,758 | Table 5 (visually confirmed) |
| Memory / FLOPs | 6.09 Mb / 21.36 MFlops | Table 5 |

### Cấu trúc Pipeline tổng thể (theo thứ tự — Fig. 5, đã visually confirmed)
| # | Block | Kernel | Filters | Stride | Activation | Ghi chú |
|---|-------|--------|---------|--------|-----------|---------|
| 1 | Input | — | — | — | — | Ảnh lá 256×256×3 |
| 2 | Conv block | 3×3 | **CẦN XÁC NHẬN** | **CẦN XÁC NHẬN** | (implied, xem note) | Conv2d + BatchNorm + Pooling (Fig. 5 / text). Là conv chuẩn (không depth-wise) — "except for the first and last conv layer" |
| 3 | MaxPool ×2 | **CẦN XÁC NHẬN** (gợi ý 2×2) | — | **CẦN XÁC NHẬN** (gợi ý 2) | — | Hai lớp MaxPool2d liên tiếp giảm spatial |
| 4 | RDWConv ×3 | 3×3 (DWConv) | **CẦN XÁC NHẬN** | 1 (Fig. 7a) | ReLU | Ba block RDWConv stride=1 (Fig. 7a) |
| 5 | Conv block | 1×1 | **CẦN XÁC NHẬN** | 1 | (BN, không nêu ReLU) | 1×1 Conv + BatchNorm. Là conv chuẩn (last conv layer) |
| 6 | SPP | bins {4×4, 2×2, 1×1} | — | — | — | AdaptiveAvgPool 3 mức, flatten, concat → 21×C |
| 7 | Linear classifier | — | num_classes | — | — | FC; PlantVillage=38, Cassava=5 |

> Note: Fig. 5 KHÔNG vẽ stride/số filters. Số filters của mọi block và stride của conv/maxpool đầu KHÔNG có trong paper.

### RDWConv block (Stride=1) — Fig. 7a (visually confirmed)
```
Input
 ├─ skip path ────────────────────────────────────┐
 └─ 3×3 DWConv(stride=1) → BN → ReLU → 1×1 Conv → BN ┤
                                                    └→ Concatenate → Channel Shuffle → Output
```
- Hai nhánh được **Concatenate** (không phải cộng element-wise) → có khả năng số kênh tăng/nhân đôi sau concat. Channel Shuffle theo cơ chế ShuffleNet.

### RDWConv block (Stride=2) — Fig. 7b (visually confirmed, dùng khi downsample)
```
Input
 ├─ 1×1 Conv → BN → ReLU → 3×3 DWConv(stride=2) → BN → 1×1 Conv → BN → ReLU ┐
 └─ 3×3 DWConv(stride=2) → BN → ReLU → 1×1 Conv → BN ───────────────────────┤
                                                          Concatenate → Channel Shuffle → Output
```
> Lưu ý: Trong pipeline Fig. 5, ba RDWConv được mô tả là stride=1 (Fig. 7a). Biến thể stride=2 (Fig. 7b) được paper đề xuất như tùy chọn downsample thay MaxPool, nhưng Fig. 5 dùng MaxPool×2 cho downsample. → Mặc định build 3× RDWConv stride=1.

### SPP — Fig. 8a (visually confirmed)
```
Feature map ┬→ AdaptiveAvgPool(4×4) → flatten ┐
            ├→ AdaptiveAvgPool(2×2) → flatten ─┼→ Concatenate → 1D vector (21×C) → Linear
            └→ AdaptiveAvgPool(1×1) → flatten ┘
```
Output dim = (16 + 4 + 1) × C = 21 × C.

### Hyperparameters
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | Adam | Mục "Model training" |
| Learning rate | 0.001 cho 5 epochs đầu, sau đó 0.0001 | Mục "Model training" |
| LR Scheduler | Step thủ công: epoch 1–5 = 1e-3; epoch 6–20 = 1e-4 | Mục "Model training" |
| Warmup epochs | Không có (5 epoch đầu là LR cao, không phải warmup) | — |
| Batch size | 16 | Mục "Model training" |
| Epochs | 20 | Mục "Model training" |
| Loss function | Cross-Entropy Loss | Mục "Model training", Eq. (1) |
| Weight decay | Không đề cập | — |
| Normalization (mean/std) | Không đề cập | — |
| Class weighting | KHÔNG dùng weighting; chỉ dùng **stratified split per sub-category** để mọi class có mặt ở cả 3 tập | Mục "Model training" |

### Data split & xử lý ảnh
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Split | 70% train / 20% val / 10% test, stratified per class | Mục "The Cassava dataset" + "Model training" |
| Resize | 256 × 256 | Mục "The Cassava dataset" |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| (Không có) | Paper KHÔNG mô tả augmentation nào | — |

### GitHub Source
**Link:** Không tìm thấy (không có mục Code/Data Availability trong nội dung trích xuất).

### Kết quả trong paper (tham khảo)
| Dataset | Precision | Recall | F1 | Params |
|---------|-----------|--------|-----|--------|
| PlantVillage (38 lớp) | 0.9813 | 0.9749 | 0.9776 | 111,758 |
| Cassava (5 lớp) | 0.5497 | 0.5322 | 0.5337 | 111,758 |

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG có trong bài báo (hình chỉ vẽ topology, không ghi số kênh/stride). Vui lòng trả lời trước khi viết notebook.

1. **Số filters/channels của từng block.** Paper chỉ cho tổng params = 111,758, KHÔNG liệt kê số kênh cho: Conv block 3×3 đầu, 3× RDWConv, Conv block 1×1 cuối. Bạn muốn chỉ định cụ thể số kênh từng block, hay để tôi dò ngược một cấu hình sao cho tổng params ≈ 111,758 (sẽ ghi rõ là "ước lượng, không phải từ paper")? (ví dụ tham khảo nội bộ paper dùng: input 16ch → output 16ch cho ví dụ DWConv)

2. **Cấu hình Conv block đầu (3×3).** Stride = ? Padding = ? Số output filters = ? Loại "pooling" trong conv block đầu là MaxPool hay AvgPool, kernel/stride bao nhiêu? Paper chỉ nói "a convolutional layer ... a pooling layer ... a batch normalization layer".

3. **Activation function của Conv block đầu và Conv block 1×1 cuối.** Fig. 7 chỉ ghi ReLU bên trong RDWConv. Conv block đầu/cuối không ghi rõ activation. Dùng ReLU cho cả hai? (gợi ý: ReLU)

4. **MaxPool ×2 cấu hình.** Kernel size và stride của 2 lớp MaxPool sau conv block đầu? (gợi ý: kernel 2×2, stride 2 cho mỗi lớp → giảm 4 lần spatial)

5. **Channel Shuffle — số groups.** Paper nói dùng channel shuffle theo cơ chế ShuffleNet nhưng không ghi số groups. Dùng groups = 2 (vì block tách 2 nhánh concat)? (gợi ý: 2)

6. **Concatenate trong RDWConv làm thay đổi số kênh thế nào.** Concat 2 nhánh → số kênh output = tổng kênh 2 nhánh? Cần xác nhận để dò params cho khớp 111,758.

7. **Linear classifier — số lớp FC & dropout.** Một FC duy nhất (21×C → num_classes) hay có hidden layer? Có dropout không? (gợi ý: 1 FC, không dropout)

8. **Normalization mean/std.** Paper không nêu. Dùng giá trị nào? (gợi ý: tính mean/std từ dataset, hoặc ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225])

9. **Dataset chạy trên Kaggle.** Notebook nên train trên dataset nào? PlantVillage (38 lớp) hay Cassava (5 lớp), hay cả hai? Đường dẫn dataset Kaggle cụ thể?

---

## GHI CHÚ IMPLEMENTATION

- **Ràng buộc params 111,758 là chìa khóa.** Vì số kênh không có trong paper, sau khi người dùng chốt cấu hình, BẮT BUỘC in `torchinfo.summary()` và so sánh tổng params với 111,758. Nếu lệch nhiều → cấu hình kênh sai, cần điều chỉnh và ghi rõ "kênh là ước lượng, không phải từ paper".
- **Conv đầu & cuối là conv chuẩn (full conv), các block giữa là depth-wise.** Paper: "except for the first and last convolutional layer, the depth-wise convolution layer in ULEN is introduced".
- **SPP output luôn cố định = 21×C** bất kể spatial size đầu vào (vì dùng AdaptiveAvgPool) → linear layer in_features = 21 × (số kênh sau conv 1×1 cuối). Điều này khiến model robust với input size, nhưng vẫn resize 256×256 theo paper.
- **RDWConv dùng Concatenate + Channel Shuffle, KHÔNG dùng phép cộng residual element-wise** (khác ResNet truyền thống) — theo Fig. 7. Đặt tên "residual" nhưng implement bằng concat 2 nhánh rồi channel shuffle, giống ShuffleNet unit.
- **LR schedule là step thủ công 2 mức**, không phải scheduler chuẩn của PyTorch. Implement bằng cách set `optimizer.param_groups[0]['lr']` ở đầu epoch 6, hoặc dùng `LambdaLR`/`MultiStepLR(milestones=[5], gamma=0.1)`.
- **Class imbalance:** Cả 2 dataset đều imbalance nặng (PlantVillage: cedar apple rust 0.5% vs tomato yellow leaf curl 9.86%; Cassava: mosaic 61.49% vs bacterial blight ~5%). Paper KHÔNG dùng class weighting, chỉ stratified split. Tuy nhiên theo core rule #11 của workspace, notebook PHẢI auto-detect imbalance (max/min ≥ 1.5) và tự bật class_weights → sẽ là điểm KHÁC paper, cần ghi chú rõ trong notebook là deviation từ workspace rule (không phải từ paper).
- **Dual experiment (core rule #3):** Scenario 1 baseline = Resize(256) + ToTensor + Normalize. Scenario 2 = thêm Rotation, Zoom/Scale, HorizontalFlip, ColorJitter. Lưu ý paper gốc KHÔNG dùng augmentation — Scenario 2 là yêu cầu workspace, không từ paper.
- **Metrics theo paper:** Precision, Recall, Macro-F1, Confusion Matrix. Paper dùng Macro-F1 → khớp với core rule #4 (F1 Macro, Recall Macro + classification_report per class).
- **Best checkpoint theo val_accuracy** (core rule #10): `best_model_s1.pth` / `best_model_s2.pth`.
