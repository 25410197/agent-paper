# IMPLEMENT PLAN: TrIncNet

**Paper:** TrIncNet: a lightweight vision transformer network for identification of plant diseases
**Tác giả:** Pushkar Gole, Punam Bedi, Sudeep Marwaha, Md. Ashraful Haque, Chandan Kumar Deb
**Journal/Conference:** Frontiers in Plant Science, Vol. 14, 2023 (Published 27 July 2023)
**DOI:** 10.3389/fpls.2023.1221557
**Ngày phân tích:** 2026-06-16

---

## PHÂN LOẠI
**Nhóm:** A
**Lý do:** Bài báo đề xuất một kiến trúc mạng mới có tên cụ thể (TrIncNet) dựa trên Vision Transformer, mô tả chi tiết block (Trans-Inception block), cung cấp 2 bảng hyperparameters (Table 1, 2) cùng các thông số training (optimizer, epochs, batch size, loss, early stopping) và kết quả thực nghiệm rõ ràng trên 2 dataset. Đủ thông tin để phục dựng. Một số chi tiết về cách nối dây (wiring) bên trong Inception module và augmentation cụ thể chưa được nêu tường minh → liệt kê ở mục CẦN XÁC NHẬN.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | TrIncNet (Trans-Inception Network) | Tiêu đề / Abstract / Mục 3.2 |
| Loại | Vision Transformer (ViT) cải tiến — Hybrid ViT + Inception | Abstract / Mục 3.2 |
| Framework | TensorFlow 2.6.0 + Keras | Mục 4.3 (Experimental Setup) |
| Input size | 256 × 256 (×3 RGB) | Mục 4.2.1, Table 1, Table 2 |
| Pretrained backbone | Không — train from scratch | Toàn bộ Mục 3, 4 (không đề cập pretrained) |
| Patch size | 16 × 16 | Table 1, Table 2 |
| Số patch (M) | (256/16)² = 256 patches | Suy ra từ image size 256 và patch 16 (Mục 3.3) |
| Embedding dim (N) | 256 | Table 2 ("Size of Embedded Patch (N)") |
| Số Trans-Inception blocks | 2 | Table 2 |
| Số heads (Multi-Head Attention) | 12 | Mục 3.2.1, Table 2 |
| LayerNorm epsilon | 1e-6 | Table 2 |
| Activation | SoftMax (output layer), ReLU (hidden layers) | Table 2 |

### Cấu trúc tổng thể TrIncNet (theo mô tả Mục 3.2)
Luồng dữ liệu:
1. **Patch Embedding:** Chia ảnh 256×256 thành các patch 16×16 → 256 patches, mỗi patch embed thành vector kích thước N=256.
2. **Position Embedding:** (Chuẩn ViT — cộng positional embedding; xem mục CẦN XÁC NHẬN về class token).
3. **Stack 2× Trans-Inception block**, mỗi block gồm:
   - Layer Normalization (epsilon=1e-6)
   - Multi-Head Attention (12 heads)
   - Skip connection bao quanh block (residual) — "each Trans-Inception block is surrounded by a skip connection" (Mục 3.2)
   - Layer Normalization
   - **Inception module** (thay thế MLP module của ViT gốc)
4. **Head phân loại:** Output layer dùng SoftMax → số class theo dataset.

### Trans-Inception block — Inception module (Mục 3.2.3 + Appendix Table A1)
Inception module thực hiện song song 4 nhánh:
| Nhánh | Operation | Kernel | Ghi chú |
|------|-----------|--------|---------|
| 1 | Conv 1×1 | 1×1 | nhánh conv 1×1 trực tiếp |
| 2 | Conv 1×1 → Conv 3×3 | 1×1 rồi 3×3 | reduce rồi 3×3 (kiểu GoogLeNet) |
| 3 | Conv 1×1 → Conv 5×5 | 1×1 rồi 5×5 | reduce rồi 5×5 (kiểu GoogLeNet) |
| 4 | MaxPool 3×3 → Conv 1×1 | pool 3×3 rồi 1×1 | pooling branch |
| — | Concatenate 4 nhánh | — | nối các feature map theo channel |

- **Số filters dùng trong Inception module:** 96, 16, 64, 128, 32, 32 (Table 2). Đây là 6 giá trị filter cho 6 phép convolution (operation 1,2,3,5,6,7 trong Table A1; operation 4 là max-pooling không có weight).
- **Padding:** Same (Table 2)
- **Stride:** 1×1 (Table 2)
- **Mapping filter ↔ nhánh:** Xem mục CẦN XÁC NHẬN (paper không ghi rõ thứ tự gán 6 filter này cho từng phép conv cụ thể).

> Tham chiếu Appendix Table A1 (7 operations, operation 4 = maxpool):
> op1: 1×1 conv (F1), op2: 1×1 conv (F2, reduce cho 3×3), op3: 1×1 conv (F3, reduce cho 5×5), op4: maxpool 3×3, op5: 3×3 conv (F5, sau F2), op6: 5×5 conv (F6, sau F3), op7: 1×1 conv (F7, sau maxpool).
> Đây là cấu trúc Inception-v1 (GoogLeNet) chuẩn với 6 conv có weight.

### Multi-Head Attention (Mục 3.2.1)
- Self-attention chuẩn: Z = softmax(QK^T / √dk) · V (Equation 1)
- Multi-head: concat các head rồi nhân W_O (Equation 2)
- Số heads m = 12

### Layer Normalization (Mục 3.2.2)
- LayerNorm chuẩn (Ba et al., 2016), epsilon = 1e-6, có learnable scale (g) và shift (b)

### Hyperparameters
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | Adam | Mục 4.3 |
| Learning rate | KHÔNG nêu trong paper | — (xem CẦN XÁC NHẬN) |
| LR Scheduler | KHÔNG nêu trong paper | — |
| Warmup epochs | KHÔNG nêu | — |
| Batch size | 32 | Mục 4.3 |
| Epochs | 500 (với Early Stopping) | Mục 4.3 |
| Early Stopping | patience = 20 (theo validation accuracy) | Mục 4.3 |
| Loss function | Categorical Cross-Entropy | Mục 4.3 |
| Weight decay | KHÔNG nêu | — |
| Normalization (ảnh) | Chia pixel cho 255 → [0,1] (KHÔNG dùng mean/std ImageNet) | Mục 4.2.2 |
| Image resize | 256 × 256 (qua Keras ImageDataGenerator) | Mục 4.2.1 |
| Data split | 70% train / 15% val / 15% test (random) | Mục 4.2.4 |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | KHÔNG nêu góc cụ thể | Mục 4.2.3 (chỉ liệt kê ví dụ "rotation, flipping, etc.") |
| Flipping | KHÔNG nêu chi tiết | Mục 4.2.3 |
| (Offline augmentation) | Chỉ áp dụng cho Maize dataset, tăng lên 100,000 ảnh | Mục 4.2.3 |

> Lưu ý: Paper KHÔNG liệt kê đầy đủ và KHÔNG nêu tham số augmentation cụ thể. Augmentation chỉ áp dụng cho Maize dataset (offline). PlantVillage không augment.
> Theo CORE RULE #3, notebook bắt buộc chạy 2 kịch bản:
> - **Scenario 1 (Baseline):** chỉ Resize(256×256) + ToTensor + Normalize(/255). KHÔNG augmentation.
> - **Scenario 2 (Augmented):** dùng 4 thuật toán bắt buộc: Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness. Val/Test giống baseline.

### GitHub Source
**Link:** Không tìm thấy link code/GitHub cho TrIncNet trong bài báo.
- PlantVillage dataset: https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw (Mục Data Availability — đây là dataset, không phải code TrIncNet)
- Maize dataset: chỉ cung cấp qua liên hệ tác giả (không public)

### Kết quả trong paper (tham khảo)
| Dataset | Accuracy | F1 | Precision | Recall | Params |
|---------|---------|-----|-----------|--------|--------|
| PlantVillage (38 class, 54,503 ảnh) | 99.93% | 99.91% | 99.92% | 99.91% | 6.95M |
| Maize (4 class, 13,971 ảnh gốc → 100k sau aug) | 96.93% | 96.90% | 96.98% | 96.83% | 6.95M |

- PlantVillage: 14 plant species, 38 classes, 54,503 leaf images (Mục 4.1.2)
- Maize: 4 classes (Healthy, TLB, BLSB, MLB), 13,971 ảnh in-field (Mục 4.1.1)
- Val accuracy đỉnh: PlantVillage 99.95%, Maize 97.0%

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG có (hoặc không rõ ràng) trong bài báo. Vui lòng trả lời trước khi viết notebook.

1. **Learning rate:** Paper chỉ ghi dùng Adam optimizer, KHÔNG nêu giá trị learning rate. Bạn muốn dùng giá trị nào? (gợi ý: 1e-3 — mặc định Adam của Keras, hoặc 1e-4)

2. **Framework triển khai:** Paper gốc dùng TensorFlow/Keras (TF 2.6.0). Tuy nhiên CORE RULE mặc định ưu tiên PyTorch nếu người dùng không yêu cầu. Bạn muốn notebook viết bằng **TensorFlow/Keras (đúng như paper)** hay **PyTorch (mặc định của hệ thống)?**

3. **Dataset chạy thực tế trên Kaggle:** Maize dataset không public (chỉ qua tác giả). Bạn muốn reproduce trên **PlantVillage (38 class)** hay một dataset cây trồng khác bạn có sẵn trên Kaggle? Cho biết tên/đường dẫn dataset Kaggle để cấu hình `ImageFolder`.

4. **Cách nối filter trong Inception module:** Paper cho 6 filter (96, 16, 64, 128, 32, 32) nhưng KHÔNG ghi rõ filter nào gán cho nhánh nào. Theo cấu trúc Inception-v1 chuẩn (Table A1) tôi đề xuất mapping:
   - Nhánh 1×1 trực tiếp (op1) = 96
   - Reduce 1×1 cho 3×3 (op2) = 16 → Conv 3×3 (op5) = 64
   - Reduce 1×1 cho 5×5 (op3) = 128 → Conv 5×5 (op6) = 32
   - Conv 1×1 sau maxpool (op7) = 32

   Bạn xác nhận mapping này, hay muốn điều chỉnh? (Đây là điểm rủi ro mismatch param count — cần khớp ~6.95M params).

5. **Class token & Positional embedding:** Paper mô tả ViT chuẩn nhưng không nói rõ có dùng `[CLS]` token hay Global Average Pooling ở head, và loại positional embedding (learnable hay sinusoidal). Bạn muốn dùng theo ViT gốc (Dosovitskiy 2021: learnable position embedding + CLS token)? (gợi ý: dùng learnable position embedding; với head có thể dùng GAP nếu không có CLS token — cần thống nhất).

6. **Cách reshape sau Attention để đưa vào Inception (2D conv):** Inception module dùng conv 2D nên cần reshape chuỗi token (M=256 patch, N=256 dim) về dạng không gian 2D (Appendix nêu reshape về √N×√N×M = 16×16×256). Bạn xác nhận theo Appendix reshape thành 16×16×256 trước khi đưa vào Inception, rồi flatten lại sau Inception?

7. **Số epochs cho môi trường Kaggle:** Paper train 500 epochs (early stopping patience=20) trên DGX V100. Trên Kaggle GPU thời gian giới hạn (~9–12h). Bạn muốn giữ nguyên 500 epochs + early stopping, hay giảm xuống (ví dụ 30–50 epochs) để fit thời gian Kaggle? (gợi ý: giữ early stopping patience=20, đặt max epochs hợp lý theo dataset).

---

## GHI CHÚ IMPLEMENTATION

1. **Normalization KHÁC ImageNet:** Paper chỉ chia pixel /255 (range [0,1]), KHÔNG dùng mean/std ImageNet. Nếu viết PyTorch, dùng `transforms.ToTensor()` (đã tự chia /255) và KHÔNG thêm `Normalize(mean_imagenet, std_imagenet)`. Đây là điểm dễ sai nếu copy template ImageNet.

2. **Train from scratch:** Không có pretrained backbone — model khởi tạo ngẫu nhiên. Cần đủ epoch để hội tụ; chú ý learning curve.

3. **Param count mục tiêu ~6.95M:** Đây là tiêu chí kiểm tra tính đúng của kiến trúc. Sau khi build, in `torchinfo.summary()` / `model.summary()` và so sánh với 6.95M. Lệch nhiều → kiểm tra lại mapping filter Inception (câu hỏi #4) và số block (=2), heads (=12), embed dim (=256).

4. **Reshape token ↔ feature map:** Đây là phần phức tạp nhất. Sau Multi-Head Attention, chuỗi (batch, 256 tokens, 256 dim) phải reshape thành (batch, 16, 16, 256) để đưa vào Inception conv2D, rồi sau Inception reshape ngược lại để skip-connection cộng với input của block. Cần đảm bảo channel/shape khớp cho residual add.

5. **Skip connection:** "each Trans-Inception block is surrounded by a skip connection" — residual bao quanh toàn bộ block (input block + output block). Cần đảm bảo output Inception có cùng số channel/dim với input để cộng được (Inception concat output phải = N=256; với mapping ở câu #4: 96+64+32+32 = 224 ≠ 256 → đây là rủi ro mismatch, cần xác nhận lại mapping filter hoặc dùng projection. ĐÂY LÀ ĐIỂM CẦN LÀM RÕ KỸ).

6. **Dual experiment (CORE RULE #3):** Bắt buộc 2 scenario. Paper chỉ augment offline cho Maize; với notebook ta dùng online augmentation theo 4 thuật toán bắt buộc cho Scenario 2.

7. **Class imbalance (CORE RULE #11):** PlantVillage mất cân bằng giữa 38 class → cần auto-detect và dùng class_weights nếu max/min ≥ 1.5.

8. **Metrics bắt buộc (CORE RULE #4):** Total Params, Accuracy, F1-macro, Recall-macro, và `classification_report` per-class.
