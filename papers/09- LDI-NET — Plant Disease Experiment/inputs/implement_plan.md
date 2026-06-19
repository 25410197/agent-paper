# IMPLEMENT PLAN: LDI-NET

**Paper:** A novel plant type, leaf disease and severity identification framework using CNN and transformer with multilabel method
**Tác giả:** Bin Yang, Mingwei Li, Fei Li, Yongbo Wang, Qiaokang Liang, Ruiyuan Zhao, Caihong Li, Jianwu Wang
**Journal/Conference:** Scientific Reports (Nature), 2024
**DOI:** 10.1038/s41598-024-62452-x
**Ngày phân tích:** 2026-06-19

---

## PHÂN LOẠI
**Nhóm:** C (có model + nhiều hyperparameter rõ ràng, NHƯNG thiếu một số thông số cốt lõi về kích thước nội bộ kiến trúc: embed_dim, số head MSA, số transformer block N, MLP hidden dim, dropout rate, số kênh Conv2d). Có thể code được khung kiến trúc nhưng cần xác nhận các giá trị bên trong.
**Lý do:** Paper mô tả đầy đủ luồng dữ liệu (3 module, số block encoder/decoder = 2+2, patch size, optimizer, lr, epochs, batch, dataset split), nhưng KHÔNG nêu các siêu tham số chiều bên trong (dimension/heads/depth/dropout) — đây là các thông số bắt buộc để khởi tạo các lớp Transformer/Linear.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Model Architecture (Kiến trúc)

Tổng quan: LDI-NET là mô hình **multi-label** single-branch nhận diện đồng thời **plant type + leaf disease + severity**. Gồm 3 module nối tiếp:

| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | LDI-NET | Tiêu đề / Abstract |
| Loại | Hybrid CNN + Transformer (encoder-decoder, multi-label) | Abstract, Method |
| Framework | PyTorch | Implementation details (dòng 197) |
| Input size | 224 x 224 x 3 | Data augmentation (dòng 185) |
| Patch size | 16 x 16 (best, FTM ps=16) — có cấu hình 32 x 32 | Feature tokenizer (dòng 86-87); Table 9 |
| Pretrained backbone | Không đề cập (model tự thiết kế, không nêu pretrained weights) | — |
| Số label đầu ra | 34 query embeddings (12 plant + 19 disease + 3 severity) | Dòng 137-140, Fig 1 |

#### Module 1 — Feature Tokenizer Module
Công thức: `Y = Conv2d( TB...TB( Conv2d(I) ) )` (Eq. 1)
Luồng (Fig 1, Fig 2 — đã xác nhận từ hình):
1. **Conv2d** đầu vào: sliding convolution window kích thước = patch size (16x16), chia ảnh thành patches.
2. **Flatten** các patch.
3. **Position Embedding**: cộng vào các patch đã flatten (giữ thông tin vị trí không gian).
4. **Transformer Block x N** (lặp N lần, mỗi block có skip connection):
   - Norm -> Multi-head Self-Attention -> residual add -> Norm -> MLP -> residual add
5. **Conv2d** cuối: bắt local information.
6. Output: "compact spatial feature tokens".

#### Module 2 — Token Encoder Module
Gồm **2 Token Encoder Block** (xác nhận: best config = 2 encoder, Table 10).
Cấu trúc 1 block (Fig 3a — xác nhận từ hình):
- Input -> Norm -> Multi-head Self-Attention -> residual add (+) -> Norm -> MLP -> residual add (+) -> Output
- Đầu ra: "context-rich tokens".
- **Fused token** = context-rich token + compact spatial token (residual addition, circle-plus, Fig 1).

#### Module 3 — Multi-Label Decoder Module
Gồm **2 Multi-Label Decoder Block** (xác nhận: best config = 2 decoder, Table 10).
Cấu trúc 1 block (Fig 3b — xác nhận từ hình):
- Input (adaptive feature embeddings = query, 1 per label) -> Norm -> Multi-head **Self-Attention** (Q,K,V = input) -> residual add (+) -> Norm -> Multi-head **Cross-Attention** (Q = self-attn output; K,V = fused tokens) -> residual add (+) -> Norm -> MLP -> residual add (+) -> Output
- Hai input: (1) discriminative adaptive feature embeddings (query, mỗi label 1 embedding); (2) fused tokens (làm K, V).
- Decoder học correlation giữa label và feature để predict.

#### Chi tiết khối phụ
- **MSA (Fig 3c):** Linear projection -> Feature Split thành n heads -> Scaled Dot-Product Attention (MatMul Q,K -> Scale 1/sqrt(dk) -> Softmax -> MatMul V) -> Concat heads -> Linear (WO). Công thức Eq. 5-7.
- **MLP (Fig 3d):** Linear -> GELU -> Dropout -> Linear -> Dropout.

### Hyperparameters
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | AdamW | Implementation details (dòng 200) |
| Learning rate (initial) | 6.25e-6 | Implementation details (dòng 201) |
| LR Scheduler | StepLR (step_size=4, gamma=0.5) | Implementation details (dòng 201) |
| Warmup epochs | Không đề cập | — |
| Batch size | 16 | Implementation details (dòng 199) |
| Epochs | 60 | Implementation details (dòng 198) |
| Loss function | Không nêu tên cụ thể (multi-label, 3 nhánh: plant + disease + severity) — paper chỉ nói "multi-label method" | Không có |
| Weight decay | Không đề cập (AdamW có default; paper không nêu giá trị) | — |
| Normalization (mean/std) | Không đề cập | — |
| Số lần chạy | 5 lần, lấy trung bình | Dòng 202 |
| Hardware | Intel i9-12900K, NVIDIA 3090ti, Ubuntu | Dòng 195-196 |

### Dataset
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Tên dataset | AI Challenger 2018 (plant disease) | Dòng 126 |
| Tổng gốc | 61 categories; 31,718 train / 4,540 val / 4,514 test | Dòng 126-127 |
| Sau lọc | 56 categories (bỏ 5 category quá ít ảnh); dùng 30,210 train + 4,324 val | Dòng 128-129 |
| Split | Training set chia 80% train / 20% val; **validation set gốc dùng làm test** | Dòng 129-130 |
| Số ảnh thực dùng | Train: 24,159 / Val: 6,054 / Test: 4,324 | Dòng 133-135 |
| Multi-label | 34 label = 12 plant + 19 disease + 3 severity | Dòng 137-140 |
| Plant classes (12) | Apple, Cherry, Citrus, Corn, Grape, Peach, Pepper, Potato, Pumpkin, Soybean, Strawberry, Tomato | Table 1 |
| Disease classes (19) | Black Blight Fungus, Black Measles Fungus, Black Rot Fungus, Black Spot, Cedar Rust, CZTD, Early Blight, Greening June, Late Blight, Leaf Blight, Leaf Mold, Leaf Spot, Powdery Mildew, Puccinia Polysora, Scab, Septoria Leaf Spot Fungus, Spider Mite Damage, Target Spot Bacteria, YLCV Virus | Table 2 |
| Severity classes (3) | Healthy, General, Serious | Table 3 |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Random Resized Crop | crop + resize về 224x224 | Dòng 185 |
| Random Horizontal Flip | (mặc định) | Dòng 186 |

> Lưu ý: paper CHỈ áp dụng augmentation cho training set. Không nêu thêm augmentation nào khác (không rotation, không colorjitter trong paper). Theo rule DUAL EXPERIMENT của workspace, Scenario 2 vẫn dùng 4 augmentation chuẩn (Rotation, Zoom/Scale, HFlip, ColorJitter).

### GitHub Source
**Link:** Không tìm thấy trong nội dung đã trích xuất.

### Kết quả trong paper (tham khảo)
| Metric | Giá trị LDI-NET (ps=16, enc=2, dec=2) | Nguồn |
|---------|---------|-------|
| Plant accuracy | 99.42% | Table 4 / Conclusion |
| Disease accuracy | 98.68% | Table 4 |
| Severity accuracy | 88.55% | Table 4 / Conclusion |
| PDSA (plant-disease-severity) | 87.40% | Table 4 / Table 9 / Table 10 |
| Precision / Recall / F1 | 95.28 / 95.29 / 95.28 | Table 9 |

---

## THÔNG SỐ ĐÃ XÁC NHẬN BỞI NGƯỜI DÙNG (2026-06-19)

| Thông số | Giá trị xác nhận |
|---------|-----------------|
| embed_dim (d_model) | **384** |
| n_heads | **6** (384 ÷ 6 = 64 ✓) |
| N (FTM transformer blocks) | **2** |
| MLP hidden dim | **4 × 384 = 1536** (mlp_ratio=4) |
| Dropout rate | **0.1** |
| Conv2d out_channels | **384** (= embed_dim) |
| Loss function | **CrossEntropyLoss** (single-label, dataset người dùng) |
| Normalization | **ImageNet** mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |
| Weight decay (AdamW) | **0.01** |
| Dataset Kaggle | `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` (đã chia train/val/test) |
| Class weighting | Auto-detect (rule workspace: áp dụng nếu max/min ≥ 1.5) |

> **Lưu ý quan trọng về kiến trúc:** Dataset người dùng là **single-label** (không phải AI Challenger 2018 multi-label). Do đó:
> - Module 1 (Feature Tokenizer) và Module 2 (Token Encoder) giữ nguyên.
> - Module 3 (Multi-Label Decoder) → thay bằng **global average pooling + Linear classifier** (num_classes tự detect từ dataset).
> - Loss: 1 `CrossEntropyLoss` thay vì 3 nhánh.

---

## Missing / Ambiguous Parameters

> ⚠️ Các thông số sau KHÔNG có trong bài báo (cả text lẫn hình). KHÔNG được tự suy diễn.

1. **embed_dim / token dimension (d_model)** của transformer — không nêu.
2. **Số head (n_heads)** trong Multi-head Self-Attention và Cross-Attention — không nêu.
3. **Số transformer block N** trong feature tokenizer (Fig 2 ghi "x N" nhưng không cho giá trị N) — không nêu.
4. **MLP hidden dimension** (expansion ratio) trong các MLP block — không nêu.
5. **Dropout rate** (trong MLP và attention) — không nêu giá trị.
6. **Số kênh (out_channels) của 2 Conv2d** trong feature tokenizer — không nêu.
7. **Loss function cụ thể** — paper chỉ nói "multi-label method", không nêu CrossEntropy/BCE hay cách kết hợp 3 nhánh loss (tổng có trọng số?).
8. **Normalization mean/std** cho preprocessing — không nêu (ImageNet hay custom?).
9. **Weight decay** của AdamW — không nêu giá trị.
10. **Cách tạo adaptive feature embeddings (query)** trong decoder — learnable parameters khởi tạo thế nào, kích thước cụ thể — không mô tả chi tiết.
11. **Class weighting** cho imbalance — paper không đề cập (dataset rất mất cân bằng, ví dụ Tomato 7932 vs Strawberry 817; Target Spot Bacteria 53 vs YLCV 3107).

> Lưu ý quan trọng về dataset cho việc tái hiện trên Kaggle: paper dùng **AI Challenger 2018** (multi-label: plant+disease+severity). Nếu Kaggle dataset thay thế là **single-label** (ví dụ PlantVillage), thì cấu trúc multi-label 3-nhánh + decoder query của LDI-NET sẽ cần điều chỉnh — cần xác nhận dataset đích.

---

## Vietnamese Questions

> ⚠️ Vui lòng trả lời các câu hỏi sau trước khi tôi viết notebook.

1. **Kích thước embedding (embed_dim / d_model)** của transformer trong LDI-NET không có trong paper. Bạn muốn dùng giá trị nào? (gợi ý chuẩn ViT: 192, 384 hoặc 768)
2. **Số head** trong Multi-head Self-Attention/Cross-Attention không có trong paper. Dùng bao nhiêu head? (gợi ý: 6 hoặc 8 — phải chia hết cho embed_dim)
3. **Số transformer block N** trong feature tokenizer module (Fig 2 ghi "x N" nhưng không cho số). Bạn muốn N bằng mấy? (gợi ý: 2 hoặc 4)
4. **MLP hidden dimension** (tỉ lệ mở rộng) trong các MLP không được nêu. Dùng expansion ratio bao nhiêu? (gợi ý: 4x embed_dim như chuẩn Transformer)
5. **Dropout rate** không có giá trị cụ thể. Dùng bao nhiêu? (gợi ý: 0.1)
6. **Số kênh out_channels của 2 lớp Conv2d** trong feature tokenizer không nêu. Bạn muốn dùng giá trị nào? (gợi ý: bằng embed_dim)
7. **Loss function**: paper chỉ nói "multi-label" với 3 nhánh (plant/disease/severity). Bạn muốn dùng tổng 3 CrossEntropy (mỗi nhánh 1 loss) hay BCEWithLogits? Và có dùng trọng số giữa 3 nhánh không?
8. **Normalization mean/std**: paper không nêu. Dùng ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] hay tự tính trên dataset?
9. **Weight decay của AdamW** không nêu. Dùng giá trị nào? (gợi ý: 0.01 hoặc 0.05)
10. **Dataset đích trên Kaggle**: bạn sẽ dùng đúng AI Challenger 2018 (multi-label plant+disease+severity), hay một dataset single-label khác (ví dụ PlantVillage)? Điều này quyết định có giữ kiến trúc multi-label decoder hay không.
11. **Class imbalance**: dataset rất mất cân bằng. Có muốn áp dụng class weighting (theo rule workspace tự động khi max/min ≥ 1.5) không?

---

## GHI CHÚ IMPLEMENTATION

- **Framework: PyTorch** (paper nêu rõ).
- **Best config đã xác nhận**: patch size = 16, **2 token encoder blocks + 2 multi-label decoder blocks** (Table 9, Table 10) — dùng cấu hình này.
- **Kiến trúc decoder kiểu DETR-like**: dùng learnable query embeddings (1 per label, 34 query), self-attention trên query rồi cross-attention với fused tokens. Đây là điểm phức tạp nhất, dễ mismatch nếu thiếu embed_dim/heads — phải chờ xác nhận câu hỏi 1-2.
- **Fused token = compact spatial token + context-rich token** (residual add, Fig 1) — cần đảm bảo hai tensor cùng shape (cùng số token và embed_dim).
- **Multi-label đặc thù**: output không phải 1 head 56-class mà là 3 nhóm dự đoán riêng (plant 12 / disease 19 / severity 3) qua các query embedding. Nếu Kaggle dataset là single-label, phải thiết kế lại head.
- **Metric paper dùng**: ngoài Accuracy còn có PDSA (đồng thời đúng cả plant+disease+severity). Theo rule workspace vẫn bắt buộc in Accuracy, F1-macro, Recall-macro, classification_report per class.
- **Rủi ro mismatch chính**: embed_dim, n_heads, N, MLP hidden, conv channels — tất cả đều thiếu trong paper → KHÔNG code khi chưa có câu trả lời cho phần Vietnamese Questions 1-6.
