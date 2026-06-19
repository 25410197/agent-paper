# IMPLEMENT PLAN: PMVT (Plant-based MobileViT)

**Paper:** PMVT: a lightweight vision transformer for plant disease identification on mobile devices
**Tác giả:** Guoqiang Li, Yuchao Wang, Qing Zhao, Peiyan Yuan, Baofang Chang
**Journal/Conference:** Frontiers in Plant Science, Vol. 14:1256773, 2023
**DOI:** 10.3389/fpls.2023.1256773
**Ngày phân tích:** 2026-06-17

---

## PHÂN LOẠI
**Nhóm:** A (có một số điểm thuộc C — xem mục CẦN XÁC NHẬN)
**Lý do:** Paper đề xuất kiến trúc cụ thể (PMVT) dựa trên MobileViT, có bảng hyperparameters đầy đủ (epochs, batch size, optimizer, LR, loss), có sơ đồ kiến trúc chi tiết (Fig 2, 3, 4A/B/C) và kết quả thực nghiệm trên 3 dataset. TUY NHIÊN, paper KHÔNG ghi giá trị channel dimension / transformer dim cụ thể cho từng stage — chỉ nói "theo MobileViT XXS/XS/S" (Mehta & Rastegari, 2021). Do đó cần xác nhận lựa chọn cấu hình (XXS/XS/S) và nguồn giá trị channel/dim.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | PMVT (plant-based MobileViT) | Abstract / Mục 2.2 |
| Loại | Hybrid CNN + ViT (lightweight MobileViT-based) | Abstract / Mục 1 |
| Framework | PyTorch 1.9.0, Python 3.8 | Mục 2.6 |
| Input size | 224×224×3 | Mục 2.4 |
| Pretrained backbone | KHÔNG đề cập pretrained — train from scratch (không nêu nguồn weights) | Toàn bài (không nhắc ImageNet pretrain) |
| Số biến thể | 3 biến thể: XXS, XS, S (theo MobileViT) | Mục 2.2.1 |
| Số classes | Wheat=7, Coffee=3, Rice=2 | Table 1 |

### Cấu trúc tổng thể (Figure 2 — đã transcribe từ hình)
Mô hình gồm 5 layer (stage). Trước mỗi block là downsample bằng Conv 3×3.

| Stage | Cấu trúc | Downsample (↓2) | Ghi chú L (số block lặp) |
|-------|----------|-----------------|--------------------------|
| Stem | Conv2d 3×3 (downsampling conv) | có | — |
| Layer 1 | Inverted Residual Block ×1 | không | 1 block |
| Layer 2 | Inverted Residual Block (×3 theo hình) | có (đầu stage) | L=2 (MobileViT notation) |
| Layer 3 | Inverted Residual Block → Mobile ViT Block | có (đầu stage) | L=4 (số transformer block trong MobileViT block) |
| Layer 4 | Inverted Residual Block → Mobile ViT Block | có (đầu stage) | L=4 |
| Layer 5 | Inverted Residual Block → Mobile ViT Block | có (đầu stage) | L=3 |
| Head | Conv2d 1×1 (mở rộng channel ×4) → AvgPool → FC (Classifier) | — | "expand channel ×4 bằng 1×1 conv ở layer cuối" (Mục 2.2.1) |

> LƯU Ý quan trọng: Trong original MobileViT, "L" là số transformer block trong MobileViT block. Hình ghi L=2/4/4/3. So sánh với MobileViT gốc (L=2,4,3 cho 3 MobileViT stage) — paper PMVT ghi L=4 cho cả layer 3 và layer 4 (theo transcription hình). Cần xác nhận giá trị L này (xem CẦN XÁC NHẬN).

### Inverted Residual Block (Figure 3 — đã transcribe)
| Bước | Thao tác | Giá trị | Nguồn |
|------|----------|---------|-------|
| 1 | Conv2d 1×1 (mở rộng channel) | expand ratio = 2× ("generally by two times") | Mục 2.2.2 |
| 2 | Depthwise Separable Conv với DW kernel = **7×7** (thay cho 3×3 của MobileViT gốc) | DWConv 7×7 | Mục 2.2.2 + Fig 3 |
| 3 | Cat (gộp các nhánh depthwise) | — | Fig 3 |
| 4 | Conv2d 1×1 (khôi phục channel) | — | Mục 2.2.2 |
| — | Residual/skip connection | có (cấu trúc inverted residual chuẩn) | Mục 2.2.2 |

> Cấu trúc: `1×1 Conv (expand ×2) → 7×7 Depthwise Conv → 1×1 Conv (project)` + skip. Padding cho DW 7×7 = 3 để giữ spatial size (không ghi trong paper nhưng là hệ quả toán học của "giữ resolution", không phải suy diễn hyperparameter). Stride: stage có downsample dùng stride=2 ở DW conv (theo MobileViT gốc) — CẦN XÁC NHẬN vị trí stride.

### MobileViT Block (Figure 4A — đã transcribe)
Luồng: Input [C,H,W] → (local rep) Conv 1×1 → Conv 3×3 → Fold (chia patch) → ViT Encoder ×L → Unfold → Conv 1×1 → **CBAM Block** → (skip) Cat với feature map gốc → Conv 3×3 → Output [C,H,W].

| Thành phần | Giá trị | Nguồn |
|-----------|---------|-------|
| Local representation | Conv 1×1 + Conv 3×3 | Mục 2.2.3 + Fig 4A |
| Patch size P | KHÔNG ghi giá trị số — chỉ ký hiệu "P" (MobileViT gốc dùng 2×2) | Mục 2.2.4 (THIẾU giá trị) |
| Transformer encoder lặp L lần | L=4/4/3 (theo Fig 2) | Fig 2 |
| CBAM | Chèn SAU 1×1 conv (fusion), TRƯỚC skip-cat | Mục 2.2.3 + Fig 4A |
| Skip fusion | Cat feature gốc [C] + output [C] → [2C] → Conv 3×3 → [C] | Mục 2.2.3 + Fig 4A |

### Vision Transformer Encoder (Figure 4B — đã transcribe)
Cấu trúc chuẩn transformer block, lặp L lần:
`Linear → Flatten → + Positional Encoding → [LayerNorm → Multi-Head Attention → residual add → LayerNorm → MLP → residual add] ×L`

| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Self-attention | softmax(QK^T / √dk) · V (chuẩn) | Eq.1 / Mục 2.2.4 |
| MLP | 2 fully-connected layers | Mục 2.2.4 |
| num_heads | **THIẾU** — paper không ghi | Mục 2.2.4 |
| head_dim | **THIẾU** — paper không ghi | — |
| transformer dim (per stage) | **THIẾU** — paper không ghi (theo MobileViT XXS/XS/S) | — |
| mlp_dim ratio | **THIẾU** — paper không ghi (MobileViT gốc dùng ratio 2) | — |
| dropout | **THIẾU** — paper không ghi | — |

### CBAM Block (Figure 4C — đã transcribe)
Gồm Channel Attention + Spatial Attention. Trước khi chèn dùng Conv 3×3 tiền xử lý.

**Channel Attention (Eq.2):** `Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )`
- AvgPool & MaxPool song song → [C,1,1]
- Shared MLP = 2 conv 1×1 (nén channel xuống C/r rồi mở lại C)
- Reduction ratio r: **THIẾU giá trị số** (paper viết "compress to R times", CBAM gốc dùng r=16) — Mục 2.2.5
- σ = Sigmoid → nhân với feature map gốc

**Spatial Attention (Eq.3):** `Ms(F) = σ( f^7×7([AvgPool(F); MaxPool(F)]) )`
- AvgPool & MaxPool theo channel → 2× [1,H,W] → Cat → [2,H,W]
- Conv **7×7** → [1,H,W] → Sigmoid → nhân với feature map
- Có residual connection cuối (Fig 4C)

### Hyperparameters (Table 2 — đầy đủ trong paper)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Epochs | 100 | Table 2 |
| Batch size | 32 | Table 2 |
| Optimizer | AdamW | Table 2 |
| Learning rate | 0.0001 (1e-4) | Table 2 |
| Loss function | Cross Entropy | Table 2 |
| LR Scheduler | KHÔNG đề cập | — |
| Warmup epochs | KHÔNG đề cập | — |
| Weight decay | KHÔNG đề cập (AdamW mặc định) | — |
| Class weighting | KHÔNG đề cập (sẽ áp dụng auto theo Rule 11) | — |
| Normalization | "standardize bằng mean & std" — KHÔNG ghi giá trị số cụ thể | Mục 2.4 |

### Preprocessing & Augmentation (paper dùng — Mục 2.4)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Resize | 224×224 (tất cả ảnh) | Mục 2.4 |
| Random rotation | range không ghi | Mục 2.4 |
| Center crop | range/size không ghi | Mục 2.4 |
| Normalize (mean/std) | giá trị không ghi | Mục 2.4 |

> LƯU Ý Rule 3 (DUAL EXPERIMENT): Scenario 1 baseline CHỈ Resize + ToTensor + Normalize (bỏ rotation/crop của paper). Scenario 2 dùng 4 aug chuẩn (Rotation, Zoom/Scale, HFlip, ColorJitter). Augmentation của paper chỉ để tham khảo.

### GitHub Source
**Link:** Không tìm thấy (Data availability chỉ nói "included in the article/supplementary material", không có URL code).

### Kết quả trong paper (tham khảo — Table 3 & Table 4)
| Biến thể | Wheat Acc | Coffee Acc | Rice Acc | Params (M) | FLOPs (G) |
|----------|-----------|------------|----------|-----------|-----------|
| PMVT-XXS | 93.6% | 85.3% | 93.1% | 0.98 | 0.31 |
| PMVT-XS | 94.7% | 86.5% | 97.7% | 2.01 | 0.85 |
| PMVT-S | 94.9% | 87.6% | 92.0% | 5.06 | 1.59 |

Ablation (Table 4) xác nhận đóng góp: +Conv7×7, +CBAM, PMVT (cả hai) đều tăng accuracy.

---

## ĐÃ XÁC NHẬN (user trả lời)

| # | Câu hỏi | Quyết định |
|---|---------|-----------|
| 1 | Biến thể | **PMVT-XS** (2.01M params) |
| 2 | Expand ratio trong InvertedResidual | **×2** (theo paper Mục 2.2.2) |
| 3 | L cho 3 MobileViT stage | **L=[4,4,3]** (theo Fig 2) |
| 4 | num_heads | **1** (MobileViT-XS gốc, head_dim = d) |
| 5 | mlp_dim ratio | **×2** (mlp_dim = 2 × dim) |
| 6 | Patch size P | **2×2** (MobileViT chuẩn) |
| 7 | CBAM reduction ratio r | **16** (CBAM gốc Woo 2018) |
| 8 | Normalization | **ImageNet** mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |
| 9 | Dataset structure | **Có sẵn train/val/test** (3 thư mục riêng) |
| 10 | LR Scheduler | **Không dùng** — LR cố định 1e-4 |

**Channel/dim config cho PMVT-XS** (tham chiếu MobileViT-XS gốc, Mehta & Rastegari 2021):
| Stage | out_ch | MV2 stride | MobileViT dim | L |
|-------|--------|-----------|---------------|---|
| Stem  | 32     | stride=1  | —             | — |
| S1    | 32     | stride=1  | —             | — |
| S2    | 48     | stride=2  | —             | — |
| S3    | 64→80  | stride=2  | d=96          | 4 |
| S4    | 80→96  | stride=2  | d=120         | 4 |
| S5    | 96→128 | stride=2  | d=144         | 3 |
| Head  | 384    | —         | —             | — |

> MobileViT-XS Head: Conv 1×1 (128→384) + GAP + FC. Target params ≈ 2.01M với expand=2.

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG có trong bài báo (paper chỉ nói "theo MobileViT XXS/XS/S" của Mehta & Rastegari 2021, hoặc theo CBAM gốc của Woo 2018). Cần xác nhận trước khi viết notebook để tránh hallucination.

1. **Chọn biến thể nào để reproduce?** PMVT có 3 biến thể XXS / XS / S. Bạn muốn implement biến thể nào? (gợi ý: PMVT-XXS — 0.98M params, là biến thể chính được nhấn mạnh trong Abstract).

2. **Channel dimensions cho từng stage (S1→S5):** Paper KHÔNG liệt kê số channel cụ thể, chỉ nói "theo MobileViT XXS/XS/S". Bạn có đồng ý lấy nguyên cấu hình channel từ paper MobileViT gốc (Mehta & Rastegari, 2021) cho biến thể tương ứng không? (ví dụ MobileViT-XXS: channels [16, 24, 48, 64, 80], expansion=2). Nếu đồng ý, đây là nguồn tham chiếu hợp lệ; nếu không, vui lòng cung cấp bảng channel.

3. **Transformer dim mỗi stage:** Không ghi trong paper. Lấy theo MobileViT gốc (XXS: d=[64,80,96] cho 3 MobileViT stage) hay bạn muốn giá trị khác?

4. **num_heads của Multi-Head Attention:** Paper không ghi. MobileViT gốc dùng num_heads=4 (head_dim=8). Dùng giá trị này hay bạn muốn khác?

5. **mlp_dim ratio trong transformer:** Không ghi. MobileViT gốc dùng mlp ratio = 2 (mlp_dim = 2×dim). Dùng giá trị này?

6. **Patch size P trong MobileViT block:** Paper chỉ ghi ký hiệu "P", không có số. MobileViT gốc dùng patch 2×2. Dùng 2×2?

7. **Số transformer block L mỗi stage:** Theo Fig 2 đọc được L=2 (layer2), L=4 (layer3), L=4 (layer4), L=3 (layer5). Lưu ý layer 2 là Inverted Residual (L này có thể là số InvRes block lặp), còn MobileViT gốc dùng L=[2,4,3] cho 3 MobileViT stage. Bạn xác nhận dùng L=[4,4,3] (theo hình) hay L=[2,4,3] (theo MobileViT gốc) cho 3 MobileViT stage (layer 3,4,5)?

8. **CBAM reduction ratio r:** Paper viết "compress to R times" nhưng không cho giá trị số. CBAM gốc (Woo 2018) dùng r=16. Dùng r=16?

9. **Giá trị Normalization mean/std:** Paper chỉ nói "normalize bằng mean & std" không cho số. Dùng ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225], hay tính mean/std từ chính dataset?

10. **Dataset để train trên Kaggle:** Paper dùng 3 dataset (wheat/coffee/rice) không công khai link. Bạn muốn reproduce trên dataset Kaggle nào? Cần cung cấp đường dẫn dataset Kaggle (số class sẽ auto-detect).

11. **LR Scheduler:** Paper không đề cập scheduler. Train với LR cố định 1e-4 suốt 100 epochs, hay bạn muốn thêm scheduler (ví dụ CosineAnnealing)?

---

## GHI CHÚ IMPLEMENTATION

1. **Điểm mấu chốt — channel/dim config:** Vì paper không ghi channel/transformer-dim, cách an toàn nhất (zero-hallucination) là tham chiếu trực tiếp config của MobileViT gốc cho biến thể tương ứng, rồi áp 2 thay đổi của PMVT lên trên: (a) đổi DW conv 3×3 → **7×7** trong inverted residual, (b) chèn **CBAM** vào sau 1×1 conv trong MobileViT block. Hai thay đổi này được paper mô tả rõ ràng và là khác biệt cốt lõi so với MobileViT.

2. **Inverted Residual với DW 7×7:** padding=3 để giữ spatial resolution. Expand ratio = 2 (paper ghi "generally by two times"). Skip connection chỉ áp dụng khi stride=1 và in_ch==out_ch (chuẩn inverted residual) — cần xác nhận stride placement (câu hỏi #7).

3. **MobileViT block + CBAM:** Khác biệt so với MobileViT gốc là chèn CBAM SAU 1×1 conv fusion và TRƯỚC skip-cat (theo Fig 4A). CBAM gồm Channel Attention (Eq.2) rồi Spatial Attention (Eq.3, conv 7×7). Lưu ý CBAM còn có Conv 3×3 tiền xử lý (Fig 4C) — cần cài đúng thứ tự.

4. **Head:** Conv 1×1 mở rộng channel ×4 → Global Average Pool → FC (num_classes). num_classes auto-detect theo dataset (Rule 12), KHÔNG hard-code 7/3/2.

5. **Rủi ro mismatch params:** Mục tiêu params PMVT-XXS ≈ 0.98M. Sau khi build, dùng `torchinfo.summary()` để đối chiếu tổng params với Table 4 (~0.97-0.98M cho XXS) — nếu lệch nhiều (>20%) nghĩa là channel/dim config sai, cần xem lại câu hỏi #2/#3.

6. **Tuân thủ Rules notebook:** PyTorch (paper dùng PyTorch 1.9.0 → khớp Rule 2); 2 scenario (baseline + augmented, Rule 3); metrics Accuracy/F1-macro/Recall-macro + classification_report per class (Rule 4); torchinfo.summary (Rule 5); learning curve + confusion matrix dpi=300 cell riêng (Rule 6); seed=42 + cudnn deterministic (Rule 7,9); device auto-detect (Rule 8); best checkpoint theo val_acc, file best_model_s1/s2.pth (Rule 10); class imbalance auto (Rule 11 — wheat dataset rõ ràng mất cân bằng, max/min > 1.5 → sẽ bật class_weights); num_workers=2 + pin_memory + empty_cache giữa 2 scenario (Rule 13).

7. **Split dữ liệu:** Paper dùng tỉ lệ 8:1:1 (train:val:test). Khi reproduce trên Kaggle nên giữ tỉ lệ này nếu dataset chưa chia sẵn.
