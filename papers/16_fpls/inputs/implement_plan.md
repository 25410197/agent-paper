# IMPLEMENT PLAN: Two-Channel Swin Transformer (RGB + Trainable-Sobel Edge) — "Han & Guo 2024"

**Paper:** Automatic classification of ligneous leaf diseases via hierarchical vision transformer and transfer learning
**Tác giả:** Dianyuan Han, Chunhua Guo (Weifang University)
**Journal/Conference:** Frontiers in Plant Science 14:1328952, 2024 (published 12 Jan 2024)
**DOI:** 10.3389/fpls.2023.1328952
**Ngày phân tích:** 2026-10-20

---

## PHÂN LOẠI
**Nhóm:** A
**Lý do:** Paper đề xuất mô hình cụ thể (two-channel Swin-T, kênh dưới nhận ảnh biên Sobel khả huấn luyện), có framework (PyTorch 2.0.1), có bảng hyperparameter (Table 1), input size, pretrained, loss, và kết quả thực nghiệm. Các thông số còn thiếu (heads, drop-path, scheduler, chi tiết ghép kênh Sobel...) đều là chi tiết phụ có default chuẩn của Swin-T, không thiếu thông số cốt lõi.

Ký hiệu: **[CONFIRMED]** = có trong text/hình của paper. **[MISSING]** = paper không nêu, cần người dùng xác nhận (có kèm default đề xuất trong mục CẦN XÁC NHẬN).

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Trạng thái | Nguồn trong paper |
|---------|--------|-----------|------------------|
| Tên model | Two-channel Swin vision transformer (không có tên riêng) | CONFIRMED | Mục 2.2, Fig. 2 |
| Loại | Vision Transformer phân cấp (Swin), 2 nhánh song song | CONFIRMED | Mục 2.2 |
| Framework | PyTorch 2.0.1, Python 3.8.3 | CONFIRMED | Mục 3.1 |
| Backbone mỗi nhánh | Swin-T | CONFIRMED | Mục 3.1 ("Swin-T ... employed for each channel") |
| Chia sẻ trọng số giữa 2 nhánh | KHÔNG (no weight sharing) | CONFIRMED | Mục 2.2 |
| Input size | 224×224×3 (cả 2 nhánh H×W×3) | CONFIRMED | Mục 2.1, 3.1, Fig. 2 |
| Pretrained | Có — ImageNet | CONFIRMED | Mục 2.1, 3.1 |
| Tên checkpoint cụ thể (IN-1k / IN-22k) | không nêu | MISSING | — |
| Stage depths | 2 / 2 / 6 / 2 | CONFIRMED | Mục 2.2.1, Fig. 2 |
| "depth" trong Table 1 | 12 (= 2+2+6+2, tổng block mỗi nhánh) | CONFIRMED (giá trị); diễn giải là suy luận nhất quán | Table 1 |
| Embed dim C | 96 | CONFIRMED | Mục 2.2.1, caption Fig. 2 |
| Patch size | 4×4 (patch dim 48) | CONFIRMED | Mục 2.2.1 |
| Window size M | 7 | CONFIRMED | Mục 2.2.3 |
| Số heads mỗi stage | không nêu | MISSING | — |
| MLP ratio | không nêu | MISSING | — |
| Dropout / attn drop / drop-path | không nêu | MISSING | — |
| Class token | KHÔNG dùng | CONFIRMED | Mục 2.2.1 |

### Cấu trúc Layer (theo thứ tự, MỖI nhánh)
| Layer / Block | Kernel / Patch | Channels | Output | Activation | Ghi chú |
|--------------|--------|---------|--------|-----------|---------|
| Patch Partition | 4×4 non-overlap | 48 | 56×56×48 | — | CONFIRMED |
| Stage 1: Linear Embedding + Swin Block ×2 | — | C=96 | 56×56×96 | (GELU trong MLP — chuẩn Swin, paper không ghi) | W-MSA / SW-MSA xen kẽ, M=7 |
| Stage 2: Patch Merging + Swin Block ×2 | ↓2 | 2C=192 | 28×28×192 | — | CONFIRMED |
| Stage 3: Patch Merging + Swin Block ×6 | ↓2 | 4C=384 | 14×14×384 | — | CONFIRMED |
| Stage 4: Patch Merging + Swin Block ×2 | ↓2 | 8C=768 | 7×7×768 | — | CONFIRMED |

Swin block (Eq. 3–6, CONFIRMED): `ẑ = W-MSA(LN(z)) + z; z = MLP(LN(ẑ)) + ẑ; ẑ' = SW-MSA(LN(z)) + z; z' = MLP(LN(ẑ')) + ẑ'`.

### Nhánh Edge (input của kênh dưới) — Fig. 3, Eq. 1–2
| Bước | Chi tiết | Trạng thái |
|------|---------|-----------|
| 1 | Chuyển ảnh gốc sang grayscale | CONFIRMED |
| 2 | Conv 3×3 hướng X, khởi tạo Gx = [[+1,0,-1],[+2,0,-2],[+1,0,-1]] | CONFIRMED |
| 3 | Conv 3×3 hướng Y, khởi tạo Gy = [[+1,+2,+1],[0,0,0],[-1,-2,-1]] | CONFIRMED |
| 4 | Trọng số Sobel **trainable** (backprop), không cố định | CONFIRMED (ablation so với fixed Sobel) |
| 5 | "Merging" 3 nhánh → ảnh 3 kênh H×W×3 → vào nhánh Swin dưới | CONFIRMED (hình vẽ); thứ tự kênh [gray, Gx, Gy] là transcription từ hình |
| Sobel áp lên gray hay lên RGB | Text: "the original image undergoes convolution"; hình: nhánh song song từ ảnh gốc | MISSING / mơ hồ |
| Padding, bias, lấy |G| / abs hay giữ dấu, chuẩn hóa output | không nêu | MISSING |

### Head
| Bước | Chi tiết | Trạng thái |
|------|---------|-----------|
| Concat output Stage 4 của 2 nhánh | 7×7×768 ⊕ 7×7×768 | CONFIRMED (Fig. 2) |
| Average Pooling → Fully Connected → softmax | thứ tự Concat → AvgPool → FC | CONFIRMED (Fig. 2) |
| Concat theo chiều kênh (→1536) hay chiều không gian/token | không nêu | MISSING |
| LayerNorm cuối mỗi nhánh trước pool (có trong Swin gốc) | không nêu | MISSING |
| Câu "linear classifier takes the first C components" | mơ hồ, không giải thích | Bỏ qua — ghi chú |

### Hyperparameters
| Thông số | Giá trị | Trạng thái | Nguồn |
|---------|--------|-----------|-------|
| Optimizer | Adam | CONFIRMED | Table 1 |
| Learning rate | 1e-4 | CONFIRMED | Table 1 |
| LR Scheduler | không nêu | MISSING | — |
| Warmup epochs | không nêu | MISSING | — |
| Batch size | 8 | CONFIRMED | Table 1 |
| Epochs | 100 | CONFIRMED | Table 1 |
| Loss function | Cross-entropy | CONFIRMED | Mục 2.2.3 |
| Weight decay | không nêu | MISSING | — |
| Class weighting | không nêu (paper thừa nhận dataset mất cân bằng) | Theo rule 11 (auto ≥1.5) | Mục 3.4 |
| Normalization | không nêu | MISSING | — |
| Hardware gốc | 4× RTX 3080 | CONFIRMED | Mục 3.1 |
| Validation scheme | 10-fold cross-validation | CONFIRMED (nhưng KHÔNG áp dụng — xem bên dưới) | Mục 3.1 |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| RandomFlip | không nêu (hướng/prob) | Mục 2.1 |
| Color Jitter | không nêu | Mục 2.1 |
| CutMix | không nêu (alpha, prob) | Mục 2.1 |
| Mixup | không nêu (alpha) | Mục 2.1 |

Rule 01 yêu cầu Scenario 2 = đúng 4 aug: Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness. Scenario 1 = chỉ Resize(224) + ToTensor + Normalize.

### GitHub Source
**Link:** Không tìm thấy (chỉ có link dataset: https://aistudio.baidu.com/datasetdetail/76075 — AI Challenger 2018)

### Kết quả trong paper (tham khảo)
| Dataset | Accuracy | Precision | Recall | F1 | Params |
|---------|---------|-----|--------|--------|--------|
| AI Challenger 2018 subset, 22 lớp (11,603 train / 1,668 test) | 86.43% | 85.73 | 87.22 | 86.47 | không nêu |
| Ablation: so với ViT +2.2% / fixed-Sobel +1.4% (25% data); +2.22% / +1.40% (50% data) | | | | | |

---

## RÀNG BUỘC TỪ NGƯỜI DÙNG / RULES (áp dụng khi implement)
- Dataset của người dùng LUÔN pre-split train/val/test → KHÔNG dùng train_test_split, KHÔNG 10-fold CV. Dùng `ImageFolder` cho từng split; checkpoint theo val_accuracy (rule 10), evaluate test với best checkpoint.
- Số lớp auto-detect: `num_classes = len(train_ds.classes)`; `class_names = train_ds.classes` (rule 12). Không hard-code 22.
- Framework: PyTorch (đúng paper). Seed 42, cudnn deterministic, DataParallel nếu >1 GPU, `num_workers=4, pin_memory=True`, torchinfo summary (input `(1,3,224,224)`), classification_report, learning curve + confusion matrix dpi=300 ở cell riêng, `torch.cuda.empty_cache()` giữa 2 scenario.

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG có trong bài báo. Vui lòng trả lời trước khi tôi viết notebook. Mỗi mục kèm default đề xuất.

1. **Số heads mỗi stage** — Paper không nêu. Default đề xuất: theo Swin-T chuẩn **3/6/12/24**.
   Câu hỏi: "Paper không ghi số attention heads từng stage. Bạn đồng ý dùng chuẩn Swin-T (3, 6, 12, 24) không?"

2. **MLP ratio / dropout / drop-path** — Paper không nêu. Default đề xuất: mlp_ratio=4, drop=0, attn_drop=0, **drop_path_rate=0.2** (mặc định Swin-T timm/official).
   Câu hỏi: "Paper không ghi MLP ratio và drop-path. Dùng mặc định Swin-T (mlp_ratio=4, drop_path_rate=0.2, dropout=0) được không, hay bạn muốn drop_path=0?"

3. **LR scheduler / weight decay / warmup** — Paper chỉ ghi Adam lr=1e-4. Default đề xuất: **Adam(lr=1e-4), weight_decay=0, không scheduler, không warmup** (bám sát paper, không thêm gì).
   Câu hỏi: "Paper chỉ nêu Adam lr=1e-4, không có scheduler hay weight decay. Bạn muốn giữ lr cố định, weight_decay=0 (bám sát paper), hay thêm CosineAnnealingLR?"

4. **Cách ghép input 3 kênh của nhánh edge** — Hình 3 vẽ 3 nhánh: Gray, Sobel-X, Sobel-Y → "Merging". Default đề xuất: tính gray từ RGB (0.299R+0.587G+0.114B, trên ảnh chưa normalize ImageNet), áp `Conv2d(1,2,3,padding=1,bias=False)` trainable khởi tạo [Gx, Gy] lên **ảnh gray**, rồi stack **[gray, Gx_out, Gy_out]** thành 3 kênh; sau đó normalize bằng ImageNet mean/std. Giữ dấu (không abs). Module này nằm trong model để gradient chảy về kernel Sobel.
   Câu hỏi: "Paper không nói rõ Sobel áp lên ảnh gray hay RGB, có lấy giá trị tuyệt đối không, và 3 kênh ghép theo thứ tự nào. Bạn đồng ý dùng [gray, Sobel-X(gray), Sobel-Y(gray)], giữ dấu, padding=1, không bias không?"

5. **Thứ tự Concat / Pool và chiều concat** — Fig. 2: Concat → AvgPool → FC. Default đề xuất: LayerNorm cuối mỗi nhánh (như Swin gốc) → concat theo **chiều kênh** (7×7×1536) → AdaptiveAvgPool → Linear(1536, num_classes). (Concat kênh rồi pool tương đương pool rồi concat.)
   Câu hỏi: "Paper vẽ Concat → AvgPool → FC nhưng không nói concat theo chiều kênh hay không gian. Bạn đồng ý concat theo kênh (768+768=1536) rồi global average pool và FC không?"

6. **Checkpoint pretrained** — Paper chỉ ghi "ImageNet". Default đề xuất: timm `swin_tiny_patch4_window7_224` (ImageNet-1k, weights `ms_in1k`), nạp riêng cho **cả 2 nhánh** (2 bản copy độc lập, không share).
   Câu hỏi: "Paper không nói checkpoint ImageNet-1k hay 22k. Dùng timm `swin_tiny_patch4_window7_224` (ImageNet-1k) cho cả hai nhánh được không?"

7. **Normalization** — Paper không nêu. Default đề xuất: ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (khớp pretrained), áp cho cả 2 nhánh.
   Câu hỏi: "Paper không ghi mean/std. Dùng ImageNet mean/std cho cả nhánh RGB và nhánh edge được không?"

8. **CutMix / Mixup vs. Scenario 2 bắt buộc 4 aug** — Paper dùng RandomFlip + ColorJitter + CutMix + Mixup; rule dự án yêu cầu Scenario 2 = Rotation, Zoom/Scale, HorizontalFlip, ColorJitter. Default đề xuất: **Scenario 2 chỉ dùng đúng 4 aug theo rule, KHÔNG dùng CutMix/Mixup**; ghi chú lệch so với paper.
   Câu hỏi: "Paper dùng CutMix và Mixup, nhưng quy chuẩn Scenario 2 chỉ có 4 aug (Rotation, Zoom, HFlip, ColorJitter). Bạn muốn chỉ dùng 4 aug chuẩn, hay thêm Scenario 3 có CutMix+Mixup giống paper?"

9. **10-fold CV vs. dataset pre-split** — Paper dùng 10-fold CV; dataset của bạn đã chia sẵn train/val/test. Default đề xuất: **bỏ CV**, train 1 lần trên train, chọn best theo val_accuracy, đánh giá trên test.
   Câu hỏi: "Paper đánh giá bằng 10-fold cross-validation, nhưng dataset của bạn đã chia sẵn train/val/test. Tôi sẽ bỏ CV và dùng split có sẵn — bạn đồng ý chứ?"

10. **Đường dẫn dataset Kaggle** — Cần biết để viết cell cấu hình.
    Câu hỏi: "Đường dẫn dataset trên Kaggle là gì (thư mục chứa train/ val/ test/)?"

---

## GHI CHÚ IMPLEMENTATION

- **Kiến trúc đề xuất (PyTorch + timm):**
  ```
  class SobelEdgeInput(nn.Module): gray(x_raw) -> conv(1->2, 3x3, pad=1, bias=False, init [Gx,Gy], requires_grad=True) -> cat[gray,gx,gy] -> normalize
  class TwoChannelSwin(nn.Module):
      rgb  = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, global_pool='')
      edge = timm.create_model(... same, independent copy ...)
      head = AdaptiveAvgPool over concat(1536) -> Linear(1536, num_classes)
  ```
  Lưu ý timm Swin trả feature dạng NHWC (B,7,7,768) sau `forward_features` — cần permute trước khi concat/pool.
- **Normalization & edge branch:** vì Sobel là trainable và nằm trong model, DataLoader nên trả ảnh `[0,1]` chưa normalize; model tự normalize nhánh RGB và nhánh edge bên trong (hoặc DataLoader normalize ImageNet rồi model de-normalize để tính gray). Chọn 1 cách nhất quán; khuyến nghị: normalize trong model.
- **Params:** ~2 × 27.5M (Swin-T backbone) + Sobel 18 + FC(1536×num_classes) ≈ 55M. Paper không báo params.
- **Bộ nhớ:** batch 8 trên 2×T4 với DataParallel → 4 ảnh/GPU, an toàn. 100 epochs × 2 scenario × 2 backbone Swin-T có thể vượt giới hạn 12h Kaggle tùy dataset — cân nhắc báo người dùng.
- **Table 1 "depth 12":** pdftotext xáo thứ tự; đã xác nhận bằng hình: Batch 8, Adam, lr 1e-4, depth 12, epochs 100.
- **Mơ hồ trong paper:** câu "linear classifier takes the first C components of this output vector" và "output vector of dimensions N = H/32 × W/32" không khớp với Fig. 2; theo Fig. 2 (Concat → AvgPool → FC) để implement.
- **Class imbalance:** paper thừa nhận dataset mất cân bằng; áp rule 11 (auto class_weights nếu max/min ≥ 1.5).
- **Rủi ro mismatch kết quả:** khác dataset, bỏ CutMix/Mixup, bỏ 10-fold CV → không kỳ vọng tái tạo đúng 86.43%.
