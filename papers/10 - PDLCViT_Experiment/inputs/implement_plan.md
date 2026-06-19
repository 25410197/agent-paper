# IMPLEMENT PLAN: PDLC-ViT

**Paper:** A Multitask Learning-Based Vision Transformer for Plant Disease Localization and Classification
**Tác giả:** S. Hemalatha, Jai Jaganath Babu Jayachandran
**Journal/Conference:** International Journal of Computational Intelligence Systems (2024) 17:188
**DOI:** https://doi.org/10.1007/s44196-024-00597-3
**Ngày phân tích:** 2026-06-19

---

## PHÂN LOẠI
**Nhóm:** C (có model với kiến trúc đặt tên rõ ràng + bảng hyperparameters đầy đủ, NHƯNG thiếu nhiều thông số định lượng cốt lõi của ViT: patch size, embedding dim, số transformer layers, số attention heads, MLP dim, input size, λ_seg/λ_cls).
**Lý do:** Bài báo mô tả kiến trúc PDLC-ViT bằng văn bản và sơ đồ (Fig 1-4) ở mức khái niệm (component nào nối với component nào), kèm Table 2 hyperparameters training rõ ràng. Tuy nhiên KHÔNG nêu các con số định lượng bắt buộc để dựng một ViT cụ thể (patch size, hidden dim, depth, heads...). Vì vậy cần người dùng xác nhận trước khi viết notebook.

---

## ĐIỀU CHỈNH PHẠM VI PHỤC DỰNG (QUAN TRỌNG — ĐỌC TRƯỚC)

Bài báo gốc là Multi-Task Learning (MTL): **vừa segmentation (localization) vừa classification**, dùng 2 dataset khác nhau:
- **PlantVillage** (54,306 ảnh, 38 classes) — dùng cho **classification**.
- **PlantDoc** (2,940 ảnh, có segmentation mask) — dùng cho **localization/segmentation**.

Theo CORE RULES của workspace (DUAL EXPERIMENT, REPORTING METRICS classification: Accuracy / F1 Macro / Recall Macro / Classification Report per class), pipeline phục dựng trong workspace này **tập trung vào nhánh CLASSIFICATION trên PlantVillage** (dataset có label class rõ ràng cho ImageFolder + train/val/test theo Kaggle). Lý do:
- Nhánh segmentation cần ground-truth mask (PlantDoc) — không phải bài toán phân loại, không khớp template dual-experiment + classification report của workspace.
- Các metrics bắt buộc của workspace đều là metrics phân loại.

=> **Mặc định đề xuất:** phục dựng **PDLC-ViT (classification branch)** = Image Splitter → Co-scale → Co-attention → Cross-attention (MHCA) → Transformer Encoder → Classification Head, train trên PlantVillage 38 classes. Phần Segmentation Head / localization là **optional / out-of-scope** trừ khi người dùng yêu cầu (xem CẦN XÁC NHẬN câu 1).

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | PDLC-ViT (Plant Disease Localization and Classification ViT) | Abstract / Tiêu đề |
| Loại | Vision Transformer (MTL, co-scale + co-attention + cross-attention) | Abstract, Sect. 4.2 |
| Framework gốc | MATLAB 2023a (Deep Learning + Image Processing Toolbox) | Sect. 3.2 |
| Framework phục dựng | **PyTorch** (theo CORE RULE #2: paper không dùng PyTorch/TF → ưu tiên PyTorch) | Quy ước workspace |
| Pretrained backbone | KHÔNG (model tự thiết kế, không nêu pretrained weights) | Toàn bài (không đề cập transfer learning cho PDLC-ViT) |
| Optimizer | Adam | Table 2, Sect. 4.1, Sect. 5.1 |
| Loss classification | Categorical cross-entropy | Eq. (2), Sect. 4.1 |

### Cấu trúc Component (theo Fig. 1-4, đã visual-confirm)

Luồng tổng thể (Fig. 1):
```
Input Image → Image Splitter → Patches X={x1..xn}
       ├──► Co-scale Mechanism      → CS={xs1..xsn}
       └──► Co-attention Mechanism  → CA={a1..an}
                  └──► Cross-Attention Mechanism → Z={z1..zn}
                            └──► Transformer Encoder → encoded features
                                      ├──► Segmentation Head → mask  (out-of-scope mặc định)
                                      └──► Classification Head → labels
```

| Component | Cấu tạo (theo paper) | Nguồn |
|-----------|----------------------|-------|
| Image Splitter | Chia ảnh thành các patch không chồng lấn X={x1..xn} | Sect. 4.2, Fig. 1 |
| Co-scale Mechanism | Bắt phụ thuộc đa tỉ lệ (multi-scale) giữa patches → CS | Sect. 4.2, Fig. 1 (chi tiết layer KHÔNG nêu) |
| Co-attention Mechanism | Bắt phụ thuộc liên-patch → CA | Sect. 4.2, Fig. 1 (chi tiết layer KHÔNG nêu) |
| Cross-Attention (Fig. 3) | 2 nhánh song song: nhánh CS → (Qs,Ks,Vs)→MHSA→Add&Norm→FeedForward→Add&Norm→Z1; nhánh CA → (QA,KA,VA)→MHSA→Add&Norm→FeedForward→Add&Norm→Z2; sau đó MHCA(Z1,Z2)→Z | Sect. 4.3, Fig. 3, Eq. (4)(5) |
| Transformer Encoder (Fig. 2) | Linear Projection → Positional Encoding → Multi-Head Self-Attention → Add&Norm → Feed-Forward (2 linear + ReLU) → Add&Norm → encoded features | Sect. 4.2 (list 1-6), Fig. 2 |
| Segmentation Head (Fig. 4) | 3 block có residual: B1=[LN→MHSA]; B2=[LN→MHCA(+Z)]; B3=[LN→MLP(FC+ReLU)] → mask | Sect. 4.4, Fig. 4 (OUT-OF-SCOPE mặc định) |
| Classification Head (Fig. 4) | 2 block: B1=[LN→MHCA(+Z)]; B2=[LN→MLP(FC + Sigmoid)] → labels | Sect. 4.4, Fig. 4 |

> Lưu ý: Attention dùng công thức scaled dot-product chuẩn: `softmax(Q·Kᵀ/√d)·V` (Eq. 4, 5).

### Hyperparameters (Table 2 — visual-confirmed, page 10)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | Adam | Table 2 / Sect. 5.1 |
| Initial learning rate | 0.001 | Table 2 / Sect. 5.1 |
| LR Scheduler | StepLR: nhân factor 0.5 mỗi 15 epochs (`gamma=0.5, step_size=15`) | Table 2 / Sect. 5.1 |
| Batch size | 16 | Table 2 / Sect. 5.1 |
| Dropout | 0.2 (áp dụng cho cả encoder và decoder) | Table 2 / Sect. 5.1 |
| L2 regularization (weight_decay) | 0.0005 | Table 2 / Sect. 5.1 |
| Loss function (classification) | Categorical cross-entropy (PyTorch: `nn.CrossEntropyLoss`) | Eq. (2) |
| Loss tổng MTL | L_total = λ_seg·L_seg + λ_cls·L_cls | Eq. (3), Fig.11 |
| Epochs | KHÔNG cố định — early stopping theo validation loss threshold = 0.001 | Table 2 / Sect. 5.1 |
| Data handling | Random shuffle khi train và test | Table 2 |
| Normalization (mean/std) | KHÔNG nêu | — (xem CẦN XÁC NHẬN) |
| Input size | KHÔNG nêu | — (xem CẦN XÁC NHẬN) |
| Patch size / embed dim / depth / heads / MLP dim | KHÔNG nêu | — (xem CẦN XÁC NHẬN) |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Affine transformation | "augmented with images by affine transformation from the original images" (không nêu range cụ thể) | Sect. 3.1 |

> Paper chỉ nói chung "affine transformation", không liệt kê range. Scenario 2 của workspace dùng bộ 4 augmentation chuẩn (Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness) theo CORE RULE #3.

### Dataset (Table 1 — PlantVillage)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Dataset classification | PlantVillage | Sect. 3.1 |
| Tổng ảnh | 54,306 | Sect. 3.1, Table 1 |
| Số classes | 38 | Sect. 3.1 |
| Split | Table 1 hiển thị số train = số test cho mỗi class (đây là per-class count, KHÔNG phải tỉ lệ split cụ thể) | Table 1 |
| Dataset localization | PlantDoc (2,940 ảnh, 13 species, có mask) — out-of-scope mặc định | Sect. 3.1 |

> Về split: notebook trên Kaggle sẽ dùng split do người dùng chọn (xem CẦN XÁC NHẬN) vì Table 1 không cho tỉ lệ train/val/test rõ ràng cho pipeline reproduce.

### GitHub Source
**Link:** Không tìm thấy (paper không có mục Code Availability / repo link).

### Kết quả trong paper (tham khảo — classification)
| Dataset | Accuracy | F1 | Precision | AUC | Params |
|---------|---------|-----|-----------|-----|--------|
| PlantVillage (Type A, binary Healthy/Unhealthy) | 99.97% | 0.9995 | 0.9995 | 0.9993 | Không nêu |
| PlantVillage (multiclass, mean ablation baseline) | 99.37% | 0.9904 | 0.9898 | — | Không nêu |
| Localization (PlantDoc) | Mean Dice 0.9921, Mean IoU 0.9973 | — | — | — | — |

---

## KẾ HOẠCH 2 KỊCH BẢN (CORE RULE #3 — DUAL EXPERIMENT)

**Scenario 1 — Baseline:** Train transform = CHỈ `Resize + ToTensor + Normalize`. Không augmentation. Val/Test transform giống hệt baseline.

**Scenario 2 — Augmented:** Train transform = Resize + 4 augmentation (RandomRotation, RandomResizedCrop/Zoom, RandomHorizontalFlip, ColorJitter brightness) + ToTensor + Normalize. Val/Test transform KHÔNG aug (giống baseline).

Cả 2 scenario dùng cùng kiến trúc PDLC-ViT (classification branch) + cùng hyperparameters Table 2.

---

## METRICS BẮT BUỘC (CORE RULE #4, #5, #6)
- Total / Trainable / Non-trainable Parameters (`torchinfo.summary`).
- Accuracy.
- F1-Score (Macro).
- Recall (Macro).
- Classification Report per class (`sklearn.metrics.classification_report`, `target_names=class_names`).
- Learning Curve (cell riêng, dpi=300, `plt.show()`).
- Confusion Matrix (cell riêng, dpi=300, `plt.show()`).
- Class imbalance auto-detect + class_weights nếu max/min ≥ 1.5 (CORE RULE #11). PlantVillage rất mất cân bằng (min ~152, max ~5507) → gần như chắc chắn sẽ bật class weighting.
- Best checkpoint theo `val_accuracy`, reload trước khi test (`best_model_s1.pth`, `best_model_s2.pth`).

---

## THÔNG SỐ ĐÃ XÁC NHẬN (DO NGƯỜI DÙNG CHỌN)

> ✅ Người dùng đã xác nhận các thông số sau (2026-06-19). Giá trị không trích từ paper — ghi disclaimer trong notebook.

| Thông số | Giá trị | Ghi chú |
|---------|--------|--------|
| Phạm vi | **CHỈ classification branch** | Bỏ segmentation head / PlantDoc |
| Dataset | `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300` | Dataset riêng của người dùng (đã pre-split) |
| Input size | **224×224** | Chuẩn ViT |
| Patch size | **16×16** | → 196 patches với input 224 |
| Embed dim | **384** | ViT-Small |
| Depth (encoder layers) | **6** | ViT-Small |
| Attention heads | **6** | ViT-Small |
| MLP dim | **1536** | = 4 × 384 |
| Normalization | ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | |
| Max epochs | **100** + early stopping (patience=7, monitor val_loss) | |
| Split | **70/15/15** stratified | Dùng nếu dataset chưa có sẵn train/val/test subfolders |
| Co-scale implementation | Multi-scale patch embedding: 2 conv branches (3×3 + 5×5) → concat → Linear projection | Diễn giải hợp lý — KHÔNG trích từ paper |
| Co-attention implementation | Inter-patch self-attention (standard MHSA trên sequence patches) | Diễn giải hợp lý — KHÔNG trích từ paper |

---

## GHI CHÚ IMPLEMENTATION

- **Rủi ro mismatch lớn nhất:** Kiến trúc ViT của paper thiếu toàn bộ con số định lượng (patch/dim/depth/heads). Không thể tái tạo chính xác accuracy 99.97% — kết quả phục dựng sẽ phụ thuộc vào cấu hình do người dùng chọn. Cần ghi disclaimer trong notebook rằng các giá trị này là lựa chọn hợp lý, không phải trích từ paper.
- **Co-scale / Co-attention mơ hồ:** Đây là điểm yếu nhất khi reproduce — paper không có pseudo-code hay layer spec cho 2 module này. Phải triển khai theo diễn giải và đánh dấu rõ (xem câu 12).
- **MTL → single-task:** Khi bỏ nhánh segmentation, loss đơn giản hóa còn `L_cls` (cross-entropy). Classification Head theo Fig.4 nhận MHCA(query_labels, Z); khi không có segmentation, có thể dùng class token / pooled Z làm query — cần ghi rõ là điều chỉnh do bỏ MTL.
- **Sigmoid ở classification head:** Paper ghi MLP classification dùng sigmoid. Với multiclass 38 lớp + CrossEntropyLoss (softmax bên trong) thì KHÔNG dùng sigmoid ở output — đây là khác biệt; sẽ dùng logits + CrossEntropyLoss theo chuẩn PyTorch và ghi chú.
- **Class imbalance:** PlantVillage rất mất cân bằng → CORE RULE #11 sẽ tự bật class_weights. Cần cell "Dataset Distribution" bar chart trước train.
- **Multi-GPU Kaggle T4×2:** Tuân thủ CORE RULE #8, #10, #13, #14 (DataParallel, num_workers=4, pin_memory, empty_cache giữa 2 scenario).
- **Tham số count:** Paper không công bố số params → chỉ in từ `torchinfo.summary`, không so sánh với paper.
