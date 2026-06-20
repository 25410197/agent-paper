# IMPLEMENT PLAN: Lightweight Meta-Ensemble (MLP-Mixer + LSTM + SVM)

**Paper:** A Lightweight Meta-Ensemble Approach for Plant Disease Detection Suitable for IoT-Based Environments
**Tác giả:** Ritesh Maurya, Satyajit Mahapatra, Lucky Rajput
**Journal/Conference:** IEEE Access, 2024
**DOI:** 10.1109/ACCESS.2024.3367443
**Ngày phân tích:** 2026-06-19

---

## PHÂN LOẠI
**Nhóm:** A (có model phục dựng được), nhưng có một số mục thuộc dạng C (mơ hồ) cần xác nhận.
**Lý do:** Paper đề xuất kiến trúc cụ thể (meta-ensemble 2 tầng: MLP-Mixer + LSTM ở Level 1, SVM ở Level 2), có bảng hyperparameters (Table 5, 6), bảng chi tiết layer Mixer (Table 7), bảng SVM grid-search, có dataset và kết quả thực nghiệm rõ ràng. Tuy nhiên một vài thông số bị mơ hồ/mâu thuẫn trong bảng trích từ hình (xem mục CẦN XÁC NHẬN) nên cần hỏi trước khi code.

---

## TỔNG QUAN MÔ HÌNH

Đây là **meta-ensemble 2 tầng (stacking)**, KHÔNG phải single model:

```
                ┌─────────────────────────────────────────────┐
LEVEL 1         │                                             │
                │  Branch A: MLP-Mixer (ảnh 64x64x3 → softmax probs)
                │            input ảnh → patch → 7 mixer layers → GAP → softmax (num_classes)
                │                                             │
                │  Branch B: LSTM                             │
                │     ảnh 64x64x3 → 4 pretrained CNN (frozen) trích feature
                │     → concat feature (1, 5632) → LSTM(30 cells, timestep=1) → softmax (num_classes)
                └─────────────────────────────────────────────┘
                          │ P_Mixer (N, num_classes)  │ P_LSTM (N, num_classes)
                          └──────────── concat ────────┘
                                       │
LEVEL 2                    P_Concat (N, 2*num_classes)
                                       │
                              SVM (RBF kernel) → Final prediction
```

**Quy trình train (stacking, KHÔNG end-to-end):**
1. Split dataset 80/20 (train/test).
2. Train MLP-Mixer trên ảnh train (đã augment) → softmax probs.
3. Trích feature từ 4 CNN pretrained (frozen) → concat (5632-d) → train LSTM → softmax probs.
4. Lấy prediction-probability của Mixer & LSTM trên **train set** → concat thành (N_train, 2*num_classes) → train SVM.
5. Test: ảnh test → Mixer & LSTM → concat probs → SVM predict → class cuối.

**LƯU Ý QUAN TRỌNG VỀ DUAL-EXPERIMENT (Rule 3 của workspace):**
Notebook BẮT BUỘC chạy 2 kịch bản:
- **Scenario 1 (Baseline):** training transform CHỈ Resize(64x64) + ToTensor + Normalize. KHÔNG augment.
- **Scenario 2 (Augmented):** thêm 4 phép Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness.
- Validation/Test transform luôn = baseline (chỉ Resize + ToTensor + Normalize).
- Cả 2 scenario đều chạy đủ pipeline 2 tầng (Mixer + LSTM + SVM) và report metrics riêng.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc tổng quát
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | Meta-ensemble (MLP-Mixer + LSTM + SVM) | Abstract, Mục IV |
| Loại | Stacking ensemble: MLP-Mixer (all-MLP) + LSTM (RNN) + SVM (ML) | Mục IV.D |
| Framework gốc của paper | TensorFlow/Keras (Table 5 ghi "sparse categorical crossentropy", "categorical crossentropy", layer names kiểu Keras: dense_1, layer_normalization, global_avg_pool, tf_op_layer_lambda) | Table 5, 6, 7 |
| **Framework dùng để code** | **Xem CẦN XÁC NHẬN #1** — paper gốc là Keras nhưng workspace mặc định ưu tiên PyTorch | Rule 2 workspace |
| Input size | 64 × 64 × 3 | Mục IV.B |
| Normalization | Pixel scale về [0, 1] (chia 255). KHÔNG dùng ImageNet mean/std | Mục IV.B |
| Python/GPU gốc | Python 3.6, Nvidia K80 16GB | Mục V |

### MLP-Mixer — kiến trúc (Branch A, Level 1)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Input image | 64 × 64 × 3 | Table 5 |
| Number of patches (N) | 9 (patch grid 3×3, mỗi patch ~21×21 sau khi compress lên 72×72, reshape thành 9 tokens) | Table 5, 7, Fig.3 |
| Patch size | Ảnh được "compress" lên 72×72 rồi reshape thành 9 tokens × 128 dim (xem Table 7 — chi tiết bên dưới) | Table 7 |
| Embedding dim (D) | 128 | Table 5, Mục IV.D.1.a |
| Hidden layer size | 128 | Table 5 |
| Số mixer layers | 7 | Table 5, 7, Mục IV.D.1.a |
| Token mixing MLP | 2 FC layers + GELU + skip connection | Mục IV.D.1.a, Fig.3, Table 7 |
| Channel mixing MLP | 2 FC layers + GELU + skip connection | Mục IV.D.1.a, Fig.3, Table 7 |
| Layer Normalization | Có (trước mỗi MLP) | Mục IV.D.1.a, Fig.3 |
| Activation | GELU (trong MLP), Softmax (output) | Mục IV.D.1.a |
| Dropout | rate = 0.25 (sau mixer layer cuối, trước GAP) | Mục IV.D.1.a |
| Head | LayerNorm → Dropout(0.25) → Global Average Pooling → Dense(num_classes, softmax) | Mục IV.D.1.a, Fig.3 |
| Params (Cotton) | ~915,631 | Table 9 |
| Params (Maize) | ~810,910 | Table 9 |
| Params (TPP) | ~815,431 | Table 9 |

**Chi tiết flow 1 mixer layer (từ Table 7 — đọc trực tiếp từ hình, ĐÃ XÁC NHẬN):**
```
Input (B, 64, 64, 3)
  → _compression (Sequential)  → (B, 72, 72, 3)    [8272 params]
  → reshape_1                  → (B, 9, 128)        [0 params]    ← tạo 9 tokens × 128 dim
  → dropout_1                  → (B, 9, 128)
  --- TOKEN MIXING ---
  → dense_1 (FC)               → (B, 9, 128)        [16512 params]
  → transpose (lambda)         → (B, 128, 9)
  → dense_2 (FC)               → (B, 128, 9)        [90 params]
  → transpose (lambda_1)       → (B, 9, 128)
  → add (skip)                 → (B, 9, 128)
  → layer_normalization        → (B, 9, 128)        [256 params]
  --- CHANNEL MIXING ---
  → dense_3 (FC)               → (B, 9, 128)        [16512 params]
  → dense_4 (FC)               → (B, 9, 128)        [16512 params]
  → add_1 (skip)               → (B, 9, 128)
  → layer_normalization_1      → (B, 9, 128)        [256 params]
  --- HEAD (sau layer cuối) ---
  → global_avg_pool            → (B, 128)
  → dense_5 (softmax)          → (B, num_classes)
```
Note: ~41,600 params / 1 mixer layer × 7 layers ≈ phù hợp tổng ~810K–915K.
**Lưu ý implement:** `_compression` resize 64→72 trước khi chia patch là chi tiết đặc thù của paper; reshape (72,72,3)→(9,128) ngụ ý patch projection. Cần tái hiện đúng để khớp N=9, D=128. Xem CẦN XÁC NHẬN #4.

### LSTM — kiến trúc (Branch B, Level 1)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Input ảnh | 64 × 64 × 3 (resize trước khi đưa vào các CNN) | Mục IV.D.1.b |
| Feature extractors (frozen, ImageNet pretrained) | MobileNet, DenseNet121, DenseNet169, DenseNet201 | Mục IV.D.1.b |
| Cách trích feature | Bỏ FC top → GAP trên output conv base cuối; conv base FREEZE | Mục IV.D.1.b |
| Dim từng feature | MobileNet=1024, DenseNet121=1024, DenseNet201=1536, DenseNet169=2048 | Mục IV.D.1.b |
| Concat feature dim | 5632 (= 1024+1024+1536+2048) | Mục IV.D.1.b |
| Input shape vào LSTM | (1, 5632) — timestep=1, features=5632 | Mục IV.D.1.b |
| Time step | 1 | Table 6, Mục IV.D.1.b |
| Number of cells (units) | 30 | Table 6, Mục IV.D.1.b |
| Head | LSTM(30) → Dense(num_classes, softmax) | Mục IV.D.1.b + Fig.2 (output = probs) |
| Params | < 0.3 triệu (~225,782 Maize / 225,881 Cotton & TPP) | Table 9 |

**Lưu ý:** Thứ tự concat trong Eq.3: `[FMobileNet, FDenseNet121, FDenseNet169, FDenseNet201]` → 1024+1024+2048+1536 = 5632 (cùng tổng, khác thứ tự với phần text liệt kê dim). Giữ tổng = 5632; thứ tự concat không ảnh hưởng kết quả train miễn nhất quán train/test.

### SVM — Level 2 (meta-learner)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Input (meta-feature) | concat[P_Mixer, P_LSTM] = (N, 2*num_classes) | Mục IV.E, Eq.6 |
| Kernel | RBF (best theo grid-search; Table 11 ghi `kernel='rbf'`) | Mục V, Table 11 |
| C (best) | 2 (Table 11) — nhưng grid-search range: [0.1, 1, 10, 100, 1000] | Mục V, Table 11 |
| gamma grid | [1, 0.1, 0.01, 0.001, 0.0001] | Mục V |
| kernel grid | ['linear', 'RBF'] | Mục V |
| Output | class cuối cùng (Eq.7) | Mục IV.E |
| So sánh baseline | SVM > KNN, RF, NB (Table 11) | Mục V |

**Khuyến nghị implement:** dùng `sklearn.svm.SVC(probability=False)` + `GridSearchCV` với param_grid như trên (C, gamma, kernel) để chọn best, hoặc cố định `SVC(C=2, kernel='rbf', gamma='scale')` theo Table 11. Đề xuất: chạy GridSearchCV để bám sát paper.

### Hyperparameters training (Level 1)
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer (Mixer) | Adam, learning rate < 0.001 | Table 5 |
| Loss (Mixer) | sparse/categorical crossentropy (paper ghi "sparse categorical crossentropy") | Table 5 |
| Batch size (Mixer) | 512 | Table 5 |
| Epochs (Mixer) | 50–100 | Table 5 |
| Optimizer (LSTM) | Adam | Table 6 |
| Loss (LSTM) | categorical crossentropy | Table 6 |
| Batch size (LSTM) | 512 | Table 6 |
| Epochs (LSTM) | 50–100 | Table 6 |
| Validation split (LSTM) | 0.1 | Table 6 |
| LR Scheduler | KHÔNG đề cập | — |
| Warmup | KHÔNG đề cập | — |
| Weight decay | KHÔNG đề cập | — |
| Class weighting | KHÔNG đề cập (nhưng workspace Rule 11 yêu cầu auto-detect — xem ghi chú) | — |

### Preprocessing & Normalization
| Bước | Giá trị | Nguồn |
|------|--------|-------|
| Resize | 64 × 64 × 3 | Mục IV.B |
| Normalize | pixel → [0, 1] (chia 255). KHÔNG ImageNet mean/std | Mục IV.B |

**LƯU Ý cho PyTorch:** nếu code PyTorch và dùng CNN pretrained ImageNet (MobileNet/DenseNet) để trích feature, thông thường cần ImageNet normalize. Paper chỉ ghi [0,1]. Xem CẦN XÁC NHẬN #5.

### Augmentation (paper dùng — cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Random flipping | Horizontal flip (không nêu xác suất) | Mục IV.C |
| Random rotation | (không nêu góc) | Mục IV.C |
| Resizing / Zoom | width-shift & zooming (không nêu range) | Mục IV.C |
| (Affine transforms tổng quát) | — | Mục IV.C |

→ Map sang 4 phép bắt buộc của Scenario 2 (Rule 3): Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness. (ColorJitter/Brightness KHÔNG được paper nêu rõ nhưng là yêu cầu cứng của workspace cho Scenario 2.)

### GitHub Source
**Link:** Không tìm thấy. (Data Availability: "data will be accessible upon request"; không có code repo.)

### Datasets
| Dataset | #Classes | Total | Train (80%) | Test (20%) | Nguồn |
|---------|---------|-------|-------------|-----------|-------|
| Cotton | 4 | 1518 | 1214 | 304 | Table 1, 4 |
| Maize | 4 | 2529 | 2023 | 506 | Table 2, 4 |
| TPP | 15 | 20637 | 16573 | 4064 | Table 3, 4 |

**Cotton classes:** Fusarium Wilt, Curl Virus, Healthy, Bacterial Blight.
**Maize classes:** Common Rust (1192), Blight (870), Healthy (21), Gray Leaf Spot (446). ⚠️ Healthy chỉ 21 ảnh → IMBALANCE NẶNG (1192/21 ≈ 57x). Rule 11 sẽ auto bật class weighting.
**TPP classes:** 15 lớp (3 potato, 2 pepper, 10 tomato). ⚠️ Bảng trích từ hình có 1 dòng class chưa đọc rõ (xem CẦN XÁC NHẬN #6). TY (Tomato Yellow Leaf Curl) = 5357 vs Potato Healthy = 152 → imbalance lớn.

### Kết quả trong paper (tham khảo)
| Dataset | Accuracy | Notes |
|---------|---------|-------|
| Cotton | 98.43% | AUC 0.98 |
| Maize | 94.27% | AUC 0.995 |
| TPP | 97.45% | AUC 0.9993 |
| Meta-ensemble params | ~0.92M–1.04M | Table 9 |
| Memory | 18.02 KB | Mục V |

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG rõ ràng / mâu thuẫn / thiếu trong bài báo. Vui lòng trả lời trước khi viết notebook.

1. **Framework:** Paper gốc viết bằng **TensorFlow/Keras** (Table 5/6/7 dùng tên layer Keras và loss "categorical crossentropy"). Tuy nhiên rule workspace mặc định ưu tiên PyTorch. Bạn muốn:
   - (a) Dùng **PyTorch** (theo default workspace), hay
   - (b) Dùng **TensorFlow/Keras** (bám sát đúng paper gốc)?

2. **Learning rate cụ thể:** Paper chỉ ghi "Adam, learning rate < 0.001". Bạn muốn dùng giá trị nào cụ thể? (gợi ý: 1e-3 hoặc 5e-4 cho Adam)

3. **Số epochs cụ thể:** Paper ghi range "50–100" cho cả Mixer và LSTM. Bạn muốn chốt bao nhiêu epochs? (gợi ý: 50, hoặc 100 nếu muốn bám max; có thể dùng EarlyStopping)

4. **Cơ chế patch của MLP-Mixer:** Table 7 cho thấy ảnh 64×64 được "_compression" lên 72×72 rồi reshape thành (9, 128). Cơ chế chính xác của bước `_compression` (8272 params) không được mô tả bằng text — có vẻ là một lớp Dense/projection để tạo 9 patch tokens 128-dim. Bạn xác nhận tôi được phép **tự thiết kế lớp patch-embedding tạo ra đúng N=9 tokens × D=128** (ví dụ: chia ảnh thành grid 3×3 patches rồi linear-project mỗi patch → 128), miễn giữ đúng N=9, D=128? (Nếu không, đây là điểm mơ hồ không thể tái hiện chính xác 1:1.)

5. **Normalization khi trích feature từ CNN pretrained:** Paper chỉ ghi normalize pixel về [0,1]. Nhưng MobileNet/DenseNet pretrained ImageNet thường cần ImageNet mean/std (hoặc preprocess_input riêng của Keras). Bạn muốn:
   - (a) Bám đúng paper: chỉ scale [0,1] cho cả CNN feature extractor, hay
   - (b) Dùng preprocess chuẩn ImageNet cho nhánh CNN (thường cho feature tốt hơn)?

6. **Lớp thứ 15 của TPP & label mapping:** Bảng TPP (Table 3) trích từ hình có 1 dòng class chưa đọc rõ (`~622`) và tổng = 20637 với 15 classes. Khi có dataset thực tế trên Kaggle, `class_names` sẽ **auto-detect từ ImageFolder** (Rule 12) nên không cần con số chính xác từ paper — nhưng bạn xác nhận TPP dataset bạn cung cấp có **đúng 15 classes** (paper) hay khác? (Table 12 lại ghi TPP "10 classes" — có mâu thuẫn trong chính paper.)

7. **Dataset trên Kaggle:** Paper không công bố link dataset (data on request). Bạn sẽ cung cấp dataset Cotton / Maize / TPP nào trên Kaggle? Cần đường dẫn dataset Kaggle để viết phần load data. (Notebook nên viết cho **1 dataset/lần chạy**, hay tham số hóa để chạy được cả 3?)

8. **SVM probability cho ROC:** Paper vẽ ROC/AUC (cần probability). Nếu dùng `SVC`, cần `probability=True` (chậm hơn) để lấy AUC. Bạn có cần vẽ ROC không, hay chỉ cần Accuracy/F1/Recall/Confusion Matrix (theo Rule 4 workspace)?

---

## GHI CHÚ IMPLEMENTATION

- **Đây là stacking ensemble, KHÔNG end-to-end.** Cần train Mixer và LSTM riêng, rồi mới train SVM trên prediction-probs của chúng. Cấu trúc notebook nên có các cell rõ ràng: (1) Load+split, (2) Transforms baseline/aug, (3) MLP-Mixer model + train, (4) CNN feature extraction + LSTM model + train, (5) Lấy probs train-set → train SVM (GridSearch), (6) Eval test-set qua toàn pipeline, (7) Metrics + plots. Lặp lại (3)–(7) cho Scenario 1 và Scenario 2.

- **Risk khớp params:** Để khớp ~810K–915K params cho Mixer cần đúng D=128, 7 layers, hidden=128. Bước `_compression`→72×72→(9,128) là nguồn rủi ro mismatch chính (xem CẦN XÁC NHẬN #4). Dùng `torchinfo.summary()` / `model.summary()` để verify số params gần với paper.

- **Imbalance:** Maize-Healthy chỉ 21 ảnh (split 80/20 → ~17 train / 4 test). TPP cũng lệch mạnh (Potato Healthy 152 vs TY 5357). Rule 11 workspace tự bật `class_weights` khi max/min ≥ 1.5 → sẽ luôn bật cho cả 3 dataset. SVM ở Level 2 nên cân nhắc `class_weight='balanced'`.

- **CNN feature extractor (LSTM branch):** 4 mạng pretrained chạy inference 1 lần để trích feature (frozen). Nên cache feature (5632-d) ra numpy để tránh trích lại mỗi epoch của LSTM (LSTM train trên feature tĩnh, không phải ảnh). Với augmentation (Scenario 2), feature phải trích lại trên ảnh đã augment — cân nhắc trích feature trên tập ảnh đã augment-expand offline, hoặc trích on-the-fly.

- **Metrics bắt buộc (Rule 4):** Total Params, Accuracy, F1 macro, Recall macro, và `classification_report` per-class. Confusion Matrix + Learning Curve lưu PNG dpi=300, mỗi plot 1 cell riêng + `plt.show()` (Rule 6).

- **Best checkpoint (Rule 10):** lưu best model theo val_accuracy cho cả Mixer và LSTM (`best_model_s1.pth`/`best_model_s2.pth` hoặc tương đương). SVM lưu bằng `joblib`.

- **Seed + device + cudnn deterministic** (Rule 7, 8, 9): đặt đầu notebook.

- **Loss "sparse categorical crossentropy" trong Table 5** thực ra là dòng ghi nhầm vào ô Optimizer trong bảng gốc; hiểu đúng: Optimizer=Adam, Loss=(sparse) categorical crossentropy. Trong PyTorch dùng `nn.CrossEntropyLoss` (tương đương sparse CCE với label dạng int).
