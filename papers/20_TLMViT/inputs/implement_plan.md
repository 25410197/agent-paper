# IMPLEMENT PLAN: TLMViT (VGG19 + ViT, primary)

**Paper:** A Deep Features Extraction Model Based on the Transfer Learning Model and Vision Transformer "TLMViT" for Plant Disease Classification
**Tác giả:** Amer Tabbakh, Soubhagya Sankar Barpanda (VIT-AP University)
**Journal/Conference:** IEEE Access, Vol. 11, 2023 (pp. 45377–45392)
**DOI:** 10.1109/ACCESS.2023.3273317
**Ngày phân tích:** 2026-10-20

---

## PHÂN LOẠI
**Nhóm:** A
**Lý do:** Paper đề xuất mô hình lai cụ thể (backbone pretrained đóng băng + ViT + MLP head) với đầy đủ thông số cốt lõi: input size, patch size, projection dim, số head, số encoder layer, MLP units, optimizer, lr, batch size, epochs, loss. Chỉ thiếu một số chi tiết phụ (activation, dropout, cách đọc output encoder, normalization) — liệt kê ở mục CẦN XÁC NHẬN.

---

## LỰA CHỌN MODEL CHÍNH ĐỂ PHỤC DỰNG

**Model chính: VGG19 + ViT (TLMViT-VGG19)** — cấu hình có validation accuracy cao nhất trên cả 2 dataset của paper:
- PlantVillage (15 lớp): val acc **98.81%**, val loss **0.0947**, train acc 98.8%, train loss 0.1 (Table 6, Abstract, Sec. VII-A).
- Wheat Rust (3 lớp): val acc **99.86%**, val loss **0.0715**, train acc 99.67% (Table 7, Abstract, Conclusion).

Lý do không chọn VGG16: VGG16+ViT chỉ tốt nhất ở train acc (99.60%), còn val acc thấp hơn (98.43%). Các backbone khác (ResNet50, AlexNet, InceptionV3) có thể làm biến thể tùy chọn — xem câu hỏi.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | TLMViT (Transfer Learning Model + Vision Transformer) | Tiêu đề, Abstract |
| Loại | Hybrid: CNN pretrained (frozen) → ViT encoder → MLP head | Sec. V, Fig. 2 |
| Framework | Keras / TensorFlow (Python 3.9.12, CUDA 11.2, GTX 1080 Ti, Spyder) | Sec. VI |
| Input size | 256×256×3 (VGG16/VGG19/ResNet50/AlexNet); InceptionV3: 299×299×3 | Sec. V-B1 (c, d), Table 4 |
| Pretrained backbone | Có — VGG19, ImageNet weights, bỏ 3 lớp FC (include_top=False), **frozen** (không train) | Sec. V-B1-c, Sec. VII-A ("frozen weights ... no training is required in the initial features extraction phase") |
| Output backbone (Initial Features) | 8×8×512 cho VGG19 với input 256 | Table 4, Sec. V-B1-c, Sec. V-B2 |
| Wheat Rust | ảnh rescale về cùng kích thước PlantVillage (256×256) | Sec. V-B1-d (đoạn cuối) |

### Cấu trúc Layer (theo thứ tự) — VGG19 + ViT
| Layer / Block | Kernel | Filters / Units | Stride | Activation | Ghi chú |
|--------------|--------|---------|--------|-----------|---------|
| Input | – | 256×256×3 | – | – | Sec. V-B1-c |
| VGG19 Block1: 2× Conv | 3×3 | 64 | 1 | ReLU (chuẩn VGG) | → 256×256×64 (Table 4); frozen |
| MaxPool | 2×2 (chuẩn VGG) | – | 2 | – | → 128×128×64 |
| Block2: 2× Conv | 3×3 | 128 | 1 | ReLU | → 128×128×128 |
| MaxPool | – | – | 2 | – | → 64×64×128 |
| Block3: 4× Conv | 3×3 | 256 | 1 | ReLU | → 64×64×256 |
| MaxPool | – | – | 2 | – | → 32×32×256 |
| Block4: 4× Conv | 3×3 | 512 | 1 | ReLU | → 32×32×512 |
| MaxPool | – | – | 2 | – | → 16×16×512 |
| Block5: 4× Conv | 3×3 | 512 | 1 | ReLU | → 16×16×512 |
| MaxPool (block5_pool) | – | – | 2 | – | → **8×8×512** = Initial Features (Table 4) |
| Patch partition | 4×4 non-overlapping | 4 patches, mỗi patch 4×4×512 | 4 | – | Eq. (1), Sec. VI ("patch size ... four") |
| Flatten patch | – | vector S = 4·4·512 = 8192 | – | – | Sec. V-B2 |
| Linear projection | Dense | 64 | – | không nêu | Eq. (2) "Dense64" |
| Position embedding | learned, map số nguyên vị trí → vector 64 | 64 | – | – | Eq. (3); Fig. 2 đánh số vị trí 0..4 |
| Class token Z0 | ghép vào đầu chuỗi, dim 64 | → chuỗi 5 token × 64 | – | – | Eq. (4), Fig. 2 |
| Transformer encoder ×8 (N = 8) | Pre-norm: X = MHSA(Norm(Z)) + Z; X = MLP(Norm(X)) + X | MHSA heads H = 4; MLP 2 lớp ẩn 128 → 64 | – | MLP activation không nêu | Eq. (5)-(6), Fig. 5, Table 5 |
| MLP head | input 256 → Dense 1024 → Dense 512 → Dense num_classes | 1024, 512 | – | hidden: không nêu; output: softmax (suy từ categorical CE — xem câu hỏi) | Sec. V-C, Eq. (7), Table 5 |

"Norm" = normalization (LayerNorm theo Table 5 transcription / Fig. 5).

### Hyperparameters
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | Adam (text). Table 5 ghi "RMSprop  adam" — mâu thuẫn, text nói rõ dùng Adam | Sec. VI, Table 5 |
| Learning rate | 1e-3 | Table 5 |
| LR Scheduler | Không đề cập | – |
| Warmup epochs | Không đề cập | – |
| Batch size | 32 | Table 5 |
| Epochs | 25 (cố định để so sánh công bằng) | Sec. VI, Table 5 |
| Loss function | Categorical cross-entropy | Sec. VI |
| Weight decay | Không đề cập | – |
| Normalization | Không đề cập (chỉ nói "rescale") | Sec. V-B1-d |
| Dataset split (paper) | 80% train / 20% val+test (dùng chung, ~4100 ảnh) | Sec. VI, Table 5 |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| Rotation | random [-25°, 25°] | Table 5 |
| Width shift | 10% | Table 5 |
| Height shift | 10% | Table 5 |
| Shear | 20% | Table 5 |
| Horizontal flip | True | Table 5 |
| Vertical flip | True | Table 5 |
| Zoom | 20% | Table 5 |

Ghi chú: paper augment offline 16,511 → 33,257 ảnh train (Sec. V-A1). Theo project rule, Scenario 2 chỉ dùng 4 phép: Rotation, Zoom, Horizontal Flip, Brightness/ColorJitter. Đề xuất lấy range từ paper: rotation ±25° (RandomRotation factor = 25/360), zoom 20%, horizontal flip; brightness không có trong paper → cần chọn range (xem câu hỏi).

### GitHub Source
**Link:** Không tìm thấy (paper không có Code Availability / link repo).
Dataset gốc: https://www.kaggle.com/datasets/emmarex/plantdisease (ref. [51]); Wheat Rust theo ref. [42].

### Kết quả trong paper (tham khảo)
| Dataset | Model | Train Acc | Val Acc | Val Loss | F1 / Recall | Params |
|---------|------|---------|---------|---------|-----|--------|
| PlantVillage (15 lớp) | VGG19+ViT (chính) | 98.8% | **98.81%** | 0.0947 | Table 8 (không trích xuất được số) | Không báo cáo |
| PlantVillage | VGG16+ViT | 99.60% | 98.43% | 0.1300 | – | – |
| PlantVillage | InceptionV3+ViT | 99.2% | 98.48% | 0.1 | – | – |
| PlantVillage | AlexNet+ViT | 92.5% | 90.49% | 0.2974 | – | – |
| PlantVillage | ResNet50+ViT | 90.405% | 89.2% | 0.3735 | – | – |
| Wheat Rust (3 lớp) | VGG19+ViT (chính) | 99.67% | **99.86%** | 0.0715 | Table 9 (không trích xuất được số) | – |
| Wheat Rust | VGG16 / Incep / AlexNet / ResNet50 + ViT | 99.93 / 99.81 / 95.48 / 93.26% | 99.44 / 99.69 / 93.39 / 91.74% | – | – | – |
| PlantVillage | TL-only baselines (ResNet50/AlexNet/VGG16/VGG19/Incep) | 80.43/83.9/90.44/88.70/92.78% | 79.84/81.91/89.31/86.55/90.77% | – | – | – |

Lưu ý: "validation" của paper = tập test chung (20%), không có test riêng.

---

## MÂU THUẪN TRONG PAPER & CÁCH XỬ LÝ

1. **Optimizer (Table 5 "RMSprop adam" vs text "Adam")** → Dùng Adam theo text Sec. VI (text giải thích lý do chọn Adam). Ghi chú trong notebook.
2. **Table 3 AlexNet: kích thước feature map không khớp stride** (227→55 hợp lệ với padding=0, nhưng 256→64 với stride 4 kernel 11 không khớp valid-padding; 27→27 sau conv2 v.v.) → Không ảnh hưởng model chính (VGG19). Chỉ liên quan nếu làm biến thể AlexNet.
3. **InceptionV3: 8×8×2048 (Sec. V-B1-d, input 299) vs R^{6×6×2048} (Sec. V-B2)** → Keras InceptionV3 include_top=False với input 299 cho 8×8×2048, khớp text Sec. V-B1-d và patch 4×4. Chỉ liên quan nếu làm biến thể Inception.
4. **Không có test split riêng (paper: 80/20, val = test)** → Dataset người dùng đã chia sẵn train/val/test: dùng `validation_data=val`, checkpoint theo val_accuracy, đánh giá cuối trên test. KHÔNG dùng train_test_split.
5. **AlexNet không có ImageNet weights chính thức trong Keras** → Chỉ liên quan nếu làm biến thể AlexNet (phải port weights từ torchvision hoặc bỏ qua).
6. **MLP head "input layer 256 neurons = dimension of deep feature vector Xi(8)"** — mâu thuẫn với encoder output 5 token × 64 = 320 (có class token) hoặc class token = 64. 256 = 4 patch token × 64 → khả năng cao là flatten 4 patch tokens (bỏ class token), nhưng paper không nói rõ → hỏi người dùng.

---

## CẦN XÁC NHẬN

> Các thông số sau KHÔNG có (hoặc mơ hồ) trong bài báo. Vui lòng trả lời trước khi viết notebook. Mỗi câu có kèm đề xuất mặc định.

## CÂU HỎI CHO NGƯỜI DÙNG

1. **Backbone:** Chỉ phục dựng VGG19+ViT (cấu hình tốt nhất của paper), hay chạy thêm các backbone khác (VGG16, ResNet50, InceptionV3; AlexNet không có weights Keras chính thức)? — *Đề xuất: chỉ VGG19+ViT.*
2. **Dataset:** Đường dẫn Kaggle dataset của bạn (thư mục train/val/test đã chia sẵn) là gì? — *Số lớp sẽ auto-detect.*
3. **Đầu vào MLP head (mơ hồ):** Paper nói input của MLP head có 256 neurons, nhưng encoder ra 5 token × 64 (= 320 nếu flatten hết, = 64 nếu chỉ lấy class token). Bạn muốn dùng cách nào? (a) Flatten 4 patch token (bỏ class token) = 256 — khớp con số 256 của paper; (b) chỉ lấy class token (64) — chuẩn ViT; (c) flatten cả 5 token (320). — *Đề xuất: (a), có LayerNorm trước khi flatten.*
4. **Activation trong MLP của encoder (128→64) và MLP head (1024, 512):** Paper không nêu. — *Đề xuất: GELU trong encoder MLP (chuẩn ViT); MLP head dùng GELU hoặc ReLU — đề xuất GELU cho nhất quán. Output layer softmax.*
5. **Dropout:** Paper không đề cập dropout ở đâu cả. Có thêm dropout không? — *Đề xuất: không thêm (bám sát paper, dropout = 0).*
6. **key_dim của MultiHeadAttention (4 heads):** Không nêu. — *Đề xuất: key_dim = 64 // 4 = 16 (chia đều projection dim cho các head).*
7. **Normalization/preprocessing đầu vào:** Paper chỉ nói "rescale", không nêu mean/std. — *Đề xuất: dùng `keras.applications.vgg19.preprocess_input` (chuẩn ImageNet caffe-style, khớp với weights pretrained) cho cả 2 scenario.*
8. **Brightness range cho Scenario 2:** Paper không dùng brightness (dùng shift/shear/vflip thay thế). — *Đề xuất: RandomBrightness factor 0.2; rotation ±25°, zoom 20%, horizontal flip lấy từ Table 5.*
9. **Fine-tune backbone:** Paper đóng băng hoàn toàn VGG19. Giữ nguyên frozen? — *Đề xuất: giữ frozen (bám sát paper).*
10. **LayerNorm epsilon:** Không nêu. — *Đề xuất: 1e-6 (mặc định ViT) — hoặc để mặc định Keras 1e-3; có thể dùng mặc định không cần hỏi lại nếu bạn không quan tâm.*

---

## GHI CHÚ IMPLEMENTATION

- **Framework:** TensorFlow/Keras (Keras 3 trên Kaggle, Python 3.12). KHÔNG pin `tensorflow==2.13.0`; KHÔNG dùng `workers`/`use_multiprocessing` trong `fit()`. Dùng `tf.data` + `.prefetch(tf.data.AUTOTUNE)`.
- **Multi-GPU:** `tf.distribute.MirroredStrategy()`; tạo model + `compile()` trong `strategy.scope()`. Batch size 32 (paper) — quyết định giữ global=32 hay 32×n_gpus; đề xuất GLOBAL_BATCH_SIZE = 32 để bám paper.
- **Backbone:** `keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(256,256,3))`, `trainable=False`. Output `block5_pool` = 8×8×512.
- **Patch partition (non-overlapping 4×4 trên map 8×8):** dùng `keras.ops.reshape`/`transpose` (8×8×512 → 2×4×2×4×512 → 4 × 8192) để tránh phụ thuộc `tf.image.extract_patches`; thứ tự patch row-major.
- **Class token:** trọng số learnable shape (1,1,64), tile theo batch, concat ở đầu → 5 token. Position embedding: `layers.Embedding(input_dim=5, output_dim=64)` trên `range(5)` (paper: "mapping integer numbers ... into vectors with size 64").
- **Encoder block (×8, pre-norm, Eq. 5-6):** LN → MHA(4 heads) → +residual → LN → Dense128 → Dense64 → +residual.
- **Cấu trúc rất giống Keras example "Image classification with Vision Transformer"** (projection_dim 64, 4 heads, 8 layers, transformer_units [128, 64]) — có thể tham khảo cách code, nhưng KHÔNG copy các giá trị không có trong paper (dropout 0.1/0.5, AdamW, weight decay, mlp_head_units [2048,1024]) — paper dùng head [1024, 512] và Adam.
- **Training:** Adam lr 1e-3, categorical CE (label one-hot), 25 epochs, batch 32, không scheduler. `ModelCheckpoint(monitor="val_accuracy", save_best_only=True)` → `best_model_s1.keras` / `best_model_s2.keras`; reload best trước khi evaluate test.
- **Dual scenario:** S1 baseline = Resize 256×256 + preprocess_input; S2 = baseline + RandomRotation(±25°) + RandomZoom(0.2) + RandomFlip("horizontal") + RandomBrightness — augmentation chỉ áp dụng cho train, đặt TRƯỚC `preprocess_input`.
- **Class imbalance:** tính phân phối; nếu max/min ≥ 1.5 → `compute_class_weight` → `class_weight=` trong `fit()`. (PlantVillage subset của paper lệch mạnh: Potato healthy 152 vs TYLCV 3208.)
- **Metrics/Artifacts:** `model.summary()` (Total/Trainable/Non-trainable — đa số params của VGG19 ~20M sẽ là non-trainable), Accuracy, F1-macro, Recall-macro, `classification_report(target_names=class_names)`; learning curve + confusion matrix dpi=300, mỗi biểu đồ 1 cell, `plt.show()`. `tf.keras.backend.clear_session(); gc.collect()` giữa S1 và S2. Seed `tf.random.set_seed(42)`.
- **Rủi ro mismatch:** (1) Paper augment offline gấp đôi tập train, còn notebook augment online → số bước/epoch khác; (2) paper báo cáo val = test; (3) cách đọc output encoder (câu hỏi 3) ảnh hưởng số params head; (4) Linear projection 8192→64 chiếm ~524K params — phần lớn trainable params nằm ở đây và MLP head.
