# IMPLEMENT PLAN: FOTCA (AFNO-based Transformer + CNN-shortcut Classifier)

**Paper:** FOTCA: hybrid transformer-CNN architecture using AFNO for accurate plant leaf disease image recognition
**Tác giả:** Bo Hu, Wenqian Jiang, Juan Zeng, Chen Cheng, Laichang He
**Journal/Conference:** Frontiers in Plant Science, 14:1231903, 2023 (published 12/09/2023)
**DOI:** 10.3389/fpls.2023.1231903
**Ngày phân tích:** 2026-10-20

---

## PHÂN LOẠI
**Nhóm:** C (gần A — có model, có bảng hyperparameter, nhưng thiếu nhiều thông số cốt lõi)
**Lý do:** Paper đề xuất model cụ thể (FOTCA) với Table 1 (LR, epochs, batch, input, embed dim, patch, heads, depth) và framework PyTorch rõ ràng, nhưng thiếu ≥ 3 thông số cốt lõi để code chính xác: chi tiết AFNO (num_blocks, sparsity/hard-thresholding, MLP ratio), độ rộng kênh của CNN classifier và cách nối token → classifier, SGD momentum/weight decay, focal gamma, dropout rate, lambda của LambdaLR, cách nạp ImageNet weights vào FOTCA. Figure 2 chỉ là flowchart khái niệm, không vẽ kích thước.

---

## THÔNG SỐ ĐÃ XÁC NHẬN

### Kiến trúc
| Thông số | Giá trị | Nguồn trong paper |
|---------|--------|------------------|
| Tên model | FOTCA (F,O = Adaptive Fourier Neural Operator; TCA = Transformer-CNN Architecture) | Tiêu đề, Mục 1 |
| Loại | Hybrid Transformer (AFNO token mixer) + CNN classifier | Abstract, Mục 3.2 |
| Framework | PyTorch 1.11.0 (Python 3.8, CUDA 11.3, 1×V100-32GB) | Mục 4.1 |
| Input size | 224×224×3 (RGB) | Mục 3.1, Table 1 |
| Pretrained | Có — "all the experiments ... are based on the pre-trained models of the ImageNet dataset" (transfer learning, Pre-train + Fine-tuning) | Mục 3.2, 4.2 |
| Embedding dimension D | 768 | Table 1 |
| Patch size P | 16 → lưới 14×14 = 196 patch tokens | Table 1 (196 = suy ra từ 224/16, phép tính) |
| Head number | 8 | Table 1 (với AFNO thường tương ứng num_blocks — xem CẦN XÁC NHẬN #4) |
| Depth L | 12 AFNO layers | Table 1, Fig. 2A ("L Layers") |
| Params tổng (tham chiếu) | 59.14M | Table 2 |
| FLOPs (tham chiếu) | 11.87G | Table 2 |

### Cấu trúc Layer (theo thứ tự — theo Fig. 2A + Mục 3.2)
| Layer / Block | Kernel | Filters / Dim | Stride | Activation | Ghi chú |
|--------------|--------|---------|--------|-----------|---------|
| Input | — | 3 kênh, 224×224 | — | — | |
| Patch Embedding: FFT → Conv2d → IFFT | Conv2d: **không ghi** (ViT chuẩn: 16×16) | D = 768 | **không ghi** (ViT chuẩn: 16) | — | Fig. 2A: "Patch embedding and flatten patches": FFT → Conv2d → IFFT, rồi flatten. Eq.(3): x_patch = W_patch · v_ij |
| Positional encoding (learnable Fourier features) | — | output 2K, cộng vào patch embedding | — | cos/sin | Eq.(4): f_pos(p) = [cos(⟨p,w1⟩+b1), sin(⟨p,w1⟩+b1), …, cos(⟨p,wK⟩+bK), sin(⟨p,wK⟩+bK)], p = ((i−1)/H, (j−1)/W) chuẩn hóa; w_k, b_k learnable. Eq.(5): x_pos = x_patch + f_pos → yêu cầu 2K = D (=768, nếu cộng trực tiếp) |
| AFNO layer × 12 | — | 768 | — | — | Mỗi layer: **Spatial Mixing** (FFT → Multiply Frequency → IFFT, Fig. 2A) → **Channel Mixing** (MLP, Eq.11). Norm/residual không được vẽ/ghi |
| Class token | — | — | — | — | Fig. 2A: "Class token" → "Classifier". Không rõ class token được thêm ở đâu và cách đưa vào CNN classifier |
| Classifier: basic block CNN shortcut | 3×3 → 3×3; shortcut 1×1 | **không ghi** | **không ghi** | ReLU | Eq.(13): x1 = ReLU(Conv3×3(x)); Eq.(14): x2 = ReLU(Conv3×3(x1)); Eq.(15): x_out = ReLU(x2 + ReLU(Conv1×1(x))). Mục 3.2.3 có nhắc BatchNorm (vị trí không ghi). Shortcut 1×1 để khớp số kênh; text nói "upsampling to a higher dimension" và "downsampling" |
| Dropout | — | — | — | — | Có dùng (Mục 4.2), rate & vị trí không ghi |
| Head cuối (→ num_classes) | — | **không ghi** | — | — | Paper không mô tả pooling/FC sau CNN block |

### Hyperparameters
| Thông số | Giá trị | Nguồn |
|---------|--------|-------|
| Optimizer | SGD | Mục 4.2 |
| Learning rate | 0.001 | Table 1 |
| LR Scheduler | LambdaLR, linear decay theo số lần cập nhật LR | Mục 4.2 (lambda chính xác không ghi) |
| Warmup epochs | Không đề cập | — |
| Batch size | 8 | Table 1 |
| Epochs | 100 | Table 1 |
| Loss function | Focal loss: Loss_fl = −(1−p_t)^γ log(p_t) (FOTCA); CE cho các model so sánh | Eq.(2), Mục 4.2 |
| Weight decay | Không đề cập | — |
| Momentum | Không đề cập | — |
| Dropout | Có dùng, rate không ghi | Mục 4.2 |
| Normalization | Không ghi mean/std (paper dùng ImageNet pretrained → ImageNet mean/std là hợp lý, xem CẦN XÁC NHẬN) | — |
| Class weighting | Không đề cập (áp dụng rule 11 tự động) | — |

### Augmentation (paper dùng — tham khảo cho Scenario 2)
| Kỹ thuật | Tham số | Nguồn |
|---------|--------|-------|
| RandomFlip | không ghi | Mục 3.1 |
| RandomCrop | không ghi | Mục 3.1 |
| RandomHorizontalFlip | không ghi | Mục 3.1 |
| RandomResizeCrop | không ghi | Mục 3.1 |
| Resize | 224×224 | Mục 3.1 |

> Theo rule 01 #3: Scenario 1 = Resize + ToTensor + Normalize; Scenario 2 = Rotation, Zoom/Scale, HorizontalFlip, ColorJitter. Aug của paper chỉ để tham khảo.

### Dataset (paper — chỉ tham khảo)
PlantVillage, 54,303 ảnh, 38 lớp (13 loài), có chèn thêm ảnh non-plant-leaf; train:val = 8:2, không có test set riêng. Notebook dùng dataset của người dùng (đã chia sẵn train/val/test — không dùng train_test_split).

### GitHub Source
**Link:** Không tìm thấy (Data availability statement chỉ ghi "included in the article/supplementary material").

### Kết quả trong paper (tham khảo)
| Dataset | Accuracy | F1 | Loss | Params | FLOPs |
|---------|---------|-----|------|--------|-------|
| PlantVillage (val 20%) | 99.8% | 99.31% | 0.005 | 59.14M | 11.87G |

Ablation (Table 3): Pap Embedding gốc 99.1% / F1 96.31; MLP-Classifier 96.1% / F1 85.91; không augmentation 99.7% / F1 98.94.
Baseline ViT (Table 2): 99.4% / F1 98.03 / 58.07M.

---

## CẦN XÁC NHẬN

> ⚠️ Các thông số sau KHÔNG có trong bài báo. Vui lòng trả lời trước khi tôi viết notebook. Giá trị gợi ý đều là **đề xuất, không phải từ paper** (chủ yếu lấy theo AFNO gốc của Guibas et al. 2021 / code FourCastNet / timm).

**Optimizer & training**
1. SGD momentum không được đề cập. Bạn muốn dùng giá trị nào? (đề xuất, không phải từ paper: momentum = 0.9)
2. Weight decay của SGD không được đề cập. Bạn muốn dùng bao nhiêu? (đề xuất, không phải từ paper: 1e-4; hoặc 0 nếu muốn bám sát "không ghi")
3. Hàm lambda của LambdaLR chỉ được mô tả là "linear decay". Bạn muốn công thức nào? (đề xuất, không phải từ paper: lr_lambda = lambda epoch: 1 − epoch/EPOCHS, cập nhật mỗi epoch, decay tuyến tính từ 0.001 về 0; không warmup)
4. Focal loss: tham số γ (gamma) không được ghi, và paper không dùng α. Bạn muốn γ bao nhiêu và có dùng α không? (đề xuất, không phải từ paper: γ = 2.0, không dùng α — đúng như Eq.(2) chỉ có γ; nếu dataset mất cân bằng thì truyền class_weights theo rule 11 làm trọng số)
5. Dropout rate và vị trí dropout không được ghi. Bạn muốn dùng bao nhiêu và đặt ở đâu? (đề xuất, không phải từ paper: drop_rate = 0.1 cho MLP trong AFNO + Dropout 0.1 trước lớp FC cuối; drop_path = 0.0)

**AFNO block**
6. Số block-diagonal (num_blocks) của AFNO không được ghi. Table 1 ghi "Head number = 8" — bạn có đồng ý coi đây là num_blocks của AFNO (768/8 = 96 kênh mỗi block) không? (đề xuất, không phải từ paper: num_blocks = 8)
7. Ngưỡng soft-shrinkage (sparsity_threshold) của AFNO không được ghi. Bạn muốn dùng bao nhiêu? (đề xuất, không phải từ paper: 0.01 như code AFNO/FourCastNet gốc)
8. Tỉ lệ giữ mode tần số (hard_thresholding_fraction) không được ghi. Bạn muốn giữ bao nhiêu? (đề xuất, không phải từ paper: 1.0 — giữ toàn bộ mode)
9. MLP ratio của channel mixing (Eq.11) không được ghi. Bạn muốn dùng bao nhiêu? (đề xuất, không phải từ paper: 4.0 → hidden 3072; với 12 layer cho tổng params ≈ 59M, khớp gần với 59.14M ở Table 2)
10. Normalization/residual trong mỗi AFNO layer không được vẽ. Bạn muốn dùng cấu trúc chuẩn của AFNO gốc không? (đề xuất, không phải từ paper: x = x + AFNO(LayerNorm(x)); x = x + MLP(LayerNorm(x)), activation GELU, LayerNorm eps=1e-6)

**Patch embedding & positional encoding**
11. Patch embedding "FFT → Conv2d → IFFT": paper không ghi kernel/stride của Conv2d, cũng không ghi FFT áp lên ảnh trước hay sau, và xử lý phần phức (complex) thế nào. Bạn muốn implement thế nào? (đề xuất, không phải từ paper: rfft2 trên ảnh đầu vào → tách real/imag thành 6 kênh → Conv2d(kernel=16, stride=16, out=768) → coi output như tín hiệu tần số và áp irfft2 trên lưới 14×14 → flatten 196 tokens. Phương án đơn giản hơn: chỉ dùng Conv2d(3→768, k=16, s=16) chuẩn ViT và bỏ FFT/IFFT)
12. Learnable Fourier features: số hàm cơ sở K và chiều đầu ra không ghi. Vì Eq.(5) cộng trực tiếp f_pos (dim 2K) vào x_patch (dim D=768), bạn có đồng ý K = 384 (2K = 768) không? Có cần thêm MLP sau Fourier features như bài "Learnable Fourier Features" (Li et al. 2021) không? (đề xuất, không phải từ paper: K = 384, khởi tạo w_k ~ N(0, 1), b_k = 0, KHÔNG thêm MLP — đúng theo Eq.(4)-(5))

**Class token & Classifier**
13. Fig. 2A ghi "Class token → Classifier" nhưng classifier là CNN (Conv3×3) cần feature map không gian. Bạn muốn nối thế nào? (đề xuất, không phải từ paper: bỏ class token khỏi phần CNN, reshape 196 patch tokens → feature map 768×14×14 đưa vào CNN classifier; hoặc phương án B: thêm class token, sau đó concat/cộng class token vào vector pooled trước FC)
14. Số kênh (filters) của 2 lớp Conv3×3 và Conv1×1 shortcut trong classifier không được ghi. Bạn muốn dùng bao nhiêu? (đề xuất, không phải từ paper: 768 → 1024 → 1024, shortcut Conv1×1 768→1024 — "upsampling to a higher dimension")
15. Stride của Conv3×3 đầu và Conv1×1 shortcut không ghi (paper nhắc cả "downsampling" lẫn "upsampling"). Bạn muốn dùng stride bao nhiêu? (đề xuất, không phải từ paper: stride = 1, padding = 1 cho 3×3, giữ 14×14)
16. Vị trí BatchNorm trong classifier không ghi (Eq.13–15 không có BN). Bạn muốn đặt BN ở đâu? (đề xuất, không phải từ paper: Conv → BN → ReLU cho cả 2 conv 3×3 và nhánh shortcut, giống ResNet BasicBlock)
17. Phần đầu ra sau CNN block (pooling + FC) không được mô tả. Bạn muốn dùng gì? (đề xuất, không phải từ paper: AdaptiveAvgPool2d(1) → Flatten → Dropout → Linear(1024, num_classes))

**Pretrained weights & preprocessing**
18. Paper nói dùng ImageNet pretrained nhưng không có checkpoint FOTCA/AFNO-ImageNet công khai và không nói cách map weights. Bạn muốn xử lý thế nào? (đề xuất, không phải từ paper: (a) nạp ViT-B/16 ImageNet từ timm, chỉ copy các phần khớp tên/shape — LayerNorm và MLP của 12 block (fc1/fc2) — còn AFNO filter, patch embed FFT, Fourier pos-enc và CNN classifier khởi tạo ngẫu nhiên; hoặc (b) train from scratch toàn bộ và ghi rõ là sai khác so với paper)
19. Mean/std normalization không được ghi. Bạn có đồng ý dùng ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] không? (đề xuất, không phải từ paper — hợp lý vì dùng ImageNet pretrained; nếu nạp ViT timm thì có thể dùng mean/std = 0.5)
20. Batch size 8 với 100 epochs rất chậm trên Kaggle T4×2 (DataParallel → mỗi GPU chỉ 4 ảnh, BN trong classifier không ổn định). Bạn muốn giữ đúng 8 / 100 epochs như paper, hay điều chỉnh? (đề xuất, không phải từ paper: giữ batch 8 & LR 0.001 để bám paper; nếu thời gian không đủ (giới hạn 12h Kaggle cho 2 scenario) thì giảm epochs, ví dụ 30, và ghi rõ sai khác)

---

## GHI CHÚ IMPLEMENTATION

1. **Mâu thuẫn mô tả AFNO:** Mục 3.2.2 (Eq.6–12) mô tả "adaptive Fourier basis functions" theo từng patch (B ∈ C^{p²×K}, A ∈ C^{N×K}, global average pooling ra w_k) — khác với AFNO gốc của Guibas et al. Trong khi đó Fig. 2A vẽ đúng AFNO gốc: Spatial Mixing = FFT → Multiply Frequency → IFFT, rồi Channel Mixing MLP. Khuyến nghị implement theo **AFNO gốc (AFNO2D, FourCastNet)** vì khớp với figure, tên gọi và tham chiếu Guibas 2021; các phương trình 6–12 không đủ rõ để code (ký hiệu lỗi, chiều tensor mâu thuẫn, ví dụ Eq.3 x_patch ∈ R^{D×K}).
2. **AFNO2D chuẩn (tham chiếu):** rfft2 trên lưới token 14×14 (norm="ortho") → block-diagonal complex MLP 2 lớp (w1, b1, w2, b2, ReLU) với num_blocks → softshrink(λ = sparsity_threshold) → irfft2 → cộng bias residual. Mọi giá trị này phải chờ người dùng xác nhận (CẦN XÁC NHẬN #6–#10).
3. **Fig. 2B/2C** vẽ loss dạng L2 trên X_ref(k+1), X_ref(k+2) (có vẻ copy từ FourCastNet, dự báo theo bước thời gian) — KHÔNG phù hợp với bài toán phân loại. Dùng Eq.(2) focal loss trên softmax cho classification; ghi chú điều này trong notebook.
4. **Kiểm tra số params:** paper báo 59.14M (ViT baseline 58.07M — ViT-B/16 chuẩn là ~86M, nên ViT của họ có thể không phải cấu hình chuẩn). Sau khi build, so tổng params với 59.14M để đánh giá độ lệch; không ép khớp.
5. **Focal loss + class weights:** nếu rule 11 kích hoạt (imbalance ≥ 1.5), truyền class_weights vào focal loss dưới dạng α per-class (CE có weight rồi nhân (1−p_t)^γ). Ghi rõ có/không trong output.
6. **Complex tensors + DataParallel + AMP:** FFT trong PyTorch cần float32 (torch.fft không hỗ trợ half trên mọi kích thước) → nếu dùng AMP, ép `.float()` trước rfft2. DataParallel với complex params: lưu weight dưới dạng real tensor (shape [2, …]) như code FourCastNet để tránh lỗi.
7. **BatchNorm với batch 8 trên 2 GPU** → 4 mẫu/GPU, BN noisy; cân nhắc giữ nguyên (bám paper) và ghi chú.
8. **Ablation (tùy chọn):** paper có 3 biến thể (Pap Embedding gốc, MLP-Classifier, không augmentation). Không bắt buộc cho notebook chính; có thể thêm sau nếu người dùng yêu cầu.
9. Paper không có test set riêng (chỉ train/val 8:2); notebook sẽ dùng split train/val/test có sẵn của dataset người dùng, checkpoint theo val_accuracy, evaluate trên test.
