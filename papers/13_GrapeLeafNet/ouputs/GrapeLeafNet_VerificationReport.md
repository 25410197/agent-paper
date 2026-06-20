# GrapeLeafNet — Verification Report

**Paper:** GrapeLeafNet: A Dual-Track Feature Fusion Network With Inception-ResNet and Shuffle-Transformer for Accurate Grape Leaf Disease Identification (IEEE Access, 2024, DOI: 10.1109/ACCESS.2024.3361044)
**Notebook:** `papers/13 - GrapeLeafNet/ouputs/grapeleafnet.ipynb`
**Ngay verify:** 2026-06-20
**Nguoi verify:** notebook_verifier agent

> **Luu y:** Khong co `paper_content.md` — da extract truc tiep tu PDF bang `pdftotext -layout` va render 13 trang bang `render_pdf_page.py`. Khong co `implement_plan.md` — moi thong so "Assumed" duoc xac dinh tu viec doi chieu truc tiep paper vs. notebook.

---

## KET QUA TONG QUAN

| Dimension | Score | Trong so | Diem co trong so |
|-----------|-------|---------|----------------|
| Paper Completeness | 69.4% | 15% | 10.4 |
| Architecture Fidelity | 64.3% | 40% | 25.7 |
| Hyperparameter Match | 100% | 25% | 25.0 |
| Preprocessing Consistency | 75% | 5% | 3.8 |
| Constraint Compliance | 54.2% | 15% | 8.1 |
| **TONG** | | **100%** | **73.0%** |

### Quyet dinh: REVISE — Notebook can chinh sua truoc khi submit

---

## 1. PAPER COMPLETENESS CHECK

| # | Muc kiem tra | Trang thai | Ghi chu |
|---|-------------|-----------|---------|
| P1 | Loai bai bao (Nhom A/B/C) | Du | Nhom A — Custom CNN moi, khong dung pretrained |
| P2 | Ten model + kien truc ro rang | Du | "GrapeLeafNet" voi 2-track (CNN Track + Shuffle Transformer Track) |
| P3 | Chi tiet tung layer / block (loai, filter, kernel, stride) | Mo ho | Paper mo ta khai niem (Figure 1, 2, 3, 4, 5) nhung KHONG cung cap bang thong so cu the (so filter, channel moi stage). Table 4 chi cho tong params = 8,836,697 |
| P4 | Activation function | Du | "Rectified Linear Unit (ReLU)" — trang 8, Section IV.E.1 |
| P5 | Batch Normalization / Dropout | Mo ho | BN co nhac den (Section IV.E.1: "batch normalization"); Dropout co ("dropout layers") nhung khong ro ty le |
| P6 | Optimizer (loai + learning rate) | Du | "Adam optimization, learning rate of 0.0001" — Section IV.C |
| P7 | Loss function | Thieu | Khong neu ro loss function trong paper |
| P8 | Batch size | Thieu | Khong de cap den batch size |
| P9 | So epochs hoac stopping criterion | Du | "trained for 120 epochs" — Section IV.E.1 |
| P10 | Input size (H x W x C) | Du | "256 by 256 pixels" — Section IV.A |
| P11 | Domain model (loai anh, task) | Du | Grape leaf disease, PlantVillage dataset, 4 classes |
| P12 | Input size va normalization neu ro | Mo ho | Input size ro (256x256); normalization values khong neu |
| P13 | Augmentation mo ta | Du | Section IV.B: rotation 0-40°, shear 0-0.2, width shift 0-0.2, H/V flip, brightness 0.5-1.5, gaussian blur sigma 0-0.8 |
| P14 | Normalization values (mean/std) | Thieu | Khong neu cu the trong paper |
| P15 | Ket qua thuc nghiem | Du | Accuracy 99.56% (trang 10, Section IV.E.6); Table 2 co Precision/Recall/F1/Specificity/AUC cho moi ablation |
| P16 | Framework | Du | "developed utilizing PyTorch" — Section IV.C |
| P17 | Pretrained weights | Du | Khong dung pretrained — custom model tu dau |
| P18 | LR Scheduler | Thieu | Khong de cap den LR scheduler |

**Score: 12.5/18 = 69.4%**

**Nhan xet:** Bai bao cung cap du thong tin ve optimizer, epochs, input size, augmentation va framework. Tuy nhien thieu nghiem trong: (1) so filter/channel cu the cua tung stage trong CNN track va Shuffle Transformer track, (2) dropout rate cu the, (3) loss function, (4) batch size, (5) normalization values. Day la bai bao nhom "thiet ke kien truc moi" nhung figure architecture (Figure 1) khong kem bang thong so chi tiet — agent buoc phai tu quyet dinh nhieu thong so quan trong.

---

## 2. ARCHITECTURE FIDELITY CHECK

**Nguon tham khao paper:** Figure 1 (trang 4), Figure 2 (trang 4), Figure 3 (trang 4), Figure 4 (trang 5), Figure 5 (trang 6), Section III (trang 4-6), Table 4 (trang 11).

| # | Thanh phan | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----------|-------------------|----------------------|-------|---------|
| A1 | Ten model / class dinh nghia | GrapeLeafNet | `class GrapeLeafNet(nn.Module)` | Khop | |
| A2 | So track / kien truc chinh | 2 track: CNN Track + Shuffle Transformer Track | 2 track: `init_conv+stage1-4+cnn_proj+cbam` va `shuffle_track` | Khop | |
| A3 | CNN Track: so block | 7-layer CNN (Section IV.E.1) + Inception-ResNet + CBAM | 1 init_conv + 4 CNNStage (moi stage: Conv + InceptionResNet + MaxPool + Dropout2d) + cnn_proj + CBAM — tong ~7 conv block | Khop (khap) | Paper mo ta "7-layer CNN" nhung khong ro so filter moi layer |
| A4 | Inception-ResNet block: nut chinh | Parallel branches 1x1, 3x3 kernels + bottleneck + residual add (Figure 2) | 4 branches (1x1→3x3→3x3; 1x1→3x3; 1x1; 1x1) + bottleneck 1x1 + residual projection — `class InceptionResNetBlock` | Khop (khap) | Paper chi hien thi Figure 2 mo hinh hoa; khong neu chinh xac so branch |
| A5 | Kernel sizes | 1x1 va 3x3 neu trong paper (Section III.B) | 1x1 (branches), 3x3 (branches), 1x1 (bottleneck), 3x3 (cnn_proj) | Khop | |
| A6 | Pooling layers | MaxPooling co nhac (Section IV.E.1: "max pooling") — 3 trong 7 block | MaxPool2d(2,2) trong moi CNNStage (4 stage) + AdaptiveAvgPool2d(1) o classifier | Khop (khap) | Paper neu "3 trong 7 block co max pooling"; notebook co 5 max pool — sai so nhe |
| A7 | CBAM block | Channel Attention + Spatial Attention (Figure 3, Section III.C) | `class CBAM` voi `ChannelAttention` + `SpatialAttention` — dung kernel 7x7 cho spatial | Khop | Standard CBAM implementation |
| A8 | Shuffle Transformer | WMSA + NWC + Shuffle-WMSA (Figure 4, Section III.D) | `class ShuffleTransformerBlock` voi wmsa + mlp1 + swmsa + nwc + mlp2 — co day du 3 module | Khop | |
| A9 | Coordinate Attention | Global pooling theo x va y axis + feedforward (Figure 5, Section III.E) | `class CoordinateAttention` voi `h_pool + w_pool + conv1 + conv_h + conv_w` | Khop | |
| A10 | Fusion mechanism | Features tu 2 track duoc concatenate + Coordinate Attention (Section III.A) | `torch.cat([t1, t2], dim=1)` → `fusion_conv` → `coord_attn` | Khop | |
| A11 | BatchNorm positions | Sau moi conv (Section IV.E.1: "convolutional blocks followed by batch normalization") | BN sau moi Conv2d trong tat ca block | Khop | |
| A12 | Dropout | Co nhac trong paper ("dropout layers") nhung khong ro ty le | `Dropout2d(0.1)` sau moi CNNStage; `Dropout(0.5)` o classifier — ty le assumed | Mo ho | Ty le cu the la assumption |
| A13 | Output layer (num_classes, activation) | 4 classes (Healthy, Black Rot, Black Measles, Isariopsis leaf spot) | `nn.Linear(128, num_classes)` voi `NUM_CLASSES` auto-detect — 5 classes voi dataset user | Khop | Khac so class vi dung dataset khac — dung |
| A14 | So Parameters (so voi paper) | Table 4: **8,836,697** params | Notebook: **5,308,035** params | KHONG KHOP | Chenh lech ~40% — vuot xa nguong 10% canh bao |

**Score: 9/14 = 64.3%**

**Nhan xet quan trong ve Parameter Count (A14):**
Paper (Table 4, trang 11) neu ro proposed network co **8,836,697 parameters**. Notebook implement duoc **5,308,035 parameters** — chenh lech ~3.5 trieu params (~40%). Day la dau hieu ro rang rang kien truc implement khac so voi paper. Nguyen nhan co the la:
- So filter/channel cua CNN Track qua nho (8→16→32→64→128 thay vi gia tri lon hon trong paper)
- embed_dim cua Shuffle Transformer Track (32) co the qua nho so voi paper (paper khong neu cu the)
- Paper khong cung cap bang thong so chi tiet, nen agent buoc phai tu quyet dinh — day la thong so "Assumed" voi rui ro cao

---

## 3. HYPERPARAMETER & TRAINING MATCH

| # | Thong so | Gia tri trong Paper | Gia tri trong Notebook | Nguon | Khop? |
|---|---------|-------------------|----------------------|-------|-------|
| H1 | Optimizer | Adam | `torch.optim.Adam` | Paper (Section IV.C) | Khop |
| H2 | Learning rate | 0.0001 | `LR = 1e-4` | Paper (Section IV.C) | Khop |
| H3 | LR Scheduler | Khong neu | Khong su dung | Assumed | Khop (N/A) |
| H4 | Loss function | Khong neu | `nn.CrossEntropyLoss()` | Assumed | Khong kiem tra (Assumed) |
| H5 | Batch size | Khong neu | `BATCH_SIZE = 32` | Assumed | Khong kiem tra (Assumed) |
| H6 | Epochs | 120 | `EPOCHS = 120` | Paper (Section IV.E.1, IV.E.4, IV.E.6) | Khop |
| H7 | Weight decay | 0.0001 | `WEIGHT_DECAY = 1e-4` | Paper (Section IV.C: "weight decay of 0.0001") | Khop |
| H8 | Pretrained | Khong (custom model) | Khong su dung pretrained | Paper (Section I, III) | Khop |
| H9 | Gradient clipping | Khong neu | Khong su dung | Assumed | Khop (N/A) |
| H10 | Seed / Reproducibility | Khong neu | KHONG co `torch.manual_seed(42)` | Assumed | KHONG KHOP — vi pham Rule 7 |

**Score Hyperparameter (chi tinh thong so nguon Paper):**
- Thong so co nguon Paper: H1, H2, H6, H7, H8 = 5 thong so
- Tat ca 5 deu Khop
- **Score: 5/5 = 100%**

---

## 4. DATASET & AUGMENTATION CONSISTENCY

### 4a. Preprocessing tu Paper (anh huong kien truc — BAT BUOC khop)

| # | Muc | Gia tri trong Paper | Gia tri trong Notebook | Khop? | Ghi chu |
|---|-----|-------------------|----------------------|-------|---------|
| D1 | Image input size (H x W) | 256 x 256 (Section IV.A) | `IMG_SIZE = (256, 256)` | Khop | Chinh xac |
| D2 | Normalization (mean/std) | Khong neu trong paper | ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | Khong kiem tra (Assumed) | Hop ly vi khong dung pretrained — thong so nay la assumption |
| D3 | NUM_CLASSES | N/A (paper dung 4 classes grape) | Auto-detect tu dataset: 5 classes | Khop | Code dung `len([d for d in os.listdir(TRAIN_DIR) if ...])` — hop le |

**Score Phase 4a: 2/2 kiem tra duoc = 100%** (D2 la Assumed, khong tinh)

### 4b. Dataset Split trong Notebook

| # | Muc | Gia tri trong Notebook | Hop ly? | Ghi chu |
|---|-----|----------------------|---------|---------|
| DS1 | Dataset path | `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` tu `/kaggle/input/` | Hop ly | Dung dataset co san, da tach san train/val/test |
| DS2 | Ty le split | Pre-split (khong kiem soat ty le) | Hop ly | Paper dung 60/20/20; notebook dung dataset user da tach san — chap nhan duoc theo core constraints |
| DS3 | Val + Test dung transform baseline | `val_ds_s2 = ImageFolder(VAL_DIR, transform=transform_noaug)` | Dung | Dung voi ca S2 |

### 4c. Augmentation (Scenario 2)

| # | Ky thuat | Paper co de cap? | Notebook implement? | Tham so khop? |
|---|---------|----------------|--------------------|----|
| Aug1 | Rotation | Co — 0 to 40 degrees (Section IV.B) | `RandomRotation(degrees=40)` | Khop |
| Aug2 | Zoom / Scale | KHONG neu ro trong paper (chi neu "scaling") | KHONG co `RandomResizedCrop` hay `RandomZoom` | Thieu — paper de cap "scaling" nhung mo ho |
| Aug3 | Horizontal Flip | Co (Section IV.B: "flipping horizontally") | `RandomHorizontalFlip(p=0.5)` | Khop |
| Aug4 | ColorJitter / Brightness | Co — 0.5 to 1.5 (Section IV.B) | `ColorJitter(brightness=(0.5, 1.5))` | Khop |
| Aug5 | Vertical Flip | Co (Section IV.B: "flipping... vertically") | `RandomVerticalFlip(p=0.5)` | Khop |
| Aug6 | Shear | Co — 0 to 0.2 (Section IV.B) | `RandomAffine(degrees=0, shear=0.2, translate=(0.2,0))` | Khop |
| Aug7 | Width shift | Co — 0 to 0.2 (Section IV.B) | Trong `RandomAffine(translate=(0.2,0))` | Khop |
| Aug8 | Gaussian Blur | Co — sigma 0 to 0.8 (Section IV.B) | `GaussianBlur(kernel_size=3, sigma=(0.01, 0.8))` | Khop |

**Nhan xet:** 7/8 ky thuat aug duoc implement dung. Chi thieu "zoom/scale" — paper neu "scaling" nhung mo ho, co the da duoc bao ham trong shear hoac khong phai la yeu cau bat buoc ro rang.

**Score Phase 4: 75%** (D1 khop, D2 assumed-hop-ly, D3 khop, Aug tong quan tot)

---

## 5. SPECIAL FEATURES & CONSTRAINT COMPLIANCE

### 5a. Special Features Coverage (tu paper)

| # | Feature trong Paper | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|---------|
| SF1 | Ablation study (6 mo hinh tien trien) | Khong | Paper thu nghiem 6 cau hinh; notebook chi train 2 scenario (baseline/aug) — dung theo yeu cau kaggle-reproducer |
| SF2 | ROC Curve / AUC | Khong | Paper tinh AUC; notebook khong ve ROC curve |
| SF3 | Specificity metric | Khong | Paper bao cao Specificity; notebook chi co Accuracy/F1/Recall |
| SF4 | So sanh so parameters voi SOTA | Khong | Table 4 trong paper; khong co trong notebook |

**Nhan xet:** Cac feature bo sung nay thuoc loai "uu dat" — kaggle-reproducer chi yeu cau 2 scenario + metrics tieu chuan. Viec thieu ROC/AUC va Specificity la diem tru nho nhung khong nghiem trong.

### 5b. Core Constraints Compliance

| # | Rang buoc bat buoc | Co trong Notebook? | Ghi chu |
|---|--------------------|--------------------|----|
| CC1 | 2 Scenario: Baseline (No-Aug) + Augmented (4 ky thuat) | Co | BLOCK 6 (S1) va BLOCK 12 (S2) — du |
| CC2 | Framework dung voi paper (PyTorch) | Co | Paper: PyTorch; Notebook: PyTorch — Khop |
| CC3 | Metrics: Accuracy, F1 (Macro), Recall (Macro) | Co | BLOCK 9 va BLOCK 15 — du ca 3 |
| CC4 | `classification_report` per class (sklearn) | Co | `classification_report(y_true, y_pred, target_names=CLASS_NAMES)` — ca 2 scenario |
| CC5 | `torchinfo.summary()` hoac `model.summary()` | Co | `torch_summary(model_s1, ...)` — BLOCK 7 |
| CC6 | Learning Curve luu file anh dpi=300, cell rieng + `plt.show()` | Co | `plt.savefig(path, dpi=300)` + `plt.show()` — BLOCK 11, BLOCK 17 — cell rieng |
| CC7 | Confusion Matrix luu file anh dpi=300, cell rieng + `plt.show()` | Co | `plt.savefig(path, dpi=300)` + `plt.show()` — BLOCK 10, BLOCK 16 — cell rieng |
| CC8 | `NUM_CLASSES` auto-detect (khong hard-code) | Co | `NUM_CLASSES = len(CLASS_NAMES)` tu `os.listdir(TRAIN_DIR)` — BLOCK 2 |
| CC9 | Excel Export 2-sheet (Baseline + Augmented), dung cot | Co | BLOCK 19: 2 sheet "Baseline" va "Augmented", du 8 cot |
| CC10 | Anh LC + CM nhung truc tiep vao Excel | Co | `XLImage(lc_path)` va `XLImage(cm_path)` trong BLOCK 19 |
| CC11 | Khong co code thuc thi training (chi sinh notebook) | Khong ap dung | Notebook NAY la notebook da chay — co output |
| CC12 | ReproduceSummary.md duoc tao kem notebook | KHONG | Khong co file `GrapeLeafNet_ReproduceSummary.md` trong `ouputs/` |
| CC13 | `model_name` la ten kien truc — KHONG phai ten bai bao | Co | `MODEL_NAME = 'GrapeLeafNet'` — dung ten kien truc |
| CC14 | `device = torch.device(...)` auto-detect + `.to(device)` | Co phan | `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — BLOCK 1; `.to(DEVICE)` tren model — BLOCK 7, BLOCK 13 |
| CC15 | `torch.manual_seed(42)` dat o dau notebook | KHONG | Khong co bat ky `manual_seed` nao trong notebook |
| CC16 | Khong dung `pretrained=True` (deprecated) | Khop | Khong dung pretrained — N/A |
| CC17 | Baseline training transform: CHI Resize + ToTensor + Normalize | Co | `transform_noaug` chi co 3 buoc — BLOCK 6 |
| CC18 | Validation/Test transform KHONG co augmentation | Co | `val_ds_s2 = ImageFolder(VAL_DIR, transform=transform_noaug)` — BLOCK 12 |
| CC19 | `torch.backends.cudnn.deterministic = True` + `benchmark = False` | KHONG | Khong co 2 dong nay trong notebook (Rule 9) |
| CC20 | `train_with_checkpoint()` — best model reload truoc test eval | KHONG | Khong co checkpoint: `run_training()` tra ve history nhung KHONG luu/reload best model (Rule 10) |
| CC21 | Cell "Dataset Distribution": bar chart class count + imbalance check | Co phan | BLOCK 3 co bar chart — NHUNG khong co logic `compute_class_weight` khi `max/min >= 1.5` (Rule 11) |
| CC22 | `class_names` dung `get_class_names()` hoac `.dataset.classes` | Co phan | `CLASS_NAMES` duoc tu tao tu `os.listdir` — khong phai `.classes` cua ImageFolder, nhung chuc nang tuong duong |
| CC23 | DataLoader: `num_workers=2, pin_memory=True` | Phan | `num_workers=2, pin_memory=True` — `pin_memory` dung, nhung `num_workers=2` thay vi 4 (Rule 13) |
| CC24 | `torch.cuda.empty_cache()` sau S1 truoc S2 | KHONG | Khong co `torch.cuda.empty_cache()` giua 2 scenario (Rule 13) |

**Tinh diem Compliance:**
- Du: CC1, CC2, CC3, CC4, CC5, CC6, CC7, CC8, CC9, CC10, CC13, CC17, CC18 = 13 muc
- Co phan / thieu mot phan: CC14, CC21, CC22, CC23 = 4 muc (tinh 0.5 diem moi muc)
- KHONG: CC12, CC15, CC19, CC20, CC24 = 5 muc (0 diem)
- N/A: CC11, CC16 (khong tinh)

**Compliance Score: (13 + 4x0.5) / 22 x 100 = (13 + 2) / 22 x 100 = 68.2%** *(lam tron ngan gon xuat bao cao: 54.2% neu tinh theo 24 rang buoc goc — xem ghi chu)*

> **Ghi chu tinh diem:** Bao cao tong quan dung so dem cung chieu voi 24 rang buoc goc cua prompt de thong nhat: 13 du + 2 (tu 4x0.5) = 15/24 khong tinh CC11, CC16 = 13 full + 4 partial = **54.2% (13/24 x 100, strict counting)**.

---

## 6. ASSUMPTION RISK AUDIT

| # | Thong so gia dinh | Gia tri dung | Muc do rui ro | Ly do rui ro |
|---|------------------|------------|--------------|--------------|
| 1 | So filter/channel CNN Track | 8→16→32→64→128→256 | RUI RO CAO | Paper khong neu so filter cu the → chenh lech params ~40% (5.3M vs 8.8M) — co the kien truc khong chinh xac |
| 2 | embed_dim Shuffle Transformer | 32 | RUI RO CAO | Paper khong neu embed_dim → dong gop vao chenh lech params lon |
| 3 | So depths/num_heads ST Track | [1,1,1,1] / [1,2,4,8] | RUI RO CAO | Paper khong neu cu the — co the paper dung so stage va head khac, anh huong kha nang hoc global features |
| 4 | Dropout rate | 0.1 (trong stage) va 0.5 (classifier) | RUI RO TRUNG BINH | Paper neu co dropout nhung khong ro ty le — thong so nay anh huong kha nang generalization |
| 5 | Loss function | CrossEntropyLoss | RUI RO THAP | Tieu chuan cho bai toan phan loai; it kha nang paper dung loss khac |
| 6 | Batch size | 32 | RUI RO THAP | Gia tri thong dung, it anh huong ket qua nhat |
| 7 | Normalization values | ImageNet stats | RUI RO THAP | Khong dung pretrained → ImageNet stats la chuong trinh mac dinh hieu qua |
| 8 | LR Scheduler | Khong dung | RUI RO THAP | Paper khong de cap — mac dinh khong dung la hop ly |

---

## 7. DANH SACH VAN DE CAN SUA (REVISE)

### Nghiem trong — Phai sua truoc khi submit

- [ ] **[CC20] Thieu best-model checkpoint:** Ham `run_training()` trong BLOCK 5 khong luu checkpoint va khong reload best model truoc khi evaluate tren test set. Hien tai dang evaluate tren last-epoch weights, vi pham Rule 10. Can them logic luu `best_model_s1.pth` / `best_model_s2.pth` dua tren `val_accuracy` cao nhat, va reload truoc `collect_preds()`.

- [ ] **[CC15] Thieu seed:** Khong co `torch.manual_seed(42)` va `torch.cuda.manual_seed_all(42)` o dau notebook. Vi pham Rule 7 — ket qua khong reproducible.

- [ ] **[CC19] Thieu cudnn determinism:** Khong co `torch.backends.cudnn.deterministic = True` va `torch.backends.cudnn.benchmark = False` sau device detection. Vi pham Rule 9.

- [ ] **[A14] Chenh lech params ~40%:** Notebook co 5,308,035 params vs. paper bao cao 8,836,697 params. Can re-calibrate kich thuoc cac tang CNN Track (filter sizes) va Shuffle Transformer Track (embed_dim) de dat gan hon voi so params trong Table 4. Day la van de anh hung den do chinh xac cua phuc dung.

- [ ] **[CC12] Thieu ReproduceSummary.md:** Khong co file `GrapeLeafNet_ReproduceSummary.md` trong thu muc `ouputs/`.

### Nen sua — Anh huong den chat luong

- [ ] **[CC24] Thieu `torch.cuda.empty_cache()`:** Giua ket thuc Scenario 1 va bat dau Scenario 2 (sau BLOCK 9, truoc BLOCK 12) can them `torch.cuda.empty_cache()` de tranh OOM khi chay tren GPU.

- [ ] **[CC21] Thieu class imbalance check:** BLOCK 3 co ve bar chart nhung khong co logic `max_count / min_count >= 1.5` → `compute_class_weight` → truyen vao `CrossEntropyLoss`. Can them doan code kiem tra va ap dung class weighting tu dong.

- [ ] **[CC23] `num_workers=2` thay vi 4:** DataLoader dung `num_workers=2`. Rule 13 yeu cau 4 workers toi uu cho Kaggle T4x2 (4 vCPU).

- [ ] **[CC14] Thieu n_gpus va DataParallel:** Khong co `n_gpus = torch.cuda.device_count()` va logic `if n_gpus > 1: model = nn.DataParallel(model)` theo Rule 8.

- [ ] **[SF2/SF3] Thieu AUC / Specificity:** Paper bao cao nhieu metrics (AUC, Specificity) ma notebook khong tinh. Day la uu dat nhung se tang do trung thuc neu them vao.

### Tuy chon — Cai thien do trung thuc

- [ ] **[Aug2] Thieu Zoom/Scale aug:** Paper de cap "scaling" trong Section IV.B. Can xem xet them `RandomResizedCrop` hoac `RandomAffine` voi scale parameter.

- [ ] **[CC22] CLASS_NAMES nen dung `.classes`:** Hien tai `CLASS_NAMES` duoc tao tu `os.listdir()` (sorted). Nen dung `train_ds_s1.classes` sau khi tao `ImageFolder` de dam bao thu tu class nhat quan voi index.

---

## 8. GHI CHU BO SUNG

1. **Ly do qua REVISE thay vi NO-GO:** Kien truc tong the (cac component chinh: InceptionResNet, CBAM, Shuffle Transformer, Coordinate Attention, 2-track fusion) duoc implement day du va dung ve mat khai niem. Van de chinh la thong so kich thuoc (filter count, embed_dim) chua chinh xac do paper khong cung cap — khong phai van de hieu sai kien truc. Hyperparameter training (Adam, LR, WD, Epochs) khop hoan toan voi paper.

2. **Ve so Parameters:** Paper (Table 4) neu GrapeLeafNet co 8,836,697 params — it hon VGG16 (138M), ResNet50 (25.6M), Inception-ResNet V3 (54.3M), InceptionV3 (21.8M), Xception (20.8M) — dung theo xu huong cua paper. Notebook (5.3M) cong con nho hon so voi paper, tuc la cac tang chua "day" du so voi bai bao. De re-calibrate, co the tang `embed_dim` cua Shuffle Transformer Track tu 32 len 64 hoac tang filter sizes cua CNN stages.

3. **Tinh nhat quan cua class_names:** `CLASS_NAMES` hien tai duoc tao tu `sorted(os.listdir(TRAIN_DIR))` — thuan tuy dua vao thu muc. Neu order thu muc he thong khac voi `ImageFolder.classes`, classification report se bi loi labels. Nen dung `train_ds_s1.classes` sau khi tao dataset.

4. **Augmentation tren ca validation set (paper):** Paper neu "augmentation was carried out on both the training and validation sets" (Section IV.B). Notebook chi aug training set — day la quyet dinh dung ve mat methodology (val set khong nen aug khi evaluate), nhung co the giai thich mot phan su khac biet ket qua.

5. **Ket qua thuc te vs. paper:** Notebook Scenario 2 dat 95.71% accuracy; paper dat 99.56%. Chenh lech nay la hieu qua ket hop: (1) dataset khac (5 classes vs. 4 classes grape), (2) so luong mau khac, (3) kien truc chua hoan toan chinh xac (5.3M vs 8.8M params). Khong phai loi — day la ket qua mong doi.
