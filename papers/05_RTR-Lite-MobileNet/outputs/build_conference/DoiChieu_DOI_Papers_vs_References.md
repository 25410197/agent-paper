# Đối chiếu DOI — Thứ tự `papers/` ↔ Reference hội nghị

> **Đã đánh số lại (phương án A):** reference `[1]–[15]` = đúng thứ tự 15 bài phục dựng trong `papers/01–15`, sau đó `[16]–[24]` = các biện pháp AI/kiến trúc/XAI.
> ⇒ **Ref số = số folder** cho 15 bài SOTA → đối chiếu 1:1 cực dễ.
> **Cập nhật:** đã bỏ SENet và ECA-Net khỏi bài (chỉ ví dụ chung ở §2, không mô tả trong phần method) → còn 24 ref, đánh số lại.
> Tất cả DOI đã verify trực tiếp qua CrossRef. Cập nhật: 2026-07-14.

## Bảng 1 — 15 bài SOTA phục dựng (Ref# = Folder#)

| Ref | Folder `papers/` | Model (Bảng 1 hội nghị) | Tác giả đầu · năm | DOI (đã verify) |
|---|---|---|---|---|
| **[1]** | 01_MetaEnsemble | Meta-Ensemble | Maurya · 2024 | 10.1109/ACCESS.2024.3367443 |
| **[2]** | 02_MaizeFormerX | MaizeFormerX | Rahman · 2026 | 10.1038/s41598-026-44550-0 |
| **[3]** | 03_TrIncNet | TrIncNet | Gole · 2023 | 10.3389/fpls.2023.1221557 |
| **[4]** | 04_MRW-CNN | MRW-CNN | Joseph · 2024 | 10.1109/ACCESS.2024.3358333 |
| **[5]** | 05_RTR-Lite-MobileNet | RTR_Lite_MobileNet (gốc) | Duhan · 2025 | 10.1016/j.cpb.2025.100459 |
| **[6]** | 06_PMVT-XS | PMVT-XS | Li · 2023 | 10.3389/fpls.2023.1256773 |
| **[7]** | 07_LiteMDC | Lite-MDC | Bhagat · 2024 | 10.1016/j.atech.2024.100408 |
| **[8]** | 08_ULEN | ULEN | **Wang** · 2023 | 10.1007/s11119-023-10020-0 |
| **[9]** | 09_LDI-NET | LDI-NET | Yang · 2024 | 10.1038/s41598-024-62452-x |
| **[10]** | 10_PDLCViT | PDLC-ViT | Hemalatha · 2024 | 10.1007/s44196-024-00597-3 |
| **[11]** | 11_AlexNet-ResNet50-EfficientNetB0 | ResNet-50 / EfficientNetB0 / AlexNet / Baseline CNN | Mishra · 2025 | 10.1038/s41598-025-32607-5 |
| **[12]** | 12_CustomMobileNetV3Large | CustomMobileNetV3Large | Rahaman · 2026 | 10.1007/s44163-025-00733-8 |
| **[13]** | 13_GrapeLeafNet | GrapeLeafNet | Karthik · 2024 | 10.1109/ACCESS.2024.3361044 |
| **[14]** | 14_VGG19 | VGG19+· / InceptionV3+· | Sujatha · 2025 | 10.1038/s41598-024-72197-2 |
| **[15]** | 15_ViT | ViT | Ali · 2025 | 10.11591/ijece.v15i2.pp2334-2344 |

**Ghi chú:** giờ **cả 15 bài đều được trích dẫn** trong Bảng 1 hội nghị (mỗi dòng so sánh gắn số `[n]` = folder tương ứng). Folder 11 (Mishra) phủ 4 model; folder 14 (Sujatha) phủ cả VGG19+· lẫn InceptionV3+·. **Tác giả đầu bài [8] đã sửa: Wang** (v2 cũ ghi nhầm "Gong" — Gong là tác giả cuối).

---

## Bảng 2 — Biện pháp AI / kiến trúc / XAI ([16]–[24], không nằm trong `papers/`)

| Ref | Nội dung | Tác giả đầu · năm | DOI (đã verify) |
|---|---|---|---|
| [16] | MobileNetV2 (xương sống) | Sandler · 2018 | 10.1109/CVPR.2018.00474 |
| [17] | Triplet Attention | Misra · 2021 | 10.1109/WACV48630.2021.00318 |
| [18] | Attention Is All You Need | Vaswani · 2017 | *(NeurIPS — không có DOI)* |
| [19] | RoFormer (RoPE) | Su · 2024 | 10.1016/j.neucom.2023.127063 |
| [20] | Grouped-Query Attention (GQA) | Ainslie · 2023 | 10.18653/v1/2023.emnlp-main.298 |
| [21] | RMSNorm | Zhang · 2019 | *(NeurIPS — không có DOI)* |
| [22] | Grad-CAM | Selvaraju · 2017 | 10.1109/ICCV.2017.74 |
| [23] | Attention Rollout | Abnar · 2020 | 10.18653/v1/2020.acl-main.385 |
| [24] | Occlusion Sensitivity | Zeiler · 2014 | 10.1007/978-3-319-10590-1_53 |

*(Đã BỎ khỏi bài: SENet [Hu 2018], ECA-Net [Wang 2020] — chỉ là ví dụ chung ở §2, RES_Block đã quy attribution về bài gốc [5]. SHAP cũng đã bỏ từ trước khi rút XAI còn 3 kỹ thuật.)*

---

## Bảng 3 — Tra nhanh: link DOI bấm được + TÊN BÀI ĐẦY ĐỦ

> Bấm link để mở trang DOI kiểm tên bài. Tên bài lấy verbatim từ CrossRef.

**15 bài SOTA (Ref# = Folder#):**

- **[1]** 01_MetaEnsemble → [doi.org/10.1109/ACCESS.2024.3367443](https://doi.org/10.1109/ACCESS.2024.3367443) → *"A Lightweight Meta-Ensemble Approach for Plant Disease Detection Suitable for IoT-Based Environments"*
- **[2]** 02_MaizeFormerX → [doi.org/10.1038/s41598-026-44550-0](https://doi.org/10.1038/s41598-026-44550-0) → *"MaizeFormerX: a lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis"*
- **[3]** 03_TrIncNet → [doi.org/10.3389/fpls.2023.1221557](https://doi.org/10.3389/fpls.2023.1221557) → *"TrIncNet: a lightweight vision transformer network for identification of plant diseases"*
- **[4]** 04_MRW-CNN → [doi.org/10.1109/ACCESS.2024.3358333](https://doi.org/10.1109/ACCESS.2024.3358333) → *"Real-Time Plant Disease Dataset Development and Detection of Plant Disease Using Deep Learning"*
- **[5]** 05_RTR-Lite-MobileNet → [doi.org/10.1016/j.cpb.2025.100459](https://doi.org/10.1016/j.cpb.2025.100459) → *"RTR_Lite_MobileNetV2: A lightweight and efficient model for plant disease detection and classification"*
- **[6]** 06_PMVT-XS → [doi.org/10.3389/fpls.2023.1256773](https://doi.org/10.3389/fpls.2023.1256773) → *"PMVT: a lightweight vision transformer for plant disease identification on mobile devices"*
- **[7]** 07_LiteMDC → [doi.org/10.1016/j.atech.2024.100408](https://doi.org/10.1016/j.atech.2024.100408) → *"Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop"*
- **[8]** 08_ULEN → [doi.org/10.1007/s11119-023-10020-0](https://doi.org/10.1007/s11119-023-10020-0) → *"An ultra-lightweight efficient network for image-based plant disease and pest infection detection"* (tác giả: B. Wang, C. Zhang, Y. Li, C. Cao, D. Huang, Y. Gong)
- **[9]** 09_LDI-NET → [doi.org/10.1038/s41598-024-62452-x](https://doi.org/10.1038/s41598-024-62452-x) → *"A novel plant type, leaf disease and severity identification framework using CNN and transformer with multi-label method"*
- **[10]** 10_PDLCViT → [doi.org/10.1007/s44196-024-00597-3](https://doi.org/10.1007/s44196-024-00597-3) → *"A Multitask Learning-Based Vision Transformer for Plant Disease Localization and Classification"*
- **[11]** 11_AlexNet-ResNet50-EfficientNetB0 → [doi.org/10.1038/s41598-025-32607-5](https://doi.org/10.1038/s41598-025-32607-5) → *"Deep learning-based disease detection in potato and mango leaves: a comparative study of CNN, AlexNet, ResNet, and EfficientNet"*
- **[12]** 12_CustomMobileNetV3Large → [doi.org/10.1007/s44163-025-00733-8](https://doi.org/10.1007/s44163-025-00733-8) → *"A customized MobileNetV3Large-based deep learning framework for plant disease detection"*
- **[13]** 13_GrapeLeafNet → [doi.org/10.1109/ACCESS.2024.3361044](https://doi.org/10.1109/ACCESS.2024.3361044) → *"GrapeLeafNet: A Dual-Track Feature Fusion Network With Inception-ResNet and Shuffle-Transformer for Accurate Grape Leaf Disease Identification"*
- **[14]** 14_VGG19 → [doi.org/10.1038/s41598-024-72197-2](https://doi.org/10.1038/s41598-024-72197-2) → *"Advancing plant leaf disease detection integrating machine learning and deep learning"*
- **[15]** 15_ViT → [doi.org/10.11591/ijece.v15i2.pp2334-2344](https://doi.org/10.11591/ijece.v15i2.pp2334-2344) → *"Plant disease detection using vision transformers"*

**Biện pháp AI / XAI (không có folder):**

- **[16]** → [doi.org/10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474) → *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*
- **[17]** → [doi.org/10.1109/WACV48630.2021.00318](https://doi.org/10.1109/WACV48630.2021.00318) → *"Rotate to Attend: Convolutional Triplet Attention Module"*
- **[18]** → *(NeurIPS 2017, không có DOI)* → *"Attention Is All You Need"*
- **[19]** → [doi.org/10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063) → *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*
- **[20]** → [doi.org/10.18653/v1/2023.emnlp-main.298](https://doi.org/10.18653/v1/2023.emnlp-main.298) → *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"*
- **[21]** → *(NeurIPS 2019, không có DOI)* → *"Root Mean Square Layer Normalization"*
- **[22]** → [doi.org/10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74) → *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization"*
- **[23]** → [doi.org/10.18653/v1/2020.acl-main.385](https://doi.org/10.18653/v1/2020.acl-main.385) → *"Quantifying Attention Flow in Transformers"*
- **[24]** → [doi.org/10.1007/978-3-319-10590-1_53](https://doi.org/10.1007/978-3-319-10590-1_53) → *"Visualizing and Understanding Convolutional Networks"*

---

## Tóm tắt

- **24 references**: `[1]–[15]` = 15 bài phục dựng (khớp folder `01–15`), `[16]–[24]` = 9 biện pháp AI/XAI.
- **15/15 bài SOTA có DOI verify**; 2 bài NeurIPS ([18], [21]) không có DOI proceedings.
- Body ↔ References khớp đúng `[1]–[24]`, không có reference chết.
- Đã sửa: tác giả đầu **[8] = Wang** (không phải Gong).
- Đã bỏ (2026-07-14): SENet, ECA-Net (chỉ ví dụ chung ở §2, không load-bearing cho method).
