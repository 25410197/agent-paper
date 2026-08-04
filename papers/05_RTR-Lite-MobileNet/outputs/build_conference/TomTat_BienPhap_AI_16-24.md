# Tóm tắt [16]–[24]: mỗi biện pháp AI là gì + áp dụng thế nào trong bài

> Giải thích vì sao mỗi ref được trích dẫn (dùng để trả lời reviewer / thầy nếu bị hỏi "cite cái này làm gì").
> Nhóm theo vai trò: **Xương sống → Khối chú ý CNN (kế thừa) → Lõi Transformer (đề xuất) → XAI (diễn giải)**.
> **Cập nhật:** đã bỏ SENet và ECA-Net (trước là [17]/[18]) — chỉ là ví dụ chung ở §2, không mô tả trong phần method; RES_Block đã quy attribution về bài gốc [5]. Còn lại đánh số lại [19–26 cũ] → [17–24 mới].

## 1. Xương sống (backbone)

**[16] MobileNetV2** — Sandler 2018 — [doi.org/10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
- **Là gì:** CNN nhẹ dùng *inverted residual* + *linear bottleneck* + tích chập tách chiều (depthwise separable), giảm mạnh tham số/FLOPs.
- **Áp dụng:** Là **xương sống trích đặc trưng** của mô hình. Bài **cắt sớm** MobileNetV2 (chỉ giữ tới lớp cho ra bản đồ đặc trưng nhỏ) để giữ nhẹ, rồi đưa các ô đặc trưng này vào khối Transformer như "token ảnh". Toàn bộ mô hình đề xuất xây **trên nền** MobileNetV2 → bắt buộc trích dẫn.

## 2. Khối chú ý CNN (kế thừa từ mô hình gốc RTR_Lite [5])

**[17] Triplet Attention** — Misra 2021 — [doi.org/10.1109/WACV48630.2021.00318](https://doi.org/10.1109/WACV48630.2021.00318)
- **Là gì:** Chú ý bắt **tương tác chéo** giữa chiều không gian và chiều kênh qua 3 nhánh hoán vị, gần như không tốn tham số.
- **Áp dụng:** Khối chú ý CNN **giữ lại trước Transformer** để lọc đặc trưng; là 1 trong 2 khối chú ý CNN được đánh giá trong phần ablation (khối còn lại là RES_Block, thuộc mô hình gốc [5], không cần trích dẫn riêng SENet/ECA).

## 3. Lõi khối Transformer (3 kỹ thuật LLM — chính là ĐÓNG GÓP đề xuất)

**[18] Attention Is All You Need (Transformer)** — Vaswani 2017 — *(NeurIPS, không có DOI)*
- **Là gì:** Kiến trúc **Transformer / self-attention** gốc.
- **Áp dụng:** Nền tảng khái niệm — khối "Tiny Transformer" của bài theo đúng tinh thần Transformer/ViT (coi ô đặc trưng là token, tự chú ý toàn cục). Trích để chỉ rõ gốc của cơ chế.

**[19] RoFormer — RoPE (Rotary Position Embedding)** — Su 2024 — [doi.org/10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063)
- **Là gì:** Mã hóa vị trí bằng cách **xoay** vector query/key theo góc phụ thuộc tọa độ, không thêm tham số học.
- **Áp dụng:** Bài mở rộng thành **2D-RoPE** (áp riêng cho trục dọc và ngang của lưới đặc trưng) để Transformer biết vị trí 2 chiều của từng vùng ảnh. **1 trong 3 kỹ thuật lõi đề xuất.**

**[20] GQA — Grouped-Query Attention** — Ainslie 2023 — [doi.org/10.18653/v1/2023.emnlp-main.298](https://doi.org/10.18653/v1/2023.emnlp-main.298)
- **Là gì:** Nhiều đầu **query** chia sẻ ít đầu **key/value** hơn → giảm tham số & bộ nhớ của lớp chú ý mà gần như không mất độ chính xác.
- **Áp dụng:** Transformer của bài dùng **4 đầu query chia sẻ 2 đầu key/value** để giữ tổng tham số < 1 triệu. **1 trong 3 kỹ thuật lõi.**

**[21] RMSNorm — Root Mean Square Normalization** — Zhang 2019 — *(NeurIPS, không có DOI)*
- **Là gì:** Chuẩn hóa theo căn quân phương, **nhẹ hơn LayerNorm** (bỏ bước trừ trung bình và tham số dịch).
- **Áp dụng:** Thay LayerNorm trong khối Transformer → nhẹ và ổn định hơn với mạng nông. **1 trong 3 kỹ thuật lõi.**

## 4. XAI — diễn giải mô hình (mục 5.1, mỗi kỹ thuật 1 hình)

**[22] Grad-CAM** — Selvaraju 2017 — [doi.org/10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
- **Là gì:** Bản đồ nhiệt **dựa trên gradient** — cho biết vùng ảnh nào làm tăng điểm lớp dự đoán.
- **Áp dụng:** XAI **nhóm gradient**, diễn giải nhánh **CNN** (Hình 2): xác nhận mô hình nhìn đúng mô bệnh, bỏ qua nền.

**[23] Attention Rollout** — Abnar 2020 — [doi.org/10.18653/v1/2020.acl-main.385](https://doi.org/10.18653/v1/2020.acl-main.385)
- **Là gì:** Gộp ma trận chú ý qua các lớp để lần theo **luồng chú ý** của Transformer.
- **Áp dụng:** XAI **nhóm chú ý**, diễn giải đúng **khối Transformer mới đề xuất** (Hình 3): xác nhận Transformer khóa vào đúng đối tượng (lá/quả), không nhiễu.

**[24] Occlusion Sensitivity** — Zeiler & Fergus 2014 — [doi.org/10.1007/978-3-319-10590-1_53](https://doi.org/10.1007/978-3-319-10590-1_53)
- **Là gì:** **Nhiễu loạn** — che lần lượt từng vùng ảnh, đo mức giảm điểm số; vùng che vào làm sụt nhiều nhất là vùng quan trọng. Không cần đạo hàm, độc lập mô hình.
- **Áp dụng:** XAI **nhóm nhiễu loạn** (Hình 4): bằng chứng thứ 3, độc lập về cơ chế với Grad-CAM và Attention Rollout → 3 nhóm cơ chế khác nhau cùng đồng thuận = độ tin cậy cao.

---

## Bảng gọn (một dòng/ref)

| Ref | Tên | Nhóm | Áp dụng trong bài |
|---|---|---|---|
| [16] | MobileNetV2 | Backbone | Xương sống CNN (cắt sớm) sinh token ảnh cho Transformer |
| [17] | Triplet Attention | Chú ý CNN | Lọc đặc trưng trước Transformer (có trong ablation) |
| [18] | Transformer | Nền tảng | Khối Tiny Transformer theo tinh thần ViT/self-attention |
| [19] | RoPE | **Lõi đề xuất** | 2D-RoPE mã hóa vị trí 2 chiều, không tham số |
| [20] | GQA | **Lõi đề xuất** | 4 query / 2 key-value → giảm tham số |
| [21] | RMSNorm | **Lõi đề xuất** | Thay LayerNorm, nhẹ hơn |
| [22] | Grad-CAM | XAI gradient | Diễn giải nhánh CNN (Hình 2) |
| [23] | Attention Rollout | XAI chú ý | Diễn giải khối Transformer (Hình 3) |
| [24] | Occlusion | XAI nhiễu loạn | Bằng chứng độc lập cơ chế (Hình 4) |

**Kết:** không ref nào thừa — [16] là nền, [17] là khối chú ý CNN thực sự nằm trong mô hình, [19]–[21] chính là 3 kỹ thuật tạo nên tên "LiteFormer", [22]–[24] là 3 XAI thuộc 3 nhóm cơ chế độc lập dùng để kiểm chứng.

*(SENet và ECA-Net đã bỏ khỏi bài ngày 2026-07-14: chỉ là ví dụ chung trong Related Work, không mô tả trong phần kiến trúc — bỏ không ảnh hưởng attribution vì RES_Block đã quy về bài gốc [5].)*
