# Rà soát định dạng — DoAnTotNghiep_ChiliEdgeNet.docx

Ngày rà soát: 2026-09-06
Đối chiếu với: `UIT_Phu-luc-2_Hinh-thuc-trinh-bay-KLTN.docx` và `UIT_Phu-luc-3_Mau-bao-cao.docx` (mẫu trường cung cấp trong thư mục "template của trường").
File được rà soát: `khoa luận/kết quả/DoAnTotNghiep_ChiliEdgeNet.docx` (77 trang).

## Đã đạt (không cần sửa)
- Lề trang đúng chuẩn: trên 3cm, dưới 3.5cm, trái 3.5cm, phải 2cm.
- Font Times New Roman toàn bài, cỡ 13pt cho nội dung, giãn dòng 1.5 lines — đúng.
- Tiêu đề chương (Heading 1): in đậm, cỡ 14 — đúng. Tiêu đề mục 2.x/2.x.x: in đậm, cỡ 13 — đúng.
- Đánh số bảng/hình theo quy tắc "số chương.số thứ tự" — đúng.
- Tổng độ dài 77 trang, nằm trong khoảng 50-100 trang yêu cầu.
- Thứ tự phần đầu (bìa chính, bìa phụ, nhận xét CBHD/CBPB, lời cảm ơn, mục lục, danh mục hình/bảng/viết tắt, tóm tắt) tồn tại đầy đủ và đúng trình tự tương đối.
- Tiêu đề "ĐỒ ÁN TỐT NGHIỆP" (thay vì "KHÓA LUẬN TỐT NGHIỆP") — **đã xác nhận với sinh viên là đúng**, không cần sửa.

## Lỗi/thiếu đã được sửa trực tiếp trong file `DoAnTotNghiep_ChiliEdgeNet_updated.docx`

### 1. Không có số trang — ĐÃ SỬA
Đã tạo section break trước trang Tóm tắt, thêm field PAGE vào footer, đánh số Ả-rập bắt đầu từ 1 tại trang Tóm tắt, tăng dần đến hết bài. Phần đầu (bìa, nhận xét, lời cảm ơn, mục lục...) không có số trang — đúng quy định.

### 2. Thiếu 2 trang bắt buộc — ĐÃ THÊM (cần điền thêm nội dung thật)
- **Danh sách Hội đồng bảo vệ khóa luận** — đã thêm trang với khung mẫu (Quyết định số..., 4 dòng Chủ tịch/Thư ký/Ủy viên/Ủy viên để trống). Cần trường/khoa điền khi có quyết định thành lập hội đồng.
- **Đề cương chi tiết** — đã thêm trang dạng bảng, đã điền sẵn Tên đề tài, Cán bộ hướng dẫn (ThS. Đỗ Minh Tiến), Sinh viên thực hiện (Tô Minh Hải – 25410197). Còn "Thời gian thực hiện", "Nội dung đề tài", "Kế hoạch thực hiện" và chữ ký CBHD cần bạn/GVHD bổ sung và ký tay trước khi in.

### 3. "KẾT LUẬN", "TÀI LIỆU THAM KHẢO", "PHỤ LỤC A" không nằm trong Mục lục — ĐÃ SỬA
Đã gán outline level 1 trực tiếp cho 3 tiêu đề này (giữ style Title, không dùng Heading 1 để tránh bị đánh số nhầm thành "Chương 9/10/11"). Sau khi cập nhật field (xem mục "Việc cần làm" bên dưới), cả 3 sẽ xuất hiện trong Mục lục.

### 4. Danh mục từ viết tắt chưa xếp theo alphabet — ĐÃ SỬA
Đã sắp lại theo A→Z: API, CI, CNN, CPU, DS, ECA, F1, FP32/FP16, GPU, GQA, INT8, LLM, MLP, pp, RMSNorm, ROI, RoPE, SE, SOTA, ViT, XAI.

### 5. Cỡ chữ trang bìa/bìa phụ — ĐÃ SỬA
| Dòng | Yêu cầu mẫu | Trước | Sau |
|---|---|---|---|
| ĐẠI HỌC QUỐC GIA TP.HCM | 14pt | 13pt | 14pt |
| TRƯỜNG ĐH CNTT / KHOA... | 16pt | 13pt | 16pt |
| Tên khóa luận (Việt) | 18-30pt | 16pt | 18pt |
| Tên khóa luận (Anh) | 15-25pt | 14pt | 15pt |
| "ĐỒ ÁN/KHÓA LUẬN TỐT NGHIỆP" | 16pt | 18pt | 16pt |
| Tên sinh viên | 14pt | 15pt | 14pt |
| GIẢNG VIÊN HƯỚNG DẪN + tên GVHD | 14pt | 13pt | 14pt |

### 6. Công thức Wilson CI dạng LaTeX thô (mục 2.7.1) — ĐÃ SỬA
Thay `\hat{p}_{Wilson} = \frac{...}` bằng biểu thức có ký hiệu mũ/chỉ số/±/√ thật: `p̂_Wilson = ( p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²) ) / (1 + z²/n)`.

### 7. Tài liệu tham khảo chưa tách Việt/Anh + chưa xếp alphabet — CHƯA SỬA
Vẫn đang dùng numbered IEEE theo thứ tự trích dẫn (chuẩn IEEE phổ biến). Nên hỏi lại GVHD/khoa vì nhiều khoa CNTT vẫn chấp nhận cách này — chưa động vào vì cần xác nhận trước.

## Việc cần bạn tự làm trong Word (bắt buộc)
Mở file `DoAnTotNghiep_ChiliEdgeNet_updated.docx`, bấm **Ctrl+A** rồi **F9** (chọn "Update entire table" nếu được hỏi) để cập nhật lại Mục lục, Danh mục Hình vẽ, Danh mục Bảng — vì tài liệu giờ có số trang thật lần đầu tiên nên các số trang cũ (15, 16, 76...) đã lệch, và 3 mục KẾT LUẬN/TÀI LIỆU THAM KHẢO/PHỤ LỤC A chỉ hiện ra trong mục lục sau bước này.

## Phát hiện thêm — ĐÃ SỬA HẾT
Đã quét lại toàn bộ tài liệu và sửa hết các công thức/ký hiệu dạng LaTeX thô còn sót lại (không chỉ Wilson CI), gồm:
- Mục 2.1: `f_\theta`, `\hat{y} = f_\theta(x) \in \{1,...,K\}` → `fθ`, `ŷ = fθ(x) ∈ {1, …, K}`.
- Công thức cross-entropy `\mathcal{L} = -\sum...` → `L = −Σk=1K yk log(ŷk)` với subscript/superscript thật.
- Mục 2.4.3: công thức RMSNorm và RMS(x) → đã có ⊙, √, subscript/superscript thật.
- Mục 2.7.3: `\alpha` (Bonferroni) và cả đoạn `p_{(1)} \le p_{(2)}...`, `\alpha/m`, `p_{Holm}` → đã có α, ≤, …, subscript thật.
- Mục 4.3: công thức Precision/Recall/F1 → gộp lại một dòng gọn, bỏ `\frac`, `\times`, `\text{...}`.
- Bảng 5.4, Bảng 5.5 (tiêu đề cột) và các đoạn văn ở Chương 5 + Kết luận nhắc `p_{Holm} = 0,0234 / 0,0105`, `p = 1,83 × 10^{-242}` → tất cả đã chuyển sang subscript/superscript thật, không còn dấu `\`, `_{`, `^{`, `$` nào trong toàn bộ tài liệu (đã quét lại xác nhận 0 chỗ còn sót).
