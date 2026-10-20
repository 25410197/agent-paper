# Rà soát lần 2 — Đối chiếu `DoAnTotNghiep_ChiliEdgeNet.docx` với yêu cầu template mới

Ngày rà soát: 2026-09-11
Tài liệu yêu cầu mới: `khoa luận/yêu cầu mới template/phuluc2_hinhthuctrinhbay (3).docx`
Đối chiếu thêm: `khoa luận/template của trường/UIT_Phu-luc-2_Hinh-thuc-trinh-bay-KLTN.docx` (bản cũ, để biết cái gì đã đổi)
File được rà soát: `khoa luận/kết quả/DoAnTotNghiep_ChiliEdgeNet.docx` — **83 trang** (tăng từ 77 trang ở lần rà soát 06/09 do đã bổ sung nội dung mục 7.1–7.3)
Rà soát trước đó: `review-DoAnTotNghiep_ChiliEdgeNet.md` (06/09) + `nhat-ky-chinh-sua-DoAnTotNghiep.md` (08/09, ghi lại các sửa đổi đã làm)

Phương pháp: đọc trực tiếp `word/document.xml`, `footer*.xml`, `sectPr` (margin, section break, page-number field) bằng script, và render toàn bộ file ra ảnh từng trang để xem trực quan phần đầu + các trang liên quan.

---

## 0. Điểm khác biệt chính của **yêu cầu mới** so với bản cũ

Đây là những điểm cần chú ý đặc biệt vì bản cũ (Phụ lục 2 cũ) đã dùng để rà soát lần 1 — nay đã thay đổi:

| Nội dung | Bản cũ (Phụ lục 2 cũ) | Bản mới |
|---|---|---|
| Danh sách hội đồng + Nhận xét CBHD + Nhận xét CBPB (3 trang riêng) | Bắt buộc, 3 trang tách biệt | **Gộp thành 1 mục duy nhất: "Thông tin hội đồng chấm Đồ án tốt nghiệp"** — không còn liệt kê "Nhận xét CBHD" / "Nhận xét CBPB" như 2 trang riêng trong CẤU TRÚC |
| Đề cương chi tiết có chữ ký CBHD | Bắt buộc, đóng ngay sau Lời cảm ơn | **Không còn xuất hiện trong danh sách CẤU TRÚC của bản mới** (có thể vẫn cần khi nộp bản đóng bìa cứng — bản mới không có phần "NỘP KHOÁ LUẬN" để đối chiếu, xem mục 5 bên dưới) |
| Bìa phụ | Bắt buộc | **Vẫn bắt buộc, liệt kê riêng** — không đổi |
| Danh mục bảng, sơ đồ, hình vẽ | 1 mục gộp chung | **Tách thành 2 mục riêng: "Danh mục hình" và "Danh mục bảng"** |
| Không đánh số trang các trang | Lời cảm ơn, nhận xét CBHD/CBPB, Đề cương chi tiết, các mục lục | Thông tin hội đồng, Lời cảm ơn, Mục lục, danh mục hình/bảng/viết tắt (đổi tên mục cho khớp cấu trúc mới, nội dung ý nghĩa như cũ) |
| Margin / font / line spacing / đánh số trang giữa-dưới / đánh số bảng-hình / quy cách TLTK (IEEE, tách VN/EN, alphabet) | — | **Giữ nguyên y hệt bản cũ**, không đổi |
| Tên gọi | "Khóa luận" | Đổi thành "Đồ án" xuyên suốt (khớp với tiêu đề "ĐỒ ÁN TỐT NGHIỆP" mà file hiện đang dùng — **không cần sửa gì ở đây**) |

Tóm lại: yêu cầu mới chủ yếu thay đổi **cấu trúc phần đầu** (gộp trang nhận xét/hội đồng, bỏ đề cương khỏi danh sách, tách danh mục hình/bảng) và giữ nguyên toàn bộ quy cách trình bày kỹ thuật (margin, font, số trang, TLTK).

---

## 1. Bảng đối chiếu CẤU TRÚC phần đầu — yêu cầu mới vs. hiện trạng file

| # | Yêu cầu mới | Hiện trạng trong file | Trạng thái |
|---|---|---|---|
| 1 | Bìa chính Đồ án | Trang 1, có khung viền, đủ nội dung | ✅ Đạt |
| 2 | **Bìa phụ Đồ án** | **Không có** — đã bị xóa ở lần sửa 08/09 vì cho là "trùng bìa chính" | ❌ **Thiếu — cần thêm lại** vì bản mới liệt kê đây là mục bắt buộc riêng |
| 3 | **Thông tin hội đồng chấm Đồ án tốt nghiệp** | Trang 2 hiện là "NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN" — chỉ có tiêu đề + các dòng kẻ chấm trống, không có bảng thông tin hội đồng/CBHD/CBPB/điểm số nào | ⚠️ **Không khớp nội dung yêu cầu** — cần mẫu cụ thể (xem mục 5) |
| — | *(không có trong yêu cầu)* Lời cam đoan | Trang 3 — "LỜI CAM ĐOAN" | Không sai (nội dung tốt, nhiều trường chấp nhận) nhưng không bắt buộc; **đang bị đánh số trang** dù nằm trước Tóm tắt |
| 4 | Lời cảm ơn | Trang 4 | ✅ Đạt nội dung, nhưng đang bị đánh số (xem mục 2) |
| 5 | Mục lục | Trang 5–8 | ⚠️ Thiếu 3 mục (KẾT LUẬN, TÀI LIỆU THAM KHẢO, PHỤ LỤC A) — xem mục 3 |
| 6 | Danh mục hình | Trang 9 | ✅ Đạt, đã tách riêng đúng yêu cầu mới |
| 7 | Danh mục bảng | Trang 10–11 | ✅ Đạt, đã tách riêng đúng yêu cầu mới |
| 8 | Danh mục từ viết tắt (theo alphabet) | Trang 12–13 | ✅ Đạt — đã xếp A→Z (API, CI, CNN, CPU, DS, ECA, F1... đúng thứ tự) |
| 9 | Nội dung Đồ án | Tóm tắt (tr.14) → Chương 1–8 → Kết luận → TLTK → Phụ lục | ✅ Đủ nội dung; xem mục 4 về cấu trúc chương |

---

## 2. Đánh số trang — vẫn sai, nghiêm trọng nhất

Yêu cầu mới (giữ nguyên yêu cầu cũ):
- Không đánh số: trang Thông tin hội đồng, Lời cảm ơn, Mục lục, các danh mục hình/bảng/viết tắt.
- **Bắt đầu đánh số từ trang Tóm tắt Đồ án**, số Ả-rập, **ở giữa bên dưới** (center).

Kiểm tra trực tiếp trong file (`footer_page.xml`, `sectPr`):
- Toàn bộ các trang từ trang 2 (Nhận xét) đến hết đều hiển thị số trang liên tục: trang 2 hiện số "2", trang 4 (Lời cảm ơn) hiện "4", trang 5 (Mục lục) hiện "5", trang 12–13 (Danh mục viết tắt) hiện "12"/"13", và **Tóm tắt hiện số "14"** thay vì phải là "1".
- Vị trí số trang hiện đang canh **phải** (`<w:jc w:val="right"/>` trong `footer_page.xml`), trong khi yêu cầu là **canh giữa**.
- Đây là do lần sửa 08/09 đã cố tình theo "bài mẫu" (đóng góc dưới bên phải, đánh liên tục từ 1, không restart ở Tóm tắt) thay vì theo văn bản quy định chính thức của trường — và bài mẫu ở đây không đáng tin hơn văn bản quy định mới vừa nhận được.

→ Cần làm lại: các trang trước Tóm tắt (bìa phụ, thông tin hội đồng, lời cam đoan nếu giữ, lời cảm ơn, mục lục, danh mục hình/bảng/viết tắt) **không hiện số trang**; số trang bắt đầu = 1 tại trang Tóm tắt, canh giữa bên dưới.

---

## 3. Mục lục thiếu 3 mục — vẫn chưa sửa

`KẾT LUẬN`, `TÀI LIỆU THAM KHẢO`, `PHỤ LỤC A` trong file vẫn dùng style **Title** (kiểm tra bằng cách export sang Markdown: 3 dòng này không được nhận là heading trong khi "TỔNG QUAN", "CƠ SỞ LÝ THUYẾT"... đều là heading) → không xuất hiện trong Mục lục hiện tại. Đây là lỗi đã nêu ở lần rà soát trước (06/09), nhật ký sửa đổi (08/09) chưa ghi nhận đã xử lý mục này.

→ Đổi style 3 tiêu đề này sang Heading 1, sau đó cập nhật lại Mục lục.

---

## 4. Cấu trúc 8 chương tùy biến — vẫn là câu hỏi mở, chưa đổi

Yêu cầu mới liệt kê đúng 8 phần chuẩn giống bản cũ: MỞ ĐẦU → TỔNG QUAN → NGHIÊN CỨU THỰC NGHIỆM HOẶC LÍ THUYẾT → TRÌNH BÀY ĐÁNH GIÁ KẾT QUẢ → KẾT LUẬN → HƯỚNG PHÁT TRIỂN → TÀI LIỆU THAM KHẢO → PHỤ LỤC.

File hiện dùng 8 chương đặt tên khác (Tổng quan → Cơ sở lý thuyết → Các nghiên cứu liên quan → Phương pháp đề xuất → Thực nghiệm và kết quả → Diễn giải mô hình → Triển khai ứng dụng → Hạn chế và hướng phát triển), sau đó mới có Kết luận/TLTK/Phụ lục riêng. Về nội dung là tương đương và đây là cách làm phổ biến ở nhiều đồ án CNTT, nhưng **không có xác nhận bằng văn bản là GVHD/khoa đã chấp nhận** — không có gì mới từ tài liệu yêu cầu mới để giải quyết câu hỏi này. **Vẫn nên hỏi lại GVHD trước khi quyết định giữ nguyên hay đổi tên/gộp lại theo đúng 8 mục chuẩn.**

---

## 5. Thông tin hội đồng chấm Đồ án tốt nghiệp — thiếu mẫu cụ thể để làm theo

Tài liệu yêu cầu mới chỉ ghi tên mục "Thông tin hội đồng chấm Đồ án tốt nghiệp" trong danh sách cấu trúc, **không kèm mẫu/bảng cụ thể** (khác với bản Phụ lục 2 cũ có kèm hẳn mẫu "NHẬN XÉT KHÓA LUẬN TỐT NGHIỆP (CỦA CÁN BỘ HƯỚNG DẪN/PHẢN BIỆN)" với bảng tên khóa luận, SV thực hiện, đánh giá 4 mục, điểm số...).

Ngoài ra, tài liệu yêu cầu mới **không có phần "NỘP KHOÁ LUẬN"** (quy định về đĩa CD, đóng bìa cứng, việc "Đề cương chi tiết phải đóng kèm khi nộp") như bản cũ — không rõ là do bản mới chỉ trích đoạn về hình thức trình bày (cắt bỏ phần nộp bài) hay quy định nộp bài đã thực sự đổi.

**→ Cần hỏi GVHD/khoa 2 việc trước khi dựng lại trang này:**
1. Xin mẫu cụ thể (nếu có) cho trang "Thông tin hội đồng chấm Đồ án tốt nghiệp", hoặc xác nhận đây có phải là gộp 3 mục cũ (Danh sách hội đồng + Nhận xét CBHD + Nhận xét CBPB) thành 1 trang hay không, và nội dung tối thiểu cần có (tên hội đồng, chức danh, hay vẫn là nhận xét dạng dòng kẻ trống).
2. Đề cương chi tiết có chữ ký CBHD — còn bắt buộc đóng kèm khi nộp bản cứng không, dù không còn trong danh sách cấu trúc chính của bản mới.

---

## 6. Tài liệu tham khảo — vẫn chưa alphabet hoá

Yêu cầu (không đổi so với bản cũ, giờ đã rõ ràng hơn): theo chuẩn IEEE, **tách riêng tài liệu tiếng Việt / tiếng Anh, xếp theo alphabet tên tác giả**.

Kiểm tra 29 tài liệu hiện tại: toàn bộ đều là tài liệu tiếng Anh (không có tài liệu tiếng Việt nào), và đang xếp theo **thứ tự trích dẫn xuất hiện trong bài** (IEEE numeric — [1] Duhan, [2] Sandler, [3] Misra, [4] Su...), không theo alphabet tên tác giả.

→ Sắp xếp lại 29 tài liệu theo alphabet họ tác giả đầu tiên (giữ định dạng trích dẫn kiểu IEEE, chỉ đổi thứ tự + số hiệu `[n]`, sau đó rà lại toàn bộ các chỗ trích dẫn `[n]` trong thân bài để trỏ đúng số mới — đây là việc tốn công nhất trong toàn bộ kế hoạch). Vì không có tài liệu tiếng Việt, có thể không cần tiêu đề phụ "Tài liệu tiếng Anh" — nên hỏi GVHD để chắc chắn.

---

## 7. Những gì đã đạt / đã tự sửa đúng — không cần đụng vào

- Margin: trên 3 cm / dưới 3.5 cm / trái 3.5 cm / phải 2 cm — đúng (kiểm tra trực tiếp trong `sectPr`, khớp mọi section).
- Font Times New Roman 13pt, line spacing 1.5 — đúng.
- Cỡ chữ trang bìa (ĐHQG 14pt, Trường/Khoa 16pt, "ĐỒ ÁN TỐT NGHIỆP" 16pt, tên đề tài VN 18pt/Anh 15pt, GVHD+SV+ngành 14pt) — đã sửa đúng theo lần chỉnh 08/09, xác nhận lại bằng ảnh trang bìa.
- Công thức Wilson CI (mục 2.7.1) — đã chuyển thành ảnh công thức chuẩn, không còn hiển thị cú pháp LaTeX thô.
- Danh mục từ viết tắt — đã xếp alphabet A→Z.
- Danh mục hình / Danh mục bảng — đã tách thành 2 danh mục riêng (vô tình khớp sẵn với yêu cầu mới).
- Đánh số bảng/hình theo quy tắc "số chương.số thứ tự" — đúng.
- Tổng độ dài 83 trang, trong khoảng 50–100 trang yêu cầu.
- Tiêu đề "ĐỒ ÁN TỐT NGHIỆP" khớp với cách gọi "Đồ án" xuyên suốt tài liệu yêu cầu mới.

---

## Tóm tắt việc cần làm (theo mức độ ưu tiên)

| Ưu tiên | Việc cần làm | Cần hỏi GVHD trước? |
|---|---|---|
| Cao | Sửa lại đánh số trang: ẩn số ở các trang trước Tóm tắt, canh giữa, restart = 1 tại Tóm tắt | Không |
| Cao | Thêm lại trang Bìa phụ | Không |
| Cao | Dựng lại trang "Thông tin hội đồng chấm Đồ án tốt nghiệp" thay cho trang "Nhận xét GVHD" hiện tại | **Có** — cần mẫu/nội dung cụ thể |
| Trung bình | Đổi style KẾT LUẬN / TÀI LIỆU THAM KHẢO / PHỤ LỤC A sang Heading 1 + cập nhật Mục lục | Không |
| Trung bình | Sắp xếp lại 29 tài liệu tham khảo theo alphabet + rà lại toàn bộ trích dẫn `[n]` trong bài | Có thể hỏi về việc có cần tách heading "Tiếng Anh" khi 0 tài liệu tiếng Việt |
| Thấp | Xác nhận Đề cương chi tiết có còn cần đóng kèm bản nộp cứng không | **Có** |
| Thấp (không đổi từ lần trước) | Xác nhận cấu trúc 8 chương tùy biến so với 8 mục chuẩn | **Có** |
