# Kế hoạch triển khai — sửa `DoAnTotNghiep_ChiliEdgeNet.docx` theo yêu cầu template mới

Tài liệu này mô tả **cách sẽ sửa**, dựa trên các vấn đề nêu trong `review-2-yeu-cau-moi-template.md`. Chưa thực hiện bước nào — cần xác nhận trước khi bắt đầu.

Nguyên tắc chung:
- Backup file gốc trước khi sửa (bản backup 08/09 đã cũ, không phản ánh trạng thái hiện tại).
- Sửa bằng cách `unzip` → sửa trực tiếp `word/document.xml` / `footer_page.xml` / `footer1.xml` → `zip` lại (không dùng Word để tránh vỡ layout với nhiều ảnh/footnote/hyperlink hiện có).
- Sau mỗi phase, render PDF và xem ảnh từng trang liên quan để xác minh trước khi qua phase kế.
- Không đổi bất cứ gì thuộc phần "Đã đạt" (mục 7 trong bản review) — chỉ đụng vào đúng các mục bị gắn cờ.

---

## Phase 0 — Chờ xác nhận từ GVHD/khoa (làm trước, không cần chờ để bắt đầu Phase 1–3)

Cần hỏi 4 việc:
1. Mẫu/nội dung cụ thể cho trang "Thông tin hội đồng chấm Đồ án tốt nghiệp" (bản yêu cầu mới không kèm mẫu).
2. Đề cương chi tiết có ký CBHD — còn phải đóng kèm bản nộp cứng không, dù không còn trong danh sách cấu trúc chính.
3. Cấu trúc 8 chương tùy biến hiện tại có được chấp nhận, hay cần đổi tên/gộp lại theo đúng 8 mục chuẩn (MỞ ĐẦU/TỔNG QUAN/.../PHỤ LỤC)?
4. Tài liệu tham khảo 100% tiếng Anh — có cần thêm tiêu đề phụ "Tài liệu tiếng Anh" hay chỉ cần xếp alphabet là đủ?

Chỉ có **Phase 4 (trang hội đồng)** thực sự bị chặn bởi câu trả lời #1. Các phase khác có thể làm song song trong lúc chờ.

---

## Phase 1 — Backup & chuẩn bị

```
cp DoAnTotNghiep_ChiliEdgeNet.docx DoAnTotNghiep_ChiliEdgeNet_BACKUP_2026-09-11.docx
unzip -q DoAnTotNghiep_ChiliEdgeNet.docx -d unpacked/
python scripts/merge_runs.py unpacked/    # gộp run rời rạc để dễ tìm/sửa text trong XML
```

---

## Phase 2 — Sửa đánh số trang (ưu tiên cao nhất, ảnh hưởng nhiều trang)

Hiện trạng kỹ thuật (đã xác minh):
- 3 `sectPr` trong `document.xml`. Section 1 (bìa) dùng `footer1.xml` (rỗng, không số) + có `pgBorders` + `<w:pgNumType w:start="1"/>`. Section 2 và 3 đều dùng `footer_page.xml` — có field `PAGE`, canh phải (`<w:jc w:val="right"/>`), không có `pgNumType` riêng nên số trang chạy liên tục từ section 1.

Cần sửa thành:
- **Section "phần đầu"** (từ sau bìa chính đến hết Danh mục từ viết tắt — gồm cả bìa phụ mới thêm ở Phase 3 và trang hội đồng mới ở Phase 4): dùng `footer1.xml` (rỗng) cho toàn bộ, không hiện số trang ở bất kỳ trang nào trong nhóm này.
- **Section "thân bài"** (bắt đầu từ trang Tóm tắt): dùng `footer_page.xml` nhưng
  - đổi `<w:jc w:val="right"/>` → `<w:jc w:val="center"/>`
  - thêm `<w:pgNumType w:start="1"/>` vào `sectPr` của chính section này (dời từ section bìa xuống đây)
- Section bìa chính: giữ `footer1.xml`, bỏ `<w:pgNumType w:start="1"/>` (không cần nữa vì không hiện số).

Có thể cần thêm 1 section break mới (ranh giới giữa "bìa+phần đầu" và "thân bài" hiện đang nằm giữa section 1 và 2 — cần kiểm tra section break đó có đúng nằm ngay trước Tóm tắt hay đang nằm ở vị trí khác trước khi sửa).

---

## Phase 3 — Thêm trang Bìa phụ

- Dựng theo mẫu "MẪU TRANG PHỤ BÌA KHÓA LUẬN" trong Phụ lục 2 cũ (đổi "KHÓA LUẬN" → "ĐỒ ÁN" cho khớp cách gọi hiện tại của file), gồm: ĐHQG/Trường/Khoa, tên SV — MSSV, "ĐỒ ÁN TỐT NGHIỆP", tên đề tài VN + Anh, ngành, GVHD, TP.HCM năm.
- Khác bìa chính: **không có khung viền trang** (`pgBorders`), không có logo lớn ở giữa (theo đúng mẫu phụ bìa cũ).
- Chèn làm trang mới ngay sau trang bìa chính (trang 1), trước trang "Thông tin hội đồng" (Phase 4).

---

## Phase 4 — Dựng lại trang "Thông tin hội đồng chấm Đồ án tốt nghiệp"

- Chờ kết quả Phase 0 #1.
- Nếu chưa có mẫu kịp: dùng tạm cấu trúc bảng nhận xét theo Phụ lục 2 cũ (tên khóa luận, SV thực hiện — MSSV, cán bộ hướng dẫn/phản biện, 4 mục đánh giá, đánh giá chung, điểm) nhưng đổi tiêu đề thành "THÔNG TIN HỘI ĐỒNG CHẤM ĐỒ ÁN TỐT NGHIỆP", và note rõ trong file là "chờ mẫu chính thức từ khoa" để dễ thay sau.
- Thay thế hoàn toàn trang "NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN" hiện tại (trang 2).

---

## Phase 5 — Sửa style 3 tiêu đề thiếu trong Mục lục

- Trong `document.xml`, tìm 3 đoạn `KẾT LUẬN`, `TÀI LIỆU THAM KHẢO`, `PHỤ LỤC A: ...` — hiện dùng style Title.
- Đổi `w:pStyle` sang cùng style Heading 1 mà "TỔNG QUAN", "CƠ SỞ LÝ THUYẾT"... đang dùng.
- Kiểm tra `numbering.xml`: nếu style Heading 1 hiện tại có gắn numbering tự động ("Chương N"), cần dùng biến thể Heading 1 **không đánh số chương** cho 3 mục này (để không bị tự thêm "Chương 9. KẾT LUẬN"), tương tự cách style Title cũ đã tránh việc này — chỉ đổi style thôi có thể chưa đủ, cần kiểm tra kỹ outline level + numPr.

---

## Phase 6 — Sắp xếp lại Tài liệu tham khảo theo alphabet

Đây là việc tốn công và dễ sai nhất — cần làm cẩn thận theo thứ tự:
1. Trích toàn bộ 29 entry `[1]`...`[29]` hiện tại ra một bảng tạm (số cũ, tên tác giả đầu, nội dung).
2. Sắp xếp lại theo alphabet họ/tên tác giả đầu tiên → gán số mới `[1]`...`[29]`. Lập bảng ánh xạ số cũ → số mới.
3. Quét toàn bộ `document.xml` tìm mọi trích dẫn `[n]` trong thân bài (đoạn văn, chú thích bảng/hình, footnote) — thay theo bảng ánh xạ. **Phải thay đồng thời** (dùng placeholder tạm hoặc thay theo thứ tự giảm dần độ dài số để tránh `[1]` thay nhầm vào trong `[10]`...`[19]`).
4. Viết lại khối Tài liệu tham khảo theo thứ tự alphabet mới, số hiệu mới.
5. Rà lại toàn bộ Chương 2 (Cơ sở lý thuyết), Chương 3 (Các nghiên cứu liên quan), phần Phụ lục — đây là nơi trích dẫn dày đặc nhất — so khớp thủ công từng số.

---

## Phase 7 — Cập nhật fields

- Đảm bảo `updateFields` bật trong `settings.xml` để Word hỏi cập nhật khi mở (Mục lục, Danh mục hình, Danh mục bảng phụ thuộc số trang mới sau Phase 2/5).
- Nếu cần file "sẵn sàng nộp" mà không phụ thuộc việc mở bằng Word: cân nhắc cập nhật số trang thủ công trong 3 danh mục sau khi mọi thay đổi cấu trúc đã chốt.

---

## Phase 8 — Kiểm tra & xác minh (bắt buộc, trước khi coi là xong)

- `scripts/office/validate.py` — kiểm tra XML hợp lệ.
- Render lại PDF, xem toàn bộ các trang từ bìa đến hết Danh mục từ viết tắt: xác nhận **không trang nào hiện số**.
- Xem trang Tóm tắt: số trang phải là "1", canh giữa.
- Xem 5–10 trang rải rác trong thân bài: số tăng dần đúng, canh giữa.
- Mở Mục lục sau khi update field: xác nhận đủ Kết luận/TLTK/Phụ lục A, số trang khớp thực tế.
- Kiểm tra ngẫu nhiên 5–10 trích dẫn `[n]` trong thân bài khớp đúng tài liệu sau khi đổi số ở Phase 6.
- Đối chiếu lại toàn bộ bảng mục 1 trong bản review — tất cả các dòng phải chuyển thành ✅.

---

## Thứ tự khuyến nghị thực hiện

Phase 1 → 2 → 3 → 5 → 6 → 7 → (Phase 4 khi có mẫu) → 8.

Lý do: Phase 4 (trang hội đồng) là phase duy nhất bị chặn bởi câu trả lời từ GVHD/khoa, nên làm các phase không phụ thuộc trước, tránh mất thời gian chờ.
