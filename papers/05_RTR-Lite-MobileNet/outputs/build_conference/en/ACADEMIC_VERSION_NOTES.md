# Bản Word mới: `ChiliEdgeNet_CEA_Draft_EN_academic.docx`

Bản này được build từ nội dung gốc trong `en/` (qua `_BaoCao_Full_EN.md` → `_BaoCao_Full_EN_academic.md`), đối chiếu văn phong/trình bày với 9 bài CEA thật (4 bài cũ + 5 bài mới tải trong `en/template/`). Hai thay đổi áp dụng, cả hai đều **chỉ về hình thức, không đổi bất kỳ số liệu/claim nào**:

1. **Sửa lỗi font heading** (đã mô tả trong `TEMPLATE_DIAGNOSIS_AND_5_NEW_REFS.md`) — heading nay đúng Times New Roman, khớp thân bài.
2. **Bỏ bôi đậm tràn lan trong văn xuôi.** Đối chiếu 9 bài CEA thật: không bài nào bôi đậm số liệu/tên mô hình trong Abstract, Introduction, Results, Discussion — bôi đậm chỉ dùng cho heading, tiêu đề bảng ("Table N."), và nhãn mở đoạn kiểu "**Prompt used.**"/"**Few-shot setup.**" (run-in label, cũng là quy ước chuẩn học thuật thấy trong bài Barley few-shot). Bản gốc bôi đậm gần như mọi con số ("**98.93%**", "**671,173 parameters**"...) trong cả văn xuôi — không giống bài báo đã xuất bản, giống văn phong ghi chú/review hơn. Đã bỏ bôi đậm này, giữ nguyên bôi đậm hợp lệ (heading, "Table N." prefix, ô kết quả tốt nhất trong bảng — đúng quy ước CEA).
3. Thêm 1 câu "roadmap" cuối phần Introduction (Section 2 trình bày..., Section 3..., v.v.) — quy ước phổ biến trong cả 9 bài tham khảo, chưa có trong bản gốc.

**Đã kiểm chứng:** so sánh văn bản gốc và bản mới sau khi bỏ hết ký tự `*` — hai file **giống hệt nhau về chữ**, xác nhận không có nội dung/số liệu nào bị thay đổi trong bước này.

**Script build:** `build_journal_docx_en_academic.py` (bản sao của `build_journal_docx_en.py`, chỉ đổi đường dẫn input/output). Chạy lại bằng `python build_journal_docx_en_academic.py` nếu chỉnh sửa thêm `_BaoCao_Full_EN_academic.md`.

**Việc chưa làm:** đây là bước chuẩn hoá hình thức/typography, chưa phải viết lại câu chữ ở mức diễn đạt sâu hơn (đổi cấu trúc câu, gộp/tách đoạn...). Rủi ro khi viết lại sâu hơn là có thể vô tình đổi phạm vi claim đã được review kỹ (theo `REVIEW_TONG_QUAT_VN.md`), nên tôi dừng ở mức an toàn này trừ khi bạn muốn đi tiếp từng mục.

## Vì sao bản Word 1 cột, còn 9 bài PDF tham khảo lại 2 cột?

Đây là 2 giai đoạn khác nhau của cùng một bài báo, không phải lỗi template:

- **Bản nộp (Word)** — cái mình đang làm — theo Guide for Authors CEA: *"Format Word files in a single-column layout. Double-column formatting is only permitted for LaTeX submissions."* → bắt buộc 1 cột.
- **Bản public (PDF)** — 9 file trong `en/template/` — là bản đã qua khâu dàn trang của Elsevier sau khi bài được chấp nhận: 2 cột, có logo tạp chí, "Contents lists available at ScienceDirect", số trang... Tác giả không tự tạo phần này.

Nên: bản Word 1 cột hiện tại là **đúng chuẩn nộp bài**, còn 2 cột chỉ xuất hiện ở giai đoạn xuất bản sau này, do tòa soạn làm.
