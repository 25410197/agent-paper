Kiểm tra chéo notebook đã sinh so với bài báo gốc để đánh giá độ chính xác phục dựng.

**Tên bài báo:** $ARGUMENTS

---

**Bước đầu tiên — kiểm tra đầu vào:**

Kiểm tra theo thứ tự:

1. `papers/$ARGUMENTS/outputs/` — tìm file `.ipynb`
2. `papers/$ARGUMENTS/inputs/paper_content.md` — nội dung bài báo đã extract (tùy chọn)
3. `papers/$ARGUMENTS/inputs/*.pdf` — PDF gốc (fallback nếu thiếu paper_content.md)

Nếu thiếu notebook → dừng và thông báo:
```
❌ Không tìm thấy notebook trong papers/$ARGUMENTS/outputs/
Hãy chạy /reproduce $ARGUMENTS trước để sinh notebook.
```

Nếu thiếu `paper_content.md` nhưng có PDF trong `inputs/`:
- Thông báo: "⚠️ Không tìm thấy paper_content.md — sẽ extract trực tiếp từ PDF."
- Agent `notebook-verifier` sẽ tự extract nội dung từ PDF (text + hình ảnh kiến trúc) rồi tiến hành verify.
- Ưu tiên đọc PDF: dùng `pdftotext -layout` cho text, render hình ảnh kiến trúc bằng `render_pdf_page.py` khi cần.

Nếu đủ điều kiện (có notebook + paper_content.md hoặc PDF) → dùng agent `notebook-verifier` kiểm tra toàn bộ 6 phase và xuất báo cáo `[ModelName]_VerificationReport.md` vào `papers/$ARGUMENTS/outputs/`.
