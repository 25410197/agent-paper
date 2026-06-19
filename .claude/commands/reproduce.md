Phục dựng bài báo khoa học từ PDF thành notebook Kaggle chạy được.

**Tên bài báo:** $ARGUMENTS

---

**Bước đầu tiên — kiểm tra PDF:**

Tìm file PDF tại `papers/$ARGUMENTS/inputs/`. Nếu không tìm thấy file `.pdf` nào trong thư mục đó → dừng ngay và thông báo:
```
❌ Không tìm thấy PDF trong papers/$ARGUMENTS/inputs/
Vui lòng đặt file PDF vào đúng thư mục trên rồi chạy lại.
```

Nếu tìm thấy → đặt `PDF_PATH = papers/$ARGUMENTS/inputs/<tên-file>.pdf` rồi chạy pipeline:

1. **Extract PDF** — dùng agent `pdf-extractor` đọc `PDF_PATH`, lưu vào `papers/$ARGUMENTS/inputs/paper_content.md`
2. **Phân tích** — dùng agent `paper-brainstorm` (Opus) trích xuất kiến trúc + hyperparameters, tạo `papers/$ARGUMENTS/inputs/implement_plan.md`; nếu có thông số còn thiếu thì hỏi người dùng bằng tiếng Việt trước khi tiếp tục
3. **Sinh notebook** — dùng agent `kaggle-reproducer` viết `[ModelName]_Experiment.ipynb` + `[ModelName]_ReproduceSummary.md` vào `papers/$ARGUMENTS/outputs/`

Tuân thủ toàn bộ quy tắc trong `.claude/rules/01_core_constraints.md`.
