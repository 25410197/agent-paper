# Chẩn đoán lỗi template EN + 5 bài CEA mới tham khảo

## 1. Lỗi template đã tìm ra và đã sửa

**Triệu chứng:** trong `ChiliEdgeNet_CEA_Draft_EN.docx`/PDF, các heading ("1. Introduction", "4.1.1. ...") hiển thị bằng font khác (font mặc định theo theme của Word, dạng sans-serif) trong khi toàn bộ phần thân bài, tiêu đề bài báo, tên tác giả đều đúng Times New Roman. So với 9 bài CEA thật đã tải (xem mục 2), heading trong bài báo xuất bản luôn cùng font serif với thân bài — sự lệch font này là điểm khác biệt trực quan rõ nhất giữa bản thảo hiện tại và một bài báo CEA thật, và nhiều khả năng là điều thầy hướng dẫn gọi là "sai template".

**Nguyên nhân kỹ thuật:** `build_journal_docx_en.py` gán font cho heading bằng `doc.styles["Heading 1/2/3"].font.name = "Times New Roman"`. Cách này chỉ ghi đè thuộc tính `w:ascii`/`w:hAnsi` thường, nhưng style Heading mặc định của python-docx (kế thừa từ template Word chuẩn) còn giữ tham chiếu font theo theme (`w:asciiTheme="majorHAnsi"`, tức Calibri Light). Nhiều trình render (kể cả LibreOffice — công cụ dùng để xuất PDF review) ưu tiên thuộc tính theo theme khi cả hai cùng tồn tại, nên heading vẫn hiện font theme dù script đã "set" Times New Roman. Các đoạn dùng `run.font.name` trực tiếp (tiêu đề bài, tên tác giả, abstract, caption) không bị lỗi này vì set ở cấp run, không qua style.

**Đã sửa trong `build_journal_docx_en.py`:**
- Thêm hàm `_force_style_font()` xoá các thuộc tính `*Theme` khỏi `w:rFonts` của style Normal/Heading 1-3/List Bullet/List Number, đồng thời set cứng `w:ascii/w:hAnsi/w:cs/w:eastAsia` = Times New Roman.
- Thêm vòng lặp ép `run.font.name = "Times New Roman"` trên từng run của heading ngay sau khi tạo (lớp bảo hiểm thứ hai).
- Đã build thử và xuất PDF kiểm tra: heading "1. Introduction" nay hiển thị đúng Times New Roman, khớp thân bài (xem file `ChiliEdgeNet_CEA_Draft_EN_fixfont.docx` được tạo trong lúc kiểm tra — có thể xoá, bản build tiếp theo sẽ ghi đè đúng tên file gốc).

**Cần làm tiếp:** chạy lại `python build_journal_docx_en.py` trên máy của bạn (môi trường của tôi chỉ có quyền tạo file mới trong thư mục này, không ghi đè/xoá file cũ) để file `ChiliEdgeNet_CEA_Draft_EN.docx` chính thức được ghi đè với font đã sửa. `build_journal_docx_vn.py` gần như chắc chắn có cùng lỗi (cùng pattern code) — nên áp fix tương tự nếu bản VN cũng bị.

**Các phần đã đúng chuẩn CEA, không cần sửa** (đối chiếu trực tiếp với 9 PDF/bài tham khảo): một cột (FILE-02), References kiểu tác giả-năm có DOI, sắp theo alphabet (REF-01–04), caption bảng đặt phía trên/ghi chú phía dưới (TABLE-03), không kẻ dọc trong bảng (TABLE-04), mục lục đánh số 1., 2., 3. còn Abstract không đánh số (STRUCT-01).

## 2. Năm bài báo CEA mới đã tải/đọc (bổ sung 4 bài đã có trong `template/`)

Đã lưu full text (HTML → markdown, vì tải PDF trực tiếp từ ScienceDirect cần phiên trình duyệt JS nên không tải nhị phân PDF được qua công cụ tự động; đã lấy nguyên văn qua trình duyệt thay vì chỉ xem tóm tắt) vào `en/template/`:

1. `AICropCAM_2023_compag_fulltext.md` — AICropCAM (Chamara et al., 2023) — edge-computing camera stack 4 mô hình, CEA vol 215.
2. `Barley_FSL_Transformer_2025_compag_fulltext.md` — Few-shot Swin transformer cho bệnh lúa mạch (Rezaei et al., 2025) — CEA vol 229.
3. `PestPVT_2025_compag_fulltext.md` — Pest-PVT, PVTv2 cho phát hiện sâu hại (Chen et al., 2025) — CEA vol 230, có thử nghiệm trên Jetson TX2.
4. `HSI_CNNTransformer_2025_compag_fulltext.md` — Gate-Shift-Fuse CNN-Transformer cho ảnh siêu phổ (Guerri et al., 2025) — CEA vol 237.
5. `Survey_LeafDisease_2025_compag_fulltext.md` — "Past, present and future of deep plant leaf disease recognition" (George et al., 2025) — CEA vol 234, bài tổng quan PRISMA.

Mỗi file có kèm phần "Style notes relevant to ChiliEdgeNet" — ghi lại các quy ước hành văn/cấu trúc cụ thể có thể áp dụng trực tiếp (ví dụ: mọi số cải thiện luôn đi kèm baseline cụ thể đang so sánh; Discussion diễn giải lại lý do chứ không lặp số liệu của Results; danh sách "contributions" đặt ngay sau đoạn nêu khoảng trống nghiên cứu; v.v.).

## 3. Ghi chú về `Elsevier-template.doc`

File `en/template/Elsevier-template.doc` bạn mới tải là template Elsevier **cũ** (2001/2017) dành cho tạp chí *Computer Physics Communications* — 2 cột sau Abstract, reference đánh số [1][2][3]. Đây **không phải** style thật của CEA (CEA dùng author-year, nộp bản thảo 1 cột). Không nên dùng file này làm chuẩn.

## 4. Đề xuất bước tiếp theo

Việc "viết lại cho chuẩn văn bản học thuật" ở mức toàn bộ ~10.000 từ là khối lượng lớn và động đến số liệu/claim đã được review kỹ (theo `JOURNAL_TEMPLATE_RULES_CEA.md` và `REVIEW_TONG_QUAT_VN.md`), nên tôi chưa tự ý viết lại toàn văn. Nếu bạn muốn, tôi có thể làm tiếp theo từng mục (Abstract → Introduction → ...), đối chiếu từng câu với văn phong của 9 bài tham khảo, và giữ nguyên mọi số liệu/phạm vi claim đã chốt.
