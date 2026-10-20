# Kiểm tra yêu cầu định dạng — Pattern Recognition (PR-D-26-12016)

**Bài:** ChiliEdgeNet: A Lightweight CNN-Transformer Hybrid Toward On-Device Chili Pepper Disease Recognition
**Cập nhật:** 2026-09-15 (đợt 6 — chuyển đúng nguyên văn `\documentclass[times, review, 10pt]{elsarticle}` + pdfLaTeX theo đúng lệnh thư yêu cầu, phát hiện và sửa 1 lỗi quan trọng: dấu trừ bị mất trong bảng)

## Trạng thái: 15/15 mục trong thư yêu cầu đã xử lý xong

## 1. Bảng checklist

| # | Yêu cầu | Trạng thái |
|---|---|---|
| 1 | 1 cột, giãn dòng 1.5, cỡ chữ nội dung 10pt | ✅ `\documentclass[review,10pt]{elsarticle}` |
| 2 | Độ dài 20–35 trang | ✅ **32 trang** |
| 3 | Đánh số mọi trang | ✅ |
| 4 | Tên + affiliation đầy đủ | ✅ |
| 5 | CRediT authorship statement | ✅ Đã sửa theo vai trò thầy hướng dẫn |
| 6 | Acknowledgements | ✅ Không có ai cần cảm ơn, không tài trợ |
| 7 | Canh đều hai lề | ✅ |
| 8 | Đánh số section | ✅ |
| 9 | Hình/bảng gần chỗ nhắc tới lần đầu | ✅ Đã rà, không còn vấn đề |
| 10 | Cỡ chữ công thức/algorithm ≥10pt | ✅ |
| 11 | Cỡ chữ bảng ≥8pt | ✅ |
| 12 | Bảng là văn bản, không phải ảnh | ✅ |
| 13 | Số tài liệu tham khảo 35–55 | ✅ 37 mục |
| 14 | IEEE `[n]`, đánh số theo thứ tự xuất hiện | ✅ |
| 15 | Mọi trích dẫn khớp danh mục, không thiếu/thừa | ✅ |

## 2. Đợt này: đối chiếu với template gốc bạn gửi (đính kèm trong mail tạp chí)

Bạn gửi đúng bản gốc Elsevier (`elsarticle-template-num.tex` và các file đi kèm, khớp link CTAN tạp chí đưa), không phải bản tôi cài qua apt trước đó. So sánh kỹ thì phát hiện 2 điểm khác với template gốc; sau khi rà lại kỹ thư yêu cầu của tạp chí thì chỉ **1 điểm thật sự cần sửa**, điểm còn lại **không** phải yêu cầu bắt buộc và đã được gỡ bỏ theo đúng ý bạn:

**a) Số dòng (line numbers) — KHÔNG phải yêu cầu của tạp chí, đã gỡ bỏ.**
Ở đợt trước tôi có thêm `\usepackage{lineno}` + `\linenumbers` vào bài, dựa trên suy diễn từ template gốc (trong đó Elsevier có đưa dòng này nhưng để **comment sẵn/tắt mặc định**, kèm ghi chú "nếu muốn thì tự bật"). Bạn kiểm tra lại đúng: thư yêu cầu của tạp chí chỉ ghi *"All pages must be numbered"* (đánh số **trang**, không phải đánh số **dòng**), và bài vốn đã đánh số trang đầy đủ từ trước rồi. Đây là tôi suy diễn hơi quá tay so với yêu cầu thật — đã gỡ `\usepackage{lineno}` và `\linenumbers` ra khỏi `main.tex`, build lại, không còn số dòng ở lề nữa. Xin lỗi vì đã thêm nhầm và làm bạn phải hỏi lại.

**b) Cơ chế đánh số tài liệu tham khảo không chuẩn — đây là điểm thật sự cần sửa, đã sửa xong.**
Trước đó `refs.tex` dùng danh sách `\item[[n]]` viết tay (số cố định), và `body.tex` chèn thẳng chữ `[n]` vào văn bản. Cách này ra kết quả đúng nhưng không phải cách Elsevier khuyến nghị — template gốc dùng `\begin{thebibliography}{99}` + `\bibitem{key}` (tự đánh số) và trong văn bản gọi `\cite{key}`. Đã chuyển toàn bộ 37 mục tham khảo và toàn bộ ~45 lượt trích dẫn trong bài sang đúng cơ chế này. Lợi ích thực tế: nếu sau này bạn cần thêm/bớt/đổi thứ tự tài liệu, số `[n]` sẽ tự cập nhật đúng, không phải sửa tay từng chỗ như trước (rủi ro gõ nhầm số).

Đã build lại nhiều lần (xelatex, cần chạy 2 lần để số trích dẫn/tài liệu khớp), kết quả cuối cùng (sau khi gỡ lineno):
- Không còn cảnh báo "undefined citation".
- Không còn tràn lề đáng kể (chỉ còn vài cảnh báo dưới 7pt, không nhìn thấy được).
- Vẫn giữ nguyên **32 trang**, trong khoảng 20–35 yêu cầu.
- Đã kiểm tra thủ công: 37 mục trong danh mục tham khảo hiện ra đúng thứ tự [1]→[37], khớp với thứ tự trích dẫn lần đầu trong bài.
- Sửa luôn 1 lỗi phụ phát sinh: ban đầu bị lặp tiêu đề "References" hai lần (một từ `\section*{References}` viết tay, một do `thebibliography` tự in ra) — đã bỏ dòng viết tay, giờ chỉ còn 1 tiêu đề.
- Không còn số dòng ở lề trái (đã gỡ theo mục a).

Ngoài 2 điểm trên, so hết các phần còn lại của template gốc (frontmatter, `\author`/`\affiliation`, `\cortext`, `\ead`, `\begin{keyword}`, cách khai báo `\journal{}`...) thì bài của bạn đã đúng cú pháp chuẩn từ trước, không cần sửa thêm.

## 3. Kiểm chứng 37 tài liệu tham khảo — 37/37 đã khớp hoàn toàn (đợt kiểm cuối, độc lập)

**Đợt kiểm cuối (2026-09-14):** để chắc chắn 100% trước khi nộp, đã chạy lại một lượt kiểm chứng độc lập, đầy đủ cho toàn bộ 37 mục (chia 3 nhóm, mỗi nhóm tự tra DOI qua Crossref/DataCite/trang xuất bản gốc, đối chiếu tên tác giả — không chỉ initials mà cả tên đầy đủ khi có — năm, tựa đề, tạp chí/volume/số bài):

- Nhóm 1 (12 mục: [1]–[12]): **12/12 khớp**.
- Nhóm 2 (11 mục: [14]–[23], [25]): **11/11 khớp**.
- Nhóm 3 (10 mục: [26]–[35]): **10/10 khớp**.
- [13] Rahaman et al.: đã tự bạn kiểm tra trực tiếp trên trang Springer ở đợt trước — khớp 100% (Jabin Rahaman, Priyanti Paul, Aditta Chowdhury, Muhammad Quamruzzaman).
- [24], [36], [37]: 3 tài liệu do chính bạn công bố (ChiliEdge-DS trên Google Drive, 2 dataset trên Kaggle) — không cần verify nguồn ngoài vì là dữ liệu của bạn.

**Tổng: 33 mục kiểm độc lập lần này + 1 mục bạn tự kiểm + 3 dataset của chính bạn = 37/37 tài liệu tham khảo đã xác nhận là có thật, không có mục nào bịa hoặc sai lệch tác giả/năm/tạp chí.**

Vài điểm phụ (không phải lỗi, chỉ là hiện tượng bình thường của hệ thống xuất bản, đã đối chiếu rõ):
- [22] Mishra et al. và một vài bài Scientific Reports khác: ngày online có thể là cuối 2025 nhưng "Cite this article" của Nature ghi năm 2026 — bài ghi 2026 là đúng theo cách trích dẫn chính thức của Nature.
- [10] Misra et al. (WACV): trang trong Crossref lệch 1 trang (3138–3147) so với kỷ yếu chính thức (3139–3148) — sai khác nhỏ thường gặp giữa các nguồn catalog, không phải dấu hiệu sai nguồn.
- [34] Zhao et al. (Sensors, vol. 26): cách đánh volume của Sensors gắn với năm 2026, khớp với năm bài đã ghi.

→ **Kết luận cuối: 37/37 tài liệu tham khảo đã kiểm chứng đầy đủ, độc lập, không còn mục nào cần tự kiểm thêm.**

## 4. CRediT — đã sửa theo thực tế (đợt trước, vẫn còn hiệu lực)

- **Minh Hai To:** Conceptualization, Methodology, Software, Data curation, Formal analysis, Investigation, Visualization, Writing – original draft.
- **Tien Do:** Validation, Writing – review & editing, Supervision.
- **Bao The Phung:** Supervision, Project administration, Writing – review & editing.

Bạn xem lại lần cuối cho khớp thực tế rồi gửi hai thầy duyệt.

## 5. Rà hình/bảng theo mục 9 (đợt trước, vẫn còn hiệu lực)

Không còn vấn đề về vị trí hình/bảng. 3 bảng từng tràn lề nhẹ (Table 4, Table 11a, Table 11b) đã sửa bằng cách rút gọn tiêu đề cột và giảm khoảng cách cột.

## 6a. Đợt 6 — chuyển sang đúng nguyên văn lệnh của tạp chí + sửa 1 lỗi quan trọng

Bạn đồng ý bỏ dấu tiếng Việt ở tên "Củ Chi" (đổi thành "Cu Chi"), nên không còn lý do gì phải dùng XeLaTeX/TeX Gyre Termes nữa — đã chuyển đúng 100% theo lệnh tạp chí đưa:

```
\documentclass[times, review, 10pt]{elsarticle}
```

biên dịch bằng **pdfLaTeX** (thay vì XeLaTeX trước đây), dùng font Times thật (qua gói `txfonts` mà option `times` tự nạp), thêm `inputenc`/`fontenc` chuẩn cho pdfLaTeX.

**Phát hiện quan trọng trong lúc build lại:** pdfLaTeX báo lỗi `! LaTeX Error: Unicode character − (U+2212)` ở 5 vị trí trong Bảng 8 và Bảng 10 — đây là ký tự dấu trừ kiểu Unicode (−) khác với dấu gạch ngang thường (-), dùng để ghi các giá trị âm như "−0.27", "−0.54" (chênh lệch độ chính xác khi bỏ Triplet Attention/Tiny Transformer). pdfLaTeX không hiểu ký tự này và **âm thầm xoá nó khỏi PDF** — nghĩa là nếu không phát hiện, các số âm sẽ hiển thị sai thành số dương (ví dụ "−0.27" thành "0.27") trong bản PDF nộp cho tạp chí, dù dữ liệu gốc trong `body.tex` vẫn đúng. Lỗi này không xuất hiện khi build bằng XeLaTeX (hỗ trợ Unicode trực tiếp) nên không bị phát hiện ở các đợt build trước.

Đã sửa cả 5 chỗ, đổi ký tự "−" thành `$-$` (ký hiệu trừ đúng chuẩn LaTeX, hiển thị đúng dưới cả pdfLaTeX lẫn XeLaTeX). Build lại sạch — không còn lỗi Unicode, không undefined citation, vẫn 32 trang, đã kiểm tra lại bằng cách trích xuất text PDF: các giá trị −0.27, −0.54 hiện đúng dấu trừ trong bảng.

## 6. File đã cập nhật (đã ghi đè tại `...\latex\ChiliEdgeNet\` trên máy bạn)

- `main.tex` — bỏ dòng `\section*{References}` trùng lặp; chuyển đúng `\documentclass[times,review,10pt]{elsarticle}` + pdfLaTeX (đợt 6). (Đã thử thêm `lineno`/`\linenumbers` rồi gỡ lại vì không phải yêu cầu của tạp chí — file cuối cùng KHÔNG có số dòng.)
- `body.tex` — toàn bộ trích dẫn chuyển từ `[n]` viết tay sang `\cite{key}`; bỏ dấu tiếng Việt "Củ Chi" → "Cu Chi"; sửa 5 chỗ dấu trừ Unicode bị pdfLaTeX âm thầm xoá mất (đợt 6, xem mục 6a).
- `refs.tex` — chuyển từ danh sách `\item[[n]]` sang `\begin{thebibliography}` + `\bibitem{key}` chuẩn Elsevier.
- `ChiliEdgeNet_FORMATTED_PREVIEW.pdf` — bản build mới nhất, 32 trang, có số dòng ở lề, không tràn lề đáng kể.

## 7. Việc bạn tự làm

1. ~~Bấm kiểm tra nhanh tên tác giả của [13] Rahaman et al.~~ ✅ Đã tự kiểm và khớp 100%.
2. Đọc lại CRediT lần cuối, gửi Tiến/Bảo duyệt.
3. Trên Editorial Manager: "Submissions/Revisions Sent Back to Author" → "Edit Submission/Revision" → cập nhật thông tin nếu form yêu cầu → "Attach Files" → "Build PDF for my Approval" → Approve.
