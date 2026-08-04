# Checklist review tổng quát bản thảo VN

Ngày review: **04/08/2026**  
Phạm vi: `outputs/build_conference/vn/`  
Trạng thái tổng quát: **chưa sẵn sàng nộp**.

Quy ước:

- `[ ]`: chưa xử lý.
- `[~]`: đã xử lý một phần hoặc đang chờ kết quả mới.
- `[x]`: hoàn tất và đã kiểm tra lại.
- **P0**: chặn nộp bài.
- **P1**: ưu tiên cao, nên hoàn tất trước bản cuối.
- **P2**: biên tập, dọn dẹp hoặc cải thiện khả năng tái lập.

---

## P0 — Các việc chặn nộp bài

### P0.1. Đánh giá lại thí nghiệm cross-dataset trên query sạch

- [x] Checkpoint few-shot hiện có đã được huấn luyện với **seed 42**; tên tệp trước đây chỉ gây hiểu nhầm, không phải khác seed.
- [x] Checkpoint few-shot dùng cùng kiến trúc và hyperparameter với checkpoint gốc; khác biệt chủ đích là bổ sung 42 ảnh support vào dữ liệu huấn luyện.
- [x] Checkpoint few-shot đã được lưu riêng, không ghi đè checkpoint gốc `c14_s2_seed42.pth`.
- [x] Trong bài chỉ định danh đây là checkpoint few-shot dùng seed 42 và ghi SHA-256; không đưa chi tiết lỗi đặt tên tệp vào nội dung học thuật. Việc đổi tên local thành `c14_s2_fewshot_seed42.pth` là tùy chọn quản lý artifact; nếu đổi, cập nhật metadata và kiểm tra SHA-256 không đổi.
- [x] Đã đánh giá checkpoint gốc và checkpoint few-shot trên đúng Kaggle query version 1 gồm **1.463 ảnh**.
- [x] Đã xác nhận `D:\ds\test` có đúng 1.463 ảnh, giao SHA-256 giữa support và query bằng **0**, không thiếu và không thừa hash so với query manifest.
- [x] Đã lưu prediction từng ảnh cho cả hai checkpoint trong `cross_dataset_predictions.csv`.
- [~] Đã lưu hash checkpoint, hash manifest và cấu hình tiền xử lý; còn cần lưu log/config huấn luyện đã thực thi để chứng minh seed 42, kiến trúc, hyperparameter và bổ sung thời gian chạy đánh giá.
- [x] Đã tính lại accuracy, precision-macro, recall-macro, F1-macro và McNemar ghép cặp.
- [x] Đã tạo lại hai confusion matrix sạch.
- [x] Đã thay Bảng 10b và Hình S1–S2 trong [`vn/05_06_cross_dataset.md`](vn/05_06_cross_dataset.md).
- [x] Đã cập nhật các số cross-dataset trong [`vn/01_tom_tat.md`](vn/01_tom_tat.md), [`vn/02_dat_van_de.md`](vn/02_dat_van_de.md), [`vn/07_han_che.md`](vn/07_han_che.md) và [`vn/08_ket_luan.md`](vn/08_ket_luan.md).
- [x] Không còn dùng các metric sơ bộ 30,94% và 89,69%; bản thảo đã thay bằng kết quả sạch 30,96% và 89,75%.
- [x] Upload gói `chili-mendeley-cross-query` thành Kaggle version mới; sau đó cập nhật version/URL trong README, CITATION và mục dữ liệu Hải (2026a) trong References.

**Tiêu chí hoàn thành:** lưu log/metadata chứng minh hai checkpoint dùng cùng seed 42 và cùng cấu hình nền; đánh giá cả hai trên cùng query sạch 1.463 ảnh; support/query không trùng hash; toàn bộ bảng, hình và số liệu trong bài đã được thay bằng kết quả mới.

### P0.2. Chốt phạm vi tuyên bố “thiết bị biên”

- [x] Chọn phạm vi tuyên bố phù hợp với bằng chứng hiện có:
  - [x] Đổi tiêu đề thành “hướng tới triển khai trên thiết bị biên”; thay vì tuyên bố đã benchmark trên thiết bị biên.
  - **Không áp dụng trong phạm vi bài hiện tại:** chưa benchmark trên điện thoại, Raspberry Pi, Jetson hoặc thiết bị nhúng tương đương; nội dung này được giữ là hướng phát triển.
- [x] Không dùng CPU máy chủ Kaggle hoặc CPU laptop như bằng chứng cho hiệu năng trên thiết bị biên; các con số này được ghi đúng theo phần cứng đã đo.
- [x] Đồng bộ tiêu đề tiếng Việt trong [`vn/00_tieu_de.md`](vn/00_tieu_de.md) và tiêu đề tiếng Anh trong [`vn/11_title_page.md`](vn/11_title_page.md).
- [x] Tóm tắt, kết luận và highlights chỉ xem mô hình là ứng viên cho phần cứng hạn chế tài nguyên và nêu benchmark thiết bị biên vật lý là công việc tương lai.

**Tiêu chí hoàn thành:** tiêu đề và claim phần cứng không vượt quá bằng chứng thực nghiệm thực tế.

### P0.3. Hoàn thiện provenance và giấy phép ChiliEdge-DS

- [ ] Ghi thời gian thu thập ảnh.
- [x] Ghi địa điểm thu thập ở mức có thể công bố: khu vực Củ Chi, Thành phố Hồ Chí Minh, Việt Nam.
- [x] Ghi thiết bị/camera và độ phân giải ảnh gốc: camera điện thoại thông minh, không lưu model, xấp xỉ 12.000 × 9.000 pixel.
- [ ] Mô tả tiêu chí chọn và loại ảnh.
- [x] Mô tả cách xác định năm lớp theo biểu hiện thị giác và giới hạn không định danh tác nhân bằng xét nghiệm.
- [~] Đã ghi tác giả gán nhãn thủ công; còn cần xác nhận có hay không bước duyệt bởi chuyên gia nông nghiệp.
- [x] Mô tả quy tắc chia train/validation/test một lần theo tỷ lệ xấp xỉ 65:20:15 và giữ cố định cho toàn bộ thí nghiệm.
- [ ] Kiểm tra exact duplicate bằng SHA-256 giữa ba tập.
- [ ] Kiểm tra near-duplicate hoặc ảnh liên tiếp của cùng đối tượng giữa ba tập.
- [ ] Ghi kết quả kiểm toán leakage vào paper hoặc supplementary material.
- [ ] Chỉ định giấy phép công khai cho ChiliEdge-DS trên Kaggle.
- [ ] Cập nhật license trong mục dữ liệu Hải (2026c); ngày truy cập đã được bổ sung.
- [ ] Xóa dòng `Ghi chú biên tập` khỏi [`vn/04_phuong_phap_de_xuat.md`](vn/04_phuong_phap_de_xuat.md) sau khi hoàn tất.

**Tiêu chí hoàn thành:** độc giả có thể hiểu nguồn gốc, cách gán nhãn, cách chia dữ liệu, tình trạng leakage và quyền tái sử dụng ChiliEdge-DS.

### P0.4. Xử lý việc lựa chọn mô hình bằng test

- **Không áp dụng theo phương án hiện tại:** không chạy lại quy trình chọn RTR_Lite_MobileNet hoặc cấu hình đại diện của nghiên cứu [14] chỉ bằng validation.
- **Không thể thực hiện hậu nghiệm:** không thể đóng băng lựa chọn trước lần đánh giá test đã xảy ra.
- [x] Công khai hai trường hợp có sự tham gia của test và giữ cách gọi “phân tích thăm dò”.
- [x] Không dùng chênh lệch +0,80 pp để khẳng định ưu thế tổng quát hoặc thứ hạng thống kê đã xác lập.
- [x] Đã kiểm tra và đồng bộ nội dung trong [`vn/05_01_thiet_lap.md`](vn/05_01_thiet_lap.md), [`vn/05_02_so_sanh_mo_hinh.md`](vn/05_02_so_sanh_mo_hinh.md), [`vn/07_han_che.md`](vn/07_han_che.md) và [`vn/08_ket_luan.md`](vn/08_ket_luan.md).

**Tiêu chí hoàn thành:** tập test không tham gia lựa chọn cấu hình cho phép so sánh cuối cùng, hoặc giới hạn này được trình bày rõ và claim được hạ xuống mức thăm dò.

### P0.5. Làm sạch cấu trúc manuscript được merge

- [x] Không đưa danh sách link nội bộ “Các mục được tách thành file riêng” vào main manuscript.
- [x] Chỉ giữ tiêu đề `## 4. Thực nghiệm và kết quả` trong [`vn/05_00_muc_luc.md`](vn/05_00_muc_luc.md).
- [x] `merge_sections.py` chỉ ghép các mục `00_`–`09_`, không ghép `10_highlights_en.md`, `10_highlights_vn.md` và `11_title_page.md` vào main manuscript.
- [x] Giữ Title Page và hai bản Highlights dưới dạng artifact Markdown riêng.
- [x] Xóa toàn bộ placeholder khỏi [`vn/11_title_page.md`](vn/11_title_page.md); giữ tên tác giả, affiliation và corresponding author đã biết. Email/ORCID chưa được cung cấp nên không tự suy đoán.
- [x] Đã chạy lại `python merge_sections.py`.
- [x] Xác nhận [`vn/_BaoCao_Full_VN.md`](vn/_BaoCao_Full_VN.md) kết thúc tại References [31], không có Highlights/Title Page nối phía sau.

**Tiêu chí hoàn thành:** main manuscript chỉ chứa nội dung bài báo theo đúng thứ tự; title page và highlights không nằm sau References.

---

## P1 — Ưu tiên cao

### P1.1. Sửa công thức độ đo

- [ ] Sửa ba công thức trong [`vn/04_phuong_phap_de_xuat.md`](vn/04_phuong_phap_de_xuat.md) sang khối LaTeX `\[` và `\]`.
- [ ] Bỏ các dòng `====` đang bị Markdown hiểu như heading.
- [ ] Viết công thức Accuracy, (F1_c) và Macro-F1 trên một cấu trúc thống nhất.
- [ ] Kiểm tra render trong DOCX/PDF sau khi build.

**Tiêu chí hoàn thành:** công thức hiển thị đúng, không tạo heading hoặc văn bản rời trong bản render.

### P1.2. Thống nhất phạm vi sử dụng McNemar

- [x] Đã sửa mâu thuẫn giữa [`vn/05_01_thiet_lap.md`](vn/05_01_thiet_lap.md) và [`vn/05_06_cross_dataset.md`](vn/05_06_cross_dataset.md).
- [x] Đã ghi rõ McNemar được dùng cho Bảng 5–7 và phép so sánh cross-dataset ghép cặp ở Bảng 10b.
- [x] P-value cross-dataset được tính trên hai checkpoint cùng seed và query sạch 1.463 ảnh.
- [x] Đã ghi rõ $b=870$, $c=10$, p-value chính xác và nêu đây là tương phản riêng, không thuộc họ kiểm định Holm của ablation.

**Tiêu chí hoàn thành:** không còn câu nói McNemar “không được dùng” cho cross-check trong khi Mục 4.5 vẫn báo p-value.

### P1.3. Chuẩn hóa citation và References

- [x] Chuyển sang reference style tác giả–năm của CEA; References không đánh số và được sắp theo alphabet.
- [x] Kiểm tra lại toàn bộ citation: 30/30 mục References đều được trích dẫn và không còn citation số trong thân bài.
- [x] Xóa reference Grad-CAM của Selvaraju et al. (2017) vì bài không sử dụng Grad-CAM.
- [x] Đã thêm citation/model card chính thức cho `gemini-3.1-flash-lite` và nêu rõ mô hình được gọi qua Gemini API với API key quản lý bằng Google AI Studio.
- [x] Giữ model ID, ngày truy cập, prompt và cấu hình sinh trong phần đối chiếu Gemini.
- [x] Đã bổ sung Heo et al. (2025) làm nguồn trực tiếp cho axial 2D-RoPE; giữ Su et al. (2024) làm nguồn RoPE 1D nền tảng.
- [x] Đã đối chiếu metadata của 32/32 tài liệu với Crossref, trang nhà xuất bản hoặc kho dữ liệu gốc; đã sửa tác giả, năm, volume/issue, article number và nguồn hội nghị bị thiếu hoặc sai.
- [ ] Kiểm tra mọi DOI/URL còn hoạt động.

**Tiêu chí hoàn thành:** không có reference dư, citation thiếu hoặc số tài liệu sai thứ tự theo template nộp bài.

### P1.4. Hoàn thiện đối chiếu Gemini

- [ ] Chạy Gemini trên ba ảnh còn thiếu để đủ 373 ảnh, hoặc giữ rõ phạm vi 370 ảnh.
- [ ] Lưu raw response cho mọi ảnh.
- [ ] Lưu model ID, ngày chạy và response metadata.
- [ ] Đặt tường minh temperature, top-p, giới hạn token và structured output nếu API hỗ trợ; không phụ thuộc hoàn toàn vào default có thể thay đổi.
- [ ] Xác định quy tắc chấm lỗi API/parse trước khi chạy.
- [ ] Dùng thuật ngữ “mô hình đa phương thức” hoặc “mô hình thị giác–ngôn ngữ” nhất quán thay cho cách gọi LLM thị giác nếu phù hợp.
- [ ] Không diễn giải latency API và latency local như chi phí tính toán kiến trúc ngang hàng.

**Tiêu chí hoàn thành:** số mẫu, cấu hình API và quy tắc chấm có thể kiểm toán; claim vẫn được giới hạn là đối chiếu tham khảo.

### P1.5. Bổ sung bằng chứng về tính nhẹ

- [ ] Báo MACs/FLOPs tại kích thước đầu vào 224×224.
- [ ] Báo peak memory hoặc ít nhất RAM khi suy luận batch 1.
- [ ] Ghi backend, precision và số thread CPU.
- [ ] Phân biệt latency suy luận thuần, tiền xử lý–suy luận và toàn pipeline camera.
- [ ] Không quy đổi 49,41 ảnh/giây thành FPS camera.
- [ ] Nếu có lượng tử hóa, benchmark bằng backend thực sự hỗ trợ INT8/FP16.

**Tiêu chí hoàn thành:** claim “lightweight” không chỉ dựa vào số tham số và kích thước file.

### P1.6. Bổ sung sơ đồ kiến trúc

- [ ] Tạo sơ đồ luồng: ảnh → truncated MobileNetV2 → Conv 1×1 → 196 token → Tiny Transformer → mean pooling → classifier.
- [ ] Ghi kích thước tensor tại các điểm chính.
- [ ] Thể hiện cấu hình 4 query head và 2 key/value head.
- [ ] Thể hiện axial 2D-RoPE và ba vị trí RMSNorm nếu sơ đồ không quá rối.
- [ ] Đặt hình tại Mục 3.2 và cập nhật lại số hình phía sau.

**Tiêu chí hoàn thành:** người đọc hiểu kiến trúc mà không phải suy luận từ danh sách bước bằng văn bản.

---

## P2 — Biên tập và dọn dẹp

### P2.1. Chuẩn hóa thuật ngữ

- [ ] Chọn cách dùng nhất quán cho `accuracy`/“độ chính xác”.
- [ ] Chọn cách dùng nhất quán cho `output`/“đầu ra”.
- [ ] Giới thiệu “support set” và “query set” bằng tiếng Việt ở lần xuất hiện đầu.
- [ ] Dùng “Transformer” và “CNN–Transformer” thống nhất về dấu nối.
- [ ] Chuẩn hóa cách viết `few-shot`, `cross-dataset`, `Macro-F1` và `F1-macro`.
- [ ] Đổi ngày `08/07/2026` thành dạng không mơ hồ, ví dụ `2026-07-08` hoặc “08 tháng 7 năm 2026”.

### P2.2. Dọn file không thuộc manuscript khỏi `vn/`

- [ ] Chuyển file `vn/new_ds` sang thư mục provenance/result phù hợp.
- [ ] Đổi tên thành `.md` hoặc `.txt` rõ nghĩa, ví dụ `cross_dataset_source_notes.md`.
- [ ] Không để ghi chú đường dẫn local `D:\ds\...` trong thư mục manuscript cuối.
- [ ] Chỉ giữ các file được merge hoặc artifact nộp riêng trong `vn/`.

### P2.3. Kiểm tra độ dài

- [x] Abstract hiện khoảng 232 từ, phù hợp nếu venue giới hạn 250 từ.
- [ ] Kiểm tra giới hạn trang/từ chính thức của venue.
- [ ] Bản chính hiện khoảng 10.340 từ nếu tính cả References và bỏ Title Page/Highlights; rút gọn nếu vượt giới hạn.
- [ ] Ưu tiên rút phần liệt kê baseline, chi tiết prompt Gemini và cảnh báo lặp lại giữa Mục 4.5 với Mục 5.
- [ ] Cân nhắc chuyển prompt đầy đủ, bảng provenance và kết quả cross-dataset chi tiết sang supplementary material.

### P2.4. Kiểm tra bản render cuối

- [x] Chạy `python merge_sections.py`.
- [ ] Build lại DOCX/PDF.
- [ ] Kiểm tra ngắt trang, bảng rộng, công thức, caption và độ phân giải hình.
- [ ] Kiểm tra Hình 1–5, S1–S2 và Bảng 1–11 không bị trùng số.
- [ ] Kiểm tra mọi hình được nhắc đến trong thân bài trước hoặc gần vị trí xuất hiện.
- [ ] Kiểm tra mọi link hình local tồn tại.
- [ ] Kiểm tra main manuscript không còn TODO, GAP, placeholder hoặc `Ghi chú biên tập`.

---

## Các phần đã đạt ở lần review này

- [x] Cấu trúc Mục 2.1–2.8 đã tách rõ cơ sở lý thuyết và nghiên cứu liên quan.
- [x] Các số liệu chính 98,93%, 671.173 tham số, 20,24 ms và 369/373 nhất quán giữa các mục chính.
- [x] Bảng được đánh số liên tục từ 1 đến 11, với Bảng 10a–10b cho cross-dataset.
- [x] Hình chính được đánh số 1–5; hình cross-dataset được tách thành S1–S2.
- [x] Các đường dẫn hình local hiện tồn tại.
- [x] Số lượng 42 support, 1.463 query và 1.505 ảnh thuộc bốn lớp nhất quán.
- [x] Hai gói Kaggle dẫn xuất đã được thêm vào References theo style CEA: query version 2 là Hải (2026a), few-shot version 1 là Hải (2026b).
- [x] Phần hạn chế đã công khai single seed, test selection, leakage cross-dataset và giới hạn phần cứng.

---

## Thứ tự xử lý khuyến nghị

1. P0.1 — Chạy lại cross-dataset sạch cùng seed.
2. P0.2 — Chốt tiêu đề và phạm vi edge claim.
3. P0.3 — Hoàn thiện provenance, annotation, leakage và license ChiliEdge-DS.
4. P0.4 — Xử lý việc lựa chọn bằng test.
5. P0.5 — Làm sạch cấu trúc merge và điền thông tin tác giả.
6. P1.1–P1.3 — Sửa công thức, McNemar và References.
7. P1.4–P1.6 — Hoàn thiện Gemini, bằng chứng lightweight và sơ đồ kiến trúc.
8. P2 — Chuẩn hóa ngôn ngữ, dọn file và kiểm tra bản render cuối.
