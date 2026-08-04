# Quy tắc template — Computers and Electronics in Agriculture

Nguồn chính thức: [Guide for Authors — Computers and Electronics in Agriculture](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/publish/guide-for-authors)

## 1. Cách dùng

- `MUST`: yêu cầu bắt buộc trước khi nộp.
- `SHOULD`: khuyến nghị mạnh về trình bày hoặc khả năng tái lập.
- `PROJECT`: quy tắc riêng của bài ChiliEdgeNet để tránh diễn giải vượt quá bằng chứng.
- Khi review, ghi kết quả theo dạng: `PASS`, `FAIL`, `N/A` hoặc `TODO` kèm mã rule.
- Không đánh `PASS` dựa trên dự định; chỉ đánh khi nội dung hoặc tệp đã tồn tại và kiểm tra được.

## 2. Loại bài và phạm vi

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| SCOPE-01 | MUST | Bản thảo phải được chuẩn bị dưới loại **Original Research Paper**, không phải Application Note. |
| SCOPE-02 | MUST | Mục tiêu phải gắn với một vấn đề trong sản xuất nông nghiệp hoặc horticulture, không chỉ trình bày benchmark thị giác máy tính. |
| SCOPE-03 | PROJECT | Phần mở đầu và kết luận phải nêu rõ giá trị đối với nhận diện bệnh cây ớt và triển khai thiết bị biên. |
| SCOPE-04 | PROJECT | Không gọi hệ thống là đã sẵn sàng triển khai ngoài thực địa nếu mới có đánh giá dữ liệu và latency trên laptop. |

## 3. Tệp và định dạng chung

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| FILE-01 | MUST | File nguồn phải chỉnh sửa được: `.doc/.docx` hoặc `.tex`; PDF không phải file nguồn. |
| FILE-02 | MUST | Nếu dùng Word, bản thảo phải trình bày **một cột**. |
| FILE-03 | MUST | Không còn chữ gạch ngang hoặc gạch chân nếu không mang ý nghĩa khoa học. |
| FILE-04 | MUST | Bản thảo phải được kiểm tra chính tả và ngữ pháp trước khi nộp. |
| FILE-05 | MUST | Tiếng Anh phải thống nhất theo Anh–Mỹ hoặc Anh–Anh, không trộn lẫn. |
| FILE-06 | PROJECT | Không còn đường dẫn cục bộ như `E:\`, `C:\`, `../../`, ghi chú nội bộ, `TODO` hoặc placeholder trong bản nộp. |

## 4. Title page, Abstract, Keywords và Highlights

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| FRONT-01 | MUST | Tiêu đề ngắn gọn, giàu thông tin; tránh viết tắt và công thức không phổ biến. |
| FRONT-02 | MUST | Title page có đủ tên tác giả, thứ tự tác giả, cơ quan, địa chỉ và quốc gia. |
| FRONT-03 | MUST | Chỉ rõ corresponding author và thông tin liên hệ cập nhật. |
| FRONT-04 | MUST | Abstract không vượt quá **250 từ**. |
| FRONT-05 | MUST | Abstract tự đứng độc lập, nêu mục tiêu, kết quả chính và kết luận chính. |
| FRONT-06 | MUST | Abstract không dùng tài liệu tham khảo; viết tắt không phổ biến phải được định nghĩa khi xuất hiện lần đầu. |
| FRONT-07 | MUST | Có từ **1 đến 7 keywords**, viết bằng tiếng Anh. |
| FRONT-08 | SHOULD | Keywords tránh cụm nhiều khái niệm và từ nối như “and” hoặc “of”. |
| FRONT-09 | MUST | Có file Highlights riêng ở định dạng chỉnh sửa được. |
| FRONT-10 | MUST | Highlights gồm **3–5 gạch đầu dòng**, mỗi dòng tối đa **85 ký tự**, kể cả khoảng trắng. |
| FRONT-11 | SHOULD | Graphical abstract, nếu có, phải nộp thành file riêng và đọc được ở kích thước thu nhỏ. |

## 5. Cấu trúc bài

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| STRUCT-01 | MUST | Abstract không đánh số; các phần nội dung được đánh số 1, 2, 3, ... |
| STRUCT-02 | MUST | Tiểu mục dùng hệ thống 1.1, 1.1.1, ...; không để đề mục nội dung quan trọng không có số. |
| STRUCT-03 | MUST | Khi dẫn lại một phần, dùng số mục cụ thể thay vì “phần trên” hoặc “đoạn bên dưới”. |
| STRUCT-04 | PROJECT | Trình tự chính: Introduction → Related work → Materials and methods → Results and discussion → Limitations → Conclusions. |
| STRUCT-05 | PROJECT | Methods phải mô tả đủ dữ liệu, phép chia, augmentation, tiêu chí checkpoint, seed, phần cứng, phần mềm và độ đo. |
| STRUCT-06 | PROJECT | Results phải tách rõ kết quả quan sát, kiểm định thống kê và diễn giải. |
| STRUCT-07 | PROJECT | Limitations không được dùng để sửa một tuyên bố quá mạnh ở phần trước; tuyên bố chính phải tự giới hạn ngay tại nơi xuất hiện. |

## 6. Hình và bảng

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| FIG-01 | MUST | Mỗi hình được trích dẫn trong bài và đánh số theo thứ tự xuất hiện. |
| FIG-02 | MUST | File hình có tên rõ ràng, liên quan trực tiếp đến nội dung. |
| FIG-03 | MUST | Caption giải thích được hình mà không phụ thuộc hoàn toàn vào nội dung chính; danh sách caption được chuẩn bị theo yêu cầu nộp bài. |
| FIG-04 | SHOULD | Biểu đồ, sơ đồ và hình có chữ được xuất dạng vector PDF/EPS khi có thể. |
| FIG-05 | MUST | Ảnh màu hoặc ảnh grayscale đạt tối thiểu 300 dpi. |
| FIG-06 | MUST | Hình hỗn hợp nét–ảnh đạt tối thiểu 500 dpi nếu không dùng vector. |
| FIG-07 | MUST | Hình nét đạt tối thiểu 1000 dpi nếu không dùng vector. |
| FIG-08 | MUST | Không nộp hình chất lượng thấp hoặc định dạng không phù hợp như GIF/BMP cho hình khoa học. |
| TABLE-01 | MUST | Bảng phải là văn bản có thể chỉnh sửa, không phải ảnh chụp. |
| TABLE-02 | MUST | Mỗi bảng được trích dẫn và đánh số theo thứ tự xuất hiện. |
| TABLE-03 | MUST | Tiêu đề bảng đặt phía trên; ghi chú đặt phía dưới bảng. |
| TABLE-04 | SHOULD | Tránh đường kẻ dọc, tô nền và lặp lại dữ liệu đã thể hiện đầy đủ trong hình hoặc nội dung. |

## 7. Tài liệu tham khảo và dữ liệu

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| REF-01 | MUST | Mọi trích dẫn trong bài đều có mục tương ứng trong References và ngược lại. |
| REF-02 | MUST | Tài liệu tham khảo dùng một kiểu thống nhất trong toàn bài. |
| REF-03 | MUST | Tên tác giả, tiêu đề, năm, tạp chí, volume, trang/article number phải chính xác khi áp dụng. |
| REF-04 | SHOULD | Bổ sung DOI cho tài liệu có DOI. |
| REF-05 | MUST | Tài liệu web có URL đầy đủ và ngày truy cập. |
| DATA-01 | MUST | Dữ liệu nghiên cứu phải được lưu trong repository phù hợp và được trích dẫn, liên kết trong bài theo chính sách Option C. |
| DATA-02 | MUST | Trích dẫn dataset có tác giả, tên dataset, repository, phiên bản nếu có, năm và định danh ổn định. |
| DATA-03 | MUST | Thêm nhãn `[dataset]` vào tài liệu tham khảo dữ liệu để hệ thống Elsevier nhận diện. |
| DATA-04 | SHOULD | Có Data availability statement nêu rõ URL/DOI, phiên bản và điều kiện truy cập. |
| DATA-05 | PROJECT | ChiliEdge-DS, cross-check và few-shot phải được đóng băng đúng phiên bản đã dùng để sinh kết quả. |
| DATA-06 | PROJECT | Phải kiểm tra trùng ảnh bằng hash giữa train, validation, test, cross-check và few-shot. |
| DATA-07 | PROJECT | Phải lưu manifest ảnh theo tập, lớp, nguồn và giấy phép để có thể truy vết. |

## 8. Các khai báo cuối bài

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| DECL-01 | MUST | Có Acknowledgements ngay trước References nếu có cá nhân hoặc tổ chức cần cảm ơn. |
| DECL-02 | MUST | Có CRediT author statement cho đóng góp của từng tác giả. |
| DECL-03 | MUST | Khai báo nguồn tài trợ và vai trò của nhà tài trợ; nếu không có phải ghi rõ. |
| DECL-04 | MUST | Có Declaration of competing interests theo biểu mẫu hoặc hướng dẫn của hệ thống nộp bài. |
| DECL-05 | MUST | Khai báo việc sử dụng AI tạo sinh nếu công cụ tham gia vượt quá kiểm tra chính tả, ngữ pháp hoặc tài liệu tham khảo cơ bản. |
| DECL-06 | PROJECT | Vì Codex đã hỗ trợ tổ chức và biên tập nội dung, bản nộp phải có Declaration of generative AI and AI-assisted technologies. |
| DECL-07 | MUST | AI không được liệt kê là tác giả; tác giả con người chịu trách nhiệm kiểm tra toàn bộ nội dung. |

## 9. Quy tắc khoa học riêng của ChiliEdgeNet

| Mã | Mức | Quy tắc kiểm tra |
|---|---|---|
| EXP-01 | PROJECT | Mọi kết quả chính của bản hội nghị hiện tại phải khớp giao thức **seed 42**. |
| EXP-02 | PROJECT | Không dùng cách viết hàm ý nhiều seed, trung bình nhiều lần chạy hoặc độ ổn định giữa seed nếu chưa có dữ liệu đó. |
| EXP-03 | PROJECT | Cấu hình được lựa chọn với sự tham gia của test phải được gọi là phân tích thăm dò, không phải ước lượng test độc lập. |
| EXP-04 | PROJECT | Không dùng kết quả lựa chọn trên test để xác lập thứ hạng thống kê hoặc tuyên bố ưu thế tổng quát. |
| EXP-05 | PROJECT | Stress test nhòe và thiếu sáng chỉ chứng minh phản ứng dưới suy giảm tổng hợp trong miền ChiliEdge-DS. |
| EXP-06 | PROJECT | Lợi ích của Tiny Transformer chỉ được khẳng định tại điều kiện có bằng chứng và kiểm định tương ứng. |
| EXP-07 | PROJECT | Cross-check phải được gọi là đánh giá chuyển miền theo lớp vì nguồn ảnh và kiểu nền gắn với từng lớp. |
| EXP-08 | PROJECT | Few-shot phải được mô tả là bổ sung có mục tiêu cho `healthy`, `leaf_curl` và `yellow_leaf`. |
| EXP-09 | PROJECT | Không tuyên bố few-shot cải thiện đồng đều nếu precision, recall hoặc F1 của một số lớp không tăng. |
| EXP-10 | PROJECT | Ba phương pháp XAI chỉ cung cấp khảo sát định tính, không chứng minh quan hệ nhân quả hoặc độ đúng bệnh lý. |
| EXP-11 | PROJECT | Đối chiếu LLM phải ghi rõ model/version, ngày chạy, prompt, ánh xạ nhãn và đầu ra không hợp lệ. |
| EXP-12 | PROJECT | Không gọi LLM zero-shot là baseline hoàn toàn ngang hàng với mô hình chuyên biệt đã huấn luyện. |
| EXP-13 | PROJECT | Kết quả camera hiện chỉ được gọi là độ trễ tiền xử lý–suy luận trên CPU laptop. |
| EXP-14 | PROJECT | Không gọi số đo hiện tại là FPS camera hoặc độ trễ toàn pipeline thu nhận–hiển thị. |

## 10. Quy tắc kiểm tra tự động tối thiểu

| Mã | Mức | Điều kiện PASS |
|---|---|---|
| AUTO-01 | MUST | Abstract tiếng Anh có số từ `<= 250`. |
| AUTO-02 | MUST | Keywords có số lượng từ `1–7`. |
| AUTO-03 | MUST | Highlights có `3–5` dòng và mọi dòng dài `<= 85` ký tự. |
| AUTO-04 | MUST | Không tìm thấy `TODO`, `placeholder`, `E:\`, `C:\` hoặc đường dẫn tương đối đến file nội bộ trong bản nộp. |
| AUTO-05 | MUST | Không có trích dẫn thiếu mục References hoặc References không được trích dẫn. |
| AUTO-06 | PROJECT | Các giá trị accuracy, F1, tham số, kích thước và latency thống nhất giữa Abstract, Results, bảng và Conclusions. |
| AUTO-07 | PROJECT | Tên checkpoint, seed và cấu hình trong bài khớp notebook, log và file kết quả. |
| AUTO-08 | PROJECT | Tất cả link dữ liệu công khai truy cập được và trỏ đúng phiên bản đã công bố. |

## 11. Quy tắc chặn nộp bài

Không đánh dấu “sẵn sàng nộp” nếu còn `FAIL` ở một trong các nhóm sau:

- Abstract, Highlights hoặc title page.
- Dữ liệu chưa công khai/cite/link đúng phiên bản.
- Chưa kiểm tra trùng dữ liệu.
- Số liệu trong bài không khớp log hoặc notebook.
- Thiếu Funding, Competing interests, CRediT hoặc khai báo AI áp dụng cho bài này.
- Còn đường dẫn máy cục bộ, placeholder hoặc ghi chú nội bộ.
- Quyền sử dụng ảnh/dữ liệu bên ngoài chưa xác định.

