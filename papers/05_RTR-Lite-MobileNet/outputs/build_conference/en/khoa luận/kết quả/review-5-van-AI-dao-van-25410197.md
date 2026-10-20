# Review 5: Rà soát văn AI, dấu vết công cụ và nguy cơ đạo văn

**File kiểm tra:** `25410197_Tô Minh Hải.docx` (bản 24/09/2026, 14:11)
**Ngày rà soát:** 27/09/2026
**Phạm vi:** toàn bộ văn bản (từ Tóm tắt đến Kết luận) cùng 19 bảng. Danh mục tài liệu tham khảo chỉ được quét ký tự.
**Cách làm:** trích XML của file docx để quét ký tự Unicode, đếm tần suất từ và độ dài câu bằng script, sau đó đọc toàn bộ từng đoạn.

## Số liệu nhanh

| Chỉ số | Giá trị | Nhận xét |
|---|---|---|
| U+200B (khoảng trắng không độ rộng) | **54** | Toàn bộ nằm trong bảng, ngay sau `-` hoặc `_` |
| U+2060 (word joiner) | **11** | Nằm trong công thức ở phần thân bài |
| Dấu gạch dài (—) | 0 | Đã dọn sạch |
| Gạch ngắn (–) | 11 | Chỉ có ở bìa, trang hội đồng, Bảng 4.12 và Bảng 6.1 |
| Ngoặc kép cong / thẳng | 0 / 110 | Thống nhất, nhưng toàn bộ là ngoặc thẳng |
| Câu sót lại của chatbot ("Dưới đây là", "Hy vọng", ...) | 0 | Sạch |
| Độ dài câu trung bình | **38 từ** (độ lệch chuẩn 20) | 188/496 câu dài trên 40 từ, **63 câu dài trên 60 từ** |
| Dấu chấm phẩy | 117 | Rất dày, câu ghép nối bằng ";" |
| "thay vì" / "không phải" | **27 / 29** | Khuôn "không phải X mà là Y", dấu hiệu AI mạnh nhất của bài |
| "trực tiếp" / "hiệu chỉnh Holm" / "khác nhau" | 34 / 23 / 20 | Lặp dày |
| Chú thích tiếng Anh trong ngoặc | **93** | Quá nhiều |

## Bảng lỗi chi tiết

### A. Dấu vết kỹ thuật

| Vị trí | Đoạn trích | Loại lỗi | Mức | Gợi ý sửa |
|---|---|---|---|---|
| Bảng 3.1, 3.2, 4.1 đến 4.10, 4.12, 4.13, 5.1 | `leaf_​curl`, `F1-​macro`, `-​0,27`, `RTR_​Lite_​MobileNet` | Ký tự ẩn U+200B (54 chỗ) | **Cao** | Mở Word, Ctrl+H, ô Find gõ `^u8203`, ô Replace để trống, chọn Replace All. Các công cụ phát hiện đều đánh dấu ký tự này. |
| 1.4.2 GQA (2 đoạn); 1.7.2 McNemar; 4.1 "Số lần chạy..."; 4.5 | `G =⁠ 1`, `H=⁠4, G=⁠2`, `b+⁠c`, `b/⁠c`, `p =⁠ 0,⁠⁠5` | Ký tự ẩn U+2060 (11 chỗ). Riêng "0,5" có tới 2 ký tự nằm **giữa con số** | **Cao** | Find `^u8288`, Replace để trống. Sau đó thống nhất khoảng trắng: viết "G = 1", "H = 4, G = 2" (hiện tại chỗ có cách, chỗ không). |
| 1.7.3 so với Bảng 4.5, 4.7, 4.12 và mục 4.6/Kết luận | `α/(m−1)` so với `-0,27` và `10-242` | Dấu trừ lẫn lộn: U+2212 ở một chỗ, gạch nối ở mọi chỗ còn lại | Trung bình | Dùng dấu trừ "−" cho số âm và số mũ: "−0,27", "10^−242". |
| Toàn văn | 110 ngoặc thẳng `"`, không có ngoặc cong | Word gõ tay sẽ tự đổi sang ngoặc cong, nên ngoặc thẳng cho thấy văn bản được dán vào hoặc sinh bằng script | Thấp | Có thể giữ vì đã thống nhất. Nếu muốn đổi: bật AutoFormat, rồi Find `"` Replace `"`. |
| Ch.6 mở đầu | "đã được đánh giá định lượng và báo cáo chính thức **trong bài báo khoa học của đề tài** (mục 4.7) ... không phải một thực nghiệm được báo cáo trong bài báo" | Dấu vết chuyển từ bài báo sang chuyên đề | **Cao** | Hội đồng sẽ hỏi "bài báo nào?". Nếu có bài báo thì trích dẫn và khai báo ở Lời cam đoan; nếu chưa có thì xóa cụm này. |
| 1.5 và 3.2.4 | "khác với báo cáo giai đoạn đầu của đề tài, nơi quyết định này chỉ dựa trên một khảo sát thiết kế sơ bộ" | Dấu vết phiên bản cũ, lặp nguyên văn 2 lần | Trung bình | Chỉ giữ ở 3.2.4 và viết ngắn lại. Người đọc chuyên đề không biết "báo cáo giai đoạn đầu" là gì. |
| Kết luận, đoạn 1 | "...là điều xuyên suốt cách chuyên đề trình bày kết quả **ở các chương sau**" | Câu được chép từ phần Mở đầu sang (Kết luận không có "chương sau") | **Cao** | Đổi thành "...trong toàn bộ chuyên đề". |
| Kết luận, đoạn cuối | "giữ lại **features[0:13]**" | Mâu thuẫn với 3.2.1 và Hình 3.2 (features[0:14], tức features[0] đến [13]) | **Cao** | Sửa thành features[0:14]. Lỗi kiểu này thường gặp khi ghép đoạn từ nhiều bản. |
| 4.8 mở đầu | "khoảng **2,84 MB** ở FP32" | Mâu thuẫn với 2,7 MB ở Tóm tắt, 3.2.5 và Ch.6 | Trung bình | Chọn một cách tính (MiB hoặc MB) rồi dùng thống nhất, có chú thích đơn vị. |
| 2.2 | "RTR_Lite_Mobile**NetV2** [7]" | Tên không thống nhất (mọi chỗ khác viết RTR_Lite_MobileNet) | Thấp | Thống nhất một tên. |
| 3.1 | "độ phân giải gốc khoảng 12.000×9.000 pixel" | Số liệu khó tin (tương đương 108 MP) | Trung bình | Kiểm tra lại EXIF. Hội đồng dễ bắt lỗi chỗ này. |

### B. Giọng văn AI

| Vị trí | Đoạn trích | Loại lỗi | Mức | Gợi ý sửa |
|---|---|---|---|---|
| Tóm tắt đ.2; Mở đầu "Lý do" đ.3; 2.8; Kết luận đ.2 | "...**thay vì chỉ báo cáo một mô hình đã lắp ráp hoàn chỉnh**" (4 lần) | Câu khẩu hiệu lặp lại, khuôn "X thay vì Y" | **Cao** | Chỉ giữ một lần ở Mở đầu và nói thẳng: "Đề tài đo đóng góp của từng thành phần bằng cách huấn luyện lại mô hình sau khi bỏ thành phần đó." |
| Mở đầu, Lý do đ.3 | "Khoảng trống ở đây **không phải là** sự vắng mặt tuyệt đối... **mà là** sự vắng mặt của..." | Khuôn "không phải X mà là Y", đặc trưng của văn AI | **Cao** | "Các công trình trước có làm từng phân tích riêng lẻ. Điều còn thiếu là một quy trình gộp chúng trên cùng một lần chia dữ liệu." |
| Mở đầu, Lý do đ.2 | Câu 1 dài khoảng 90 từ, "...self-attention làm được, điều quan trọng với các triệu chứng lan tỏa..." | Câu quá dài, ý phụ gắn đuôi bằng dấu phẩy (vết của việc xóa gạch dài) | **Cao** | Tách thành 3 câu. Câu ngắn nhất chỉ cần: "Điều này quan trọng với bệnh xoăn lá và vàng lá." |
| 3.2.1 | "Việc cắt trước features[14] **mang tính cơ học chứ không tùy tiện**" | Cách nói hoa mỹ, giọng AI | **Cao** | "Chọn cắt ở features[14] vì tầng này giảm lưới đặc trưng từ 14×14 xuống 7×7." |
| 4.2 "So sánh với kiến trúc nền tảng"; 7.1 đ.1; Kết luận đ.1 | "Phần không phụ thuộc vào độ nhiễu của tập test là hiệu quả tham số: **một con số chính xác tuyệt đối, không cần kiểm định thống kê**" | Một ý lặp 3 lần với 3 cách diễn đạt | **Cao** | Chỉ nói ở 4.2, một câu ngắn. Ở 7.1 và Kết luận chỉ cần nhắc lại con số 35%. |
| 5.2 "Về độ trễ" | "Hai con số này không tương đương... không phần nào... không phải một khẳng định... không cấu thành..." | Chồng 4 câu rào đón liên tiếp | **Cao** | Gộp lại còn 2 câu: "Con số 105 lần so sánh hai cách triển khai (cục bộ và gọi API qua mạng), không so sánh tốc độ tính toán của hai kiến trúc. Gemini cũng không được huấn luyện cho bài toán này." |
| Ch.6 câu mở | "Kết quả trên notebook là một chuyện, chạy được thành sản phẩm thật lại là chuyện khác. Để kiểm chứng điều đó,..." | Câu mở kiểu "câu móc" của AI | **Cao** | "Chương này mô tả ứng dụng web minh họa được xây dựng quanh ChiliEdgeNet." |
| 6.1.3; 6.1.4; 6.2.3; Kết luận đ.1 | "Một chi tiết nhỏ nhưng cần thiết cho trải nghiệm:", "Điểm quan trọng nhất của thiết kế này là", "Điểm đáng chú ý nhất của DFD là", "Điều đáng chú ý hơn con số tuyệt đối này là" | Khuôn "Điểm X nhất là..." dùng để mở đoạn | Trung bình | Bỏ phần đệm, vào thẳng nội dung: "Khi người dùng kéo thanh trượt, frontend chỉ gửi..." |
| 4.7 đoạn cuối; 5.1 đoạn cuối | "...chỉ cung cấp bằng chứng sơ bộ..., không phải một phép đo đại diện cho toàn bộ pipeline hay cho phần cứng biên chuyên dụng"; "không chứng minh mô hình đã học đúng..., hoàn toàn bỏ qua nền, hay nắm bắt một quan hệ nhân quả" | Rào đón xếp thành bộ ba | Trung bình | Chỉ giữ một câu hạn chế cho mỗi mục; các hạn chế đầy đủ để ở Chương 7. |
| Kết luận, 4 điểm đánh số | "Không phải khối kế thừa nào cũng có ích." / "Thích nghi miền hoạt động rất hiệu quả với chỉ một tập hỗ trợ nhỏ." | Mỗi ý mở bằng một câu khẩu hiệu rồi mới giải thích | Trung bình | Viết thành 2 đoạn văn liền mạch; số liệu p, Holm đã có ở Ch.4, không cần nhắc lại. |
| Kết luận, đoạn cuối | Câu về XAI và Gemini dài khoảng 75 từ, kết bằng "; dù đây là so sánh..., không phải một tuyên bố..." | Câu quá dài kèm đuôi rào đón | **Cao** | Tách thành 3 câu, bỏ đuôi rào đón (đã nói ở 5.2). |
| Mở đầu của Ch.1, 2, 3, 4, 5, 7 | "Chương này trình bày / khảo sát / xem xét / tổng hợp..." rồi liệt kê mục con | Mở chương theo cùng một khuôn, lặp nội dung đã có ở "Bố cục chuyên đề" | Trung bình | Viết ngắn còn 1 câu, mỗi chương mở theo một cách khác. Ví dụ Ch.4: "Mọi thí nghiệm dưới đây dùng cùng lần chia dữ liệu và seed 42." |
| Toàn văn (các đoạn nêu ở mục D bên dưới) | "..., một rủi ro đặc biệt đáng lưu ý khi...", "..., chính là cách tiếp cận của ChiliEdgeNet", "..., phát hiện ablation rõ ràng nhất trong toàn bộ Chương 4", "..., lợi ích trực tiếp cho khả năng diễn giải" | Ngữ đồng vị gắn đuôi bằng dấu phẩy, thường là chỗ trước đây dùng gạch dài | Trung bình | Tách thành câu riêng: "Rủi ro này càng đáng lưu ý vì phần lớn ảnh được chụp trên nền trắng." |
| Toàn văn | 117 dấu chấm phẩy, 63 câu dài trên 60 từ | Nhịp văn đều và dài | **Cao** | Mỗi đoạn nên có ít nhất 1 câu dưới 12 từ. Hiện chỉ có 23/496 câu như vậy. |
| Tóm tắt đ.1 | Các câu đều có dạng "Chủ ngữ + đạt/có + số liệu" | Nhịp đều, liệt kê số liệu dồn dập | Trung bình | Bớt số liệu: Tóm tắt chỉ cần 98,93%, 671.173 tham số, và 30,96% lên 89,75%. |
| Mở đầu "Đóng góp", gạch đầu dòng thứ 6 | "Hệ thống hóa cơ sở lý thuyết đầy đủ... phục vụ mục đích trình bày chuyên đề tốt nghiệp" | Đóng góp chung chung, dễ bị coi là lời độn | Trung bình | Xóa gạch đầu dòng này. |
| 1.6 (3 gạch đầu dòng XAI) | "• Nhóm nhiễu loạn...", "• Nhóm dựa trên gradient...", "• Nhóm đọc nội tại Transformer..." | Liệt kê máy móc trong khi mỗi ý dài 60–80 từ | Thấp | Viết thành 3 đoạn văn ngắn, hoặc 1 đoạn có câu chuyển ý. |
| Ch.4 và Ch.7 | "Khung đánh giá chung.", "Số lần chạy và cách kiểm định.", "Phân tích lỗi.", "Thiết lập few-shot.", "Về độ trễ." | Tiêu đề chìm kiểu IEEE, là vết của bản bài báo | Thấp | Chuyển thành Heading 3, hoặc in đậm cho thống nhất với template khoa. |
| Ch.6 (6.1.1 đến 6.1.5) | "mới là phần việc chính", "khẳng định chắc nịch", "giao diện sẽ giật", "bấm-là-chạy" | Giọng khẩu ngữ, lệch hẳn so với Ch.1 đến Ch.5 | Thấp (ít giống AI nhưng lệch giọng) | Nâng giọng lên một chút cho đồng đều. Đây lại là phần đọc "người" nhất, không cần sửa nhiều. |

### C. Lỗi dấu gạch

| Vị trí | Đoạn trích | Loại lỗi | Mức | Gợi ý sửa |
|---|---|---|---|---|
| Toàn văn | Không có gạch dài (—) | Đã sạch | Không có | Giữ nguyên. Lưu ý là khi xóa gạch dài, bài đang thay bằng dấu hai chấm và ngữ đồng vị (xem 2 dòng dưới). |
| 1.2 (Depthwise), 1.3 (tự chú ý), Mở đầu "Đóng góp" gạch đầu dòng 3, Kết luận đoạn cuối | "tách... thành hai bước: depthwise..."; "trong chuỗi: mỗi token được chiếu thành ba vector: truy vấn..." | **Hai dấu hai chấm trong một câu**, gần như chắc chắn là chỗ đã thay gạch dài | Trung bình | Tách câu: "Cơ chế tự chú ý tổ hợp mọi token trong chuỗi theo trọng số. Mỗi token được chiếu thành ba vector: truy vấn, khóa và giá trị." |
| Bìa và trang phụ bìa so với 4.6 | "Tô Minh Hải **–** MSSV" (gạch ngắn có cách) so với "Healthy Leaf **-** healthy" (gạch nối có cách) | Lẫn gạch ngắn và gạch nối | Trung bình | Ở 4.6 viết "Healthy Leaf tương ứng healthy" hoặc dùng mũi tên "→" (đã dùng ở Bảng 4.8). |
| 1.7, 2.1, 2.3, 4.1, 4.2, 6.1.4, 6.3, 3.3 | "1-2 ảnh", "98-99%", "2023-2026", "5-9 triệu", "150-250 chữ", "Bảng 4.5-4.9", "mục 1.7.2-1.7.3", "Hình 6.8-6.9" | Khoảng số dùng gạch nối, trong khi Bảng 4.12 và 6.1 dùng gạch ngắn ("17,33–122,76", "độ thưa–độ chính xác") | Trung bình | Văn tiếng Việt nên viết "từ 1 đến 2 ảnh", "Bảng 4.5 đến 4.9". Nếu giữ ký hiệu thì thống nhất một kiểu gạch ngắn "–", không cách. |
| 1.5; 2.4; 4.3, 3.2.4, Kết luận; 6.x (7 lần); 6.1.3; 5.1 | "kênh-cao, kênh-rộng, cao-rộng"; "backbone-bộ phân loại"; "thêm-lại"; "hỏi-đáp"; "kéo-thả", "bấm-là-chạy"; "trong-miền" | Gạch nối chen vào từ ghép tiếng Việt (tiếng Việt không dùng gạch nối cho từ ghép) | Trung bình | "kênh với chiều cao", "hỏi đáp", "kéo thả", "thêm lại", "trong miền". "Bấm-là-chạy" viết lại thành "chạy ngay khi bấm". |
| Bảng 4.5, 4.7, 4.12 | "-​0,27", "-​1,88 pp" | Dấu trừ bằng gạch nối, kèm U+200B | Cao (vì có ký tự ẩn) | Như mục A: dùng "−". |

### D. Tiếng Anh chen vào câu tiếng Việt

| Vị trí | Đoạn trích | Loại lỗi | Mức | Gợi ý sửa |
|---|---|---|---|---|
| 6.1.2 | "tức **model** còn phân vân"; "khi khởi động **app**"; "ngay trên **server**" | Từ tiếng Anh thông dụng đã có từ Việt tương ứng | **Cao** | mô hình, ứng dụng, máy chủ |
| Toàn văn | ablation (23), baseline (29), checkpoint (26), few-shot (18), stress test (8), benchmark (6) | Thuật ngữ lặp dày | Trung bình | Chú thích 1 lần ở lần đầu, sau đó dùng từ Việt: "thí nghiệm loại bỏ thành phần", "mô hình đối chứng", "bản trọng số đã lưu", "học với ít mẫu", "kiểm thử độ bền", "phép đối chuẩn". |
| Toàn văn | "accuracy" (13) xen với "độ chính xác"; "offline" (5) xen với "ngoại tuyến"; "heatmap" xen với "bản đồ nhiệt"; "file" xen với "tệp"; "train/val/test" | Dùng lẫn hai ngôn ngữ cho cùng một khái niệm | Trung bình | Chọn từ Việt rồi dùng thống nhất: độ chính xác, ngoại tuyến, bản đồ nhiệt, tệp, tập huấn luyện/kiểm định/kiểm thử. |
| 1.x, 2.x (93 chỗ chú thích trong ngoặc) | "(exploratory)", "(macro)", "(rotate)", "(benchmark)", "(normalization)", "(kernel)", "(axial)", "(include)", "(batching)", "(fully-connected)" | Viết tiếng Việt rồi mở ngoặc chú thích tiếng Anh cho từ phổ thông | **Cao** | Bỏ chú thích cho từ phổ thông. Chỉ giữ cho thuật ngữ khó (depthwise separable convolution, inverted residual, family-wise error rate, KV-cache), và chỉ ở lần đầu. |
| 1.3; Mở đầu và 1.2; Mở đầu và 3.1; 1.2 và 4.3 | "(self-attention)", "(inductive bias)", "(phenotype-based)", "(skip connection)" | Mỗi thuật ngữ được chú thích 2 lần | Trung bình | Chỉ chú thích ở lần đầu tiên. |
| 4.1, 4.2, 5.2 | "batch size", "optimizer", "class weighting", "round-trip", "prompt", "ground-truth", "SOTA", "pipeline", "notebook", "demo", "augmented", "resize", "fine-tune", "margin" | Từ tiếng Anh có cách dịch phổ biến | Trung bình | kích thước lô, bộ tối ưu, trọng số theo lớp, thời gian khứ hồi, câu lệnh gợi ý, nhãn thật, các mô hình tiên tiến, chuỗi xử lý, môi trường thử nghiệm, bản minh họa, có tăng cường dữ liệu, đổi kích thước, tinh chỉnh, khoảng cách xác suất. |
| Mở đầu, Phạm vi | "phân loại ảnh (classification), ... phát hiện đối tượng (detection) hay phân vùng (segmentation)" | Chú thích không cần thiết | Thấp | Bỏ cả 3 ngoặc. |
| Giữ nguyên | token, patch, embedding, RoPE, GQA, RMSNorm, MHA, zero-shot, endpoint, backend/frontend, use case, Docker image | Thuật ngữ chuyên ngành chưa có cách dịch phổ biến | Không phải lỗi | Giữ, chỉ cần có trong Danh mục từ viết tắt hoặc thuật ngữ. |

### E. Lặp từ và lặp ý

| Vị trí | Đoạn trích | Loại lỗi | Mức | Gợi ý sửa |
|---|---|---|---|---|
| Tóm tắt, Mở đầu, 3.2.5, 4.1, 4.2, Ch.6, Kết luận | 98,93% (8 lần), 671.173 (8), 35% (5), 20,24 ms (5), 2,7 MB (6), "không cần GPU" (5) | Lặp số liệu chủ chốt | **Cao** | Mỗi số nên xuất hiện ở Tóm tắt, ở chỗ đo, và 1 lần ở Kết luận. Ch.6 chỉ cần tham chiếu "mục 4.1". |
| Mở đầu Lý do đ.2, 1.2, 1.3, 2.2 (CNN nhẹ), 2.3, 3.2 | Lập luận "CNN chỉ tổng hợp cục bộ, không trực tiếp mô hình hóa quan hệ theo cặp như tự chú ý" | Cùng một ý lặp **6 lần** | **Cao** | Giữ đầy đủ ở 1.2 và 1.3. Các chỗ khác chỉ cần một vế câu kèm tham chiếu mục. |
| 4.7 (đoạn 2) và 6.4.2 | "không bao gồm thời gian đọc khung hình từ camera, cắt ROI, chuyển đổi BGR sang RGB, dựng giao diện, hay ghi file" | Giao thức đo lặp gần nguyên văn | **Cao** | 4.7 chỉ giữ kết quả và dẫn sang 6.4 để xem giao thức. |
| 1.7.2 và 4.1 "Số lần chạy" | Định nghĩa b/c của McNemar | Lặp định nghĩa | Trung bình | Ở 4.1 chỉ cần viết "ký hiệu b/c như mục 1.7.2". |
| 2.7 và 4.6 đ.1 | "Zhao và cộng sự [44]... Yang và cộng sự [41]... xuất phát từ cùng động cơ... giới hạn ở phép ánh xạ bốn lớp đặc thù cho cây ớt" | Đoạn tổng quan tài liệu lặp gần nguyên văn | **Cao** | Xóa câu này ở 4.6. |
| 3.2.1, 5.1, Kết luận | "196 vị trí, gấp bốn lần một lưới 7×7" | Lặp 3 lần | Trung bình | Chỉ giữ ở 3.2.1. |
| 1.6, 5.1 (gạch đầu dòng và chú thích Hình 5.3) | "mean pooling, không có token CLS, lấy trung bình trên 196 token" | Lặp 3 lần | Trung bình | Giữ ở 5.1, cắt ở chú thích hình. |
| 5.1 (gạch đầu dòng và chú thích Hình 5.2) | "20 mẫu và hai tensor baseline hằng số (0 và 1)" | Lặp nguyên văn | Thấp | Cắt ở chú thích hình. |
| 4.2 và 4.3; 7.1 đ.1–2; Kết luận 1–3 | Chênh 0,80 điểm, CI chồng lấn, pHolm = 0,0234, pHolm = 0,0105 | Kết quả được kể lại 3 lần | **Cao** | 7.1 chỉ nêu hạn chế và không nhắc lại p. Kết luận chỉ giữ 2 hoặc 3 phát hiện. |
| Mở đầu, 1.6, 3.1, 5.1 | "nền trắng có kiểm soát... manh mối giả" | Lặp 4 lần | Trung bình | Giữ ở 3.1 và 5.1. |
| Toàn văn | "thay vì" 27, "không phải" 29, "trực tiếp" 34, "khác nhau" 20 ("ba nhóm cơ chế khác nhau" 9 lần), "vì vậy" 11 (thường ở giữa câu: "X vì vậy là...") | Lặp từ | **Cao** | Mỗi trang giữ tối đa 1 lần "thay vì". "Ba nhóm cơ chế khác nhau" rút thành "ba nhóm cơ chế". |
| Mở đầu: Mục tiêu 4, Phương pháp, Đóng góp 3 | Cùng mô tả ablation, McNemar, Holm, SHA-256 | Mục tiêu, Phương pháp và Đóng góp nói lại cùng nội dung | Trung bình | Phương pháp: nói cách làm. Đóng góp: chỉ nói kết quả, không nhắc lại quy trình. |

### F. Nguy cơ đạo văn và thiếu trích dẫn

| Vị trí | Đoạn trích | Loại lỗi | Mức | Gợi ý sửa |
|---|---|---|---|---|
| **Toàn bộ Ch.1 đến Ch.5** | Nội dung trùng với bài báo tiếng Anh của chính bạn và các bản đồ án trước (DoAnTotNghiep, KhoaLuanTotNghiep đang nằm cùng thư mục) | **Tự đạo văn**: nếu bản cũ đã nộp lên hệ thống của trường thì Turnitin sẽ báo trùng | **Cao** | Hỏi thầy xem bản cũ đã nộp chưa. Khai báo việc tái sử dụng ở Lời cam đoan và trích dẫn bài báo nếu đã công bố. |
| 2.2 đến 2.5 (ULEN, TrIncNet, PMVT, MaizeFormerX, LDI-NET, Meta-Ensemble) | "ULEN [37] là mạng siêu nhẹ dùng tích chập depth-wise có kết nối tắt và spatial pyramid pooling"; "TrIncNet thay khối MLP trong bộ mã hóa ViT bằng một khối inception" | Dịch sát tóm tắt của bài gốc. Công cụ đối chiếu xuyên ngôn ngữ vẫn có thể phát hiện | Trung bình | Đã có trích dẫn. Nên viết lại theo góc nhìn so sánh (điểm giống và khác với ChiliEdgeNet) thay vì dịch phần tóm tắt. **Nên đưa qua công cụ kiểm tra.** |
| 1.2 (MobileNetV2), 1.3 (ViT, tự chú ý), 1.5 (Triplet Attention) | "nhằm tránh mất thông tin khi biểu diễn đi qua không gian nhiều chiều bị nén bởi ReLU"; "tính toán tương tác chéo giữa ba cặp chiều... thông qua ba nhánh xoay và nén" | Định nghĩa kiểu sách giáo khoa, gần với tóm tắt của bài gốc | Trung bình | Đã trích [33], [6], [28], nhưng đoạn 1.3 thứ 3 (ưu và nhược của Transformer) không có trích dẫn. **Nên kiểm tra.** |
| 1.4.3 | "LayerNorm, phép chuẩn hóa phổ biến trong Transformer, chuẩn hóa mỗi vector..." | Thiếu trích dẫn LayerNorm (Ba và cộng sự, 2016) | Trung bình | Bổ sung trích dẫn. |
| 1.6 (Gradient SHAP) | "thay vì một baseline cố định như Integrated Gradients... cho kết quả xấp xỉ giá trị Shapley" | Thiếu trích dẫn Integrated Gradients (Sundararajan 2017) và SHAP (Lundberg & Lee 2017) | Trung bình | Bổ sung 2 trích dẫn. |
| 1.7.3 | "Hiệu chỉnh Bonferroni đơn giản... khá bảo thủ"; "family-wise error rate" | Định nghĩa sách giáo khoa, Bonferroni chưa có trích dẫn | Thấp | Trích dẫn một giáo trình thống kê, hoặc Dunn (1961). |
| Mở đầu, Lý do đ.1 | "Cây ớt là một trong những loại cây trồng phổ biến ở nhiều vùng nông nghiệp Việt Nam, mang lại giá trị kinh tế đáng kể" | Câu mở chung chung, không có số liệu và nguồn. Câu kiểu này rất dễ trùng với luận văn khác | Trung bình | Thêm số liệu diện tích hoặc sản lượng ớt (Tổng cục Thống kê, Bộ NN&PTNT) kèm trích dẫn. |
| Mở đầu, Lý do đ.2 | "Học sâu... đã chứng minh hiệu quả cao trong bài toán phân loại bệnh cây trồng từ ảnh" | Khẳng định không có trích dẫn | Trung bình | Trích 1 hoặc 2 bài tổng quan. |
| 1.4 mở | "LLM hiện đại... quy mô hàng tỉ tham số khiến mọi khoản tiết kiệm đều có giá trị lớn" | Không có trích dẫn | Thấp | Trích một bài về LLM (ví dụ bài GQA [3] đã có sẵn). |
| 4.8 đoạn cuối | "PyTorch trên CPU không hỗ trợ suy luận FP16 hiệu quả"; lý do INT8 chậm hơn | Khẳng định kỹ thuật không có nguồn | Thấp | Ghi "theo tài liệu PyTorch" kèm liên kết, hoặc viết lại thành giả thuyết. |
| 3.1 (mô tả nhãn) | "anthracnose thể hiện các vết hoại tử sẫm màu..." | Mô tả triệu chứng không có nguồn nông học | Thấp | Trích một tài liệu bảo vệ thực vật. |
| 4.6 đ.1 | "thu thập tại một khu vườn ở Charpolisha, Jamalpur... từ tháng 10/2024 đến tháng 2/2025" | Dịch từ trang Mendeley, đã trích [29] | Thấp | Không cần sửa. |

## Nhận xét tổng thể

Về mặt kỹ thuật, văn bản đã sạch hơn các bản trước: không còn gạch dài, không còn câu sót lại của chatbot, và ngoặc kép đã thống nhất. Tuy vậy vẫn còn **65 ký tự ẩn** (54 U+200B trong bảng và 11 U+2060 trong công thức). Đây là loại dấu vết mà GPTZero và Turnitin bắt được ngay, nên cần xóa trước tiên. Giọng văn AI không nằm ở các từ nối sáo như "hơn nữa" hay "tóm lại" (bài gần như không dùng). Nó nằm ở ba thói quen khác: khuôn "không phải X mà là Y" và "thay vì" (hơn 55 lần), chuỗi câu rào đón xếp chồng, và câu rất dài nối bằng dấu chấm phẩy (trung bình 38 từ, 63 câu dài trên 60 từ). Theo ước lượng của tôi, **khoảng 40–45% phần thân bài có giọng AI ở mức trung bình đến cao**. Phần nặng nhất là Tóm tắt, phần "Lý do chọn đề tài", mục 2.8, các mục 4.2 đến 4.7, mục 5.2 và Kết luận. Ch.6 lại đọc tự nhiên nhất. Nguy cơ đạo văn theo nghĩa chép của người khác ở mức vừa phải, vì các định nghĩa đều đã có trích dẫn. Rủi ro lớn hơn là **tự trùng lặp** với bài báo và các bản đồ án trước của chính bạn, cộng với mấy dấu vết ghép bản mà hội đồng dễ nhận ra ("bài báo khoa học của đề tài", "ở các chương sau", features[0:13]).

**Ba việc nên sửa trước:**

1. **Xóa ký tự ẩn và thống nhất dấu** (khoảng 15 phút): trong Word dùng Find `^u8203` và `^u8288` rồi Replace để trống; đổi mọi dấu trừ sang "−"; khoảng số viết thành "từ ... đến ..."; bỏ gạch nối trong "hỏi-đáp", "kéo-thả", "thêm-lại".
2. **Viết lại 6 đoạn nặng nhất** theo nguyên tắc: mỗi đoạn tối đa 1 lần "thay vì" hoặc "không phải", mỗi đoạn có ít nhất 1 câu ngắn, tách các câu dài trên 50 từ. Sáu đoạn đó là: Tóm tắt đ.2, Lý do chọn đề tài đ.2–3, mục 5.2 "Về độ trễ", câu mở Ch.6, Kết luận đ.1 và đoạn cuối.
3. **Cắt phần lặp và sửa mâu thuẫn**: bớt số liệu lặp (98,93%, 671.173, 2,7 MB), bỏ giao thức đo lặp giữa 4.7 và 6.4, bỏ đoạn tổng quan lặp ở 4.6, rút 7.1 và Kết luận để không kể lại p-value. Sửa features[0:13], 2,84 MB, "ở các chương sau", "bài báo khoa học của đề tài". Bổ sung trích dẫn LayerNorm, SHAP/IG và số liệu cây ớt Việt Nam. Sau đó chạy lại Turnitin qua thư viện trường, có kèm bản đồ án cũ.
