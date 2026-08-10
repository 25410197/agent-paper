### 4.4. Diễn giải mô hình

Các phân tích trong mục này sử dụng **checkpoint gốc của ChiliEdgeNet**, được huấn luyện chỉ trên ChiliEdge-DS và dùng để tạo kết quả trong miền ở Bảng 3. Checkpoint bổ sung dữ liệu cho thí nghiệm chuyển miền tại mục 4.5 là một mô hình thích nghi riêng và không được sử dụng để tạo các kết quả diễn giải dưới đây.

#### 4.4.1. AI giải thích được (XAI)

Độ chính xác cao chưa đủ để diễn giải hành vi của mô hình, đặc biệt khi ChiliEdge-DS chủ yếu sử dụng nền trắng có kiểm soát. Nghiên cứu triển khai **ba phương pháp XAI bổ trợ** trên checkpoint ChiliEdgeNet augmented seed 42, mỗi phương pháp sử dụng một ảnh đại diện cho mỗi lớp. **Gradient SHAP** (Erion et al., 2021) ước lượng đóng góp của pixel từ gradient đối với một phân phối baseline; **Occlusion Sensitivity** (Zeiler and Fergus, 2014) che lần lượt các vùng 32×32 pixel và đo thay đổi đầu ra; **Attention Rollout** (Abnar and Zuidema, 2020) tổng hợp dòng chú ý qua khối Transformer trên toàn bộ lưới 14×14. Do kiến trúc sử dụng mean pooling và không có token `CLS`, bản đồ Rollout được tạo bằng cách lấy trung bình dòng chú ý qua 196 token, không truy vết từ một token đầu ra riêng.

Ba bản đồ cung cấp các góc nhìn không giống nhau (Hình 4–6): Gradient SHAP cho quy thuộc mức pixel phân bố trên bề mặt đối tượng; Occlusion biểu diễn độ nhạy theo vùng; Attention Rollout phản ánh sự lan truyền chú ý nội tại của Transformer. Trên năm ảnh minh họa, các vùng có đáp ứng thường giao với lá/quả hoặc vùng biểu hiện bệnh, nhưng mức tập trung khác nhau giữa phương pháp và lớp. Vì đánh giá chỉ dựa trên một ảnh mỗi lớp và chưa có ground truth phân vùng triệu chứng, kết quả này chỉ là khảo sát định tính thăm dò; nó không chứng minh mô hình đã học đúng đặc trưng bệnh lý, hoàn toàn bỏ qua nền hoặc sử dụng quan hệ nhân quả đúng. Lưới 14×14 cung cấp 196 vị trí không gian, nhiều gấp bốn lần lưới 7×7, giúp bản đồ Rollout có độ phân giải trực quan chi tiết hơn.

<!-- IMG2 -->
![Hình 4. Occlusion Sensitivity cho năm lớp bệnh (trên: ảnh gốc; dưới: bản đồ nhạy cảm chồng lên). Vùng sáng là nơi che đi làm mô hình mất tự tin nhất; kỹ thuật này coi mô hình như hộp đen.](../../results/notebook_c2/xai_occlusion.png)

![Hình 5. Gradient SHAP cho năm lớp (trên: ảnh gốc; dưới: độ lớn quy thuộc pixel chồng lên ảnh), được tính với 20 mẫu và hai baseline tensor hằng 0 và 1 trong không gian đầu vào đã chuẩn hóa.](../../results/notebook_c2/xai_shap.png)

![Hình 6. Attention Rollout của khối Tiny Transformer trên lưới 14×14, được tổng hợp bằng cách lấy trung bình dòng chú ý qua 196 token do mô hình sử dụng mean pooling và không có token `CLS`.](../../results/notebook_c2/xai_attn_rollout.png)
<!-- /IMG2 -->

#### 4.4.2. Đối chiếu với một mô hình đa phương thức tổng quát

Nghiên cứu sử dụng mô hình đa phương thức tổng quát **Gemini 3.1 Flash Lite**, model ID API `gemini-3.1-flash-lite`, thông qua Gemini API; API key được tạo và quản lý bằng Google AI Studio. Mô hình được truy cập ngày **08/07/2026** (Google DeepMind, 2026) để phân loại trực tiếp ảnh gốc ở chế độ zero-shot. Không đặt riêng temperature, top-p hoặc giới hạn token; lần chạy sử dụng cấu hình sinh mặc định của endpoint tại ngày truy cập. Đây là đối chiếu tham khảo, không phải baseline hoàn toàn ngang hàng: Gemini không được fine-tune trên ChiliEdge-DS, trong khi ChiliEdgeNet là bộ phân loại chuyên biệt được huấn luyện có giám sát cho đúng năm lớp.

**Prompt được sử dụng.** Nội dung dưới đây được giữ nguyên như trong mã chạy, gồm cả cách viết tiếng Việt không dấu:

```text
Ban la chuyen gia benh hoc cay trong. Anh dua vao la mot la/qua ot co the bi benh.
Hay xac dinh MOT vung hinh chu nhat duy nhat bao quanh vung bat thuong/trieu chung benh
ro nhat tren anh (dom benh, vet thoi, vung vang la, vung xoan/nhan la...).
Neu la hoan toan khoe manh, khoanh vung toan bo la/qua.

Sau do phan loai anh vao DUNG MOT trong 5 nhan sau (bat buoc chon dung 1 trong 5,
khong duoc tao nhan khac): anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf.

Chi tra ve DUY NHAT mot JSON array dung format sau, khong viet gi ngoai JSON:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "ten_trieu_chung_ngan_gon_tieng_viet", "predicted_class": "mot_trong_5_nhan_tren", "reason": "mot cau ngan giai thich CAN CU thi giac vi sao chon nhan do"}]
Toa do box_2d theo thang 0-1000 (chuan hoa theo chieu cao/rong anh).
```

**Ánh xạ nhãn và đầu ra không hợp lệ.** Trường `predicted_class` được chấm trực tiếp theo đúng một trong năm chuỗi `anthracnose`, `healthy`, `leaf_curl`, `leaf_spot` và `yellow_leaf`; không dịch đồng nghĩa, không sửa nhãn thủ công và không ánh xạ hậu nghiệm. Trường `label` và câu `reason` chỉ được lưu để kiểm tra, không tham gia tính accuracy. Bộ phân tích ưu tiên đọc JSON; nếu JSON sai cú pháp thì dùng biểu thức chính quy để trích các trường. Lỗi API, thiếu trường bắt buộc, không parse được hoặc `predicted_class` nằm ngoài năm nhãn sẽ được thử lại tối đa ba lần; nếu vẫn thất bại, bản ghi được gắn `gemini_error`, loại khỏi mẫu số accuracy của Gemini và số lượng lỗi phải được báo cáo riêng cùng độ phủ.

Artifact hiện lưu đủ **373 ảnh**, chạy ngày **08/07/2026**, toàn bộ **100%** lượt gọi hợp lệ và không có nhãn ngoài tập cho phép. Gemini đạt accuracy **78,55%**, còn ChiliEdgeNet đạt **98,93%** trên đúng tập test 373 ảnh — khớp với Bảng 3. Trong ba ảnh `yellow_leaf` của tập test, hai ảnh cũng là hai trong ba lỗi ChiliEdgeNet dự đoán nhầm thành `leaf_curl` đã nêu ở Mục 4.2.2, còn một ảnh ChiliEdgeNet dự đoán đúng; cả ba đều được Gemini phân loại đúng `yellow_leaf`.

**Về độ trễ.** Con số xấp xỉ 105 lần đối chiếu **20,24 ms suy luận cục bộ** với **2.123 ms trung vị round-trip API đám mây**. Hai con số không tương đương: round-trip bao gồm độ trễ mạng, hàng đợi phía máy chủ và gộp lô phía nhà cung cấp — không cái nào là phần tính toán riêng của mô hình. Đây là đối chiếu giữa hai phương thức triển khai, không phải phát biểu rằng một kiến trúc tính toán nhanh hơn kiến trúc kia khoảng 105 lần. Tương tự, dù accuracy được tính trên cùng 373 ảnh, chênh lệch accuracy không tạo thành so sánh ngang hàng vì hai hệ thống có chế độ huấn luyện và mức độ chuyên biệt khác nhau. Endpoint thương mại cũng có thể được cập nhật, nên phép đối chiếu chỉ tái lập đúng với model ID, prompt và ngày truy cập đã nêu.

**Bảng 9. Đối chiếu ChiliEdgeNet với Gemini 3.1 Flash Lite trên toàn bộ 373 ảnh test (truy cập 08/07/2026).**

| Hệ thống | Accuracy | Latency/ảnh |
|---|---|---|
| **ChiliEdgeNet (CPU cục bộ)** | **98,93%** | **20,24 ms** (suy luận cục bộ) |
| Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) | 78,55% | 2.123 ms (trung vị round-trip API) |
