# Một số góp ý
**(TS. Phùng Thế Bảo)**

## 1. Không nhất quán về số liệu trong tuyên bố cải thiện chính so với mô hình gốc

Mục 4.2 viết:

> "The seed-controlled gain over the already-strong original is small but consistent, +0.36 pp mean-to-mean (98.93% vs. 98.57%)."

Câu này đang so sánh ChiliEdgeNet với RTR_Lite_MobileNet ("the already-strong original", mô hình nền tảng mà bài báo kế thừa). Tuy nhiên, theo chú thích của Bảng 1, con số **98,57%** lại chính là **three-seed mean** của **bản thân ChiliEdgeNet** ở kịch bản **baseline** (không augment), còn **98,93%** là **three-seed mean** của ChiliEdgeNet ở kịch bản **augmented**.

Như vậy, đây là hai con số **so sánh nội bộ** giữa hai kịch bản huấn luyện của cùng một mô hình đề xuất, **không phải** so sánh với RTR_Lite_MobileNet.

Trong khi đó, RTR_Lite_MobileNet (Bảng 1, dòng 2) chỉ có kết quả **một lần chạy**:

- Baseline: **97,32%**
- Augmented: **98,12%**

Nếu phép so sánh "gain over the original" được tính đúng theo các con số đã công bố thì kết quả sẽ khác đáng kể so với **+0,36 điểm phần trăm** được nêu. Ví dụ:

- 98,93% so với 98,12% → **+0,81 điểm phần trăm**
- 99,20% so với 98,12% → **+1,08 điểm phần trăm**

Đây là một điểm **không nhất quán** cần được tác giả rà soát và sửa lại cho chính xác.

---

## 2. Thiếu ablation ở mức từng kỹ thuật cấu thành (2D-RoPE, GQA, RMSNorm)

Đóng góp kỹ thuật cốt lõi trong tiêu đề, tóm tắt và phần giới thiệu là việc tích hợp **ba kỹ thuật hiệu quả tham số** vốn có nguồn gốc từ LLM (**2D-RoPE, GQA, RMSNorm**) vào khối Transformer.

Tuy nhiên, ablation trong Mục 4.3 chỉ thực hiện ở mức **khối lớn** (toàn bộ Tiny Transformer so với các khối CNN attention), mà **không có thực nghiệm tách riêng vai trò của từng kỹ thuật** trong số ba kỹ thuật này.

Cụ thể, các câu hỏi sau chưa được trả lời:

- Nếu thay **2D-RoPE** bằng learned/absolute positional embedding thông thường, độ chính xác thay đổi ra sao?
- Nếu thay **GQA** bằng multi-head attention tiêu chuẩn (không chia sẻ key/value), độ chính xác và số tham số thay đổi ra sao? Liệu GQA có thực sự cần thiết ở quy mô mô hình nhỏ như thế này, nơi ưu điểm về bộ nhớ của GQA (vốn được thiết kế cho LLM cỡ lớn) có thể không còn rõ rệt?
- Nếu thay **RMSNorm** bằng LayerNorm tiêu chuẩn, độ chính xác và độ ổn định huấn luyện thay đổi ra sao?

Cần bổ sung để làm rõ nội dung "tích hợp ba kỹ thuật hiệu quả tham số" như tên gọi và tóm tắt của bài báo đã nhấn mạnh.

---

## 3. So sánh với mô hình nền tảng (RTR_Lite_MobileNet) chưa cân xứng về độ chặt chẽ thống kê

Mô hình đề xuất được chạy **3 seed** và báo cáo **mean ± SD**, trong khi RTR_Lite_MobileNet – phép so sánh quan trọng nhất vì đây chính là kiến trúc nền tảng mà ChiliEdgeNet kế thừa và cải tiến – chỉ được chạy **một lần (single-run)**, giống như 26 cấu hình baseline còn lại.

Điều này khiến mức cải thiện tuyệt đối (khoảng **0,8–1,1 điểm phần trăm**, tùy theo cách tính) không có ước lượng phương sai để đối chiếu, trong khi bản thân SD của ChiliEdgeNet qua 3 seed đã là **0,27–0,41 điểm phần trăm**, cùng bậc độ lớn với mức cải thiện được báo cáo.

Nên chạy RTR_Lite_MobileNet với cùng **3 seed** (ít nhất đối với phép so sánh này) để có thể khẳng định chắc chắn hơn rằng mức cải thiện là thực chất chứ không nằm trong biên độ dao động ngẫu nhiên giữa các lần chạy.

---

## 4. Ablation "knockout" đo lường sự phụ thuộc, không đo đóng góp biên – cần bổ sung ablation huấn luyện lại

Bài báo đã nêu rõ giới hạn này (Mục 4.3, đoạn cuối).

Tuy nhiên, về mặt yêu cầu bình duyệt, một ablation kiểu **retrain-from-scratch** (huấn luyện lại mô hình sau khi loại bỏ từng khối, thay vì chỉ thay bằng identity mapping trên mạng đã huấn luyện) vẫn là tiêu chuẩn kỳ vọng ở hầu hết các tạp chí AI để khẳng định giá trị thực sự của một thành phần kiến trúc.

Hiện tại nghiên cứu để việc này ở phần **future work**, nhưng vì đây là ablation **duy nhất** trong bài, và Tiny Transformer là thành phần cốt lõi của đóng góp, nên ít nhất cần bổ sung **một ablation huấn luyện lại** (ví dụ: ChiliEdgeNet không có Tiny Transformer, huấn luyện lại từ đầu, sau đó so sánh độ chính xác tối đa đạt được) trước khi công bố chính thức, thay vì chỉ dựa vào knockout ở thời điểm suy luận.

---

## 5. Khoảng cách miền (domain gap) giữa ảnh chụp studio nền trắng và điều kiện chụp thực tế ngoài đồng

Mục 3.1 mô tả ảnh trong ChiliEdge-DS được:

> "photographed against a white background to make the diseased region stand out."

Đây là điều kiện chụp ảnh có kiểm soát (studio-like), khác biệt đáng kể so với điều kiện thực tế mà nông dân sẽ chụp bằng điện thoại ngoài đồng ruộng, nơi:

- nền lá hoặc đất phức tạp,
- ánh sáng tự nhiên thay đổi,
- góc chụp không cố định,
- có thể xuất hiện nhòe hoặc rung ảnh.

Trong khi mục tiêu của bài báo (Mục 1) là:

> "farmers need a diagnostic tool that runs directly on an edge device"

để sử dụng ngoài thực địa, thì độ chính xác **99,20%** đạt được trên ảnh nền trắng có kiểm soát **có thể không phản ánh đúng hiệu năng khi triển khai thực tế**.

Đây là một khoảng trống quan trọng cần được thảo luận rõ ràng như một giới hạn của nghiên cứu, và tốt nhất nên bổ sung thử nghiệm trên một tập ảnh có nền tự nhiên hoặc ngoài đồng (dù với số lượng nhỏ) để đánh giá sơ bộ mức suy giảm hiệu năng khi chuyển miền.

---

## 6. Tập test rất nhỏ, không có cross-validation – chênh lệch giữa các mô hình có thể nằm trong biên độ nhiễu

Tập test chỉ có **373 ảnh** cho **5 lớp**, và mô hình đề xuất chỉ sai **3 ảnh** (tương ứng mỗi ảnh sai hoặc đúng làm thay đổi khoảng **0,27 điểm phần trăm accuracy**).

Với độ phân giải như vậy, mức chênh lệch **0,3–1 điểm phần trăm** được dùng để xếp hạng giữa nhiều mô hình trong Bảng 1 chỉ tương ứng với **1–4 ảnh khác biệt**, rất dễ nằm trong biên độ nhiễu ngẫu nhiên của một lần chia dữ liệu.

Bài báo chỉ dùng:

- một lần chia train/validation/test cố định (single split),
- không có k-fold cross-validation,
- không có khoảng tin cậy hoặc kiểm định thống kê cho các so sánh trong Bảng 1.

Nên cân nhắc bổ sung ít nhất:

- khoảng tin cậy (ví dụ Wilson score interval cho accuracy với **n = 373**), hoặc
- lặp lại việc chia dữ liệu (nếu tài nguyên cho phép),

để tăng độ tin cậy của kết luận "đứng đầu trong 27 cấu hình".

---

## 7. Phần Related Work còn sơ lược so với quy mô so sánh thực nghiệm (27 cấu hình từ 15 nghiên cứu)

Mục 2 (Related Work) chỉ có một đoạn văn ngắn, phân loại các hướng nghiên cứu thành bốn nhóm chung:

- CNN nhẹ,
- Vision Transformer,
- Hybrid,
- So sánh rộng.

Tuy nhiên, phần này chưa thảo luận cụ thể các phương pháp trong **15 nghiên cứu** được dùng làm baseline ở Bảng 1.

Với quy mô so sánh lớn như vậy, phần Related Work nên được mở rộng để:

1. Mô tả ngắn gọn ý tưởng chính của một số nghiên cứu tiêu biểu được chọn làm baseline.
2. Nêu rõ tiêu chí lựa chọn 15 nghiên cứu (ví dụ: phạm vi tìm kiếm tài liệu, tiêu chí đưa vào hoặc loại trừ, khoảng thời gian xuất bản) để người đọc và phản biện có thể đánh giá tính đại diện của nhóm baseline, tránh nghi ngờ về thiên lệch lựa chọn (selection bias).

---

## 8. Thiếu thông tin về phân bố lớp và phần cứng suy luận

### Phân bố lớp

Chưa thấy số lượng ảnh cụ thể theo từng lớp:

- anthracnose,
- healthy,
- leaf_curl,
- leaf_spot,
- yellow_leaf.

Bài báo chỉ nêu tổng số ảnh của các tập train/validation/test.

Cần bổ sung bảng phân bố lớp để người đọc đánh giá được:

- mức độ mất cân bằng dữ liệu,
- độ tin cậy của Macro-F1 và Macro-Recall.

### Phần cứng suy luận

Thời gian suy luận **22,9 ms/ảnh** được nhấn mạnh nhiều lần (tóm tắt và kết luận), nhưng chưa thấy cấu hình CPU cụ thể (model CPU, số lõi, xung nhịp) trong phần Setup (Mục 4.1).

Hiện chỉ có cấu hình GPU Kaggle (Tesla T4) dùng cho huấn luyện.

Đây là thông tin cần thiết để đảm bảo khả năng tái lập của kết quả suy luận, vốn là một trong những điểm nhấn quan trọng của bài báo ("runs fully offline on a CPU at 22.9 ms/image").

---

## 9. Các vấn đề nhỏ về hình thức

### Đánh số Bảng/Hình

Đánh số Bảng/Hình hiện đang bị lặp:

- cả ba bảng thực nghiệm đều được ghi là **Table 1**,
- trong khi phần nội dung lại tham chiếu **Table 2** và **Table 3**.

Tương tự:

- cả hai hình XAI đều được ghi là **Fig. 1** (Grad-CAM và Attention Rollout).

Cần rà soát và cập nhật lại toàn bộ số thứ tự Bảng/Hình trong bản thảo trước khi nộp.

### So sánh "92× faster than a vision LLM"

Cụm từ:

> "92× faster than a vision LLM"

đang so sánh:

- thời gian suy luận cục bộ (**22,9 ms**),
- với thời gian round-trip của API đám mây (**2.117 ms**),

trong khi round-trip API bao gồm:

- độ trễ mạng,
- hàng đợi phía máy chủ,
- thời gian xử lý của mô hình.

Nên ghi chú rõ đây là **so sánh độ trễ hệ thống (system-level latency)**, không phải so sánh thuần túy tốc độ suy luận giữa hai kiến trúc mô hình.

### Diễn đạt

Cụm:

> "Trained end-to-end with no actual language model"

khá khó hiểu.

Có thể viết rõ hơn, ví dụ:

> "Although ChiliEdgeNet adopts three parameter-efficient techniques inspired by LLM architectures, it is trained entirely as a vision model and contains no language-model component."

### Phiên bản Vision-LLM

Tên mô hình Vision-LLM dùng để so sánh ("Gemini 3.1 Flash Lite") nên ghi rõ:

- phiên bản API,
- hoặc ngày truy cập,

vì các mô hình thương mại thường được cập nhật liên tục theo thời gian, ảnh hưởng đến khả năng tái lập của kết quả so sánh.