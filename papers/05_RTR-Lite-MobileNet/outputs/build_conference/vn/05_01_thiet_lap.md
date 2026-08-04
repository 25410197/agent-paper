### 4.1. Thiết lập

Các thí nghiệm được thực hiện trên Kaggle với GPU NVIDIA Tesla T4; mã phục dựng sử dụng PyTorch hoặc TensorFlow tùy theo kiến trúc. Mỗi baseline được đánh giá theo hai kịch bản:

- **Baseline:** không áp dụng data augmentation.
- **Augmented:** áp dụng các phép tăng cường dữ liệu trong Bảng 2.

Tập `val` và `test` không được tăng cường. Đối với các mô hình huấn luyện theo epoch, checkpoint tốt nhất theo tiêu chí validation của từng cấu hình được nạp lại trước khi đánh giá trên tập test.

**Khung đánh giá chung.** Mười bảy baseline sử dụng cùng bộ ChiliEdge-DS, lần chia `train`/`val`/`test` cố định, seed 42 và các độ đo accuracy, F1-macro cùng số tham số. Cả hai kịch bản đều được đánh giá trên cùng tập test không tăng cường. Các cấu hình phụ thuộc kiến trúc như kích thước đầu vào, batch size, số epoch, optimizer và cường độ augmentation được giữ theo thiết kế phục dựng tương ứng. Vì vậy, phép so sánh được kiểm soát theo dữ liệu, lần chia và độ đo, nhưng không sử dụng một công thức huấn luyện đồng nhất hoàn toàn cho mọi kiến trúc. Hai trường hợp có sự tham gia của kết quả test trong quá trình lựa chọn — RTR_Lite_MobileNet (Duhan et al., 2025) làm baseline chính và InceptionV3 + SVM làm đại diện cho nghiên cứu (Sujatha et al., 2025) — được công khai tại Mục 4.2 và chỉ được diễn giải như phân tích thăm dò.

Khung đánh giá chung của 17 baseline được giới hạn ở kết quả trong miền trên tập test ChiliEdge-DS, gồm accuracy, F1-macro và số tham số. Các phân tích bổ sung về ảnh nhoè, thiếu sáng, ablation kiến trúc, XAI và chuyển miền được thực hiện riêng cho ChiliEdgeNet cùng các biến thể kiểm soát tương ứng. Vì vậy, nghiên cứu không diễn giải các thí nghiệm này như một benchmark độ bền hoặc chuyển miền đồng đều cho toàn bộ 17 baseline.

Các baseline được phục dựng hoặc chuyển thể từ mô tả của công trình gốc để phù hợp với bài toán phân loại năm lớp trên ChiliEdge-DS. Những thay đổi đáng kể được tóm tắt dưới đây; vì vậy, các phiên bản này không được xem là bản tái tạo nguyên trạng của mô hình gốc.

| Baseline | Thành phần được giữ lại | Phần được chuyển thể |
|---|---|---|
| PDLC-ViT (classification-only; Hemalatha and Jayachandran, 2024) | Co-scale, co-attention, cross-attention, Transformer encoder và classification head | Bỏ nhánh segmentation/localization; chuyển từ đa nhiệm sang phân loại năm lớp và chỉ sử dụng classification loss; các thông số ViT không được công bố đầy đủ được xác lập khi phục dựng. |
| LDI-NET (Yang et al., 2024) | Feature Tokenizer và Token Encoder | Thay multi-label decoder dự đoán loại cây, bệnh và mức độ bằng global average pooling cùng linear classifier năm lớp. |
| InceptionV3 + SVM (Sujatha et al., 2025) | Backbone đóng băng để trích đặc trưng và SVM để phân loại | Chuyển sang ChiliEdge-DS; áp dụng StandardScaler, PCA giữ 95% phương sai và lựa chọn thăm dò được mô tả tại Mục 4.2. |
| Meta-Ensemble (Maurya et al., 2024) | MLP-Mixer, nhánh CNN–LSTM và SVM meta-classifier | Đổi đầu ra thành năm lớp; hiện thực patch embedding theo cấu hình được xác lập do mô tả gốc chưa cung cấp đầy đủ chi tiết. |

Mô hình đề xuất và toàn bộ biến thể ablation sử dụng **seed 42** và được khởi tạo lại trước mỗi lần chạy. Vì vậy, trong các thí nghiệm ablation, các cấu hình chỉ khác nhau ở thành phần được loại bỏ, thêm lại hoặc thay thế.

**Bảng 2. Cấu hình huấn luyện ChiliEdgeNet và các biến thể ablation.**

| Mục | Giá trị |
|---|---|
| Kích thước ảnh đầu vào | 224 × 224 |
| Transform baseline | Resize, ToTensor, Normalize |
| Transform augmented | Xoay 30°, ColorJitter, lật ngang, affine |
| Số epoch | 30 |
| Batch size | 32 |
| Hàm mất mát | Cross-entropy, label smoothing 0,05 |
| Learning-rate schedule | Warmup tuyến tính 10% → cosine decay |
| Learning rate backbone / khối mới | 1e-4 / 1e-3 |
| Seed | **42** |

#### 4.1.1. Số lần chạy và kiểm định thống kê

Kết quả của mô hình đề xuất và các biến thể ablation được báo cáo từ **một lần chạy với seed 42**. Nghiên cứu không báo cáo trung bình hoặc độ lệch chuẩn giữa nhiều seed và không diễn giải các kết quả này như bằng chứng về tính ổn định giữa các lần khởi tạo. Kiểm định McNemar được sử dụng để đánh giá sự khác biệt giữa hai mô hình trên từng ảnh của cùng tập test gồm 373 ảnh.

Đối với mỗi phép so sánh, ký hiệu \(b/c\) biểu thị:

- \(b\): số ảnh mô hình thứ nhất dự đoán đúng, trong khi mô hình thứ hai dự đoán sai;
- \(c\): số ảnh mô hình thứ nhất dự đoán sai, trong khi mô hình thứ hai dự đoán đúng.

McNemar kiểm tra liệu số trường hợp bất đồng theo hai hướng có khác biệt có ý nghĩa thống kê hay không. Kiểm định này không thay thế việc đánh giá phương sai giữa nhiều seed; giới hạn đó được thảo luận tại Mục 5.

Trong nghiên cứu này, McNemar được áp dụng cho các so sánh giữa ChiliEdgeNet và những biến thể ablation trên cùng 373 ảnh test, tương ứng với Bảng 5–7, và cho so sánh ghép cặp hai checkpoint trên cùng 1.463 ảnh query cross-dataset ở Bảng 10b. Kiểm định không được sử dụng để xếp hạng toàn bộ 17 baseline trong Bảng 3.

Kiến trúc nền tảng RTR_Lite_MobileNet (Duhan et al., 2025) trong Mục 4.2 cũng được phục dựng với **seed 42**. Vì vậy, phép so sánh khớp seed và dùng cùng quy trình, nhưng mỗi kiến trúc vẫn chỉ được chạy một lần.

#### 4.1.2. Đo độ trễ suy luận

Độ trễ CPU của ChiliEdgeNet là **20,24 ms/ảnh**, tương đương **49,41 ảnh/giây**, được đo trên CPU Intel Xeon 2,00 GHz với 4 lõi logic của máy ảo Kaggle. Thiết lập đo sử dụng PyTorch FP32, batch size bằng 1, gồm 10 vòng warmup và 50 vòng đo. Giá trị 49,41 ảnh/giây là thông lượng suy luận quy đổi từ benchmark ảnh đơn trên máy chủ, không phải FPS của ứng dụng camera.

Trên cùng môi trường, GPU Tesla T4 đạt:

- **5,46 ms/ảnh** với batch size bằng 1;
- **0,82 ms/ảnh** với batch size bằng 32.

CPU được sử dụng là CPU x86 của máy chủ, không phải phần cứng biên. Vì vậy, các kết quả này chỉ phản ánh chi phí suy luận trong môi trường thực nghiệm hiện tại.
