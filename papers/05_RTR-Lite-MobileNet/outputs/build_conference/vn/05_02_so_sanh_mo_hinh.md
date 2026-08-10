### 4.2. So sánh với các mô hình tiên tiến

Với seed 42, ChiliEdgeNet đạt **97,86%** trong kịch bản baseline và **98,93%** trong kịch bản augmented. Macro-F1 tương ứng là 0,9790 và 0,9897.

Tổng cộng 17 baseline từ 15 nghiên cứu gần đây được phục dựng, đánh giá trong cùng một khung và trình bày đầy đủ trong Bảng 3. Vì các baseline chỉ được chạy một lần và không được tinh chỉnh riêng cho ChiliEdge-DS, bảng này phản ánh kết quả dưới cùng dữ liệu, lần chia và độ đo, không phải bảng xếp hạng tuyệt đối giữa các kiến trúc.

RTR_Lite_MobileNet (Duhan et al., 2025) được chọn làm baseline chính vì đây là kiến trúc nền tảng trực tiếp của ChiliEdgeNet và có accuracy quan sát cao nhất trong nhóm baseline được đánh giá trên tập test ChiliEdge-DS. Do kết quả test đã tham gia vào quá trình lựa chọn baseline chính, phép so sánh trực tiếp với RTR_Lite_MobileNet được xem là phân tích thăm dò trên lần chia hiện tại, không phải một đánh giá test độc lập dùng để xác lập thứ hạng thống kê hoặc khẳng định ưu thế tổng quát. Mức giảm số tham số vẫn là một so sánh cấu trúc và không phụ thuộc vào việc lựa chọn bằng test.

Đối với nghiên cứu của Sujatha et al. (2025), mười tổ hợp giữa hai backbone VGG19 và InceptionV3 với năm bộ phân loại SVM, kNN, Random Forest, AdaBoost và Decision Tree được khảo sát trên cùng lần chia dữ liệu. Cấu hình InceptionV3 + SVM có accuracy quan sát cao nhất trên tập test được chọn làm đại diện trong Bảng 3. Do việc lựa chọn có sử dụng kết quả test, kết quả này cũng được xem là phân tích thăm dò trên lần chia hiện tại và không được dùng để khẳng định ưu thế thống kê hoặc thiết lập thứ hạng tuyệt đối giữa các kiến trúc.

#### 4.2.1. So sánh với kiến trúc nền tảng

Trong lần chạy seed 42, ChiliEdgeNet đạt **98,93% augmented accuracy**, cao hơn RTR_Lite_MobileNet (Duhan et al., 2025) đạt **98,12%**, tương ứng chênh lệch **0,80 điểm phần trăm hay 3 ảnh** trên tập test 373 ảnh.

ChiliEdgeNet sử dụng **671.173 tham số**, trong khi RTR_Lite_MobileNet sử dụng **1.032.568 tham số**, tương ứng mức giảm khoảng **35%**. Hai mô hình dùng cùng seed 42 và cùng quy trình, nhưng đều mới chỉ được chạy một lần; khoảng tin cậy Wilson 95% chồng lấn nên chênh lệch accuracy phải được diễn giải thận trọng.

**Bảng 3. Độ chính xác hai kịch bản, F1-macro và số tham số trên bộ dữ liệu ớt.** *(Mỗi mô hình là một lần chạy seed 42 trong cùng khung đánh giá; chênh lệch dưới khoảng 1 pp chỉ tương ứng 1–4 ảnh trên 373.)*

| Mô hình | Acc Baseline (%) | Acc Augmented (%) | F1-macro (Aug) | Tham số |
|---|---|---|---|---|
| **ChiliEdgeNet (của chúng tôi)** | **97,86** | **98,93** | **0,9897** | **671.173** |
| RTR_Lite_MobileNet (Duhan et al., 2025) | 97,32 | 98,12 | 0,9821 | 1.032.568 |
| GrapeLeafNet (Karthik et al., 2024) | 91,15 | 97,59 | 0,9769 | 5.308.035 |
| PMVT-XS (Li et al., 2023) | 82,31 | 96,25 | 0,9644 | 2.367.515 |
| ResNet-50 (Mishra et al., 2026) | 95,98 | 95,71 | 0,9586 | 23.597.957 |
| EfficientNetB0 (Mishra et al., 2026) | 94,37 | 94,37 | 0,9463 | 4.055.976 |
| TrIncNet (Gole et al., 2023) | 80,70 | 90,62 | 0,9092 | 7.186.661 |
| InceptionV3 + SVM (Sujatha et al., 2025) | 93,83 | 90,35 | 0,9069 | 25.112.264 |
| Meta-Ensemble (Maurya et al., 2024) | 88,74 | 88,74 | 0,8908 | 1.252.638 |
| CustomMNetV3L (Rahaman et al., 2026) | 87,67 | 86,86 | 0,8639 | 3.275.909 |
| ULEN (Wang et al., 2023) | 93,03 | 85,52 | 0,8603 | 28.069 |
| MaizeFormerX (Rahman et al., 2026) | 79,89 | 82,04 | 0,8185 | 3.553.061 |
| AlexNet (Mishra et al., 2026) | 85,52 | 77,75 | 0,7747 | 46.767.493 |
| Lite-MDC (Bhagat et al., 2024) | 68,10 | 76,68 | 0,7658 | 2.288.965 |
| ViT (Ali et al., 2025) | 72,39 | 73,99 | 0,7472 | 50.013.061 |
| MRW-CNN (Joseph et al., 2024) | 74,26 | 73,73 | 0,7377 | 797.413 |
| PDLC-ViT (classification-only; Hemalatha and Jayachandran, 2024) | 74,26 | 70,78 | 0,7054 | 17.547.269 |
| LDI-NET (Yang et al., 2024) | 70,78 | 68,63 | 0,6839 | 8.798.597 |

#### 4.2.2. Phân tích lỗi

Trong kịch bản augmented, ChiliEdgeNet dự đoán sai các ảnh sau:

- một ảnh `leaf_spot` bị dự đoán thành `healthy`;
- ba ảnh `yellow_leaf` bị dự đoán thành `leaf_curl`.

Accuracy đạt 98,93% với khoảng tin cậy Wilson 95% **[97,28%; 99,58%]**. Các lỗi còn lại tập trung tại ranh giới giữa triệu chứng vàng lá và xoăn lá.

![Hình 2. Ma trận nhầm lẫn của ChiliEdgeNet trên tập test trong kịch bản augmented.](../../results/notebook_c1/c14_cm_s2.png)
