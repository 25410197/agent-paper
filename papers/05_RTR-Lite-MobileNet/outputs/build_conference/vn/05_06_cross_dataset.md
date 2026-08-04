### 4.5. Đánh giá cross-dataset bốn lớp sau thích nghi few-shot

Các stress test tại Mục 4.3 chỉ tạo nhoè và giảm sáng tổng hợp từ ảnh ChiliEdge-DS, do đó chưa phản ánh đầy đủ chênh lệch miền thu nhận. Nghiên cứu thực hiện thêm một phép đánh giá trên **Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh**, phiên bản 1, phát hành trên Mendeley Data theo giấy phép CC BY 4.0 (Nirob et al., 2025). Ảnh gốc được thu thập tại vườn ở Charpolisha, Jamalpur (Bangladesh) trong giai đoạn tháng 10/2024–02/2025; phần bệnh lá gồm 1.856 ảnh thuộc sáu nhãn nguồn.

Chỉ bốn nhãn có tương ứng trực tiếp với không gian nhãn ChiliEdgeNet được sử dụng. Hai nhãn `Bacterial Spot` (156 ảnh) và `White spot` (195 ảnh) bị loại vì mô hình không có lớp đích tương ứng. Bộ dữ liệu nguồn không có lớp thán thư, nên không có hàng ground-truth `anthracnose`. Tuy nhiên, checkpoint vẫn giữ đủ năm đầu ra đã dùng khi huấn luyện; mọi dự đoán sang `anthracnose` vẫn được tính là sai và cột dự đoán này được giữ trong ma trận nhầm lẫn.

**Bảng 10a. Phép chia support/query đã kiểm tra từ bốn lớp có ánh xạ.**

| Nhãn nguồn (Nirob et al., 2025) | Nhãn ChiliEdgeNet | Ảnh gốc | Few-shot support | Query sạch | Thư mục test đã sửa |
|---|---|---:|---:|---:|---:|
| `Healthy Leaf` | `healthy` | 458 | 8 | 450 | 450 |
| `Curl Virus` | `leaf_curl` | 423 | 9 | 414 | 414 |
| `Cercospora Leaf Spot` | `leaf_spot` | 180 | 16 | 164 | 164 |
| `Nutrition Deficiency` | `yellow_leaf` | 444 | 9 | 435 | 435 |
| **Tổng** |  | **1.505** | **42** | **1.463** | **1.463** |

**Thiết lập few-shot.** Tập support trong `D:\ds\new_ds` gồm 42 ảnh lấy nguyên bản từ bộ dữ liệu của Nirob et al. (2025): 8 ảnh `healthy`, 9 ảnh `leaf_curl`, 16 ảnh `leaf_spot` và 9 ảnh `yellow_leaf`. Các ảnh support được bổ sung vào tập huấn luyện ChiliEdge-DS để tạo checkpoint thích nghi riêng; validation và test trong miền của ChiliEdge-DS không thay đổi. Đây là thích nghi miền có giám sát bằng một tập hỗ trợ ít mẫu, không phải meta-learning theo episode. Số shot khác nhau giữa các lớp nên thí nghiệm không phải thiết lập *N*-way *K*-shot cân bằng.

Các ảnh support được chính tác giả nghiên cứu lựa chọn thủ công, nhưng tiêu chí lựa chọn chưa được ghi lại tại thời điểm tạo tập. Vì vậy, phiên bản hiện tại được mô tả là **manually author-selected support**, không phải lấy mẫu ngẫu nhiên và cũng không phải nhãn do một annotator độc lập cung cấp. Trước bản nộp cuối, cần bổ sung tiêu chí thực tế đã sử dụng, chẳng hạn quy tắc về chất lượng ảnh, mức độ triệu chứng hoặc tính đa dạng cảnh; nếu không thể truy hồi, đây phải được giữ như một hạn chế về selection bias.

Để kiểm soát rò rỉ dữ liệu, nghiên cứu tạo `fewshot_support_manifest.csv` cho 42 ảnh support và `cross_dataset_query_manifest.csv` cho 1.463 ảnh query. Mỗi dòng lưu nhãn nguồn, nhãn ánh xạ, vai trò support/query, đường dẫn tương đối, kích thước tệp, giấy phép và SHA-256. Query được định nghĩa là phần bù của 42 ảnh support trong 1.505 ảnh thuộc bốn lớp, đối chiếu bằng SHA-256 thay vì chỉ dựa vào tên tệp. Kiểm toán xác nhận giao hash giữa support và query bằng 0. Tệp `cross_dataset_run_metadata.csv` liên kết checkpoint, seed, hash support/query và trạng thái tái lập. Checkpoint few-shot được huấn luyện với seed 42; notebook hoặc log huấn luyện đã thực thi vẫn cần được lưu kèm để xác nhận seed, cấu hình và đúng manifest support đã được nạp.

Để hỗ trợ tái lập độc lập, query sạch 1.463 ảnh được công bố dưới dạng **Chili Leaf Disease Cross-Dataset Query Set**, Kaggle version 2 (Hải, 2026a), còn 42 ảnh support được công bố dưới dạng **Chili Leaf Disease Few-Shot**, Kaggle version 1 (Hải, 2026b). Hai gói này là các phép chia dẫn xuất từ bộ Mendeley Data gốc, không thay thế yêu cầu trích dẫn bộ dữ liệu nguồn (Nirob et al., 2025).

**Kết quả trên query sạch.** Cả hai checkpoint được đánh giá trên cùng 1.463 ảnh trong `D:\ds\test`. Checkpoint gốc dùng seed 42 (SHA-256 `9dee091394ac8e888ab683a6fde3c251df34b8e952c9e693a4ffc6f35dadcea3`); checkpoint sau thích nghi few-shot cũng dùng seed 42 (SHA-256 `c31f643e601fd39d6f2e9f910a7fdd2cbeb889e0bd5b2adb2db8ec694766f9c9`). Ảnh được resize về 224×224 pixel và chuẩn hoá theo thống kê ImageNet. Macro precision, recall và F1 chỉ được lấy trung bình trên bốn lớp có ground truth; dự đoán sang `anthracnose` vẫn làm giảm accuracy.

**Bảng 10b. Kết quả trước và sau few-shot trên query sạch 1.463 ảnh.**

| Cấu hình | Seed | Accuracy | Precision-macro | Recall-macro | F1-macro | Số ảnh đúng |
|---|---:|---:|---:|---:|---:|---:|
| Checkpoint gốc, không few-shot | 42 | 30,96% | 0,4337 | 0,2843 | 0,1766 | 453/1.463 |
| Checkpoint sau few-shot | 42 | 89,75% | 0,8837 | 0,8683 | 0,8744 | 1.313/1.463 |
| **Chênh lệch** |  | **+58,78 pp** | **+0,4500** | **+0,5840** | **+0,6978** | **+860** |

Ở checkpoint gốc, `leaf_curl` có recall 1,0000 nhưng precision chỉ 0,3013, trong khi `yellow_leaf` có recall bằng 0. Sau few-shot, F1 của `healthy`, `leaf_curl`, `leaf_spot` và `yellow_leaf` lần lượt đạt 0,9168, 0,9619, 0,7557 và 0,8633. So sánh prediction theo từng ảnh cho thấy 443 ảnh được cả hai checkpoint dự đoán đúng, 140 ảnh bị cả hai dự đoán sai, 870 ảnh chuyển từ sai ở checkpoint gốc thành đúng sau few-shot và 10 ảnh theo chiều ngược lại. Kiểm định McNemar chính xác hai phía cho $p=1{,}83\times10^{-242}$. Đây là một tương phản ghép cặp riêng trên bộ dữ liệu ngoài miền, không thuộc họ kiểm định Holm của ablation. Prediction của toàn bộ 1.463 ảnh và các thống kê ghép cặp được lưu trong gói metadata đi kèm nghiên cứu.

![Hình S1. Ma trận nhầm lẫn của checkpoint gốc trên query sạch 1.463 ảnh.](<../../results/Cross check default/cross_dataset_local_cm.png>)

![Hình S2. Ma trận nhầm lẫn của checkpoint sau few-shot trên cùng query sạch 1.463 ảnh.](<../../results/cross check them data/cross_dataset_local_cm.png>)

> **Kiểm toán tính độc lập của phép chia:** ảnh trùng `leaf_spot/Cercospora Leaf Spot00176.JPG` đã được loại khỏi `D:\ds\test`. Kiểm tra SHA-256 ngày 04/08/2026 xác nhận thư mục này có đúng 1.463 ảnh, giao với support bằng 0, không thiếu và không thừa hash so với `cross_dataset_query_manifest.csv`; mọi mục trong `cross_dataset_split_audit.csv` đều PASS. Bảng 10b và Hình S1–S2 được tạo lại từ query sạch này.

> **Xác minh seed và cấu hình:** cả checkpoint gốc và checkpoint few-shot đều được huấn luyện với seed 42, cùng kiến trúc và hyperparameter nền; khác biệt chủ đích là 42 ảnh support được bổ sung vào dữ liệu huấn luyện của checkpoint few-shot. SHA-256 checkpoint, SHA-256 manifest, prediction từng ảnh và kết quả McNemar đã được lưu. Log hoặc cấu hình huấn luyện đã thực thi vẫn cần được lưu kèm làm bằng chứng tái lập cho seed và cấu hình nền.
