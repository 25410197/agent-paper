### 4.3. Phân tích ablation

Mọi biến thể được huấn luyện lại từ đầu với **seed 42**, cùng dữ liệu, công thức huấn luyện và 30 epoch. Mô hình hiện tại không chứa Triplet Attention và `RES_Block`. Do accuracy trên ảnh sạch đã gần bão hòa, vai trò của Tiny Transformer còn được khảo sát bằng một stress test nhiều mức trên ảnh suy giảm tổng hợp.

#### 4.3.1. Ablation ở cấp khối

Hai phép cắt trực tiếp lần lượt bỏ Tiny Transformer và vô hiệu hóa mọi phép cộng skip của các `InvertedResidual` trong backbone. Với Triplet Attention và `RES_Block`, nghiên cứu dùng thiết kế thêm-lại 2×2: có cả hai, chỉ Triplet, chỉ `RES_Block`, và không có cả hai (mô hình hiện tại).

**Bảng 4. Kết quả ablation kiến trúc trên ảnh sạch, seed 42.**

| Cấu hình | Tham số | Accuracy (%) |
|---|---:|---:|
| **Hiện tại: không Triplet/RES_Block** | **671.173** | **98,93** |
| Bỏ Tiny Transformer | 555.845 | 99,20 |
| Tắt residual MobileNetV2 | 671.173 | 96,51 |
| Thêm Triplet + RES_Block | 671.732 | 97,59 |
| Chỉ Triplet | 671.473 | 98,12 |
| Chỉ RES_Block | 671.432 | 98,93 |

**Bảng 5. McNemar cho hai phép cắt trực tiếp trên ảnh sạch.** Delta dương nghĩa mô hình hiện tại tốt hơn biến thể; \(p_{Holm}\) hiệu chỉnh cho hai tương phản đã định trước.

| Phép cắt | Delta (pp) | b / c | \(p\) | \(p_{Holm}\) |
|---|---:|---:|---:|---:|
| Bỏ Tiny Transformer | −0,27 | 0 / 1 | 1,0000 | 1,0000 |
| Tắt residual MobileNetV2 | +2,41 | 10 / 1 | 0,0117 | **0,0234** |

Trên ảnh sạch, biến thể bỏ Transformer cao hơn một ảnh nên chưa cho thấy lợi ích của khối này. Ngược lại, tắt residual connections làm giảm chín dự đoán đúng ròng và khác biệt vẫn có ý nghĩa sau hiệu chỉnh Holm. Kết quả này ủng hộ việc giữ các nhánh skip đang hoạt động của MobileNetV2.

**Bảng 6. Ba tương phản loại bỏ Triplet Attention/RES_Block từ cấu hình có cả hai.** Giá trị \(p\) đã hiệu chỉnh Holm cho ba tương phản trên ảnh sạch.

| Tương phản | Delta (pp) | \(p_{Holm}\) |
|---|---:|---:|
| Bỏ `RES_Block` | +0,54 | 0,625 |
| Bỏ Triplet Attention | +1,34 | 0,375 |
| Bỏ cả hai về mô hình hiện tại | +1,34 | 0,375 |

Không tương phản nào đạt ý nghĩa sau hiệu chỉnh Holm. Trong lần chạy này, `RES_Block` đứng riêng cho cùng accuracy với mô hình hiện tại, còn Triplet Attention tạo xu hướng giảm hiệu năng. Kết quả ủng hộ việc loại hai khối để tránh tăng độ phức tạp không cần thiết, nhưng không chứng minh các kiến trúc tương đương.

#### 4.3.2. Stress test vai trò của Tiny Transformer

Stress test sử dụng cùng **373 ảnh test** và hai checkpoint seed 42: ChiliEdgeNet đầy đủ và biến thể huấn luyện lại sau khi bỏ Tiny Transformer. Gaussian blur được tạo bằng `cv2.GaussianBlur` với `sigmaX=0` tại các kernel 3, 5, 7 và 9; giảm sáng được tạo bằng cách nhân toàn bộ cường độ pixel với 0,8; 0,6; 0,4 và 0,2. Tám điều kiện được xác định trước khi chấm điểm. Ảnh sạch chỉ là mốc tham chiếu và không thuộc họ tám kiểm định; McNemar chính xác được hiệu chỉnh Holm trên tám điều kiện suy giảm.

![Hình 3. Accuracy của ChiliEdgeNet có và không có Tiny Transformer theo mức nhòe Gaussian và hệ số giảm sáng tổng hợp.](../../results/stress/c14_tf_stress_curves.png)

**Bảng 7. Stress test Tiny Transformer trên các mức suy giảm tổng hợp.** Delta dương nghĩa mô hình có Tiny Transformer tốt hơn.

| Điều kiện | Có Transformer (%) | Bỏ Transformer (%) | Delta (pp) | b / c | \(p_{Holm}\) |
|---|---:|---:|---:|---:|---:|
| Ảnh sạch | 98,93 | 99,20 | −0,27 | 0 / 1 | – |
| Nhoè, kernel 3 | 97,86 | 98,12 | −0,27 | 2 / 3 | 1,0000 |
| Nhoè, kernel 5 | 97,32 | 96,25 | +1,07 | 6 / 2 | 1,0000 |
| Nhoè, kernel 7 | 94,91 | 92,76 | +2,14 | 15 / 7 | 0,8028 |
| Nhoè, kernel 9 | 92,23 | 89,28 | +2,95 | 22 / 11 | 0,5610 |
| Sáng × 0,8 | 98,66 | 99,20 | −0,54 | 0 / 2 | 1,0000 |
| Sáng × 0,6 | 98,66 | 98,93 | −0,27 | 1 / 2 | 1,0000 |
| Sáng × 0,4 | 98,66 | 98,12 | +0,54 | 3 / 1 | 1,0000 |
| Sáng × 0,2 | **97,05** | 93,30 | **+3,75** | 16 / 2 | **0,0105** |

Trên ảnh sạch và các mức suy giảm nhẹ, Tiny Transformer chưa tạo ra khác biệt rõ rệt. Từ kernel 5 trở đi, lợi thế dưới nhoè tăng theo mức suy giảm nhưng từng tương phản chưa đạt ý nghĩa sau hiệu chỉnh Holm. Ở hệ số sáng 0,2, mô hình đầy đủ đạt 97,05%, cao hơn 3,75 pp và giảm số lỗi từ 25 xuống 11; đây là tương phản duy nhất còn ý nghĩa sau hiệu chỉnh. Kết quả cho thấy Tiny Transformer hỗ trợ duy trì biểu diễn khi thông tin thị giác bị suy giảm nghiêm trọng, thay vì tạo ưu thế ổn định trên ảnh sạch. Lợi ích này đi kèm 115.328 tham số bổ sung so với biến thể bỏ Transformer.

#### 4.3.3. Ablation ở cấp kỹ thuật

Mỗi biến thể thay đúng một kỹ thuật trong Tiny Transformer bằng phương án tiêu chuẩn và giữ nguyên các thành phần còn lại.

**Bảng 8. Ablation từng kỹ thuật trong Tiny Transformer trên ảnh sạch, seed 42.**

| Biến thể | Thành phần thay thế | Tham số | Delta tham số | Accuracy (%) |
|---|---|---:|---:|---:|
| **ChiliEdgeNet** | 2D-RoPE + GQA + RMSNorm | **671.173** | – | **98,93** |
| `learned_pe` | 2D-RoPE → learned PE | 696.261 | +25.088 | 98,93 |
| `mha` | GQA → MHA | 687.557 | +16.384 | 98,93 |
| `layernorm` | RMSNorm → LayerNorm | 671.557 | +384 | 98,39 |

Trên ảnh sạch, learned PE và MHA cho cùng dự đoán đúng/sai với ChiliEdgeNet; LayerNorm thấp hơn hai ảnh nhưng McNemar không có ý nghĩa (\(p=0,5\)). Ba lựa chọn 2D-RoPE, GQA và RMSNorm không được dữ liệu một seed chứng minh vượt trội về accuracy. Xét cấu trúc tham số, chúng lần lượt tránh 25.088, 16.384 và 384 tham số so với các phương án tiêu chuẩn, tương ứng tổng cộng **41.856 tham số** nếu đồng thời dùng cả ba phương án thay thế; biến thể kết hợp ba thay thế chưa được huấn luyện riêng.
