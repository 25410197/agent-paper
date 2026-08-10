## 3. Phương pháp đề xuất

### 3.1. Bộ dữ liệu

**ChiliEdge-DS** gồm 2.461 ảnh lá và quả ớt thuộc năm lớp: `anthracnose` (thán thư), `healthy` (khỏe), `leaf_curl` (xoăn lá), `leaf_spot` (đốm lá) và `yellow_leaf` (vàng lá). Dữ liệu do tác giả trực tiếp thu nhận tại khu vực Củ Chi, Thành phố Hồ Chí Minh, Việt Nam. Dữ liệu được chia một lần theo tỷ lệ xấp xỉ 65:20:15 thành `train`, `val` và `test`, với số lượng lần lượt là 1.597, 491 và 373 ảnh. Lần chia cố định này được giữ nguyên trong toàn bộ thí nghiệm chính.

Toàn bộ ảnh gốc trong ChiliEdge-DS do tác giả trực tiếp chụp và tác giả sở hữu quyền đối với các ảnh này. Ảnh được chụp trên phông nền trắng bằng camera điện thoại thông minh; thông tin model thiết bị không được lưu lại. Ảnh có độ phân giải gốc xấp xỉ 12.000 × 9.000 pixel, sau đó được lưu ở kích thước 300 × 300 pixel và resize về 224 × 224 khi đưa vào mô hình. Bộ dữ liệu được công bố công khai trên Kaggle (Hải, 2026c).

Các nhãn được tác giả gán thủ công theo biểu hiện thị giác: `anthracnose` có vùng tổn thương sẫm màu hoặc hoại tử phù hợp với biểu hiện thán thư; `healthy` không có triệu chứng quan sát rõ; `leaf_curl` có hiện tượng cong hoặc xoăn lá; `leaf_spot` có các đốm tổn thương rời rạc trên bề mặt lá; và `yellow_leaf` có biểu hiện vàng lá chiếm ưu thế. Đây là các nhãn kiểu hình dựa trên quan sát, không phải kết quả định danh tác nhân gây bệnh bằng xét nghiệm.

ChiliEdge-DS được công bố công khai trên Kaggle (Hải, 2026c) nhưng không gắn license mở (mục license tại nguồn giữ ở trạng thái "Unknown"). Đây là lựa chọn có chủ đích của tác giả nhằm giữ toàn quyền kiểm soát việc tái sử dụng ảnh gốc do chính tác giả thu thập, không phải thiếu sót biên tập.

Tỉ lệ giữa lớp lớn nhất và nhỏ nhất là 1,29, do đó nghiên cứu không áp dụng class weighting. Macro-F1 được báo cáo cùng accuracy để mỗi lớp đóng góp như nhau vào kết quả tổng hợp. Phông nền trắng là điều kiện thu thập có kiểm soát; khoảng cách với ảnh khác nguồn và nền cảnh được khảo sát tại Mục 4.5 và thảo luận tại Mục 5.

**Bảng 1. Phân bố theo lớp của ChiliEdge-DS trên lần chia cố định.**

| Lớp         |     Train |     Val |    Test |      Tổng |
| ----------- | --------: | ------: | ------: | --------: |
| anthracnose |       264 |      81 |      62 |       407 |
| healthy     |       334 |     102 |      78 |       514 |
| leaf_curl   |       328 |     104 |      82 |       514 |
| leaf_spot   |       326 |     101 |      76 |       503 |
| yellow_leaf |       345 |     103 |      75 |       523 |
| **Tổng**    | **1.597** | **491** | **373** | **2.461** |

### 3.2. Kiến trúc ChiliEdgeNet

CNN khai thác tốt đặc trưng cục bộ nhưng còn hạn chế trong việc mô hình hóa trực tiếp quan hệ giữa các vùng cách xa nhau. Vì vậy, ChiliEdgeNet kết hợp truncated MobileNetV2 với một Tiny Transformer block.

#### 3.2.1. Điểm cắt backbone

ChiliEdgeNet sử dụng MobileNetV2 pretrained trên ImageNet và giữ các tầng từ `features[0]` đến `features[13]`. Với ảnh đầu vào 224 × 224, backbone tạo ra feature map kích thước **96 × 14 × 14**, tương ứng với 196 vị trí không gian.

Backbone được cắt trước `features[14]`, vì tầng này làm giảm feature map xuống 7 × 7 và số token từ 196 xuống còn 49. Điểm cắt `cut = 14` được lựa chọn qua phép quét sơ bộ và đồng thời cung cấp bản đồ Attention Rollout có số vị trí không gian nhiều gấp bốn lần lưới 7 × 7.

#### 3.2.2. Luồng xử lý đặc trưng

Ảnh được xử lý theo luồng:

1. MobileNetV2 `features[0:14]`;
2. Conv 1 × 1 ánh xạ 96 lên 128 kênh;
3. BatchNorm, GELU và dropout 0,1;
4. Trải feature map thành 196 image token, mỗi token có 128 chiều;
5. Tiny Transformer block;
6. RMSNorm và mean pooling theo chiều token;
7. Dropout 0,2 và linear layer dự đoán năm lớp.

#### 3.2.3. Tiny Transformer block

ChiliEdgeNet là mô hình thuần thị giác, không sử dụng text encoder, tokenizer hoặc trọng số ngôn ngữ tiền huấn luyện. Tiny Transformer sử dụng cấu trúc pre-norm gồm:

* `RMSNorm → GQA kết hợp 2D-RoPE → residual connection`;
* `RMSNorm → MLP với GELU → residual connection`.

Mô hình sử dụng embedding dimension bằng 128, bốn query head, hai key/value head, MLP ratio bằng 2 và dropout bằng 0,1.

Ba kỹ thuật tối ưu hóa tham số gồm:

* **2D-RoPE:** sử dụng biến thể axial cho lưới ảnh, thay thế learned positional embedding và giảm **25.088 tham số** (Heo et al., 2025).
* **GQA:** cho bốn query head chia sẻ hai key/value head, giảm **16.384 tham số** (Ainslie et al., 2023).
* **RMSNorm:** thay thế LayerNorm và giảm tổng cộng **384 tham số** (Zhang and Sennrich, 2019).

Ba kỹ thuật giúp giảm tổng cộng **41.856 tham số**, tương đương khoảng **6,2%** số tham số của ChiliEdgeNet. Mô hình có **671.173 tham số**, ít hơn khoảng **35,0%** so với kiến trúc nền tảng (Duhan et al., 2025) có 1.032.568 tham số.

#### 3.2.4. Các khối được loại bỏ

Kiến trúc nền tảng (Duhan et al., 2025) sử dụng thêm Triplet Attention (Misra et al., 2021) và RES_Block kết hợp ECA với SE. Hai khối này không xuất hiện trong ChiliEdgeNet.

Sự cần thiết của Triplet Attention và `RES_Block` được khảo sát bằng thiết kế thêm-lại 2×2 gồm bốn cấu hình: có cả hai khối, chỉ Triplet Attention, chỉ `RES_Block`, và không có cả hai. Mỗi cấu hình được huấn luyện lại từ đầu với seed 42 theo cùng giao thức; kết quả và các tương phản loại bỏ được trình bày tại Mục 4.3. Trong lần chạy seed 42 này, việc thêm hai khối không cải thiện accuracy trong các tương phản được đánh giá, nên mô hình cuối không sử dụng chúng; kết quả không được diễn giải thành bằng chứng về tính ổn định giữa nhiều lần chạy hoặc về sự tương đương của hai kiến trúc.

![Hình 1. Sơ đồ kiến trúc ChiliEdgeNet. Panel A: luồng xử lý tổng thể từ ảnh 224×224×3 qua backbone MobileNetV2 cắt ngắn (features[0:14]), Conv 1×1 giảm kênh, 196 token, khối Tiny Transformer, mean pooling và lớp phân loại, kèm kích thước tensor tại mỗi bước. Panel B: chi tiết một khối Tiny Transformer, gồm RMSNorm, Grouped Query Attention kết hợp 2D-RoPE dạng axial (4 query-head, 2 key/value-head) với residual, và MLP GELU (tỉ lệ mở rộng 2.0) với residual.](../../results/architecture/chiliedgenet_architecture.png)

### 3.3. Độ đo đánh giá

Nghiên cứu báo cáo top-1 accuracy, macro-F1 và tổng số tham số. Mọi giá trị accuracy trên tập test đều đi kèm khoảng tin cậy Wilson 95% với (N = 373).

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i = y_i)$$

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C}F1_c,\qquad C = 5$$
