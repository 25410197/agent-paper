# ChiliEdgeNet: Mô hình lai CNN-Transformer nhẹ hướng tới nhận diện bệnh cây ớt trên thiết bị biên

**Từ khóa:** chili pepper disease classification, lightweight deep learning, CNN–Transformer, edge AI, few-shot domain adaptation, explainable AI.

## Tóm tắt

Chẩn đoán bệnh cây trồng hướng tới các thiết bị hạn chế tài nguyên cần mô hình chính xác và gọn nhẹ. Nghiên cứu đề xuất **ChiliEdgeNet**, kiến trúc lai CNN–Transformer được xây dựng bằng cách cắt gọn MobileNetV2 và tích hợp một khối Tiny Transformer. Nghiên cứu cũng giới thiệu bộ dữ liệu công khai **ChiliEdge-DS** gồm 2.461 ảnh lá và quả ớt thuộc năm lớp. ChiliEdgeNet đạt độ chính xác **98,93%** với **671.173 tham số**, giảm khoảng **35%** so với kiến trúc nền tảng. Mô hình có kích thước 2,7 MB và độ trễ suy luận 20,24 ms/ảnh trên CPU máy chủ. Trong phép thử suy giảm độ sáng tổng hợp ở hệ số 0,2, mô hình đầy đủ cao hơn biến thể không có Transformer 3,75 điểm phần trăm. Trên query cross-dataset sạch gồm 1.463 ảnh thuộc bốn lớp chung với Mendeley Data, accuracy tăng từ 30,96% ở checkpoint gốc lên 89,75% sau khi bổ sung 42 ảnh support; cả hai checkpoint dùng seed 42. Kiểm định McNemar ghép cặp ghi nhận 870 trường hợp chuyển từ sai thành đúng và 10 trường hợp theo chiều ngược lại ($p=1{,}83\times10^{-242}$). Kết quả cho thấy ChiliEdgeNet là ứng viên cho suy luận ngoại tuyến trên phần cứng hạn chế tài nguyên, nhưng cần đánh giá trực tiếp trên phần cứng biên và xác nhận chuyển miền trên nhiều bộ dữ liệu độc lập hơn.

## 1. Đặt vấn đề

Nhận diện sớm bệnh cây trồng từ ảnh có thể hỗ trợ theo dõi và xử lý bệnh. Tuy nhiên, triển khai tại nơi thu nhận ảnh đòi hỏi mô hình có chi phí tính toán thấp, hoạt động ngoại tuyến và độ chính xác đủ cao (Joseph et al., 2024; Maurya et al., 2024; Duhan et al., 2025). Đây là các yêu cầu quan trọng đối với thiết bị có tài nguyên hạn chế.

CNN nhẹ có lợi thế về số tham số và thiên kiến không gian cục bộ; mạng đủ sâu vẫn có thể đạt receptive field lớn nhưng không mô hình hóa trực tiếp tương tác cặp giữa mọi vị trí như self-attention. Vision Transformer hỗ trợ khai thác quan hệ toàn cục, song cấu hình tiêu chuẩn thường có chi phí tính toán cao hơn (Gole et al., 2023; Li et al., 2023). Kiến trúc lai CNN–Transformer vì vậy có thể dung hòa hai đặc tính này.

Trong phạm vi các công trình khảo sát ở Mục 2, kết quả được báo cáo trên những bộ dữ liệu và giao thức khác nhau, nên chưa tạo thành một đối chiếu chung đồng thời phân tích đóng góp của từng thành phần, phản ứng trước suy giảm ảnh tổng hợp và chênh lệch miền nền cảnh. Khoảng trống được xem xét ở đây không phải sự thiếu vắng tuyệt đối của từng phép phân tích, mà là thiếu một giao thức kết hợp các hướng đánh giá này trên cùng lần chia dữ liệu và quy trình huấn luyện. Đặc biệt, độ chính xác cao trên dữ liệu nền có kiểm soát chưa bảo đảm hiệu năng trên ảnh nền tự nhiên. XAI được dùng để khảo sát hành vi dự đoán, còn mô hình thị giác tổng quát cung cấp một đối chiếu tham khảo.

Từ kiến trúc RTR_Lite_MobileNet (Duhan et al., 2025) dựa trên MobileNetV2 (Sandler et al., 2018), nghiên cứu đề xuất **ChiliEdgeNet**, một mô hình lai CNN–Transformer dưới 700 nghìn tham số. Nghiên cứu tập trung vào bốn nội dung chính sau:

- **Kiến trúc gọn nhẹ:** ChiliEdgeNet đạt accuracy **98,93%** với **671.173 tham số**, giảm khoảng **35%** số tham số so với kiến trúc nền tảng trong cùng giao thức seed 42. Mô hình có kích thước 2,7 MB và thời gian suy luận 20,24 ms/ảnh trên CPU máy ảo Kaggle.
- **Bộ dữ liệu ChiliEdge-DS:** gồm 2.461 ảnh lá và quả ớt thuộc năm lớp, được thu thập tại miền Nam Việt Nam trên nền trắng có kiểm soát, chia cố định thành train/validation/test và công bố trên Kaggle (Hải, 2026c).
- **Đánh giá có kiểm soát và chuyển miền:** nghiên cứu thực hiện benchmark ảnh suy giảm và ablation huấn luyện lại từ đầu với seed 42. Trên query sạch 1.463 ảnh thuộc bốn lớp chung với Mendeley Data, checkpoint gốc đạt 30,96% và checkpoint sau khi bổ sung 42 ảnh support đạt 89,75%. Cả hai checkpoint dùng seed 42, cùng kiến trúc và hyperparameter nền; giao SHA-256 giữa support và query bằng 0. So sánh ghép cặp ghi nhận 870 ảnh chuyển từ sai thành đúng và 10 ảnh theo chiều ngược lại. Checkpoint thích nghi chỉ phục vụ phân tích chuyển miền và không thay thế mô hình chính.
- **Diễn giải và đối chiếu:** ba phương pháp XAI được dùng để phân tích định tính. Một LLM thị giác tổng quát được đánh giá zero-shot trên tập con 370 ảnh có bản ghi API hợp lệ như đối chiếu tham khảo, không phải baseline ngang hàng; ba ảnh test còn lại chưa có bản ghi API và được ghi nhận là GAP cần chạy bổ sung.
- **Tích hợp camera laptop:** trong 62 lần dự đoán với batch size 1 trên CPU Intel Core i7-13650HX, trung vị độ trễ tiền xử lý–suy luận là 29,70 ms. Phép đo không bao gồm thu nhận khung hình, cắt ROI, chuyển đổi định dạng trước ROI, vẽ giao diện hoặc ghi tệp; vì vậy con số này không phải FPS camera hoặc độ trễ toàn pipeline.

## 2. Cơ sở lý thuyết và nghiên cứu liên quan

### 2.1. CNN nhẹ và MobileNetV2

Với feature map đầu vào có kích thước (H\times W\times C_{in}), kernel (k\times k) và (C_{out}) kênh đầu ra, tích chập tiêu chuẩn cần xấp xỉ

\[
HWk^2C_{in}C_{out}
\]

phép nhân–cộng. Tích chập tách kênh (*depthwise separable convolution*) phân rã phép tính thành depthwise convolution trên từng kênh và pointwise convolution (1\times1) để trộn thông tin giữa các kênh. Chi phí khi đó xấp xỉ

\[
HW\left(k^2C_{in}+C_{in}C_{out}\right),
\]

cho tỉ lệ so với tích chập tiêu chuẩn bằng

\[
\frac{1}{C_{out}}+\frac{1}{k^2}.
\]

Đây là cơ sở giúp họ MobileNet giảm số phép tính trên thiết bị hạn chế tài nguyên. MobileNetV2 tổ chức các phép tích chập này trong khối *inverted residual with linear bottleneck* (Sandler et al., 2018). Mỗi khối trước hết dùng pointwise convolution để mở rộng số kênh, sau đó dùng depthwise convolution để trích xuất đặc trưng không gian và cuối cùng chiếu tuyến tính về không gian chiều thấp. Khi stride bằng 1 và kích thước đầu vào–đầu ra tương thích, shortcut nối hai biểu diễn bottleneck mỏng. Phép chiếu cuối không dùng hàm kích hoạt nhằm hạn chế mất thông tin trong không gian chiều thấp. ChiliEdgeNet kế thừa thiên kiến cục bộ và hiệu quả tính toán này từ phần MobileNetV2 được giữ lại.

### 2.2. Token ảnh và self-attention

Giả sử backbone CNN tạo feature map (F\in\mathbb{R}^{C\times H\times W}). Sau phép chiếu kênh, feature map được trải thành chuỗi

\[
X\in\mathbb{R}^{N\times d},\qquad N=HW,
\]

trong đó mỗi hàng là một token đại diện cho một vị trí không gian. Từ (X), các ma trận query, key và value được tạo bởi

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

Scaled dot-product attention (Vaswani et al., 2017) được xác định bởi

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)V,
\]

với (d_h) là số chiều của mỗi attention head. Khác với tích chập chỉ tổng hợp trực tiếp một lân cận cục bộ ở mỗi tầng, self-attention cho phép mỗi token tương tác trực tiếp với mọi token khác. Đổi lại, ma trận attention có độ phức tạp theo số token là (O(N^2)). ChiliEdgeNet sử dụng lưới (14\times14), tức (N=196), để giữ lại nhiều vị trí không gian hơn lưới (7\times7) nhưng vẫn giới hạn chi phí của một Transformer block nhỏ.

Khối Transformer của nghiên cứu dùng cấu trúc pre-norm và hai nhánh residual:

\[
X' = X + \operatorname{GQA}(\operatorname{RMSNorm}(X)),
\]

\[
Y = X' + \operatorname{MLP}(\operatorname{RMSNorm}(X')).
\]

MLP gồm hai lớp tuyến tính, GELU và dropout. Residual connection tạo đường truyền trực tiếp cho biểu diễn và gradient; pre-norm chuẩn hóa đầu vào của từng phép biến đổi thay vì chuẩn hóa sau phép cộng.

### 2.3. Grouped-query attention

Multi-head attention thông thường tạo key và value riêng cho mọi query head. Grouped-query attention (GQA) đặt số query head (h_q) lớn hơn số key/value head (h_{kv}), sau đó cho một nhóm (h_q/h_{kv}) query head dùng chung một cặp key/value (Ainslie et al., 2023). GQA nằm giữa multi-head attention, trong đó (h_q=h_{kv}), và multi-query attention, trong đó mọi query head dùng chung một key/value head.

Nếu embedding có chiều (d) và các phép chiếu không dùng bias, phần query và output của GQA vẫn cần (2d^2) tham số, trong khi key và value cần

\[
2d^2\frac{h_{kv}}{h_q}.
\]

Tổng số tham số của bốn phép chiếu attention do đó là

\[
P_{GQA}=2d^2+2d^2\frac{h_{kv}}{h_q}.
\]

Với (d=128), (h_q=4) và (h_{kv}=2), mỗi key/value head phục vụ hai query head và mô-đun tránh (16.384) tham số so với multi-head attention bốn head. Trong hiện thực của nghiên cứu, key và value được chiếu với hai head rồi lặp theo nhóm trước khi tính attention. Vì vậy, lợi ích được khẳng định ở đây là giảm tham số của phép chiếu; nghiên cứu không suy rộng thành kết luận về tăng tốc độ suy luận.

### 2.4. Mã hóa vị trí 2D-RoPE và RMSNorm

Self-attention không tự mang thông tin về thứ tự hoặc tọa độ token. Rotary position embedding (RoPE) mã hóa vị trí bằng cách quay từng cặp thành phần của query và key theo góc phụ thuộc vị trí (Su et al., 2024). Với một cặp hai chiều (z=(z_1,z_2)), phép quay tại vị trí (p) có dạng

\[
R(p\theta)z=
\begin{bmatrix}
\cos(p\theta) & -\sin(p\theta)\\
\sin(p\theta) & \cos(p\theta)
\end{bmatrix}z.
\]

RoPE gốc được trình bày cho chuỗi một chiều (Su et al., 2024). Đối với lưới ảnh, nghiên cứu sử dụng **2D-RoPE dạng axial**: mỗi query/key head được chia thành hai nửa; một nửa được quay theo tọa độ hàng (y), nửa còn lại theo tọa độ cột (x), phù hợp với cách mở rộng RoPE-Axial cho Vision Transformer của Heo et al. (2025). Cách này đưa tọa độ 2D vào query và key mà không học một bảng positional embedding. Với 196 token và embedding dimension 128, bảng vị trí học được tương ứng cần (196\times128=25.088) tham số.

RMSNorm chuẩn hóa theo căn trung bình bình phương thay vì trừ trung bình như LayerNorm (Zhang and Sennrich, 2019). Với token (x\in\mathbb{R}^{d}), phép chuẩn hóa được dùng trong mô hình là

\[
\operatorname{RMSNorm}(x)_i
=g_i\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon}},
\]

trong đó (g_i) là hệ số scale học được và (arepsilon=10^{-6}). Hiện thực không có bias dịch chuyển. ChiliEdgeNet có ba vị trí chuẩn hóa chiều 128, vì vậy RMSNorm tránh (3\times128=384) tham số bias so với ba LayerNorm tương ứng. Đây là khác biệt cấu trúc chính xác; hiệu quả accuracy của RMSNorm vẫn phải được xác nhận bằng ablation thay vì suy ra trực tiếp từ công thức.

### 2.5. Học chuyển giao và thích nghi miền ít mẫu

Học chuyển giao tận dụng tri thức từ miền hoặc nhiệm vụ nguồn để hỗ trợ miền hoặc nhiệm vụ đích (Pan and Yang, 2010). Trong nghiên cứu này có hai mức chuyển giao. Thứ nhất, backbone MobileNetV2 sử dụng trọng số tiền huấn luyện ImageNet trước khi được tinh chỉnh trên ChiliEdge-DS. Thứ hai, phép cross-dataset xem ChiliEdge-DS là miền nguồn và bộ ảnh Mendeley Data là miền đích có khác biệt về điều kiện thu nhận, nền cảnh và phân bố biểu hiện.

Gọi tập huấn luyện nguồn là (\mathcal{D}_s), tập support có nhãn của miền đích là (S_t), và tập query miền đích là (Q_t). Quy trình thích nghi hiện tại tạo

\[
\mathcal{D}_{adapt}=\mathcal{D}_s\cup S_t
\]

và tối ưu cross-entropy trung bình trên tập hợp này:

\[
\mathcal{L}_{CE}
=-\frac{1}{|\mathcal{D}_{adapt}|}
\sum_{(x,y)\in\mathcal{D}_{adapt}}
\log p_{\theta}(y\mid x).
\]

Sau thích nghi, (Q_t) chỉ được dùng để đánh giá. Điều kiện độc lập tối thiểu là

\[
S_t\cap Q_t=\varnothing,
\]

được kiểm tra trong nghiên cứu bằng SHA-256 của tệp ảnh. Nếu một ảnh query xuất hiện trong support, kết quả không còn phản ánh đúng khả năng tổng quát hóa tới ảnh đích chưa thấy.

Trong few-shot learning theo episode, một episode thường gồm bài toán (N)-way (K)-shot với số support được kiểm soát cho mỗi lớp; Prototypical Networks là một ví dụ học metric theo thiết lập này (Snell et al., 2017). Thí nghiệm của nghiên cứu không huấn luyện theo episode, không học prototype và có số support không đều giữa bốn lớp. Vì vậy, thuật ngữ chính xác được dùng trong bài là **thích nghi miền có giám sát bằng tập support ít mẫu**, không phải meta-learning hoặc (N)-way (K)-shot cân bằng.

### 2.6. Tiêu chí lựa chọn baseline

**Tiêu chí lựa chọn nhóm baseline.** Bảng 3 gồm 17 baseline từ 15 nghiên cứu, được chọn theo: (i) công trình bình duyệt công bố 2023–2026 về phân loại bệnh cây trồng dựa trên ảnh; (ii) có trọng tâm kiến trúc rõ ràng về khả năng triển khai (CNN nhẹ / ViT / lai) hoặc là nghiên cứu so sánh diện rộng CNN-ViT; (iii) mô tả kiến trúc đủ chi tiết (cấu hình lớp, số chiều nhúng, số đầu chú ý) để phục dựng hoặc chuyển thể cho bài toán đích mà không cần liên hệ tác giả; (iv) phủ đủ bốn hướng dưới đây thay vì chỉ chọn họ mạnh nhất. Các nghiên cứu bị loại khi hướng tới phát hiện/phân vùng thay vì phân loại, hoặc khi không thể xây dựng một đầu ra phân loại tương ứng. Nhóm baseline vì vậy đại diện cho dòng nghiên cứu **hướng triển khai** trong ba năm gần đây, không phải khảo sát toàn diện; các mô hình được nhóm tự triển khai và đánh giá trong cùng một khung được mô tả tại Mục 4.1, nên Bảng 3 không sử dụng số liệu do tác giả gốc công bố.

Các công trình chỉ thực hiện phát hiện hoặc phân vùng mà không có đầu ra phân loại mức ảnh được loại khỏi nhóm baseline. Mô hình đa nhiệm vẫn được giữ nếu kiến trúc gốc có nhánh phân loại rõ ràng và có thể chuyển thể sang bài toán năm lớp. Theo tiêu chí này, PDLC-ViT (Hemalatha and Jayachandran, 2024) được giữ vì có classification head bên cạnh nhánh segmentation. Do ChiliEdge-DS không cung cấp mask, thí nghiệm chỉ triển khai nhánh phân loại và loại bỏ segmentation head cùng segmentation loss; phiên bản được so sánh vì vậy được ký hiệu **PDLC-ViT (classification-only)**, không phải mô hình đa nhiệm nguyên trạng.

### 2.7. Các hướng kiến trúc liên quan

**CNN nhẹ** giữ tham số thấp nhờ chú ý hiệu quả hoặc khối gọn. RTR_Lite_MobileNetV2 (Duhan et al., 2025) — kiến trúc nền tảng của chúng tôi — gắn triplet attention (Misra et al., 2021) và một khối residual có cổng chú ý không gian vào MobileNetV2 (Sandler et al., 2018). ULEN (Wang et al., 2023) đẩy thiết kế siêu nhẹ xuống 28 nghìn tham số; MRW-CNN (Joseph et al., 2024) và Lite-MDC (Bhagat et al., 2024) hướng tới triển khai thời gian thực ngoài đồng; CustomMNetV3L (Rahaman et al., 2026) tùy biến MobileNetV3-Large. CNN sâu có thể mở rộng receptive field để tổng hợp ngữ cảnh rộng, nhưng không mô hình hóa trực tiếp tương tác cặp giữa mọi vị trí như self-attention. **Vision Transformer** khai thác trực tiếp quan hệ giữa các token nhưng thường có chi phí cao hơn: ViT (Ali et al., 2025), TrIncNet (Gole et al., 2023) với khối Trans-Inception, PMVT (Li et al., 2023) hướng di động và PDLC-ViT (Hemalatha and Jayachandran, 2024) đa nhiệm định vị kèm phân loại. **Kiến trúc lai CNN–Transformer** gần với hướng nghiên cứu này gồm GrapeLeafNet (Karthik et al., 2024), MaizeFormerX (Rahman et al., 2026) và LDI-NET (Yang et al., 2024), với số tham số thường ở mức 3–9 triệu. **Nghiên cứu so sánh diện rộng** gồm Mishra et al. (2026), Sujatha et al. (2025), cùng meta-ensemble nhẹ cho IoT của Maurya et al. (2024).

### 2.8. Khoảng trống đánh giá

Các công trình trên sử dụng bộ dữ liệu, cách chia và giao thức huấn luyện khác nhau; vì vậy, số liệu do từng tác giả công bố không tạo thành một phép so sánh chung có thể đồng thời đánh giá đóng góp kiến trúc, phản ứng trước suy giảm ảnh tổng hợp và chênh lệch miền nền cảnh. Nhận định này không có nghĩa từng công trình đều thiếu toàn bộ các phép phân tích, mà cho thấy nhu cầu về một giao thức kết hợp các trục đánh giá trên cùng dữ liệu và quy trình huấn luyện. Theo hướng đó, ChiliEdgeNet giữ quy mô dưới 700 nghìn tham số bằng cách cắt MobileNetV2 và sử dụng ba kỹ thuật Transformer hiệu quả tham số gồm 2D-RoPE, GQA và RMSNorm. Phần thực nghiệm huấn luyện lại các biến thể ablation với seed 42, kiểm tra ảnh suy giảm tổng hợp và sử dụng McNemar ghép cặp cho các so sánh ablation có kiểm soát, đồng thời khảo sát chuyển miền, XAI và đối chiếu tham khảo với mô hình thị giác tổng quát.

## 3. Phương pháp đề xuất

### 3.1. Bộ dữ liệu

**ChiliEdge-DS** gồm 2.461 ảnh lá và quả ớt thuộc năm lớp: `anthracnose` (thán thư), `healthy` (khỏe), `leaf_curl` (xoăn lá), `leaf_spot` (đốm lá) và `yellow_leaf` (vàng lá). Dữ liệu do tác giả trực tiếp thu nhận tại khu vực Củ Chi, Thành phố Hồ Chí Minh, Việt Nam. Dữ liệu được chia một lần theo tỷ lệ xấp xỉ 65:20:15 thành `train`, `val` và `test`, với số lượng lần lượt là 1.597, 491 và 373 ảnh. Lần chia cố định này được giữ nguyên trong toàn bộ thí nghiệm chính.

Toàn bộ ảnh gốc trong ChiliEdge-DS do tác giả trực tiếp chụp và tác giả sở hữu quyền đối với các ảnh này. Ảnh được chụp trên phông nền trắng bằng camera điện thoại thông minh; thông tin model thiết bị không được lưu lại. Ảnh có độ phân giải gốc xấp xỉ 12.000 × 9.000 pixel, sau đó được lưu ở kích thước 300 × 300 pixel và resize về 224 × 224 khi đưa vào mô hình. Bộ dữ liệu được công bố công khai trên Kaggle (Hải, 2026c).

Các nhãn được tác giả gán thủ công theo biểu hiện thị giác: `anthracnose` có vùng tổn thương sẫm màu hoặc hoại tử phù hợp với biểu hiện thán thư; `healthy` không có triệu chứng quan sát rõ; `leaf_curl` có hiện tượng cong hoặc xoăn lá; `leaf_spot` có các đốm tổn thương rời rạc trên bề mặt lá; và `yellow_leaf` có biểu hiện vàng lá chiếm ưu thế. Đây là các nhãn kiểu hình dựa trên quan sát, không phải kết quả định danh tác nhân gây bệnh bằng xét nghiệm.

> **Ghi chú biên tập:** cần chỉ định và ghi rõ giấy phép phát hành của ChiliEdge-DS tại nguồn (Hải, 2026c) trước khi nộp bài.

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

### 3.3. Độ đo đánh giá

Nghiên cứu báo cáo top-1 accuracy, macro-F1 và tổng số tham số. Mọi giá trị accuracy trên tập test đều đi kèm khoảng tin cậy Wilson 95% với (N = 373).

[
\text{Accuracy}
===============

\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}(\hat{y}_i = y_i)
]

[
F1_c
====

2 \cdot
\frac{\text{Precision}_c \cdot \text{Recall}_c}
{\text{Precision}_c + \text{Recall}_c}
]

[
\text{Macro-F1}
===============

\frac{1}{C}
\sum_{c=1}^{C}F1_c,
\qquad C = 5
]

## 4. Thực nghiệm và kết quả

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

### 4.2. So sánh với các mô hình tiên tiến

Với seed 42, ChiliEdgeNet đạt **97,86%** trong kịch bản baseline và **98,93%** trong kịch bản augmented. Macro-F1 tương ứng là 0,9790 và 0,9897; số ảnh dự đoán đúng lần lượt là 365 và 369 trên 373 ảnh sạch.

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

Trong kịch bản augmented, ChiliEdgeNet dự đoán sai **4 trên 373 ảnh**:

- một ảnh `leaf_spot` bị dự đoán thành `healthy`;
- ba ảnh `yellow_leaf` bị dự đoán thành `leaf_curl`.

Accuracy đạt 98,93% với khoảng tin cậy Wilson 95% **[97,28%; 99,58%]**. Các lỗi còn lại tập trung tại ranh giới giữa triệu chứng vàng lá và xoăn lá.

![Hình 1. Ma trận nhầm lẫn của ChiliEdgeNet trên tập test trong kịch bản augmented.](../../results/notebook_c1/c14_cm_s2.png)

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

![Hình 2. Accuracy của ChiliEdgeNet có và không có Tiny Transformer theo mức nhòe Gaussian và hệ số giảm sáng tổng hợp.](../../results/stress/c14_tf_stress_curves.png)

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

### 4.4. Diễn giải mô hình

Các phân tích trong mục này sử dụng **checkpoint gốc của ChiliEdgeNet**, được huấn luyện chỉ trên ChiliEdge-DS và dùng để tạo kết quả trong miền ở Bảng 3. Checkpoint bổ sung dữ liệu cho thí nghiệm chuyển miền tại mục 4.5 là một mô hình thích nghi riêng và không được sử dụng để tạo các kết quả diễn giải dưới đây.

#### 4.4.1. AI giải thích được (XAI)

Độ chính xác cao chưa đủ để diễn giải hành vi của mô hình, đặc biệt khi ChiliEdge-DS chủ yếu sử dụng nền trắng có kiểm soát. Nghiên cứu triển khai **ba phương pháp XAI bổ trợ** trên checkpoint ChiliEdgeNet augmented seed 42, mỗi phương pháp sử dụng một ảnh đại diện cho mỗi lớp. **Gradient SHAP** (Erion et al., 2021) ước lượng đóng góp của pixel từ gradient đối với một phân phối baseline; **Occlusion Sensitivity** (Zeiler and Fergus, 2014) che lần lượt các vùng 32×32 pixel và đo thay đổi đầu ra; **Attention Rollout** (Abnar and Zuidema, 2020) tổng hợp dòng chú ý qua khối Transformer trên toàn bộ lưới 14×14. Do kiến trúc sử dụng mean pooling và không có token `CLS`, bản đồ Rollout được tạo bằng cách lấy trung bình dòng chú ý qua 196 token, không truy vết từ một token đầu ra riêng.

Ba bản đồ cung cấp các góc nhìn không giống nhau (Hình 3–5): Gradient SHAP cho quy thuộc mức pixel phân bố trên bề mặt đối tượng; Occlusion biểu diễn độ nhạy theo vùng; Attention Rollout phản ánh sự lan truyền chú ý nội tại của Transformer. Trên năm ảnh minh họa, các vùng có đáp ứng thường giao với lá/quả hoặc vùng biểu hiện bệnh, nhưng mức tập trung khác nhau giữa phương pháp và lớp. Vì đánh giá chỉ dựa trên một ảnh mỗi lớp và chưa có ground truth phân vùng triệu chứng, kết quả này chỉ là khảo sát định tính thăm dò; nó không chứng minh mô hình đã học đúng đặc trưng bệnh lý, hoàn toàn bỏ qua nền hoặc sử dụng quan hệ nhân quả đúng. Lưới 14×14 cung cấp 196 vị trí không gian, nhiều gấp bốn lần lưới 7×7, giúp bản đồ Rollout có độ phân giải trực quan chi tiết hơn.

<!-- IMG2 -->
![Hình 3. Occlusion Sensitivity cho năm lớp bệnh (trên: ảnh gốc; dưới: bản đồ nhạy cảm chồng lên). Vùng sáng là nơi che đi làm mô hình mất tự tin nhất; kỹ thuật này coi mô hình như hộp đen.](../../results/notebook_c2/xai_occlusion.png)

![Hình 4. Gradient SHAP cho năm lớp (trên: ảnh gốc; dưới: độ lớn quy thuộc pixel chồng lên ảnh), được tính với 20 mẫu và hai baseline tensor hằng 0 và 1 trong không gian đầu vào đã chuẩn hóa.](../../results/notebook_c2/xai_shap.png)

![Hình 5. Attention Rollout của khối Tiny Transformer trên lưới 14×14, được tổng hợp bằng cách lấy trung bình dòng chú ý qua 196 token do mô hình sử dụng mean pooling và không có token `CLS`.](../../results/notebook_c2/xai_attn_rollout.png)
<!-- /IMG2 -->

#### 4.4.2. Đối chiếu với một LLM thị giác

Nghiên cứu sử dụng LLM đa phương thức tổng quát **Gemini 3.1 Flash Lite**, model ID API `gemini-3.1-flash-lite`, thông qua Gemini API; API key được tạo và quản lý bằng Google AI Studio. Mô hình được truy cập ngày **08/07/2026** (Google DeepMind, 2026) để phân loại trực tiếp ảnh gốc ở chế độ zero-shot. Không đặt riêng temperature, top-p hoặc giới hạn token; lần chạy sử dụng cấu hình sinh mặc định của endpoint tại ngày truy cập. Đây là đối chiếu tham khảo, không phải baseline hoàn toàn ngang hàng: Gemini không được fine-tune trên ChiliEdge-DS, trong khi ChiliEdgeNet là bộ phân loại chuyên biệt được huấn luyện có giám sát cho đúng năm lớp.

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

Artifact hiện lưu có **370 ảnh**, trong đó **370/370** lượt gọi hợp lệ và không có nhãn ngoài tập cho phép. Gemini dự đoán đúng **290/370 ảnh (78,38%)**, còn ChiliEdgeNet đúng **368/370 ảnh (99,46%)** trên chính tập con này. Ba ảnh còn lại của tập test 373 ảnh chưa có bản ghi API trong artifact hiện tại; chúng không được xem là đầu ra lỗi và chưa được đưa vào phép đối chiếu. Vì vậy, kết quả này chỉ được báo cáo trên tập con 370 ảnh cho đến khi chạy bổ sung ba ảnh còn thiếu.

**Về độ trễ.** Con số xấp xỉ 105 lần đối chiếu **20,24 ms suy luận cục bộ** với **2.116 ms trung vị round-trip API đám mây**. Hai con số không tương đương: round-trip bao gồm độ trễ mạng, hàng đợi phía máy chủ và gộp lô phía nhà cung cấp — không cái nào là phần tính toán riêng của LLM. Đây là đối chiếu giữa hai phương thức triển khai, không phải phát biểu rằng một kiến trúc tính toán nhanh hơn kiến trúc kia khoảng 105 lần. Tương tự, dù accuracy được tính trên cùng 370 ảnh, chênh lệch accuracy không tạo thành so sánh ngang hàng vì hai hệ thống có chế độ huấn luyện và mức độ chuyên biệt khác nhau. Endpoint thương mại cũng có thể được cập nhật, nên phép đối chiếu chỉ tái lập đúng với model ID, prompt và ngày truy cập đã nêu.

**Bảng 9. Đối chiếu ChiliEdgeNet với Gemini 3.1 Flash Lite trên tập con 370 ảnh có bản ghi API hợp lệ (truy cập 08/07/2026).**

| Hệ thống | Accuracy | Latency/ảnh |
|---|---|---|
| **ChiliEdgeNet (CPU cục bộ)** | **99,46% (368/370)** | **20,24 ms** (suy luận cục bộ) |
| Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) | 78,38% (290/370) | 2.116 ms (trung vị round-trip API) |

> **GAP cần hoàn tất:** chạy Gemini trên ba ảnh test còn thiếu, lưu raw response và cập nhật lại Bảng 9 nếu muốn tuyên bố đối chiếu trên toàn bộ 373 ảnh.

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

### 4.6. Đánh giá độ trễ tiền xử lý và suy luận trong ứng dụng camera laptop

ChiliEdgeNet được tích hợp vào một ứng dụng camera trên laptop, trong đó người dùng đặt lá trong vùng quan tâm (ROI) và kích hoạt dự đoán trên từng khung hình. Thí nghiệm chỉ đánh giá độ trễ tiền xử lý và suy luận như một minh chứng về khả năng tích hợp, không sử dụng các khung hình camera để ước lượng độ chính xác thực địa.

Phiên thử nghiệm gồm **62 lần dự đoán**, thực hiện với batch size 1 trên CPU **13th Gen Intel Core i7-13650HX**. Khoảng thời gian đo bắt đầu từ bước chuyển ảnh RGB của ROI sang PIL, tiếp tục qua resize, chuẩn hóa, suy luận mô hình, softmax và lấy dự đoán top-1. Phép đo không bao gồm thời gian đọc khung hình từ camera, cắt ROI, chuyển BGR sang RGB, vẽ giao diện hoặc ghi tệp.

**Bảng 11. Độ trễ tiền xử lý và suy luận của ChiliEdgeNet trên CPU laptop.**

| Chỉ số | Latency (ms) |
|---|---:|
| Số lần đo | 62 |
| Trung bình | 32,18 |
| Trung vị | 29,70 |
| Phân vị 95% | 46,49 |
| Nhỏ nhất | 17,33 |
| Lớn nhất | 122,76 |

Trung vị 29,70 ms và phân vị 95% 46,49 ms cho thấy phần tiền xử lý và suy luận có thể phản hồi trong thời gian ngắn trên CPU laptop mà không cần GPU. Tuy nhiên, kết quả chỉ đến từ một phiên đo trên một thiết bị và chưa bao gồm toàn bộ chi phí thu nhận–hiển thị của ứng dụng; thiết bị thử nghiệm cũng chưa phải phần cứng biên chuyên dụng. Vì vậy, thí nghiệm chỉ cung cấp bằng chứng ban đầu về khả năng tích hợp ChiliEdgeNet vào ứng dụng camera, không đại diện cho hiệu năng của toàn bộ pipeline hoặc thiết bị biên chuyên dụng. Dữ liệu gán nhãn trong phiên demo không cân bằng và chứa nhiều khung hình lặp của cùng đối tượng, nên accuracy và ma trận nhầm lẫn không được báo cáo trong mục này.

## 5. Hạn chế

**Năng lực phân giải thống kê của tập test.** Với 373 ảnh, một ảnh trị giá 0,27 pp và mô hình đề xuất sai 4 ảnh. Chênh lệch 0,80 pp (3 ảnh) so với kiến trúc nền tảng (Duhan et al., 2025) nằm trong dải nhiễu của một lần chia: khoảng Wilson 95% của ChiliEdgeNet [97,28; 99,58] chồng lấn với khoảng [96,18; 99,09] của Duhan et al. (2025). Hai mô hình đã được khớp seed 42 nhưng đều mới chạy một lần. Ngoài ra, RTR_Lite_MobileNet (Duhan et al., 2025) được chọn làm baseline chính sau khi xem xét kết quả test của nhóm baseline, nên phép so sánh này mang tính thăm dò và không tạo thành một đánh giá test độc lập. Cần lựa chọn bằng validation rồi đánh giá một lần trên test, cùng với nhiều seed và/hoặc McNemar trực tiếp giữa hai bộ dự đoán, để khẳng định thứ hạng. Mức giảm 35% tham số không phụ thuộc vào nhiễu huấn luyện hoặc tiêu chí lựa chọn baseline.

**Số lần chạy và năng lực thống kê.** Mô hình đề xuất và toàn bộ ablation đều đến từ một lần chạy seed 42. Ở cấp kiến trúc, tắt residual MobileNetV2 làm giảm accuracy 2,41 pp trên ảnh sạch và khác biệt vẫn có ý nghĩa sau hiệu chỉnh Holm cho hai phép cắt trực tiếp (\(p_{Holm}=0,0234\)). Trong tám điều kiện stress test, lợi thế của Tiny Transformer chỉ đạt ý nghĩa sau Holm tại mức sáng 0,2 (\(p_{Holm}=0,0105\)); các mức còn lại phản ánh xu hướng phụ thuộc độ nghiêm trọng, không phải hiệu ứng đã được xác lập rộng. Ba tương phản Triplet/`RES_Block` đều không có ý nghĩa sau Holm. Các kết luận này là quan sát một-seed và cần được xác nhận qua nhiều lần chạy.

**Phạm vi stress test.** Thí nghiệm khảo sát bốn mức nhoè Gaussian và bốn hệ số giảm sáng được tạo tổng hợp từ cùng tập test. Thiết kế chưa bao phủ nhiễu cảm biến, nén JPEG, nhòe chuyển động hoặc ảnh thu nhận thực tế trong điều kiện thiếu sáng; vì vậy kết quả không được ngoại suy sang mọi điều kiện chụp.

**Khoảng cách miền và phạm vi phép đánh giá few-shot.** Phép đánh giá chỉ sử dụng bốn nhãn Mendeley Data có thể ánh xạ trực tiếp sang `healthy`, `leaf_curl`, `leaf_spot` và `yellow_leaf`; nó không đánh giá `anthracnose` và loại hai nhãn nguồn mà mô hình không hỗ trợ. Trên query sạch 1.463 ảnh có giao hash với support bằng 0, accuracy đạt 30,96% ở checkpoint gốc và 89,75% sau few-shot. Cả hai checkpoint được huấn luyện với seed 42, cùng kiến trúc và hyperparameter nền; các thông tin huấn luyện vẫn cần được củng cố bằng log hoặc cấu hình đã thực thi. Tập support do tác giả lựa chọn và tiêu chí chi tiết chưa được lập hồ sơ, tạo nguy cơ selection bias. Do chỉ có một bộ dữ liệu ngoài miền và một tập support cố định, mức cải thiện chưa chứng minh khả năng khái quát của giao thức few-shot sang các nguồn dữ liệu hoặc cách chọn support khác.

**Phần cứng.** Con số 20,24 ms/ảnh đo trên CPU máy ảo Kaggle hạng máy chủ, không phải thiết bị biên. INT8 và FP16 trong lần chạy hiện tại giảm kích thước lưu trữ nhưng chưa tạo tăng tốc độ trễ; cần benchmark trên điện thoại hoặc máy tính nhúng với backend triển khai thực tế.

## 6. Kết luận

Với seed 42, ChiliEdgeNet đạt **98,93% accuracy**, F1-macro **0,9897**, sử dụng **671.173 tham số** và suy luận CPU ở **20,24 ms/ảnh**. So với RTR_Lite_MobileNet (Duhan et al., 2025) cùng seed, mô hình cao hơn **0,80 pp (3 ảnh)** và ít hơn khoảng **35% tham số**; do mỗi kiến trúc mới chạy một lần và khoảng tin cậy chồng lấn, ưu thế accuracy được xem là xu hướng chứ chưa phải thứ hạng thống kê đã xác lập.

Các kết quả ablation cho thấy:

1. Tắt residual connections của MobileNetV2 làm giảm 2,41 pp trên ảnh sạch; khác biệt vẫn có ý nghĩa sau hiệu chỉnh Holm cho hai phép cắt trực tiếp.
2. Tiny Transformer chưa tạo lợi thế trên ảnh sạch hoặc suy giảm nhẹ, nhưng khoảng cách tăng khi suy giảm nghiêm trọng. Tại hệ số sáng 0,2, mô hình đầy đủ cao hơn 3,75 pp và khác biệt có ý nghĩa sau hiệu chỉnh Holm.
3. Triplet Attention và `RES_Block` không mang lại lợi ích ổn định. Mô hình hiện tại bỏ cả hai đạt kết quả bằng hoặc cao hơn các cấu hình thêm lại, nhưng các tương phản không có ý nghĩa sau Holm.
4. 2D-RoPE, GQA và RMSNorm tránh tổng cộng 41.856 tham số so với việc đồng thời dùng ba phương án tiêu chuẩn tương ứng; kết quả một seed chưa cho thấy ưu thế accuracy trên ảnh sạch.
5. Trên query cross-dataset sạch 1.463 ảnh, thích nghi bằng 42 ảnh support làm accuracy tăng từ 30,96% lên 89,75% (+58,78 pp); McNemar ghép cặp ghi nhận 870 cải thiện và 10 suy giảm ($p=1{,}83\times10^{-242}$). Kết quả này chỉ áp dụng cho bốn lớp ánh xạ và một tập support do tác giả lựa chọn.

Hướng phát triển là lặp lại thí nghiệm trên nhiều seed, kiểm tra trùng lặp/near-duplicate giữa các tập, mở rộng stress test sang suy giảm thu nhận thực tế và đo độ trễ, bộ nhớ trên thiết bị biên vật lý với backend lượng tử hóa thực tế.

## References

Abnar, S., Zuidema, W., 2020. Quantifying attention flow in transformers. In: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pp. 4190–4197. https://doi.org/10.18653/v1/2020.acl-main.385.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., Sanghai, S., 2023. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4895–4901. https://doi.org/10.18653/v1/2023.emnlp-main.298.

Ali, M., Salma, M., El Haji, M., Jamal, B., 2025. Plant disease detection using vision transformers. International Journal of Electrical and Computer Engineering 15 (2), 2334–2344. https://doi.org/10.11591/ijece.v15i2.pp2334-2344.

Bhagat, S., Kokare, M., Haswani, V., Hambarde, P., Taori, T., Ghante, P.H., Patil, D.K., 2024. Advancing real-time plant disease detection: A lightweight deep learning approach and novel dataset for pigeon pea crop. Smart Agricultural Technology 7, 100408. https://doi.org/10.1016/j.atech.2024.100408.

Duhan, S., Gulia, P., Gill, N.S., Narwal, E., 2025. RTR_Lite_MobileNetV2: A lightweight and efficient model for plant disease detection and classification. Current Plant Biology 42, 100459. https://doi.org/10.1016/j.cpb.2025.100459.

Erion, G., Janizek, J.D., Sturmfels, P., Lundberg, S.M., Lee, S.-I., 2021. Improving performance of deep learning models with axiomatic attribution priors and expected gradients. Nature Machine Intelligence 3 (7), 620–631. https://doi.org/10.1038/s42256-021-00343-w.

Gole, P., Bedi, P., Marwaha, S., Haque, M.A., Deb, C.K., 2023. TrIncNet: A lightweight vision transformer network for identification of plant diseases. Frontiers in Plant Science 14, 1221557. https://doi.org/10.3389/fpls.2023.1221557.

Google DeepMind, 2026. Gemini 3.1 Flash-Lite: Model Card. https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/ (accessed 8 July 2026).

[dataset] Hải, T.M., 2026a. Chili Leaf Disease Cross-Dataset Query Set, Kaggle dataset version 2. Kaggle. https://www.kaggle.com/datasets/tminhhi/chili-leaf-disease-cross-dataset-query-set/version/2 (accessed 4 August 2026). Derived from Nirob et al. (2025); license: CC BY 4.0.

[dataset] Hải, T.M., 2026b. Chili Leaf Disease Few-Shot, Kaggle dataset version 1. Kaggle. https://www.kaggle.com/datasets/tminhhi/chili-leaf-disease-few-shot/version/1 (accessed 4 August 2026). Derived from Nirob et al. (2025); license: CC BY 4.0.

[dataset] Hải, T.M., 2026c. ChiliEdge-DS: Chili Pepper Leaf Disease Dataset, version 1.0 (Kaggle dataset version 2). Kaggle. https://www.kaggle.com/datasets/tminhhi/ot-21-06/versions/2 (accessed 4 August 2026).

Hemalatha, S., Jayachandran, J.J.B., 2024. A multitask learning-based vision transformer for plant disease localization and classification. International Journal of Computational Intelligence Systems 17 (1), 188. https://doi.org/10.1007/s44196-024-00597-3.

Heo, B., Park, S., Han, D., Yun, S., 2025. Rotary position embedding for Vision Transformer. In: Computer Vision – ECCV 2024, Lecture Notes in Computer Science, vol. 15068, pp. 289–305. https://doi.org/10.1007/978-3-031-72684-2_17.

Joseph, D.S., Pawar, P.M., Chakradeo, K., 2024. Real-time plant disease dataset development and detection of plant disease using deep learning. IEEE Access 12, 16310–16333. https://doi.org/10.1109/ACCESS.2024.3358333.

Karthik, R., Menaka, R., Ompirakash, S., Bala Murugan, P., Meenakashi, M., Lingaswamy, S., Won, D., 2024. GrapeLeafNet: A dual-track feature fusion network with Inception-ResNet and Shuffle-Transformer for accurate grape leaf disease identification. IEEE Access 12, 19612–19624. https://doi.org/10.1109/ACCESS.2024.3361044.

Li, G., Wang, Y., Zhao, Q., Yuan, P., Chang, B., 2023. PMVT: A lightweight vision transformer for plant disease identification on mobile devices. Frontiers in Plant Science 14, 1256773. https://doi.org/10.3389/fpls.2023.1256773.

Maurya, R., Mahapatra, S., Rajput, L., 2024. A lightweight meta-ensemble approach for plant disease detection suitable for IoT-based environments. IEEE Access 12, 28096–28108. https://doi.org/10.1109/ACCESS.2024.3367443.

Mishra, U., Pandey, A., Logeswari, G., Tamilarasi, K., 2026. Deep learning-based disease detection in potato and mango leaves: A comparative study of CNN, AlexNet, ResNet, and EfficientNet. Scientific Reports 16, 2788. https://doi.org/10.1038/s41598-025-32607-5.

Misra, D., Nalamada, T., Arasanipalai, A.U., Hou, Q., 2021. Rotate to attend: Convolutional triplet attention module. In: Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 3139–3148. https://doi.org/10.1109/WACV48630.2021.00318.

[dataset] Nirob, M.A.S., Siam, A.K.M.F.K., Bishshash, P., Assaduzzaman, M., 2025. Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh, version 1. Mendeley Data. https://doi.org/10.17632/w9mr3vf56s.1. Published 19 February 2025; license: CC BY 4.0.

Pan, S.J., Yang, Q., 2010. A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering 22 (10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191.

Rahaman, J., Paul, P., Chowdhury, A., Quamruzzaman, M., 2026. A customized MobileNetV3Large-based deep learning framework for plant disease detection. Discover Artificial Intelligence 6, 110. https://doi.org/10.1007/s44163-025-00733-8.

Rahman, M.M., Gony, M.N., Ullah, M.S., Shuvra, S.M.K., Haque, R., Ahmed, M.R., et al., 2026. MaizeFormerX: A lightweight vision transformer with cross-scale attention for explainable maize leaf disease diagnosis. Scientific Reports 16, 15160. https://doi.org/10.1038/s41598-026-44550-0.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C., 2018. MobileNetV2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510–4520. https://doi.org/10.1109/CVPR.2018.00474.

Snell, J., Swersky, K., Zemel, R.S., 2017. Prototypical networks for few-shot learning. In: Advances in Neural Information Processing Systems 30, pp. 4077–4087.

Su, J., Ahmed, M., Lu, Y., Pan, S., Wen, B., Liu, Y., 2024. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063. https://doi.org/10.1016/j.neucom.2023.127063.

Sujatha, R., Krishnan, S., Chatterjee, J.M., Gandomi, A.H., 2025. Advancing plant leaf disease detection integrating machine learning and deep learning. Scientific Reports 15, 11552. https://doi.org/10.1038/s41598-024-72197-2.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., et al., 2017. Attention is all you need. In: Advances in Neural Information Processing Systems 30, pp. 5998–6008.

Wang, B., Zhang, C., Li, Y., Cao, C., Huang, D., Gong, Y., 2023. An ultra-lightweight efficient network for image-based plant disease and pest infection detection. Precision Agriculture 24 (5), 1836–1861. https://doi.org/10.1007/s11119-023-10020-0.

Yang, B., Li, M., Li, F., Wang, Y., Liang, Q., Zhao, R., et al., 2024. A novel plant type, leaf disease and severity identification framework using CNN and transformer with multi-label method. Scientific Reports 14, 11664. https://doi.org/10.1038/s41598-024-62452-x.

Zeiler, M.D., Fergus, R., 2014. Visualizing and understanding convolutional networks. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 818–833. https://doi.org/10.1007/978-3-319-10590-1_53.

Zhang, B., Sennrich, R., 2019. Root mean square layer normalization. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 12360–12371.
