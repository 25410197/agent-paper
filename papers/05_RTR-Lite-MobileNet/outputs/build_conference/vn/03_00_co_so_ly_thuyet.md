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

trong đó (g_i) là hệ số scale học được và (\varepsilon=10^{-6}). Hiện thực không có bias dịch chuyển. ChiliEdgeNet có ba vị trí chuẩn hóa chiều 128, vì vậy RMSNorm tránh (3\times128=384) tham số bias so với ba LayerNorm tương ứng. Đây là khác biệt cấu trúc chính xác; hiệu quả accuracy của RMSNorm vẫn phải được xác nhận bằng ablation thay vì suy ra trực tiếp từ công thức.

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
