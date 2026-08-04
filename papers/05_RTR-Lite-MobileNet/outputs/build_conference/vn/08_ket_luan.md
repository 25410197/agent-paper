## 6. Kết luận

Với seed 42, ChiliEdgeNet đạt **98,93% accuracy**, F1-macro **0,9897**, sử dụng **671.173 tham số** và suy luận CPU ở **20,24 ms/ảnh**. So với RTR_Lite_MobileNet (Duhan et al., 2025) cùng seed, mô hình cao hơn **0,80 pp (3 ảnh)** và ít hơn khoảng **35% tham số**; do mỗi kiến trúc mới chạy một lần và khoảng tin cậy chồng lấn, ưu thế accuracy được xem là xu hướng chứ chưa phải thứ hạng thống kê đã xác lập.

Các kết quả ablation cho thấy:

1. Tắt residual connections của MobileNetV2 làm giảm 2,41 pp trên ảnh sạch; khác biệt vẫn có ý nghĩa sau hiệu chỉnh Holm cho hai phép cắt trực tiếp.
2. Tiny Transformer chưa tạo lợi thế trên ảnh sạch hoặc suy giảm nhẹ, nhưng khoảng cách tăng khi suy giảm nghiêm trọng. Tại hệ số sáng 0,2, mô hình đầy đủ cao hơn 3,75 pp và khác biệt có ý nghĩa sau hiệu chỉnh Holm.
3. Triplet Attention và `RES_Block` không mang lại lợi ích ổn định. Mô hình hiện tại bỏ cả hai đạt kết quả bằng hoặc cao hơn các cấu hình thêm lại, nhưng các tương phản không có ý nghĩa sau Holm.
4. 2D-RoPE, GQA và RMSNorm tránh tổng cộng 41.856 tham số so với việc đồng thời dùng ba phương án tiêu chuẩn tương ứng; kết quả một seed chưa cho thấy ưu thế accuracy trên ảnh sạch.
5. Trên query cross-dataset sạch 1.463 ảnh, thích nghi bằng 42 ảnh support làm accuracy tăng từ 30,96% lên 89,75% (+58,78 pp); McNemar ghép cặp ghi nhận 870 cải thiện và 10 suy giảm ($p=1{,}83\times10^{-242}$). Kết quả này chỉ áp dụng cho bốn lớp ánh xạ và một tập support do tác giả lựa chọn.

Hướng phát triển là lặp lại thí nghiệm trên nhiều seed, kiểm tra trùng lặp/near-duplicate giữa các tập, mở rộng stress test sang suy giảm thu nhận thực tế và đo độ trễ, bộ nhớ trên thiết bị biên vật lý với backend lượng tử hóa thực tế.
