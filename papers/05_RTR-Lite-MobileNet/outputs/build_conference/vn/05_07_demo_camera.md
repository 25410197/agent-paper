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
