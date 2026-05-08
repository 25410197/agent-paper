# CORE RULES: STRICT CONSTRAINTS FOR AI
1. ZERO HALLUCINATION: Tuyệt đối không tự suy diễn thông số model từ bài báo. Nếu thiếu thông số, phải hỏi lại người dùng.
2. FRAMEWORK FLEXIBILITY (QUAN TRỌNG): 
   - Đọc kỹ bài báo để xác định tác giả dùng **PyTorch** hay **TensorFlow/Keras**. 
   - Viết code dựa trên đúng framework mà bài báo đề cập. 
   - Nếu bài báo không đề cập rõ ràng, hãy tự động ưu tiên chọn **PyTorch** vì đây là tiêu chuẩn hiện tại cho Research, trừ khi người dùng yêu cầu khác.
3. DUAL EXPERIMENT REQUIREMENT: Bất kỳ notebook nào cũng PHẢI chứa 2 kịch bản chạy:
   - Kịch bản 1: Baseline (Không sử dụng Data Augmentation, chỉ Resize/Normalize).
   - Kịch bản 2: Augmented (Sử dụng 04 thuật toán Augmentation: Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness).
4. REPORTING METRICS: Bắt buộc code phải tính toán và in ra chính xác: Total Parameters, Accuracy, F1-Score (Macro), Recall (Macro).
5. EXPORT ARTIFACTS: Bắt buộc dùng matplotlib để lưu biểu đồ Learning Curve và Confusion Matrix thành file ảnh chất lượng cao (dpi=300).