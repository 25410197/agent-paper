# WORKFLOW: KAGGLE NOTEBOOK GENERATOR (TWO-SCENARIO EXPERIMENT)

Nhiệm vụ của bạn CHỈ LÀ VIẾT CODE VÀ TẠO FILE `.ipynb`. Tuyệt đối không cố gắng thực thi (run/execute) các block code huấn luyện mô hình. Người dùng sẽ mang file bạn tạo ra lên Kaggle để chạy.

**OUTPUT: 1 FILE DUY NHẤT** chứa 2 scenario chạy tuần tự.
- `[ModelName]_Experiment.ipynb` — Scenario 1 (No-Aug) → Scenario 2 (Aug)

---

- BƯỚC 1: ĐỌC VÀ TRÍCH XUẤT (EXTRACTION)
  - Đọc file PDF trong thư mục `.claude/inputs/`.
  - Xác định Framework tác giả sử dụng (PyTorch hay TensorFlow).
  - Trích xuất toàn bộ cấu trúc mạng, hyper-parameters (Learning rate, Batch size, Epochs, Optimizer, Loss) và các phép Data Augmentation.

- BƯỚC 2: XÁC MINH THÔNG SỐ (VERIFICATION) - BẮT BUỘC DỪNG LẠI
  - Đọc đối chiếu thông tin. NẾU THIẾU BẤT KỲ THÔNG SỐ NÀO, AI KHÔNG ĐƯỢC TỰ SUY DIỄN.
  - In ra danh sách các câu hỏi (1, 2, 3...) cho người dùng để làm rõ các điểm còn thiếu.
  - Dừng hoàn toàn việc generate code và chờ người dùng phản hồi. Chỉ đi tiếp khi đã có đủ thông số.

- BƯỚC 3: SINH CODE SETUP DÙNG CHUNG (SHARED BLOCKS)
  - Viết các block dùng chung: Imports, Config (DATASET_PATH, hyperparams), Dataset Exploration (auto-detect class), Feature Extractor / Model Builder, Evaluation & Artifact Functions.
  - Đảm bảo biến thư mục (TRAIN_DIR, VAL_DIR, TEST_DIR) theo chuẩn Kaggle (`/kaggle/input/...`).

- BƯỚC 4: SINH CODE SCENARIO 1 — KHÔNG AUGMENTATION
  - Định nghĩa transform No-Aug (Chỉ Resize/Normalize).
  - Viết code gọi model, đếm Total Parameters.
  - Viết code train/fit với tập No-Aug.
  - Viết code lưu ảnh `[ModelName]_NoAug_Curve.png` và `[ModelName]_NoAug_CM.png` (dpi=300).
  - Viết code in ra: Total Parameters, Accuracy, F1-Score (Macro), Recall (Macro).

- BƯỚC 5: SINH CODE SCENARIO 2 — CÓ AUGMENTATION
  - Định nghĩa transform Aug (4 thuật toán: Rotation, Zoom/Scale, Horizontal Flip, ColorJitter).
  - Khởi tạo lại model từ đầu (tránh Data Leakage từ Scenario 1).
  - Viết code train/fit với tập Augmented.
  - Viết code lưu ảnh `[ModelName]_Aug_Curve.png` và `[ModelName]_Aug_CM.png` (dpi=300).
  - Viết code in ra: Total Parameters, Accuracy, F1-Score (Macro), Recall (Macro).

- BƯỚC 6: XUẤT FILE NOTEBOOK (FINAL OUTPUT)
  - Tổng hợp toàn bộ code từ Bước 3, 4, 5 theo cấu trúc template đã định.
  - Xuất ra thành **1 file duy nhất** `.ipynb`, lưu vào thư mục `outputs/`.
  - Thông báo cho người dùng biết file đã sẵn sàng để upload lên Kaggle.
