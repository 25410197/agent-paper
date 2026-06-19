# CORE RULES: STRICT CONSTRAINTS FOR AI
1. ZERO HALLUCINATION: Tuyệt đối không tự suy diễn thông số model từ bài báo. Nếu thiếu thông số, phải hỏi lại người dùng.
2. FRAMEWORK FLEXIBILITY (QUAN TRỌNG): 
   - Đọc kỹ bài báo để xác định tác giả dùng **PyTorch** hay **TensorFlow/Keras**. 
   - Viết code dựa trên đúng framework mà bài báo đề cập. 
   - Nếu bài báo không đề cập rõ ràng, hãy tự động ưu tiên chọn **PyTorch** vì đây là tiêu chuẩn hiện tại cho Research, trừ khi người dùng yêu cầu khác.
3. DUAL EXPERIMENT REQUIREMENT: Bất kỳ notebook nào cũng PHẢI chứa 2 kịch bản chạy:
   - Kịch bản 1: Baseline — Training transform: **CHỈ Resize + ToTensor + Normalize** (KHÔNG dùng CenterCrop, RandomCrop hay bất kỳ augmentation nào trong bước training). Validation/Test transform luôn giống baseline.
   - Kịch bản 2: Augmented (Sử dụng 04 thuật toán Augmentation: Rotation, Zoom/Scale, Horizontal Flip, ColorJitter/Brightness). Validation/Test transform KHÔNG aug — giống baseline.
4. REPORTING METRICS: Bắt buộc code phải tính toán và in ra chính xác: Total Parameters, Accuracy, F1-Score (Macro), Recall (Macro), **và Classification Report per class** (precision/recall/f1/support cho từng class bằng `sklearn.metrics.classification_report`).
5. MODEL SUMMARY: Bắt buộc dùng `torchinfo.summary()` (PyTorch) hoặc `model.summary()` (TF/Keras) để in bảng Layer / Output Shape / Param #. In đủ: Total params, Trainable params, Non-trainable params.
6. EXPORT ARTIFACTS: Bắt buộc dùng matplotlib để lưu biểu đồ Learning Curve và Confusion Matrix thành file ảnh chất lượng cao (dpi=300). Mỗi loại biểu đồ phải nằm trong **cell riêng biệt** và gọi `plt.show()` để hiển thị inline.
7. REPRODUCIBILITY: Bắt buộc đặt seed ở đầu notebook: `torch.manual_seed(42)` (PyTorch) hoặc `tf.random.set_seed(42)` (TF/Keras). Nếu dùng CUDA: thêm `torch.cuda.manual_seed_all(42)`. Mục đích: kết quả chạy lại phải nhất quán.
8. DEVICE DETECTION & MULTI-GPU:
   **PyTorch** — đặt ngay sau import:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   n_gpus = torch.cuda.device_count()
   print(f"Device: {device} | GPUs available: {n_gpus}")
   ```
   Sau khi tạo model và `.to(device)`, nếu `n_gpus > 1` → wrap với `nn.DataParallel`:
   ```python
   if n_gpus > 1:
       model = nn.DataParallel(model)
       print(f"[Multi-GPU] DataParallel on {n_gpus} GPUs")
   ```
   Không hard-code `"cuda"` / `"cpu"`. Mọi tensor phải `.to(device)`.

   **TensorFlow/Keras** — đặt ngay sau import:
   ```python
   strategy = tf.distribute.MirroredStrategy()
   n_gpus = strategy.num_replicas_in_sync
   print(f"GPUs: {n_gpus}")
   ```
   Mọi model creation **và** `model.compile()` phải nằm trong `with strategy.scope():`.

9. FULL REPRODUCIBILITY (GPU): Ngay sau device detection (PyTorch), bắt buộc thêm:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
   Seed một mình không đủ — GPU operation vẫn non-deterministic nếu thiếu 2 dòng này.

10. BEST MODEL CHECKPOINT: Bắt buộc lưu trọng số model tốt nhất trong training loop dựa trên `val_accuracy`. Khi training kết thúc, **reload checkpoint tốt nhất** trước khi evaluate trên test set. KHÔNG được evaluate trên last-epoch weights. Tên file: `best_model_s1.pth` (Scenario 1) và `best_model_s2.pth` (Scenario 2).
   **PyTorch + DataParallel** — lưu và load `module.state_dict()`:
   ```python
   # Save
   _state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
   torch.save(_state, checkpoint_path)
   # Load
   _target = model.module if isinstance(model, nn.DataParallel) else model
   _target.load_state_dict(torch.load(checkpoint_path, map_location=device))
   ```
11. CLASS IMBALANCE AUTO-DETECTION: Trước khi train, bắt buộc tính phân phối class và vẽ bar chart trong cell "Dataset Distribution". Nếu `max_count / min_count ≥ 1.5` → tự động tính `class_weights` bằng `sklearn.utils.class_weight.compute_class_weight` và truyền vào `CrossEntropyLoss(weight=class_weights.to(device))`. Ghi rõ có/không dùng class weighting trong output.
12. CLASS NAMES AUTO-DETECT: `class_names` bắt buộc auto-detect, không hard-code. Quy tắc:
    - `ImageFolder` trực tiếp: `class_names = dataset.classes`
    - `Subset` (từ `make_splits()`): `class_names = dataset.dataset.classes`
    Truyền `target_names=class_names` vào `classification_report` và dùng làm labels cho Confusion Matrix.
13. DATALOADER & GPU MEMORY:
    - **PyTorch:** Bắt buộc dùng `num_workers=4, pin_memory=True` trong mọi DataLoader (4 workers tối ưu cho Kaggle T4×2 — 4 vCPU có sẵn).
    - **TF/Keras (ImageDataGenerator):** Thêm `workers=4, use_multiprocessing=True` vào `model.fit()`.
    - **TF/Keras (tf.data):** Dùng `.prefetch(tf.data.AUTOTUNE)` — đã tối ưu tự động, không cần thêm.
    - Bắt buộc thêm `torch.cuda.empty_cache()` (PyTorch) hoặc `tf.keras.backend.clear_session(); gc.collect()` (TF) SAU khi kết thúc training + evaluation Scenario 1, TRƯỚC khi khởi tạo Scenario 2. Tránh OOM error.

14. MULTI-GPU NUANCES (BẮT BUỘC ĐỌC KHI DÙNG MULTI-GPU):
    **PyTorch DataParallel:**
    - `torchinfo.summary()` dùng `model.module` (không phải wrapped model):
      ```python
      _m = model.module if isinstance(model, nn.DataParallel) else model
      summary(_m, input_size=(1, C, H, W))
      ```
    - Với 2 GPU: effective per-GPU batch = `BATCH_SIZE // 2` (DataParallel split tự động).
    - `class_weights_tensor.to(device)` vẫn hoạt động bình thường với DataParallel.

    **TensorFlow MirroredStrategy:**
    - Model tạo **bên ngoài** `strategy.scope()` sẽ KHÔNG dùng multi-GPU → luôn wrap đúng.
    - `GLOBAL_BATCH_SIZE = BATCH_SIZE * n_gpus` nếu muốn giữ nguyên per-GPU batch size.
    - Checkpoint `.keras` load bình thường để evaluate (không cần scope khi load).
    - `model.fit(..., workers=4, use_multiprocessing=True)` chỉ dùng với `ImageDataGenerator`, không dùng với `tf.data.Dataset`.