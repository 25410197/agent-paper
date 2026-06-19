---
name: user-dataset
description: Thông tin về dataset của người dùng — kích thước ảnh gốc 300x300, đường dẫn Kaggle, và quy tắc xử lý resize
metadata:
  type: project
  updated: 2026-06-16
---

# Dataset của người dùng

**Đường dẫn Kaggle (hiện tại, xác nhận lại 2026-06-16):** `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300`
**Kích thước ảnh gốc:** 300×300 pixels
**5 class:** anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf (cây ớt — "ot")

> Đường dẫn cũ `/kaggle/input/datasets/tminhhi/ot-mixed/dataset_smartcrop_300` (flat, không train/val/test) đã lỗi thời — KHÔNG dùng nữa. Dataset hiện tại đã được pre-split sẵn.

## Quy tắc khi tạo notebook mới

**Không cần thay đổi kiến trúc model** khi dataset là 300×300. Lý do:
- Notebook luôn có resize (`transforms.Resize` hoặc `target_size` trong `flow_from_directory`) trong mọi pipeline
- Ảnh 300×300 sẽ tự động được scale xuống kích thước model yêu cầu (ví dụ 64×64, 224×224, 256×256...)
- Chỉ cần đặt đúng `DATASET_ROOT`/`DATASET_PATH` và `IMG_SIZE` theo paper — resize lo phần còn lại

## Cấu trúc thư mục dataset (pre-split, xác nhận qua `papers/03 - TrIncNet`)

Dataset đã được **chia sẵn train/val/test**, mỗi folder chứa 5 class subfolder:

```
dataset_smartcrop_plit_300/
├── train/
│   ├── anthracnose/
│   ├── healthy/
│   ├── leaf_curl/
│   ├── leaf_spot/
│   └── yellow_leaf/
├── val/
│   └── ... (5 class)
└── test/
    └── ... (5 class)
```

**KHÔNG cần tự chia train/val/test trong code** (không cần `TRAIN_RATIO`/`VAL_RATIO`/`TEST_RATIO`/`make_splits()`) — chỉ trỏ trực tiếp vào 3 subfolder có sẵn:

```python
DATASET_ROOT = "/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300"
TRAIN_DIR    = os.path.join(DATASET_ROOT, "train")
VAL_DIR      = os.path.join(DATASET_ROOT, "val")
TEST_DIR     = os.path.join(DATASET_ROOT, "test")
```

- PyTorch: `datasets.ImageFolder(TRAIN_DIR, transform=...)`, tương tự cho val/test — không cần `Subset`.
- TF/Keras: `ImageDataGenerator(...).flow_from_directory(TRAIN_DIR, ...)`, tương tự cho val/test.

**Why:** Người dùng xác nhận lại đường dẫn này ngày 2026-06-16 khi phục dựng paper 04 (Real-Time Plant Disease Dataset). Notebook `TrIncNet_Experiment.ipynb` (paper 03) đã dùng đúng path + cấu trúc pre-split này và chạy thành công.
**How to apply:**
- `kaggle-reproducer`: điền sẵn `DATASET_ROOT` này, dùng pattern pre-split (KHÔNG cần `make_splits()`/flat-detection nữa — đó là quy tắc cũ, lỗi thời).
- Nếu thấy notebook cũ dùng flat-detection (`_root = _TRAIN_DIR if os.path.isdir(_TRAIN_DIR) else DATASET_PATH`), đó là pattern lỗi thời từ trước khi dataset được pre-split.
