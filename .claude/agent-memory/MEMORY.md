# Agent Memory Index

## Verification & Fix Lessons
- [verification_lessons.md](verification_lessons.md) — Kinh nghiệm verify + fix MetaEnsemble: lỗi LSTM hyperparameter, channel_hidden dim, dpi, bug chain-replace Python script, **quy tắc báo cáo bằng tiếng Việt**

## Dataset người dùng
- [user_dataset.md](user_dataset.md) — Dataset 300×300px tại `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300`; pre-split train/val/test, 5 class ớt; không cần sửa kiến trúc vì resize xử lý tự động

## Agents
- Agent `aug-analyzer` đã được tạo tại `.claude/agents/aug_analyzer.md` — phân tích ảnh thực → sinh preview → đề xuất tham số augmentation → cập nhật notebook

## Augmentation Strategy
- [augmentation_strategy.md](augmentation_strategy.md) — Template augmentation tối ưu cho dataset ớt: +VerticalFlip, +Perspective, +GaussianBlur, rotation 45°, saturation=0.1/hue=0.02 (màu là đặc trưng chẩn đoán), tránh ElasticTransform/RandomErasing

## Multi-GPU & Performance
- [multi_gpu_template.md](multi_gpu_template.md) — Template DataParallel cho Kaggle 2×T4: pattern khởi tạo, save/load checkpoint với `.module`, lưu ý SVM là CPU bottleneck không fix được bằng GPU
