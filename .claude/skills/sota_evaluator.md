# SKILL: SOTA MODEL EXPERIMENT & EVALUATION
Nhiệm vụ của bạn là viết code đánh giá mô hình linh hoạt theo framework:

NẾU DÙNG TENSORFLOW:
- Dùng `tf.keras.preprocessing.image.ImageDataGenerator` cho 2 kịch bản Aug và No-Aug.
- Dùng `tf.keras.applications` để gọi 15 SOTA models (ResNet, EfficientNet, ViT...).
- Lấy số parameters bằng `model.count_params()`.

NẾU DÙNG PYTORCH:
- Dùng `torchvision.transforms` để định nghĩa 2 kịch bản:
  + No-Aug: Resize, CenterCrop, ToTensor, Normalize.
  + Aug: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize.
- Dùng `torchvision.models` (ví dụ: `models.resnet50(pretrained=True)`) để gọi SOTA models. Sửa lại lớp FC cuối cùng cho khớp số class.
- Khai báo hàm `train_one_epoch` và `validate` rõ ràng để vẽ Learning Curve.
- Đếm parameters bằng: `sum(p.numel() for p in model.parameters() if p.requires_grad)`
- Tính toán Confusion Matrix và F1, Recall bằng cách thu thập toàn bộ `preds` và `targets` vào một list sau đó dùng thư viện `sklearn.metrics`.