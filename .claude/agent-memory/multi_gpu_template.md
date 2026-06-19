---
name: multi-gpu-template
description: Template và kinh nghiệm tận dụng 2 GPU trên Kaggle — DataParallel pattern, checkpoint với .module, và lưu ý SVM bottleneck
metadata:
  type: feedback
  updated: 2026-05-26
---

# Multi-GPU trên Kaggle (2× T4)

## Nguyên nhân không tận dụng được 2 GPU

| Nguyên nhân | Biểu hiện | Giải thích |
|------------|-----------|-----------|
| Không dùng `DataParallel` | GPU 1 idle, chỉ GPU 0 dùng | `torch.device('cuda')` chỉ pick cuda:0 |
| SVM GridSearchCV | CPU đỏ trong khi GPU nhàn | sklearn SVM chạy hoàn toàn trên CPU — không thể GPU-accelerate |
| Model quá nhỏ | GPU utilization thấp dù có DataParallel | MLP-Mixer D=128, LSTM 30 cells không đủ lớn để bão hòa T4 |

## Template: Khởi tạo model với DataParallel

Đặt ngay sau `device = torch.device(...)` và các dòng cudnn:

```python
# ── Multi-GPU setup ────────────────────────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
print(f'Available GPUs: {NUM_GPUS}')
USE_MULTI_GPU = NUM_GPUS > 1
```

Khi khởi tạo model (sau `.to(device)`):

```python
model = MyModel(...).to(device)

if USE_MULTI_GPU:
    model = nn.DataParallel(model)
    print(f'DataParallel enabled on {NUM_GPUS} GPUs')
```

## Template: Lưu và load checkpoint với DataParallel

**QUAN TRỌNG:** Khi wrap `DataParallel`, weights nằm trong `model.module` — nếu lưu `model.state_dict()` trực tiếp sẽ có prefix `module.` gây lỗi khi load trên single GPU.

```python
# ── Lưu checkpoint (dùng mọi nơi thay cho torch.save trực tiếp) ──────────
def save_checkpoint(model, path):
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, path)

# ── Load checkpoint (dùng mọi nơi thay cho load_state_dict trực tiếp) ────
def load_checkpoint(model, path, device):
    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(torch.load(path, map_location=device))
```

Dùng trong training loop:
```python
# Thay vì:
torch.save(model.state_dict(), checkpoint_path)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Dùng:
save_checkpoint(model, checkpoint_path)
load_checkpoint(model, checkpoint_path, device)
```

## Template: Lấy probabilities / logits từ model DataParallel

```python
# Không cần thay đổi gì — DataParallel transparent với forward pass
logits = model(x)          # hoạt động bình thường
probs  = F.softmax(logits, dim=1)
```

## Template: torchinfo.summary với DataParallel

```python
# summary cần model gốc, không phải DataParallel wrapper
_model_for_summary = model.module if hasattr(model, 'module') else model
summary(_model_for_summary, input_size=(...), device='cpu')
```

## Khi nào DataParallel THỰC SỰ có ích

| Điều kiện | DataParallel có ích? |
|----------|-------------------|
| Model lớn (ResNet50+, ViT, EfficientNet-L) | Có — đáng kể |
| Batch size lớn (≥ 256 per GPU) | Có |
| Model nhỏ (MLP-Mixer D=128, LSTM 30 cells) | Ít — overhead ~ lợi ích |
| Phase SVM / sklearn | Không — chạy CPU, không thể GPU |
| CNN feature extraction (frozen, eval mode) | Có thể giúp một chút |

## Lưu ý Kaggle cụ thể

- Kaggle cung cấp 2× T4 (16GB mỗi GPU) khi bật GPU accelerator
- `torch.cuda.device_count()` trả về 2 khi Kaggle cấp 2 GPU
- DataParallel tự chia batch theo số GPU: batch 512 → mỗi GPU xử lý 256
- SVM phase luôn là CPU bottleneck — không fix được bằng GPU
