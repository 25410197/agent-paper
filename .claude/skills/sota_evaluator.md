# SKILL: SOTA MODEL EXPERIMENT & EVALUATION

## DEVICE DETECTION + FULL REPRODUCIBILITY (BẮT BUỘC ĐẶT ĐẦU NOTEBOOK)
```python
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Rule 9 — Full GPU reproducibility (seed alone is not enough)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## TRANSFORMS — QUY TẮC PHÂN BIỆT TRAINING / VALIDATION / BASELINE

### Scenario 1 — Baseline (No-Aug): Resize + Normalize ONLY
```python
# Training transform — KHÔNG dùng RandomCrop, CenterCrop, hay bất kỳ augmentation nào
train_transform_baseline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # resize thẳng về kích thước target
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Validation/Test transform — luôn giống Baseline (Resize + Normalize)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Scenario 2 — Augmented (4 kỹ thuật bắt buộc theo core_constraints):
```python
train_transform_aug = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # Zoom/Scale
    transforms.RandomHorizontalFlip(p=0.5),                    # Horizontal Flip
    transforms.RandomRotation(degrees=15),                     # Rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3,       # ColorJitter
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Val/Test transform cho Scenario 2 — LUÔN GIỐNG val_transform (không aug validation)
```

> ⚠️ Nếu paper dùng normalization khác (custom mean/std trên dataset riêng), dùng giá trị của paper thay cho ImageNet default.

---

## LOADING PRETRAINED MODELS (API HIỆN TẠI — torchvision ≥ 0.13)

> ⚠️ `pretrained=True` đã **DEPRECATED**. Dùng `weights=` thay thế. Code cũ sẽ gây `UserWarning` và có thể lỗi trong tương lai.

```python
import torchvision.models as models

# ──── Cú pháp ĐÚNG (torchvision ≥ 0.13) ────
# ResNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# EfficientNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
# MobileNet
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
# Không dùng pretrained
model = models.resnet50(weights=None)

# Sửa lớp FC cuối cho đúng NUM_CLASSES
# ResNet / EfficientNet
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
# MobileNetV3
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
```

### Transfer Learning: Freeze / Unfreeze backbone
```python
# Freeze toàn bộ backbone, chỉ train classifier
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():  # hoặc model.fc cho ResNet
    param.requires_grad = True

# Unfreeze dần (Fine-tuning toàn bộ sau N epochs)
for param in model.parameters():
    param.requires_grad = True
```

---

## PYTORCH TRAINING LOOP (CHUẨN)

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_targets
```

---

## LR SCHEDULERS (dùng đúng loại scheduler mà paper đề cập)

```python
# StepLR — giảm LR mỗi step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# CosineAnnealingLR — phổ biến trong papers mới
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ReduceLROnPlateau — dựa trên validation loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

# Gọi scheduler.step() sau mỗi epoch (hoặc sau optimizer.step() với OneCycleLR)
```

---

## CLASS NAMES AUTO-DETECT (Rule 12)
```python
# Lấy class_names — phân biệt ImageFolder vs Subset
def get_class_names(dataset):
    if hasattr(dataset, 'classes'):          # ImageFolder trực tiếp
        return dataset.classes
    elif hasattr(dataset, 'dataset'):        # Subset từ make_splits()
        return dataset.dataset.classes
    else:
        raise AttributeError("Không tìm được class_names từ dataset")

class_names = get_class_names(train_ds_s1)
NUM_CLASSES  = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}")
```

---

## CLASS IMBALANCE AUTO-DETECTION + WEIGHTED LOSS (Rule 11)
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(dataset, device):
    """Tính class_weights nếu dataset mất cân bằng (max/min >= 1.5)."""
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'dataset'):        # Subset
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = [label for _, label in dataset]

    counts = np.bincount(targets)
    ratio  = counts.max() / counts.min()
    print(f"Class distribution: {dict(zip(class_names, counts))}")
    print(f"Max/Min ratio: {ratio:.2f}")

    if ratio >= 1.5:
        print("⚠️  Mất cân bằng phát hiện → dùng weighted CrossEntropyLoss")
        weights = compute_class_weight('balanced',
                                       classes=np.unique(targets),
                                       y=targets)
        return torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        print("✅  Dataset cân bằng → dùng CrossEntropyLoss thông thường")
        return None

class_weights = get_class_weights(train_ds_s1, device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

> ⚠️ **Quan trọng:** Gọi lại `get_class_weights(train_ds_s2, device)` trước Scenario 2 (cùng dataset, weight không đổi, nhưng phải re-assign criterion vì `criterion` có thể đã bị gc).

---

## BEST MODEL CHECKPOINT (Rule 10)
```python
import os

CHECKPOINT_S1 = "/kaggle/working/best_model_s1.pth"
CHECKPOINT_S2 = "/kaggle/working/best_model_s2.pth"

def train_with_checkpoint(model, train_loader, val_loader, optimizer, criterion,
                           scheduler, num_epochs, checkpoint_path, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Lưu best model dựa trên val_accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
              + (" ← best" if val_acc == best_val_acc else ""))

    # Reload best weights trước khi evaluate
    print(f"\nReloading best model (val_acc={best_val_acc:.4f}) from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history
```

> ⚠️ **KHÔNG dùng `model` trực tiếp sau training loop** — phải dùng `model` đã được reload từ checkpoint.

---

## DATALOADER OPTIMIZATION (Rule 13)
```python
from torch.utils.data import DataLoader

def make_loaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader
```

## GPU CACHE CLEAR GIỮA 2 SCENARIO (Rule 13)
```python
# ── Đặt ở cuối Scenario 1, trước khi bắt đầu Scenario 2 ─────────────────────
del model_s1, train_loader_s1, val_loader_s1, test_loader_s1
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("GPU cache cleared. Bắt đầu Scenario 2...")
```

---

## METRICS CALCULATION (bắt buộc dùng sklearn)

```python
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

# Evaluate trên test_ds — BẮT BUỘC dùng model đã reload từ checkpoint (Rule 10)
_, _, all_preds, all_targets = validate(model, test_loader, criterion, device)

accuracy     = accuracy_score(all_targets, all_preds)
f1_macro     = f1_score(all_targets, all_preds, average='macro')
recall_macro = recall_score(all_targets, all_preds, average='macro')

# class_names auto-detect từ Rule 12 — KHÔNG hard-code
report = classification_report(all_targets, all_preds,
                                target_names=class_names, digits=4)
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test F1 Macro : {f1_macro:.4f}")
print(f"Test Recall   : {recall_macro:.4f}")
print("\nClassification Report:\n", report)
```

---

## TENSORFLOW / KERAS (nếu paper dùng TF)

```python
# Transforms — dùng ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Baseline (No-Aug)
baseline_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Augmented
aug_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    validation_split=0.2,
)

# Load SOTA models (TF)
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV3Small
base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.count_params()  # đếm parameters
```
