import json, sys

NB_PATH = 'E:/UIT/study/claude/learn-claude/papers/13 - GrapeLeafNet/ouputs/grapeleafnet.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ─── Cell 3 — BLOCK 2: Config — auto-detect dataset path ─────────────────────
cells[3]['source'] = (
    "# BLOCK 2: Config\n"
    "import glob as _glob\n"
    "\n"
    "def _find_dataset(base='/kaggle/input'):\n"
    "    for train_path in sorted(_glob.glob(f'{base}/**/train', recursive=True)):\n"
    "        bp = os.path.dirname(train_path)\n"
    "        if (os.path.isdir(os.path.join(bp, 'val')) and\n"
    "                os.path.isdir(os.path.join(bp, 'test'))):\n"
    "            return bp\n"
    "    return None\n"
    "\n"
    "DATASET_PATH = _find_dataset()\n"
    "if DATASET_PATH is None:\n"
    "    raise FileNotFoundError(\n"
    "        'Cannot auto-detect dataset. Attach a dataset with train/val/test folders.'\n"
    "    )\n"
    "print(f'[Auto] Dataset: {DATASET_PATH}')\n"
    "\n"
    "TRAIN_DIR      = os.path.join(DATASET_PATH, 'train')\n"
    "VAL_DIR        = os.path.join(DATASET_PATH, 'val')\n"
    "TEST_DIR       = os.path.join(DATASET_PATH, 'test')\n"
    "PAPER_CITATION = 'Karthik et al. (2024) - IEEE Access'\n"
    "MODEL_NAME     = 'GrapeLeafNet'\n"
    "\n"
    "IMG_SIZE     = (256, 256)\n"
    "BATCH_SIZE   = 32\n"
    "EPOCHS       = 120\n"
    "LR           = 1e-4\n"
    "WEIGHT_DECAY = 1e-4\n"
    "DROPOUT      = 0.5\n"
    "OUTPUT_DIR   = '/kaggle/working'\n"
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
    "\n"
    "MEAN = [0.485, 0.456, 0.406]\n"
    "STD  = [0.229, 0.224, 0.225]\n"
    "\n"
    "# Auto-detect classes tu TRAIN_DIR\n"
    "CLASS_NAMES = sorted([\n"
    "    d for d in os.listdir(TRAIN_DIR)\n"
    "    if os.path.isdir(os.path.join(TRAIN_DIR, d))\n"
    "])\n"
    "NUM_CLASSES = len(CLASS_NAMES)\n"
    "print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')\n"
    "\n"
    "BASELINE_LC_PATH = f'{OUTPUT_DIR}/{MODEL_NAME}_NoAug_LC.png'\n"
    "BASELINE_CM_PATH = f'{OUTPUT_DIR}/{MODEL_NAME}_NoAug_CM.png'\n"
    "AUG_LC_PATH      = f'{OUTPUT_DIR}/{MODEL_NAME}_Aug_LC.png'\n"
    "AUG_CM_PATH      = f'{OUTPUT_DIR}/{MODEL_NAME}_Aug_CM.png'\n"
)
cells[3]['outputs'] = []

# ─── Cell 2 — BLOCK 1: Imports ───────────────────────────────────────────────
cells[2]['source'] = """\
# BLOCK 1: Imports
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary as torch_summary
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f'Device: {DEVICE} | GPUs available: {n_gpus}')
"""
cells[2]['outputs'] = []

# ─── Cell 4 — BLOCK 3: Dataset Exploration — add imbalance check ─────────────
orig_block3 = ''.join(cells[4]['source'])
imbalance_code = """
# Class imbalance auto-detection (Rule 11)
from sklearn.utils.class_weight import compute_class_weight
train_counts_list = list(stats['Train'].values())
max_count = max(train_counts_list)
min_count = min(train_counts_list)
imbalance_ratio = max_count / min_count
print(f'\\nImbalance ratio (max/min): {imbalance_ratio:.2f}')
if imbalance_ratio >= 1.5:
    all_labels_flat = []
    for i, cnt in enumerate(train_counts_list):
        all_labels_flat.extend([i] * cnt)
    all_labels_flat = np.array(all_labels_flat)
    cw = compute_class_weight('balanced', classes=np.unique(all_labels_flat), y=all_labels_flat)
    CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float32)
    print(f'[Imbalanced] Using class weights: {CLASS_WEIGHTS}')
else:
    CLASS_WEIGHTS = None
    print('[Balanced] No class weighting applied.')
"""
cells[4]['source'] = orig_block3 + imbalance_code
cells[4]['outputs'] = []

# ─── Cell 10 — BLOCK 5: Training & Evaluation Functions ──────────────────────
cells[10]['source'] = """\
# BLOCK 5: Training & Evaluation Functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def collect_preds(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def run_training(model, train_loader, val_loader, epochs, lr, wd,
                 checkpoint_path, class_weights=None):
    weight_tensor = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    history   = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = validate(model, val_loader, criterion)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            _state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(_state, checkpoint_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{epochs} | '
                  f'Loss: {tr_loss:.4f}/{va_loss:.4f} | '
                  f'Acc: {tr_acc:.4f}/{va_acc:.4f}')
    # Reload best checkpoint before evaluation
    _target = model.module if isinstance(model, nn.DataParallel) else model
    _target.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f'[Checkpoint] Best val_acc={best_val_acc:.4f} — reloaded from {checkpoint_path}')
    return history


def save_confusion_matrix(y_true, y_pred, title, path):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=30, colorbar=True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


def save_learning_curve(history, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_acc']) + 1)
    axes[0].plot(epochs, history['train_acc'], label='Train')
    axes[0].plot(epochs, history['val_acc'],   label='Val')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].plot(epochs, history['train_loss'], label='Train')
    axes[1].plot(epochs, history['val_loss'],   label='Val')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
"""
cells[10]['outputs'] = []

# ─── Cell 12 — BLOCK 6: S1 Transforms — num_workers=4, CLASS_NAMES sync ──────
cells[12]['source'] = """\
# BLOCK 6: Transforms & DataLoaders — Scenario 1 (No-Aug)
transform_noaug = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

train_ds_s1 = datasets.ImageFolder(TRAIN_DIR, transform=transform_noaug)
val_ds_s1   = datasets.ImageFolder(VAL_DIR,   transform=transform_noaug)
test_ds_s1  = datasets.ImageFolder(TEST_DIR,  transform=transform_noaug)

# Sync CLASS_NAMES with ImageFolder index order
CLASS_NAMES = train_ds_s1.classes
NUM_CLASSES  = len(CLASS_NAMES)

train_loader_s1 = DataLoader(train_ds_s1, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader_s1   = DataLoader(val_ds_s1,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader_s1  = DataLoader(test_ds_s1,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f'Train: {len(train_ds_s1)} | Val: {len(val_ds_s1)} | Test: {len(test_ds_s1)}')
print(f'Classes: {CLASS_NAMES}')
"""
cells[12]['outputs'] = []

# ─── Cell 13 — BLOCK 7: Build Model & Summary — DataParallel ─────────────────
cells[13]['source'] = """\
# BLOCK 7: Build Model & Summary — Scenario 1
model_s1 = GrapeLeafNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
if n_gpus > 1:
    model_s1 = nn.DataParallel(model_s1)
    print(f'[Multi-GPU] DataParallel on {n_gpus} GPUs')

_m_s1 = model_s1.module if isinstance(model_s1, nn.DataParallel) else model_s1
torch_summary(
    _m_s1,
    input_size=(1, 3, *IMG_SIZE),
    col_names=['output_size', 'num_params', 'trainable'],
    verbose=1,
)
total_params = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
print(f'\\nTotal Trainable Parameters: {total_params:,}')
"""
cells[13]['outputs'] = []

# ─── Cell 14 — BLOCK 8: Train S1 — add checkpoint + class_weights ────────────
cells[14]['source'] = """\
# BLOCK 8: Train — Scenario 1 (No-Aug)
print('=== SCENARIO 1: Training without Augmentation ===')
history_s1 = run_training(
    model_s1, train_loader_s1, val_loader_s1,
    epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY,
    checkpoint_path=f'{OUTPUT_DIR}/best_model_s1.pth',
    class_weights=CLASS_WEIGHTS,
)
"""
cells[14]['outputs'] = []

# ─── Cell 19 — BLOCK 12: S2 Transforms — num_workers=4 ──────────────────────
orig_block12 = ''.join(cells[19]['source'])
orig_block12 = orig_block12.replace('num_workers=2', 'num_workers=4')
cells[19]['source'] = orig_block12
cells[19]['outputs'] = []

# ─── Cell 20 — BLOCK 13: Re-initialize Model S2 — empty_cache + DataParallel ─
cells[20]['source'] = """\
# BLOCK 13: Re-initialize Model — Scenario 2 (prevent data leakage)
torch.cuda.empty_cache()
print('[GPU] Cache cleared between scenarios.')

model_s2 = GrapeLeafNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
if n_gpus > 1:
    model_s2 = nn.DataParallel(model_s2)
    print(f'[Multi-GPU] DataParallel on {n_gpus} GPUs')

total_params_s2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
print(f'Model re-initialized. Total Trainable Parameters: {total_params_s2:,}')
"""
cells[20]['outputs'] = []

# ─── Cell 21 — BLOCK 14: Train S2 — add checkpoint + class_weights ───────────
cells[21]['source'] = """\
# BLOCK 14: Train — Scenario 2 (Aug)
print('=== SCENARIO 2: Training WITH Augmentation ===')
history_s2 = run_training(
    model_s2, train_loader_s2, val_loader_s2,
    epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY,
    checkpoint_path=f'{OUTPUT_DIR}/best_model_s2.pth',
    class_weights=CLASS_WEIGHTS,
)
"""
cells[21]['outputs'] = []

# Write back
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done — notebook updated successfully.")
