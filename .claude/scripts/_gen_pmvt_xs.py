"""One-off generator for PMVT-XS Kaggle notebook. Writes .ipynb JSON directly (no nbformat)."""
import json, os

cells = []
def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})
def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.rstrip("\n").splitlines(keepends=True)})

# ───────────────────────── TITLE ─────────────────────────
md("""# PMVT-XS — Plant Disease Experiment
**Paper:** PMVT: a lightweight vision transformer for plant disease identification on mobile devices (2023)
**DOI:** 10.3389/fpls.2023.1256773
**Variant:** PMVT-XS (~2.01M params, paper Table 3)
**Framework:** PyTorch

| Scenario | Augmentation |
|----------|--------------|
| Baseline | None (Resize + Normalize only) |
| Augmented | Rotation + Zoom/Scale + HFlip + ColorJitter |

**Config xác nhận (PMVT-XS):** expand=2 · MobileViT L=[4,4,3] · num_heads=1, head_dim=dim · mlp_dim=2×dim · patch=2 · CBAM r=16 (spatial conv 7×7)
""")

code("""# ── Cell: Install dependencies ─────────────────────────────────────────────
!pip install -q torchinfo openpyxl Pillow""")

code('''# ── Cell: Imports + Seed + Device ─────────────────────────────────────────
import os, math, time
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Using device: {device}")''')

code('''# ── Cell: Config ───────────────────────────────────────────────────────────
# ← ĐIỀN tên dataset Kaggle của bạn vào đây
DATASET_NAME = "your-dataset-name"
DATASET_PATH = f"/kaggle/input/{DATASET_NAME}"
TRAIN_DIR    = os.path.join(DATASET_PATH, "train")
VAL_DIR      = os.path.join(DATASET_PATH, "val")
TEST_DIR     = os.path.join(DATASET_PATH, "test")

MODEL_NAME  = "PMVT_XS"
OUTPUT_DIR  = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters — từ paper (Table 2)
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
LR           = 1e-4    # paper Table 2
WEIGHT_DECAY = 0.01    # AdamW default (paper không nêu)

# Normalization — ImageNet (xác nhận user; paper không nêu giá trị cụ thể)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Auto-detect số classes từ TRAIN_DIR (Rule 12 — override lại ở cell DataLoader)
NUM_CLASSES = len([d for d in os.listdir(TRAIN_DIR)
                   if os.path.isdir(os.path.join(TRAIN_DIR, d))])
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print(f"NUM_CLASSES = {NUM_CLASSES}")
print(f"CLASS_NAMES = {CLASS_NAMES}")''')

code('''# ── Cell: Dataset Distribution + Class Imbalance Check (Rule 11) ───────────
splits = {"Train": TRAIN_DIR, "Val": VAL_DIR, "Test": TEST_DIR}
counts = {}
for split, path in splits.items():
    c = {}
    for cls in CLASS_NAMES:
        cls_path = os.path.join(path, cls)
        c[cls] = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
    counts[split] = c
    total = sum(c.values())
    print(f"{split}: {total} images")
    for k, v in c.items():
        print(f"  {k}: {v}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))
for ax, (split, c) in zip(axes, counts.items()):
    vals = list(c.values())
    bars = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_title(f"{split} Set", fontsize=13, fontweight='bold')
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                str(v), ha='center', va='bottom', fontsize=9)
plt.suptitle("Dataset Distribution", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dataset_statistics.png", dpi=300, bbox_inches='tight')
plt.show()

train_counts_list = list(counts["Train"].values())
max_count = max(train_counts_list)
min_count = min(x for x in train_counts_list if x > 0)
imbalance_ratio = max_count / min_count
print(f"\\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
if imbalance_ratio >= 1.5:
    from sklearn.utils.class_weight import compute_class_weight
    all_labels = []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        cls_path = os.path.join(TRAIN_DIR, cls)
        n = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
        all_labels.extend([cls_idx] * n)
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=np.array(all_labels))
    CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float32)
    print(f"Imbalanced (ratio={imbalance_ratio:.2f}) -> dùng class weights: {np.round(cw,3)}")
else:
    CLASS_WEIGHTS = None
    print(f"Balanced (ratio={imbalance_ratio:.2f}) -> không cần class weighting")''')

# ───────────────────────── ARCHITECTURE ─────────────────────────
code('''# ── Cell: PMVT-XS Architecture ─────────────────────────────────────────────
# Phục dựng từ: PMVT paper (10.3389/fpls.2023.1256773), biến thể XS.
# Khác biệt cốt lõi so với MobileViT-XS gốc (2 thay đổi PMVT mô tả rõ trong paper):
#   1. InvertedResidual: depthwise 3×3 -> 7×7 (Mục 2.2.2, Fig 3)
#   2. MobileViTBlock: chèn CBAM SAU 1×1 proj, TRƯỚC skip-cat (Mục 2.2.3, Fig 4A)
#
# Config PMVT-XS (xác nhận user):
#   expand=2 (toàn bộ InvertedResidual) · MobileViT L=[4,4,3] · patch=2
#   num_heads=1, head_dim=dim (inner_dim = dim) · mlp_dim = 2×dim · CBAM r=16

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class InvertedResidual(nn.Module):
    """MV2 block với depthwise 7×7 (PMVT Mục 2.2.2). expand=2 cho PMVT-XS."""
    def __init__(self, in_ch, out_ch, stride=1, expand=2):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel=1))        # 1×1 expand
        layers += [
            ConvBNAct(mid_ch, mid_ch, kernel=7, stride=stride, padding=3, groups=mid_ch),  # DW 7×7
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),                # 1×1 project
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class ChannelAttention(nn.Module):
    """CBAM channel attention (PMVT Mục 2.2.5, Eq.2). r=16."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * w


class SpatialAttention(nn.Module):
    """CBAM spatial attention (PMVT Mục 2.2.5, Eq.3). Conv 7×7."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel -> spatial)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))


class TransformerBlock(nn.Module):
    """Standard ViT encoder block (PMVT Mục 2.2.4, Eq.1).
    PMVT-XS: num_heads=1, head_dim=dim => inner_dim=dim. mlp_dim=2×dim."""
    def __init__(self, dim, heads=1):
        super().__init__()
        head_dim  = dim          # head_dim = dim (num_heads=1)
        inner_dim = heads * head_dim
        mlp_dim   = dim * 2
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, dim),
        )
    def forward(self, x):
        h = self.norm1(x)
        B, N, _ = h.shape
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, self.heads, -1).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = x + self.to_out(out)
        x = x + self.ff(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """MobileViT block + CBAM (PMVT Mục 2.2.3, Fig 4A).
    in_ch có thể != out_ch (fusion conv ở cuối điều chỉnh channel)."""
    def __init__(self, in_ch, out_ch, dim, depth, patch_size=2):
        super().__init__()
        self.ph = self.pw = patch_size
        # Local representation: Conv 3×3 + Conv 1×1 (in_ch -> dim)
        self.local_rep = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel=3),
            nn.Conv2d(in_ch, dim, 1, bias=False),
        )
        # Global representation: L transformer blocks (num_heads=1, mlp=2×dim)
        self.transformer = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # Project back dim -> in_ch, then CBAM, then concat với input, fusion -> out_ch
        self.proj = nn.Sequential(
            nn.Conv2d(dim, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.cbam   = CBAM(in_ch, reduction=16)
        self.fusion = ConvBNAct(in_ch * 2, out_ch, kernel=3)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        nph, npw = H // ph, W // pw

        y = self.local_rep(x)            # B, dim, H, W
        dim = y.shape[1]

        # Unfold thành các patch (gom pixel cùng vị trí trong mỗi patch)
        y = y.reshape(B, dim, nph, ph, npw, pw)
        y = y.permute(0, 3, 5, 2, 4, 1).contiguous().reshape(B * ph * pw, nph * npw, dim)

        for blk in self.transformer:
            y = blk(y)
        y = self.norm(y)

        # Fold lại
        y = y.reshape(B, ph, pw, nph, npw, dim)
        y = y.permute(0, 5, 3, 1, 4, 2).contiguous().reshape(B, dim, H, W)

        # Project -> CBAM -> concat với input -> fusion
        y = self.proj(y)                 # B, in_ch, H, W
        y = self.cbam(y)                 # CBAM SAU proj, TRƯỚC skip-cat (Fig 4A)
        return self.fusion(torch.cat([x, y], dim=1))   # -> out_ch


class PMVT_XS(nn.Module):
    """
    PMVT-XS — Plant-based MobileViT (Li et al. 2023, Frontiers in Plant Science).
    Target ~2.01M params. Channel/dim config xác nhận user (tham chiếu MobileViT-XS gốc).
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem: 3 -> 32, kernel 3, stride 1 (KHÔNG downsample)
        self.stem = ConvBNAct(3, 32, kernel=3, stride=1)
        # S1: 1× MV2 (32->32, stride 1, expand 2)
        self.s1 = InvertedResidual(32, 32, stride=1, expand=2)
        # S2: 3× MV2 (32->48 stride 2, rồi 48->48 ×2), expand 2
        self.s2 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expand=2),
            InvertedResidual(48, 48, stride=1, expand=2),
            InvertedResidual(48, 48, stride=1, expand=2),
        )
        # S3: MV2 (48->64, stride 2) + MobileViTBlock(in=64, out=80, dim=96, L=4)
        self.s3 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expand=2),
            MobileViTBlock(64, 80, dim=96, depth=4, patch_size=2),
        )
        # S4: MV2 (80->96, stride 2) + MobileViTBlock(in=96, out=96, dim=120, L=4)
        self.s4 = nn.Sequential(
            InvertedResidual(80, 96, stride=2, expand=2),
            MobileViTBlock(96, 96, dim=120, depth=4, patch_size=2),
        )
        # S5: MV2 (96->128, stride 2) + MobileViTBlock(in=128, out=128, dim=144, L=3)
        self.s5 = nn.Sequential(
            InvertedResidual(96, 128, stride=2, expand=2),
            MobileViTBlock(128, 128, dim=144, depth=3, patch_size=2),
        )
        # Head: Conv 1×1 (128->384) + BN + SiLU + GAP + Flatten + FC
        self.head = nn.Sequential(
            nn.Conv2d(128, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        return self.head(x)


def build_model(num_classes):
    return PMVT_XS(num_classes).to(device)


print("PMVT-XS architecture defined.")''')

# ───────────────────────── SHARED TRAINING FUNCS ─────────────────────────
code('''# ── Cell: Shared Training Functions ───────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_targets


def train_with_checkpoint(model, train_loader, val_loader, num_epochs, lr, weight_decay,
                          checkpoint_path, class_weights=None):
    """Train, lưu best checkpoint theo val_accuracy, reload trước khi return (Rule 10)."""
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), checkpoint_path)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]  "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}  |  "
                  f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}  ({time.time()-t0:.1f}s)")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Best checkpoint loaded (val_acc={best_val_acc:.4f}) <- {checkpoint_path}")
    return history


def plot_confusion_matrix(targets, preds, class_names, title, save_path):
    cm   = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(8, NUM_CLASSES), max(7, NUM_CLASSES - 1)))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curve(history, title, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_acc'], label='Train Acc', color='steelblue')
    axes[0].plot(epochs, history['val_acc'],   label='Val Acc',   color='darkorange')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
    axes[1].plot(epochs, history['val_loss'],   label='Val Loss',   color='darkorange')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


print("Training functions defined.")''')

# ───────────────────────── SCENARIO 1 ─────────────────────────
md("""---
## SCENARIO 1: BASELINE (No Augmentation)
Transform: **Resize → ToTensor → Normalize only**""")

code('''# ── Cell: Baseline Transforms ─────────────────────────────────────────────
# Scenario 1: KHÔNG augmentation trong training (Rule 3)
train_transform_baseline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])
print("Baseline transforms ready.")''')

code('''# ── Cell: Baseline DataLoaders ─────────────────────────────────────────────
train_ds_base = datasets.ImageFolder(TRAIN_DIR, transform=train_transform_baseline)
val_ds        = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
test_ds       = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

# Rule 12: class_names lấy từ ImageFolder (nguồn chính xác nhất)
CLASS_NAMES = train_ds_base.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"CLASS_NAMES (from ImageFolder): {CLASS_NAMES}")
print(f"NUM_CLASSES: {NUM_CLASSES}")

train_loader_base = DataLoader(train_ds_base, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
val_loader        = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)
test_loader       = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)
print(f"Train: {len(train_ds_base)} | Val: {len(val_ds)} | Test: {len(test_ds)}")''')

code('''# ── Cell: Build Model + Summary (Scenario 1) ───────────────────────────────
model_base = build_model(NUM_CLASSES)
torch_summary(
    model_base,
    input_size=(1, 3, IMG_SIZE, IMG_SIZE),
    col_names=["output_size", "num_params", "trainable"],
    row_settings=["var_names"],
)
TOTAL_PARAMS = sum(p.numel() for p in model_base.parameters())
print(f"\\nTotal parameters: {TOTAL_PARAMS:,}  (target PMVT-XS ~2.01M)")''')

code('''# ── Cell: Train Scenario 1 (Baseline) ─────────────────────────────────────
print("="*60)
print("  TRAINING — SCENARIO 1: BASELINE (No Augmentation)")
print("="*60)

CKPT_S1 = f"{OUTPUT_DIR}/best_model_s1.pth"
history_base = train_with_checkpoint(
    model_base, train_loader_base, val_loader,
    num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
    checkpoint_path=CKPT_S1, class_weights=CLASS_WEIGHTS,
)

criterion_eval = nn.CrossEntropyLoss()
_, test_acc_base, test_preds_base, test_targets_base = evaluate(model_base, test_loader, criterion_eval)
f1_base     = f1_score(test_targets_base, test_preds_base, average='macro')
recall_base = recall_score(test_targets_base, test_preds_base, average='macro')

BASELINE_LEARNING_CURVE_PATH   = f"{OUTPUT_DIR}/{MODEL_NAME}_NoAug_LC.png"
BASELINE_CONFUSION_MATRIX_PATH = f"{OUTPUT_DIR}/{MODEL_NAME}_NoAug_CM.png"

baseline_metrics = {
    "model_name"   : MODEL_NAME,
    "accuracy"     : test_acc_base,
    "f1_macro"     : f1_base,
    "recall_macro" : recall_base,
    "total_params" : TOTAL_PARAMS,
}
print(f"\\n[Test] Accuracy={test_acc_base:.4f} | F1={f1_base:.4f} | Recall={recall_base:.4f}")''')

code('''# ── Cell: Classification Report — Scenario 1 ──────────────────────────────
print("Classification Report — PMVT-XS (Baseline, No Augmentation)")
print(classification_report(
    test_targets_base, test_preds_base,
    target_names=CLASS_NAMES, digits=4
))''')

code('''# ── Cell: Confusion Matrix — Scenario 1 ───────────────────────────────────
plot_confusion_matrix(
    test_targets_base, test_preds_base, CLASS_NAMES,
    title=f"Confusion Matrix — {MODEL_NAME} (No-Aug)",
    save_path=BASELINE_CONFUSION_MATRIX_PATH,
)''')

code('''# ── Cell: Learning Curve — Scenario 1 ─────────────────────────────────────
plot_learning_curve(
    history_base,
    title=f"Learning Curve — {MODEL_NAME} (No-Aug)",
    save_path=BASELINE_LEARNING_CURVE_PATH,
)''')

# ───────────────────────── SCENARIO 2 ─────────────────────────
md("""---
## SCENARIO 2: AUGMENTED
Transform: **Rotation + Zoom/Scale + HFlip + ColorJitter → ToTensor → Normalize**
*(Val/Test transform không thay đổi — giống Baseline)*""")

code('''# ── Cell: Augmented Transforms ─────────────────────────────────────────────
# 4 kỹ thuật augmentation bắt buộc (Rule 3)
train_transform_aug = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # Zoom/Scale
    transforms.RandomHorizontalFlip(p=0.5),                     # Horizontal Flip
    transforms.RandomRotation(degrees=15),                      # Rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3,        # ColorJitter/Brightness
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])
# Val/Test giữ nguyên baseline transform (không aug)
print("Augmented transforms ready.")''')

code('''# ── Cell: Augmented DataLoaders ────────────────────────────────────────────
train_ds_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_transform_aug)
train_loader_aug = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
print(f"Aug Train: {len(train_ds_aug)} | Val: {len(val_ds)} | Test: {len(test_ds)}")''')

code('''# ── Cell: Reinitialize Model (Scenario 2) ─────────────────────────────────
# Rule 13: giải phóng VRAM trước Scenario 2 để tránh OOM
torch.cuda.empty_cache()

# Khởi tạo lại model từ cùng seed để so sánh công bằng
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model_aug = build_model(NUM_CLASSES)
torch_summary(
    model_aug,
    input_size=(1, 3, IMG_SIZE, IMG_SIZE),
    col_names=["output_size", "num_params", "trainable"],
    row_settings=["var_names"],
)
print(f"\\nTotal parameters: {TOTAL_PARAMS:,}")''')

code('''# ── Cell: Train Scenario 2 (Augmented) ────────────────────────────────────
print("="*60)
print("  TRAINING — SCENARIO 2: AUGMENTED")
print("="*60)

CKPT_S2 = f"{OUTPUT_DIR}/best_model_s2.pth"
history_aug = train_with_checkpoint(
    model_aug, train_loader_aug, val_loader,
    num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
    checkpoint_path=CKPT_S2, class_weights=CLASS_WEIGHTS,
)

criterion_eval = nn.CrossEntropyLoss()
_, test_acc_aug, test_preds_aug, test_targets_aug = evaluate(model_aug, test_loader, criterion_eval)
f1_aug     = f1_score(test_targets_aug, test_preds_aug, average='macro')
recall_aug = recall_score(test_targets_aug, test_preds_aug, average='macro')

AUG_LEARNING_CURVE_PATH   = f"{OUTPUT_DIR}/{MODEL_NAME}_Aug_LC.png"
AUG_CONFUSION_MATRIX_PATH = f"{OUTPUT_DIR}/{MODEL_NAME}_Aug_CM.png"

aug_metrics = {
    "model_name"   : MODEL_NAME,
    "accuracy"     : test_acc_aug,
    "f1_macro"     : f1_aug,
    "recall_macro" : recall_aug,
    "total_params" : TOTAL_PARAMS,
}
print(f"\\n[Test] Accuracy={test_acc_aug:.4f} | F1={f1_aug:.4f} | Recall={recall_aug:.4f}")''')

code('''# ── Cell: Classification Report — Scenario 2 ──────────────────────────────
print("Classification Report — PMVT-XS (Augmented)")
print(classification_report(
    test_targets_aug, test_preds_aug,
    target_names=CLASS_NAMES, digits=4
))''')

code('''# ── Cell: Confusion Matrix — Scenario 2 ───────────────────────────────────
plot_confusion_matrix(
    test_targets_aug, test_preds_aug, CLASS_NAMES,
    title=f"Confusion Matrix — {MODEL_NAME} (Aug)",
    save_path=AUG_CONFUSION_MATRIX_PATH,
)''')

code('''# ── Cell: Learning Curve — Scenario 2 ─────────────────────────────────────
plot_learning_curve(
    history_aug,
    title=f"Learning Curve — {MODEL_NAME} (Aug)",
    save_path=AUG_LEARNING_CURVE_PATH,
)''')

# ───────────────────────── ABLATION ─────────────────────────
md("""---
## ABLATION STUDY (Paper Table 4)
4 variants để định lượng đóng góp của Conv 7×7 và CBAM:
1. **MobileViT-XS baseline** — DW 3×3, không CBAM
2. **+Conv7×7** — DW 7×7, không CBAM
3. **+CBAM** — DW 3×3, có CBAM
4. **PMVT-XS full** — DW 7×7 + CBAM

*Lưu ý: ablation train rút gọn (30 epochs/variant) để tiết kiệm thời gian — xu hướng ranking vẫn đúng, accuracy tuyệt đối có thể thấp hơn paper (100 epochs).*""")

code('''# ── Cell: Ablation helper blocks (3×3 DW + no-CBAM variants) ───────────────
class _MV2_3x3(nn.Module):
    """MV2 với depthwise 3×3 (MobileViT-XS gốc). expand=2 cho XS."""
    def __init__(self, in_ch, out_ch, stride=1, expand=2):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel=1))
        layers += [
            ConvBNAct(mid_ch, mid_ch, kernel=3, stride=stride, groups=mid_ch),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class _MobileViTBlock_NoCBAM(nn.Module):
    """MobileViT block KHÔNG CBAM (cấu hình XS: num_heads=1, mlp=2×dim)."""
    def __init__(self, in_ch, out_ch, dim, depth, patch_size=2):
        super().__init__()
        self.ph = self.pw = patch_size
        self.local_rep = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel=3),
            nn.Conv2d(in_ch, dim, 1, bias=False),
        )
        self.transformer = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm   = nn.LayerNorm(dim)
        self.proj   = nn.Sequential(nn.Conv2d(dim, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch))
        self.fusion = ConvBNAct(in_ch * 2, out_ch, kernel=3)
    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        nph, npw = H // ph, W // pw
        y = self.local_rep(x); dim = y.shape[1]
        y = y.reshape(B, dim, nph, ph, npw, pw)
        y = y.permute(0, 3, 5, 2, 4, 1).contiguous().reshape(B * ph * pw, nph * npw, dim)
        for blk in self.transformer: y = blk(y)
        y = self.norm(y)
        y = y.reshape(B, ph, pw, nph, npw, dim)
        y = y.permute(0, 5, 3, 1, 4, 2).contiguous().reshape(B, dim, H, W)
        y = self.proj(y)
        return self.fusion(torch.cat([x, y], dim=1))


def _build_variant(dw7, use_cbam, num_classes):
    """Lắp ráp 1 biến thể ablation. dw7=True dùng 7×7, use_cbam=True chèn CBAM."""
    MV2  = InvertedResidual if dw7 else _MV2_3x3
    MViT = MobileViTBlock   if use_cbam else _MobileViTBlock_NoCBAM
    class _Variant(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.stem = ConvBNAct(3, 32, kernel=3, stride=1)
            self.s1 = MV2(32, 32, stride=1, expand=2)
            self.s2 = nn.Sequential(
                MV2(32, 48, stride=2, expand=2),
                MV2(48, 48, stride=1, expand=2),
                MV2(48, 48, stride=1, expand=2),
            )
            self.s3 = nn.Sequential(MV2(48, 64, stride=2, expand=2),
                                    MViT(64, 80, dim=96, depth=4))
            self.s4 = nn.Sequential(MV2(80, 96, stride=2, expand=2),
                                    MViT(96, 96, dim=120, depth=4))
            self.s5 = nn.Sequential(MV2(96, 128, stride=2, expand=2),
                                    MViT(128, 128, dim=144, depth=3))
            self.head = nn.Sequential(
                nn.Conv2d(128, 384, 1, bias=False), nn.BatchNorm2d(384), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(384, nc),
            )
        def forward(self, x):
            for m in [self.stem, self.s1, self.s2, self.s3, self.s4, self.s5, self.head]:
                x = m(x)
            return x
    return _Variant(num_classes).to(device)


print("Ablation helper blocks defined.")''')

code('''# ── Cell: Run Ablation Study (4 variants × 30 epochs) ─────────────────────
ABLATION_EPOCHS = 30

def _quick_train_eval(model, tr_loader, te_loader, epochs=ABLATION_EPOCHS):
    crit = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device) if CLASS_WEIGHTS is not None else None)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(epochs):
        train_one_epoch(model, tr_loader, opt, crit)
    eval_crit = nn.CrossEntropyLoss()
    _, acc, preds, tgts = evaluate(model, te_loader, eval_crit)
    f1  = f1_score(tgts, preds, average='macro')
    rec = recall_score(tgts, preds, average='macro')
    prm = sum(p.numel() for p in model.parameters())
    return acc, f1, rec, prm

print(f"Running ablation (4 variants × {ABLATION_EPOCHS} epochs)...")
ablation_results = {}

variants = [
    ("MobileViT-XS", dict(dw7=False, use_cbam=False)),
    ("+Conv7x7",     dict(dw7=True,  use_cbam=False)),
    ("+CBAM",        dict(dw7=False, use_cbam=True)),
    ("PMVT-XS",      dict(dw7=True,  use_cbam=True)),
]
for name, cfg in variants:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    m = _build_variant(num_classes=NUM_CLASSES, **cfg)
    ablation_results[name] = _quick_train_eval(m, train_loader_base, test_loader)
    print(f"  {name:<14} done: Acc={ablation_results[name][0]:.4f} | "
          f"Params={ablation_results[name][3]:,}")
    del m

print("\\n" + "="*70)
print(f"{'Variant':<16} {'Accuracy':>10} {'F1 Macro':>10} {'Recall':>10} {'Params':>14}")
print("-"*70)
for name, (acc, f1, rec, prm) in ablation_results.items():
    print(f"{name:<16} {acc:>10.4f} {f1:>10.4f} {rec:>10.4f} {prm:>14,}")
print("="*70)''')

# ───────────────────────── INFERENCE BENCHMARK ─────────────────────────
md("""---
## INFERENCE BENCHMARK
Đo FPS và latency batch=1 của PMVT-XS (tham chiếu Table 3 paper).""")

code('''# ── Cell: Inference Benchmark (FPS / Latency, batch=1) ─────────────────────
model_bench = build_model(NUM_CLASSES)
model_bench.eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

with torch.no_grad():
    for _ in range(10):
        _ = model_bench(dummy)   # warmup

N_RUNS = 200
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_start = time.time()
with torch.no_grad():
    for _ in range(N_RUNS):
        _ = model_bench(dummy)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - t_start

fps     = N_RUNS / elapsed
latency = elapsed / N_RUNS * 1000

print(f"Inference Benchmark (batch=1, input={IMG_SIZE}x{IMG_SIZE})")
print(f"  Device  : {device}")
print(f"  FPS     : {fps:.1f} img/s")
print(f"  Latency : {latency:.2f} ms/img")
print(f"  Params  : {TOTAL_PARAMS:,} (target PMVT-XS ~2.01M)")''')

# ───────────────────────── EXCEL EXPORT ─────────────────────────
md("""---
## EXCEL EXPORT
Tạo `results_summary.xlsx` với 2 sheet (Baseline / Augmented) và ảnh LC + CM nhúng trực tiếp.""")

code("""# ── Cell: Install Excel dependencies ──────────────────────────────────────
!pip install -q openpyxl Pillow""")

code('''# ── Cell: Export results_summary.xlsx ─────────────────────────────────────
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter

EXCEL_OUTPUT_PATH = f"{OUTPUT_DIR}/results_summary.xlsx"

HEADERS    = ["STT", "Tên Model", "Accuracy", "F1-score", "Recall",
              "Learning Curve", "Confusion Matrix", "Parameters"]
COL_WIDTHS = [6, 35, 12, 12, 12, 38, 38, 18]
IMG_ROW_HEIGHT = 200

def _style_header(ws):
    fill   = PatternFill("solid", fgColor="1F4E79")
    font   = Font(bold=True, color="FFFFFF", size=11)
    border = Border(bottom=Side(style="medium", color="FFFFFF"),
                    right=Side(style="thin", color="FFFFFF"))
    for col, header in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill, cell.font, cell.border = fill, font, border
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def _write_row(ws, row_num, stt, metrics, lc_path, cm_path):
    data = [
        stt, metrics["model_name"],
        round(metrics["accuracy"], 4),
        round(metrics["f1_macro"], 4),
        round(metrics["recall_macro"], 4),
        "", "",
        f"{metrics['total_params']:,}",
    ]
    for col, val in enumerate(data, start=1):
        cell = ws.cell(row=row_num, column=col, value=val)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row_num].height = IMG_ROW_HEIGHT
    if os.path.exists(lc_path):
        img = XLImage(lc_path); img.width, img.height = 280, 190
        ws.add_image(img, f"F{row_num}")
    if os.path.exists(cm_path):
        img = XLImage(cm_path); img.width, img.height = 280, 190
        ws.add_image(img, f"G{row_num}")

def _build_sheet(wb, sheet_name, metrics, lc_path, cm_path):
    ws = wb.create_sheet(title=sheet_name)
    for col, width in enumerate(COL_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[1].height = 30
    _style_header(ws)
    _write_row(ws, row_num=2, stt=1, metrics=metrics, lc_path=lc_path, cm_path=cm_path)
    return ws

wb = Workbook(); wb.remove(wb.active)
_build_sheet(wb, "Baseline",  metrics=baseline_metrics,
             lc_path=BASELINE_LEARNING_CURVE_PATH, cm_path=BASELINE_CONFUSION_MATRIX_PATH)
_build_sheet(wb, "Augmented", metrics=aug_metrics,
             lc_path=AUG_LEARNING_CURVE_PATH, cm_path=AUG_CONFUSION_MATRIX_PATH)
wb.save(EXCEL_OUTPUT_PATH)

print(f"[Excel] Saved: {EXCEL_OUTPUT_PATH}")
print(f"  Baseline  -> Acc={baseline_metrics['accuracy']:.4f} | F1={baseline_metrics['f1_macro']:.4f} | Recall={baseline_metrics['recall_macro']:.4f} | Params={baseline_metrics['total_params']:,}")
print(f"  Augmented -> Acc={aug_metrics['accuracy']:.4f} | F1={aug_metrics['f1_macro']:.4f} | Recall={aug_metrics['recall_macro']:.4f} | Params={aug_metrics['total_params']:,}")''')

# ───────────────────────── WRITE NOTEBOOK ─────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = r"E:\UIT\study\claude\learn-claude\papers\06 - PMVT - CNN\outputs\PMVT_XS_Experiment.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Notebook written: {out_path}")
print(f"Total cells: {len(cells)}")
