"""Fix ViT notebook: seed, cudnn, DataParallel, class imbalance, checkpoint, num_workers=4, empty_cache."""

import json
from pathlib import Path

NB_PATH = Path(r'E:\UIT\study\claude\learn-claude\papers\15 - Vision Transformer (ViT)\outputs\vision-transformer-vit-plant-disease-detection.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def find_cell(nb, cell_id):
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == cell_id:
            return i, cell
    return None, None

def set_src(cell, lines):
    cell['source'] = lines
    cell['outputs'] = []
    cell['execution_count'] = None

def make_code_cell(cell_id, lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# ── 1. cell-imports: add n_gpus, seed, cudnn ──────────────────────────────────
_, cell = find_cell(nb, 'cell-imports')
set_src(cell, [
    "# BLOCK 1: Imports\n",
    "import os, math, numpy as np, matplotlib.pyplot as plt\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torchvision import transforms\n",
    "from torchinfo import summary as torch_summary\n",
    "from PIL import Image\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, f1_score, recall_score,\n",
    "    confusion_matrix, ConfusionMatrixDisplay,\n",
    "    classification_report\n",
    ")\n",
    "\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "n_gpus = torch.cuda.device_count()\n",
    "print(f'Device: {DEVICE} | GPUs available: {n_gpus}')\n",
    "\n",
    "torch.manual_seed(42)\n",
    "torch.cuda.manual_seed_all(42)\n",
    "torch.backends.cudnn.deterministic = True\n",
    "torch.backends.cudnn.benchmark     = False\n",
    "print('Seed=42 | cudnn.deterministic=True | benchmark=False')",
])

# ── 2. cell-explore: append class imbalance detection ─────────────────────────
_, cell = find_cell(nb, 'cell-explore')
addition = [
    "\n",
    "\n",
    "# Class imbalance auto-detection (Rule 11)\n",
    "train_counts_arr = np.array([train_counts[c] for c in CLASS_NAMES])\n",
    "max_count = int(train_counts_arr.max())\n",
    "min_count = int(train_counts_arr.min())\n",
    "imbalance_ratio = max_count / min_count\n",
    "print(f'Class imbalance ratio (max/min): {imbalance_ratio:.2f}')\n",
    "if imbalance_ratio >= 1.5:\n",
    "    from sklearn.utils.class_weight import compute_class_weight\n",
    "    all_labels = []\n",
    "    for cls_idx, cls in enumerate(CLASS_NAMES):\n",
    "        all_labels.extend([cls_idx] * train_counts[cls])\n",
    "    CLASS_WEIGHTS = compute_class_weight(\n",
    "        'balanced', classes=np.arange(NUM_CLASSES), y=np.array(all_labels)\n",
    "    )\n",
    "    CLASS_WEIGHTS_TENSOR = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)\n",
    "    print(f'[Class Weighting] ENABLED — ratio={imbalance_ratio:.2f} >= 1.5')\n",
    "    print(f'  weights: {CLASS_WEIGHTS.round(3)}')\n",
    "else:\n",
    "    CLASS_WEIGHTS_TENSOR = None\n",
    "    print(f'[Class Weighting] DISABLED — balanced (ratio={imbalance_ratio:.2f} < 1.5)')",
]
existing = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
cell['source'] = existing + addition
cell['outputs'] = []
cell['execution_count'] = None

# ── 3. cell-model: fix norm1×3, DataParallel, summary with .module ────────────
_, cell = find_cell(nb, 'cell-model')
set_src(cell, [
    "# BLOCK 5: ViT Architecture (from scratch)\n",
    "\n",
    "class PatchEmbedding(nn.Module):\n",
    "    def __init__(self, img_size, patch_size, in_channels, embed_dim):\n",
    "        super().__init__()\n",
    "        self.num_patches = (img_size // patch_size) ** 2\n",
    "        self.proj = nn.Conv2d(in_channels, embed_dim,\n",
    "                              kernel_size=patch_size, stride=patch_size)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.proj(x)       # (B, embed_dim, H/P, W/P)\n",
    "        x = x.flatten(2)       # (B, embed_dim, num_patches)\n",
    "        x = x.transpose(1, 2) # (B, num_patches, embed_dim)\n",
    "        return x\n",
    "\n",
    "\n",
    "class TransformerEncoderBlock(nn.Module):\n",
    "    def __init__(self, embed_dim, num_heads, ffn_dim):\n",
    "        super().__init__()\n",
    "        self.norm1 = nn.LayerNorm(embed_dim)\n",
    "        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)\n",
    "        self.norm2 = nn.LayerNorm(embed_dim)\n",
    "        self.ffn   = nn.Sequential(\n",
    "            nn.Linear(embed_dim, ffn_dim),\n",
    "            nn.GELU(),\n",
    "            nn.Linear(ffn_dim, embed_dim),\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        normed = self.norm1(x)\n",
    "        attn_out, _ = self.attn(normed, normed, normed)\n",
    "        x = x + attn_out\n",
    "        x = x + self.ffn(self.norm2(x))\n",
    "        return x\n",
    "\n",
    "\n",
    "class ViT(nn.Module):\n",
    "    def __init__(self, img_size, patch_size, in_channels, embed_dim,\n",
    "                 num_heads, num_layers, ffn_dim, num_classes):\n",
    "        super().__init__()\n",
    "        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)\n",
    "        num_patches = (img_size // patch_size) ** 2\n",
    "        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)\n",
    "        self.encoder   = nn.Sequential(\n",
    "            *[TransformerEncoderBlock(embed_dim, num_heads, ffn_dim)\n",
    "              for _ in range(num_layers)]\n",
    "        )\n",
    "        flat_dim = num_patches * embed_dim  # 256 * 512 = 131072\n",
    "        self.head = nn.Sequential(\n",
    "            nn.Flatten(),\n",
    "            nn.Linear(flat_dim, 256), nn.ReLU(),\n",
    "            nn.Linear(256, 2048),     nn.ReLU(),\n",
    "            nn.Linear(2048, 1024),    nn.ReLU(),\n",
    "            nn.Linear(1024, 512),     nn.ReLU(),\n",
    "            nn.Linear(512, 256),      nn.ReLU(),\n",
    "            nn.Linear(256, 128),      nn.ReLU(),\n",
    "            nn.Linear(128, 64),       nn.ReLU(),\n",
    "            nn.Linear(64, 32),        nn.ReLU(),\n",
    "            nn.Linear(32, num_classes),\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.patch_embed(x) + self.pos_embed\n",
    "        x = self.encoder(x)\n",
    "        return self.head(x)\n",
    "\n",
    "\n",
    "def build_model():\n",
    "    model = ViT(\n",
    "        img_size=IMG_SIZE[0], patch_size=PATCH_SIZE,\n",
    "        in_channels=3, embed_dim=EMBED_DIM,\n",
    "        num_heads=NUM_HEADS, num_layers=NUM_LAYERS,\n",
    "        ffn_dim=FFN_DIM, num_classes=NUM_CLASSES\n",
    "    ).to(DEVICE)\n",
    "    if n_gpus > 1:\n",
    "        model = nn.DataParallel(model)\n",
    "        print(f'[Multi-GPU] DataParallel on {n_gpus} GPUs')\n",
    "    return model\n",
    "\n",
    "\n",
    "# Print model summary (dùng .module khi DataParallel — Rule 14)\n",
    "_tmp = build_model()\n",
    "_m = _tmp.module if isinstance(_tmp, nn.DataParallel) else _tmp\n",
    "torch_summary(_m, input_size=(1, 3, *IMG_SIZE),\n",
    "              col_names=['output_size', 'num_params', 'trainable'])\n",
    "del _tmp",
])

# ── 4. cell-eval-fn: train_model with checkpoint, load_best_model ─────────────
_, cell = find_cell(nb, 'cell-eval-fn')
set_src(cell, [
    "# BLOCK 6: Evaluation & Artifact Functions\n",
    "\n",
    "def evaluate(model, loader):\n",
    "    model.eval()\n",
    "    y_true, y_pred = [], []\n",
    "    with torch.no_grad():\n",
    "        for imgs, labels in loader:\n",
    "            imgs = imgs.to(DEVICE)\n",
    "            preds = model(imgs).argmax(dim=1).cpu().numpy()\n",
    "            y_pred.extend(preds)\n",
    "            y_true.extend(labels.numpy())\n",
    "    return np.array(y_true), np.array(y_pred)\n",
    "\n",
    "\n",
    "def load_best_model(model, checkpoint_path):\n",
    "    _target = model.module if isinstance(model, nn.DataParallel) else model\n",
    "    _target.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))\n",
    "    print(f'[Checkpoint] Loaded best weights from {checkpoint_path}')\n",
    "    return model\n",
    "\n",
    "\n",
    "def train_model(model, train_loader, val_loader, epochs, checkpoint_path,\n",
    "                class_weights_tensor=None):\n",
    "    weight    = class_weights_tensor.to(DEVICE) if class_weights_tensor is not None else None\n",
    "    criterion = nn.CrossEntropyLoss(weight=weight)\n",
    "    optimizer = optim.Adam(model.parameters(), lr=LR)\n",
    "    history   = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}\n",
    "    best_val_acc = 0.0\n",
    "\n",
    "    for epoch in range(1, epochs + 1):\n",
    "        model.train()\n",
    "        running_loss, correct, total = 0.0, 0, 0\n",
    "        for imgs, labels in train_loader:\n",
    "            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n",
    "            optimizer.zero_grad()\n",
    "            out  = model(imgs)\n",
    "            loss = criterion(out, labels)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            running_loss += loss.item() * imgs.size(0)\n",
    "            correct      += (out.argmax(1) == labels).sum().item()\n",
    "            total        += imgs.size(0)\n",
    "\n",
    "        train_loss = running_loss / total\n",
    "        train_acc  = correct / total\n",
    "\n",
    "        model.eval()\n",
    "        val_loss, val_correct, val_total = 0.0, 0, 0\n",
    "        with torch.no_grad():\n",
    "            for imgs, labels in val_loader:\n",
    "                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n",
    "                out  = model(imgs)\n",
    "                loss = criterion(out, labels)\n",
    "                val_loss    += loss.item() * imgs.size(0)\n",
    "                val_correct += (out.argmax(1) == labels).sum().item()\n",
    "                val_total   += imgs.size(0)\n",
    "\n",
    "        val_loss /= val_total\n",
    "        val_acc   = val_correct / val_total\n",
    "\n",
    "        star = ''\n",
    "        if val_acc > best_val_acc:\n",
    "            best_val_acc = val_acc\n",
    "            _state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()\n",
    "            torch.save(_state, checkpoint_path)\n",
    "            star = ' ★ saved'\n",
    "\n",
    "        history['train_loss'].append(train_loss)\n",
    "        history['val_loss'].append(val_loss)\n",
    "        history['train_acc'].append(train_acc)\n",
    "        history['val_acc'].append(val_acc)\n",
    "        print(f'Epoch {epoch:02d}/{epochs} | '\n",
    "              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '\n",
    "              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{star}')\n",
    "\n",
    "    print(f'Best val_acc: {best_val_acc:.4f} → {checkpoint_path}')\n",
    "    return history\n",
    "\n",
    "\n",
    "def save_learning_curve(history, title, save_path):\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "    axes[0].plot(history['train_acc'], label='Train')\n",
    "    axes[0].plot(history['val_acc'],   label='Val')\n",
    "    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')\n",
    "    axes[0].legend(); axes[0].grid(True)\n",
    "\n",
    "    axes[1].plot(history['train_loss'], label='Train')\n",
    "    axes[1].plot(history['val_loss'],   label='Val')\n",
    "    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')\n",
    "    axes[1].legend(); axes[1].grid(True)\n",
    "\n",
    "    plt.suptitle(title, fontsize=15, fontweight='bold')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(save_path, dpi=300, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "def save_confusion_matrix(y_true, y_pred, title, save_path):\n",
    "    cm   = confusion_matrix(y_true, y_pred)\n",
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)\n",
    "    fig, ax = plt.subplots(figsize=(14, 12))\n",
    "    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)\n",
    "    ax.set_title(title, fontsize=16, fontweight='bold')\n",
    "    ax.set_xlabel('Predicted Label', fontsize=14)\n",
    "    ax.set_ylabel('True Label', fontsize=14)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(save_path, dpi=300, bbox_inches='tight')\n",
    "    plt.show()",
])

# ── 5. cell-s1-transform: num_workers 2 → 4 ──────────────────────────────────
_, cell = find_cell(nb, 'cell-s1-transform')
src = ''.join(cell['source']).replace('num_workers=2', 'num_workers=4')
cell['source'] = [src]
cell['outputs'] = []
cell['execution_count'] = None

# ── 6. cell-s1-train: checkpoint + reload best ────────────────────────────────
_, cell = find_cell(nb, 'cell-s1-train')
set_src(cell, [
    "# BLOCK 8: Train — Scenario 1\n",
    "CKPT_S1   = f'{OUTPUT_DIR}/best_model_s1.pth'\n",
    "model_1   = build_model()\n",
    "history_1 = train_model(model_1, train_loader_1, val_loader_1, EPOCHS, CKPT_S1, CLASS_WEIGHTS_TENSOR)\n",
    "model_1   = load_best_model(model_1, CKPT_S1)  # reload best before evaluate",
])

# ── 7. Insert cleanup cell between cell-s1-lc and cell-s2-header ──────────────
s2_header_idx, _ = find_cell(nb, 'cell-s2-header')
nb['cells'].insert(s2_header_idx, make_code_cell('cell-s1-cleanup', [
    "# Cleanup Scenario 1 — free GPU memory before Scenario 2 (Rule 13)\n",
    "del train_loader_1, val_loader_1\n",
    "torch.cuda.empty_cache()\n",
    "print('[GPU] Cache cleared after Scenario 1')",
]))

# ── 8. cell-s2-transform: num_workers 2 → 4 ──────────────────────────────────
_, cell = find_cell(nb, 'cell-s2-transform')
src = ''.join(cell['source']).replace('num_workers=2', 'num_workers=4')
cell['source'] = [src]
cell['outputs'] = []
cell['execution_count'] = None

# ── 9. cell-s2-train: checkpoint + reload best ────────────────────────────────
_, cell = find_cell(nb, 'cell-s2-train')
set_src(cell, [
    "# BLOCK 13: Train — Scenario 2 (model khởi tạo lại để tránh data leakage)\n",
    "CKPT_S2   = f'{OUTPUT_DIR}/best_model_s2.pth'\n",
    "model_2   = build_model()\n",
    "history_2 = train_model(model_2, train_loader_2, val_loader_2, EPOCHS, CKPT_S2, CLASS_WEIGHTS_TENSOR)\n",
    "model_2   = load_best_model(model_2, CKPT_S2)  # reload best before evaluate",
])

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Done! All 9 fixes applied.')
print('  [1] cell-imports : seed=42, cudnn.deterministic, n_gpus')
print('  [2] cell-explore : class imbalance auto-detection + class_weights')
print('  [3] cell-model   : norm1 fix, DataParallel, summary with .module')
print('  [4] cell-eval-fn : train_model with checkpoint, load_best_model')
print('  [5] cell-s1-transform: num_workers=4')
print('  [6] cell-s1-train: CKPT_S1, reload best')
print('  [7] new cell-s1-cleanup: empty_cache between scenarios')
print('  [8] cell-s2-transform: num_workers=4')
print('  [9] cell-s2-train: CKPT_S2, reload best')
