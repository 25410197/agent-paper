"""Fix RTR_Lite_MobileNet notebook based on VerificationReport findings."""
import json, copy

NB_PATH = "papers/05 - RTR_Lite_MobileNet/outputs/RTR_Lite_MobileNet_Experiment.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }

cells = nb["cells"]

# ─────────────────────────────────────────────────────────────────────────────
# [2] Imports — add gc, manual_seed, cudnn
# ─────────────────────────────────────────────────────────────────────────────
cells[2]["source"] = [
    "# ── Imports ─────────────────────────────────────────────────────────────────\n",
    "import os, gc, math, warnings\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision\n",
    "import torchvision.transforms as T\n",
    "from torchvision.datasets import ImageFolder\n",
    "from torch.utils.data import DataLoader\n",
    "from torchinfo import summary as torchinfo_summary\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix,\n",
    "    f1_score, recall_score, accuracy_score,\n",
    ")\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Reproducibility\n",
    "torch.manual_seed(42)\n",
    "torch.cuda.manual_seed_all(42)\n",
    "torch.backends.cudnn.deterministic = True\n",
    "torch.backends.cudnn.benchmark = False\n",
    "print('Imports OK | seed=42 | cudnn.deterministic=True')",
]

# ─────────────────────────────────────────────────────────────────────────────
# [3] Configuration — add n_gpus + DataParallel info
# ─────────────────────────────────────────────────────────────────────────────
cells[3]["source"] = [
    "# ── Configuration ───────────────────────────────────────────────────────────\n",
    "\n",
    "# Dataset paths — UPDATE dataset name\n",
    "DATASET_PATH = '/kaggle/input/YOUR-DATASET-NAME'   # ← update\n",
    "TRAIN_DIR    = os.path.join(DATASET_PATH, 'train')\n",
    "VAL_DIR      = os.path.join(DATASET_PATH, 'val')\n",
    "TEST_DIR     = os.path.join(DATASET_PATH, 'test')\n",
    "\n",
    "# Hyperparameters (from paper)\n",
    "IMG_SIZE     = 224\n",
    "BATCH_SIZE   = 32\n",
    "EPOCHS       = 30\n",
    "LR           = 1e-3\n",
    "LR_STEP      = 5       # decay every 5 epochs\n",
    "LR_GAMMA     = 0.9     # multiply LR by 0.9\n",
    "DROPOUT      = 0.2\n",
    "PRETRAINED   = True    # use ImageNet1K_V1 weights\n",
    "\n",
    "MEAN = [0.485, 0.456, 0.406]\n",
    "STD  = [0.229, 0.224, 0.225]\n",
    "\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "n_gpus = torch.cuda.device_count()\n",
    "print(f'Device: {DEVICE} | GPUs available: {n_gpus}')",
]

# ─────────────────────────────────────────────────────────────────────────────
# [4] Dataset Exploration — fix class_names to use ImageFolder directly
#     Keep mostly the same; class_names will be overwritten in distribution cell
# ─────────────────────────────────────────────────────────────────────────────
# (no change needed to [4], class_names will be confirmed in distribution cell)

# ─────────────────────────────────────────────────────────────────────────────
# Insert new cell [5-new] after [4]: Dataset Distribution + class_weights
# ─────────────────────────────────────────────────────────────────────────────
dist_cell = code_cell([
    "# ── Dataset Distribution — imbalance check + auto class_weights ─────────────\n",
    "import numpy as np\n",
    "\n",
    "_tmp_ds = ImageFolder(TRAIN_DIR)\n",
    "class_names = _tmp_ds.classes          # authoritative order from ImageFolder\n",
    "NUM_CLASSES  = len(class_names)\n",
    "\n",
    "# Count images per class\n",
    "counts = np.zeros(NUM_CLASSES, dtype=int)\n",
    "for _, label in _tmp_ds.samples:\n",
    "    counts[label] += 1\n",
    "\n",
    "# Bar chart\n",
    "fig, ax = plt.subplots(figsize=(max(8, NUM_CLASSES * 0.8), 4))\n",
    "ax.bar(class_names, counts, color='steelblue', edgecolor='white')\n",
    "ax.set_title('Training Set — Class Distribution', fontweight='bold')\n",
    "ax.set_xlabel('Class'); ax.set_ylabel('# Images')\n",
    "plt.xticks(rotation=45, ha='right'); plt.tight_layout()\n",
    "plt.savefig('/kaggle/working/class_distribution.png', dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f'NUM_CLASSES = {NUM_CLASSES}')\n",
    "for n, c in zip(class_names, counts):\n",
    "    print(f'  {n}: {c}')\n",
    "\n",
    "# Imbalance check — Rule 11: if max/min >= 1.5, use class_weights\n",
    "ratio = counts.max() / counts.min()\n",
    "print(f'\\nImbalance ratio (max/min) = {ratio:.2f}')\n",
    "if ratio >= 1.5:\n",
    "    labels_array = np.array([lbl for _, lbl in _tmp_ds.samples])\n",
    "    cw = compute_class_weight('balanced',\n",
    "                               classes=np.arange(NUM_CLASSES),\n",
    "                               y=labels_array)\n",
    "    CLASS_WEIGHTS = torch.tensor(cw, dtype=torch.float32)\n",
    "    print(f'[Imbalanced] Class weights applied: {CLASS_WEIGHTS.numpy()}')\n",
    "else:\n",
    "    CLASS_WEIGHTS = None\n",
    "    print('[Balanced] No class weighting needed.')\n",
    "\n",
    "del _tmp_ds",
])

# Insert after cell index 4
cells.insert(5, dist_cell)
# Now original indices shift by 1: [5]→[6], [6]→[7], etc.

# ─────────────────────────────────────────────────────────────────────────────
# [6] Architecture (was [5]) — fix RES_Block + add res4/triplet3/res5
# ─────────────────────────────────────────────────────────────────────────────
cells[6]["source"] = [
    "# ── RTR_Lite_MobileNet Architecture ─────────────────────────────────────────\n",
    "# Ref: Duhan et al. (2025). Current Plant Biology.\n",
    "# Modified MobileNetV2 with Triplet Attention + RES_Block (SENet + ECA)\n",
    "\n",
    "# ─── Modified SENet ──────────────────────────────────────────────────────────\n",
    "class ModifiedSENet(nn.Module):\n",
    "    def __init__(self, channels, reduction=16):\n",
    "        super().__init__()\n",
    "        rd = max(channels // reduction, 4)\n",
    "        self.avg_pool = nn.AdaptiveAvgPool2d(1)\n",
    "        self.max_pool = nn.AdaptiveMaxPool2d(1)\n",
    "        self.fc = nn.Sequential(\n",
    "            nn.Linear(channels, rd, bias=False),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Linear(rd, channels, bias=False),\n",
    "            nn.Sigmoid(),\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        b, c = x.shape[:2]\n",
    "        avg = self.avg_pool(x).view(b, c)\n",
    "        mxp = self.max_pool(x).view(b, c)\n",
    "        w = self.fc(avg + mxp).view(b, c, 1, 1)\n",
    "        return x * w\n",
    "\n",
    "\n",
    "# ─── ECA Module ──────────────────────────────────────────────────────────────\n",
    "class ECA(nn.Module):\n",
    "    def __init__(self, channels, k_size=3):\n",
    "        super().__init__()\n",
    "        self.avg_pool = nn.AdaptiveAvgPool2d(1)\n",
    "        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size,\n",
    "                                  padding=k_size // 2, bias=False)\n",
    "        self.sigmoid  = nn.Sigmoid()\n",
    "\n",
    "    def forward(self, x):\n",
    "        y = self.avg_pool(x)\n",
    "        y = y.squeeze(-1).transpose(-1, -2)\n",
    "        y = self.conv(y)\n",
    "        y = y.transpose(-1, -2).unsqueeze(-1)\n",
    "        y = self.sigmoid(y)\n",
    "        return x * y.expand_as(x)\n",
    "\n",
    "\n",
    "# ─── RES_Block (Eq. 1: x ⊕ ECA(SENet(x))) ───────────────────────────────────\n",
    "class RES_Block(nn.Module):\n",
    "    def __init__(self, channels):\n",
    "        super().__init__()\n",
    "        self.se  = ModifiedSENet(channels)\n",
    "        self.eca = ECA(channels)\n",
    "\n",
    "    def forward(self, x):\n",
    "        return x + self.eca(self.se(x))   # residual skip — Eq.(1) paper\n",
    "\n",
    "\n",
    "# ─── Triplet Attention ───────────────────────────────────────────────────────\n",
    "class _ZPool(nn.Module):\n",
    "    def forward(self, x):\n",
    "        return torch.cat(\n",
    "            [x.max(dim=1, keepdim=True)[0],\n",
    "             x.mean(dim=1, keepdim=True)], dim=1\n",
    "        )\n",
    "\n",
    "\n",
    "class _AttGate(nn.Module):\n",
    "    def __init__(self, kernel_size=7):\n",
    "        super().__init__()\n",
    "        pad = kernel_size // 2\n",
    "        self.zpool = _ZPool()\n",
    "        self.conv  = nn.Conv2d(2, 1, kernel_size=kernel_size,\n",
    "                               padding=pad, bias=False)\n",
    "        self.bn    = nn.BatchNorm2d(1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        attn = torch.sigmoid(self.bn(self.conv(self.zpool(x))))\n",
    "        return x * attn\n",
    "\n",
    "\n",
    "class TripletAttention(nn.Module):\n",
    "    def __init__(self, kernel_size=7):\n",
    "        super().__init__()\n",
    "        self.hc = _AttGate(kernel_size)\n",
    "        self.wc = _AttGate(kernel_size)\n",
    "        self.hw = _AttGate(kernel_size)\n",
    "\n",
    "    def forward(self, x):\n",
    "        xp1 = x.permute(0, 2, 1, 3).contiguous()\n",
    "        y1  = self.hc(xp1).permute(0, 2, 1, 3).contiguous()\n",
    "        xp2 = x.permute(0, 3, 1, 2).contiguous()\n",
    "        y2  = self.wc(xp2).permute(0, 2, 3, 1).contiguous()\n",
    "        y3  = self.hw(x)\n",
    "        return (y1 + y2 + y3) / 3.0\n",
    "\n",
    "\n",
    "# ─── RTR_Lite_MobileNet ──────────────────────────────────────────────────────\n",
    "class RTR_Lite_MobileNet(nn.Module):\n",
    "    '''\n",
    "    RTR_Lite_MobileNet: Attention-fused Residual Lite MobileNetV2.\n",
    "    Modifications (Section 2.5, Fig 5b):\n",
    "      - After features[0]: TripletAttention + RES_Block\n",
    "      - features[1..15] kept intact (160 ch output)\n",
    "      - features[16,17,18] replaced by:\n",
    "          (RES_Block + TripletAttention + RES_Block) × 2  [repeated twice]\n",
    "      - Classifier: AdaptiveAvgPool -> Dropout(0.2) -> Linear(160, num_classes)\n",
    "    Target params: ~1,050,547 (Duhan et al. 2025, ablation Model 6)\n",
    "    '''\n",
    "    def __init__(self, num_classes, pretrained=True):\n",
    "        super().__init__()\n",
    "        weights = (\n",
    "            torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1\n",
    "            if pretrained else None\n",
    "        )\n",
    "        base = torchvision.models.mobilenet_v2(weights=weights)\n",
    "\n",
    "        self.first_conv = base.features[0]      # (B, 32, 112, 112)\n",
    "\n",
    "        # Attention after first conv (32 channels)\n",
    "        self.triplet1 = TripletAttention(kernel_size=7)\n",
    "        self.res1     = RES_Block(32)\n",
    "\n",
    "        # Backbone: features[1..15] → 160 channels\n",
    "        self.backbone = nn.Sequential(*list(base.features[1:16]))\n",
    "\n",
    "        # Replacement for features[16,17,18] — repeated TWICE (Section 2.5)\n",
    "        # First repetition\n",
    "        self.res2     = RES_Block(160)\n",
    "        self.triplet2 = TripletAttention(kernel_size=7)\n",
    "        self.res3     = RES_Block(160)\n",
    "        # Second repetition\n",
    "        self.res4     = RES_Block(160)\n",
    "        self.triplet3 = TripletAttention(kernel_size=7)\n",
    "        self.res5     = RES_Block(160)\n",
    "\n",
    "        # Classifier head\n",
    "        self.pool = nn.AdaptiveAvgPool2d(1)\n",
    "        self.drop = nn.Dropout(0.2)\n",
    "        self.fc   = nn.Linear(160, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.first_conv(x)    # (B, 32, 112, 112)\n",
    "        x = self.triplet1(x)\n",
    "        x = self.res1(x)\n",
    "        x = self.backbone(x)      # (B, 160, 7, 7)\n",
    "        # First repetition\n",
    "        x = self.res2(x)\n",
    "        x = self.triplet2(x)\n",
    "        x = self.res3(x)\n",
    "        # Second repetition\n",
    "        x = self.res4(x)\n",
    "        x = self.triplet3(x)\n",
    "        x = self.res5(x)\n",
    "        x = self.pool(x).flatten(1)\n",
    "        x = self.drop(x)\n",
    "        return self.fc(x)\n",
    "\n",
    "\n",
    "print('RTR_Lite_MobileNet architecture loaded.')",
]

# ─────────────────────────────────────────────────────────────────────────────
# [7] torchinfo — handle DataParallel (was [6])
# ─────────────────────────────────────────────────────────────────────────────
cells[7]["source"] = [
    "# ── Model Summary (torchinfo) ────────────────────────────────────────────────\n",
    "_tmp = RTR_Lite_MobileNet(num_classes=NUM_CLASSES,\n",
    "                           pretrained=PRETRAINED).to(DEVICE)\n",
    "_m = _tmp.module if isinstance(_tmp, nn.DataParallel) else _tmp\n",
    "torchinfo_summary(\n",
    "    _m,\n",
    "    input_size=(1, 3, IMG_SIZE, IMG_SIZE),\n",
    "    col_names=['input_size', 'output_size', 'num_params'],\n",
    "    depth=4,\n",
    "    device=DEVICE,\n",
    "    verbose=1,\n",
    ")\n",
    "del _tmp, _m",
]

# ─────────────────────────────────────────────────────────────────────────────
# [8] Training functions (was [7]) — checkpoint by val_acc + DataParallel save/load
# ─────────────────────────────────────────────────────────────────────────────
cells[8]["source"] = [
    "# ── Training & Evaluation Functions ─────────────────────────────────────────\n",
    "\n",
    "def train_one_epoch(model, loader, optimizer, criterion, device):\n",
    "    model.train()\n",
    "    total_loss, correct, total = 0.0, 0, 0\n",
    "    for imgs, labels in loader:\n",
    "        imgs, labels = imgs.to(device), labels.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        logits = model(imgs)\n",
    "        loss   = criterion(logits, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item() * imgs.size(0)\n",
    "        correct    += (logits.argmax(1) == labels).sum().item()\n",
    "        total      += imgs.size(0)\n",
    "    return total_loss / total, correct / total\n",
    "\n",
    "\n",
    "@torch.no_grad()\n",
    "def evaluate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    total_loss, correct, total = 0.0, 0, 0\n",
    "    preds_all, labels_all = [], []\n",
    "    for imgs, labels in loader:\n",
    "        imgs, labels = imgs.to(device), labels.to(device)\n",
    "        out   = model(imgs)\n",
    "        loss  = criterion(out, labels)\n",
    "        total_loss += loss.item() * imgs.size(0)\n",
    "        preds       = out.argmax(1)\n",
    "        correct    += (preds == labels).sum().item()\n",
    "        total      += imgs.size(0)\n",
    "        preds_all.extend(preds.cpu().numpy())\n",
    "        labels_all.extend(labels.cpu().numpy())\n",
    "    return total_loss / total, correct / total, preds_all, labels_all\n",
    "\n",
    "\n",
    "def train_model(model, train_loader, val_loader,\n",
    "                optimizer, scheduler, criterion,\n",
    "                device, epochs, ckpt_path):\n",
    "    history = {'train_loss':[], 'val_loss':[],\n",
    "               'train_acc':[],  'val_acc':[]}\n",
    "    best_val_acc = 0.0\n",
    "\n",
    "    for epoch in range(1, epochs + 1):\n",
    "        tr_loss, tr_acc       = train_one_epoch(\n",
    "            model, train_loader, optimizer, criterion, device)\n",
    "        vl_loss, vl_acc, _, _ = evaluate(\n",
    "            model, val_loader, criterion, device)\n",
    "        scheduler.step()\n",
    "\n",
    "        history['train_loss'].append(tr_loss)\n",
    "        history['val_loss'].append(vl_loss)\n",
    "        history['train_acc'].append(tr_acc)\n",
    "        history['val_acc'].append(vl_acc)\n",
    "\n",
    "        print(f'Epoch {epoch:3d}/{epochs} | '\n",
    "              f'Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | '\n",
    "              f'Val   Loss={vl_loss:.4f} Acc={vl_acc:.4f}')\n",
    "\n",
    "        if vl_acc > best_val_acc:          # checkpoint by val_accuracy (Rule 10)\n",
    "            best_val_acc = vl_acc\n",
    "            _state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()\n",
    "            torch.save(_state, ckpt_path)\n",
    "\n",
    "    # Reload best checkpoint before test evaluation\n",
    "    _target = model.module if isinstance(model, nn.DataParallel) else model\n",
    "    _target.load_state_dict(torch.load(ckpt_path, map_location=device))\n",
    "    return history\n",
    "\n",
    "\n",
    "def count_params(model):\n",
    "    m = model.module if isinstance(model, nn.DataParallel) else model\n",
    "    return sum(p.numel() for p in m.parameters())\n",
    "\n",
    "\n",
    "print('Functions loaded.')",
]

# ─────────────────────────────────────────────────────────────────────────────
# [10] S1 DataLoaders (was [9]) — num_workers 2→4
# ─────────────────────────────────────────────────────────────────────────────
cells[10]["source"] = [
    "# ── Scenario 1: No-Aug DataLoaders ─────────────────────────────────────────\n",
    "tf_noaug = T.Compose([\n",
    "    T.Resize((IMG_SIZE, IMG_SIZE)),\n",
    "    T.ToTensor(),\n",
    "    T.Normalize(mean=MEAN, std=STD),\n",
    "])\n",
    "\n",
    "train_ds1 = ImageFolder(TRAIN_DIR, transform=tf_noaug)\n",
    "val_ds1   = ImageFolder(VAL_DIR,   transform=tf_noaug)\n",
    "test_ds1  = ImageFolder(TEST_DIR,  transform=tf_noaug)\n",
    "\n",
    "assert train_ds1.classes == class_names, 'Class order mismatch!'\n",
    "\n",
    "train_loader1 = DataLoader(train_ds1, batch_size=BATCH_SIZE,\n",
    "                            shuffle=True,  num_workers=4, pin_memory=True)\n",
    "val_loader1   = DataLoader(val_ds1,   batch_size=BATCH_SIZE,\n",
    "                            shuffle=False, num_workers=4, pin_memory=True)\n",
    "test_loader1  = DataLoader(test_ds1,  batch_size=BATCH_SIZE,\n",
    "                            shuffle=False, num_workers=4, pin_memory=True)\n",
    "\n",
    "print(f'S1 ready — Train:{len(train_ds1)}, Val:{len(val_ds1)}, Test:{len(test_ds1)}')",
]

# ─────────────────────────────────────────────────────────────────────────────
# [11] S1 Model Init (was [10]) — class_weights + DataParallel
# ─────────────────────────────────────────────────────────────────────────────
cells[11]["source"] = [
    "# ── Scenario 1: Initialize + Train ─────────────────────────────────────────\n",
    "model_s1 = RTR_Lite_MobileNet(NUM_CLASSES, pretrained=PRETRAINED).to(DEVICE)\n",
    "if n_gpus > 1:\n",
    "    model_s1 = nn.DataParallel(model_s1)\n",
    "    print(f'[Multi-GPU] DataParallel on {n_gpus} GPUs')\n",
    "\n",
    "optimizer_s1  = torch.optim.Adam(model_s1.parameters(), lr=LR)\n",
    "scheduler_s1  = torch.optim.lr_scheduler.StepLR(\n",
    "                    optimizer_s1, step_size=LR_STEP, gamma=LR_GAMMA)\n",
    "\n",
    "# Class weighting (auto from distribution cell)\n",
    "if CLASS_WEIGHTS is not None:\n",
    "    criterion_s1 = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))\n",
    "    print('[S1] CrossEntropyLoss with class_weights')\n",
    "else:\n",
    "    criterion_s1 = nn.CrossEntropyLoss()\n",
    "    print('[S1] CrossEntropyLoss (no weighting)')\n",
    "\n",
    "print('=== Scenario 1 — Baseline (No Augmentation) ===')\n",
    "print(f'Total parameters: {count_params(model_s1):,}')\n",
    "\n",
    "history_s1 = train_model(\n",
    "    model_s1, train_loader1, val_loader1,\n",
    "    optimizer_s1, scheduler_s1, criterion_s1,\n",
    "    DEVICE, EPOCHS,\n",
    "    ckpt_path='/kaggle/working/best_model_s1.pth',\n",
    ")\n",
    "print('Training complete.')",
]

# ─────────────────────────────────────────────────────────────────────────────
# Insert empty_cache cell between S1 metrics dict and S2 markdown
# Currently: [14]=S1 metrics dict, [15]=S2 markdown
# After insert new dist cell earlier, original indices shifted, so:
# Cell index for "S1 metrics dict" = 15, "S2 markdown" = 16
# ─────────────────────────────────────────────────────────────────────────────
# Find S2 markdown cell
s2_md_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and 'Scenario 2' in ''.join(c['source']):
        s2_md_idx = i
        break

empty_cache_cell = code_cell([
    "# ── Cleanup between scenarios — free GPU memory ──────────────────────────────\n",
    "del model_s1, optimizer_s1, scheduler_s1, criterion_s1\n",
    "del train_loader1, val_loader1, test_loader1\n",
    "del train_ds1, val_ds1, test_ds1\n",
    "torch.cuda.empty_cache()\n",
    "gc.collect()\n",
    "print('GPU memory cleared. Ready for Scenario 2.')",
])

cells.insert(s2_md_idx, empty_cache_cell)
# Now s2 markdown has shifted +1

# ─────────────────────────────────────────────────────────────────────────────
# Update S2 DataLoaders — num_workers 2→4
# ─────────────────────────────────────────────────────────────────────────────
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and '# ── Scenario 2: Augmented DataLoaders' in ''.join(c['source']):
        cells[i]["source"] = [
            "# ── Scenario 2: Augmented DataLoaders ──────────────────────────────────────\n",
            "tf_aug = T.Compose([\n",
            "    T.Resize((IMG_SIZE, IMG_SIZE)),\n",
            "    T.RandomRotation(degrees=30),\n",
            "    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),\n",
            "    T.RandomHorizontalFlip(p=0.5),\n",
            "    T.RandomAffine(degrees=0, translate=(0.1, 0.1),\n",
            "                   scale=(0.85, 1.15), shear=10),\n",
            "    T.ToTensor(),\n",
            "    T.Normalize(mean=MEAN, std=STD),\n",
            "])\n",
            "\n",
            "tf_val = T.Compose([\n",
            "    T.Resize((IMG_SIZE, IMG_SIZE)),\n",
            "    T.ToTensor(),\n",
            "    T.Normalize(mean=MEAN, std=STD),\n",
            "])\n",
            "\n",
            "train_ds2 = ImageFolder(TRAIN_DIR, transform=tf_aug)\n",
            "val_ds2   = ImageFolder(VAL_DIR,   transform=tf_val)\n",
            "test_ds2  = ImageFolder(TEST_DIR,  transform=tf_val)\n",
            "\n",
            "train_loader2 = DataLoader(train_ds2, batch_size=BATCH_SIZE,\n",
            "                            shuffle=True,  num_workers=4, pin_memory=True)\n",
            "val_loader2   = DataLoader(val_ds2,   batch_size=BATCH_SIZE,\n",
            "                            shuffle=False, num_workers=4, pin_memory=True)\n",
            "test_loader2  = DataLoader(test_ds2,  batch_size=BATCH_SIZE,\n",
            "                            shuffle=False, num_workers=4, pin_memory=True)\n",
            "\n",
            "print(f'S2 ready — Train:{len(train_ds2)}, Val:{len(val_ds2)}, Test:{len(test_ds2)}')",
        ]
        break

# ─────────────────────────────────────────────────────────────────────────────
# Update S2 Model Init — class_weights + DataParallel
# ─────────────────────────────────────────────────────────────────────────────
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'Re-initialize Model' in ''.join(c['source']):
        cells[i]["source"] = [
            "# ── Scenario 2: Re-initialize Model + Train (fresh weights, no data leakage) ─\n",
            "model_s2 = RTR_Lite_MobileNet(NUM_CLASSES, pretrained=PRETRAINED).to(DEVICE)\n",
            "if n_gpus > 1:\n",
            "    model_s2 = nn.DataParallel(model_s2)\n",
            "    print(f'[Multi-GPU] DataParallel on {n_gpus} GPUs')\n",
            "\n",
            "optimizer_s2  = torch.optim.Adam(model_s2.parameters(), lr=LR)\n",
            "scheduler_s2  = torch.optim.lr_scheduler.StepLR(\n",
            "                    optimizer_s2, step_size=LR_STEP, gamma=LR_GAMMA)\n",
            "\n",
            "if CLASS_WEIGHTS is not None:\n",
            "    criterion_s2 = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))\n",
            "    print('[S2] CrossEntropyLoss with class_weights')\n",
            "else:\n",
            "    criterion_s2 = nn.CrossEntropyLoss()\n",
            "    print('[S2] CrossEntropyLoss (no weighting)')\n",
            "\n",
            "print('=== Scenario 2 — Augmented (Rotation + ColorJitter + HFlip + Affine) ===')\n",
            "print(f'Total parameters: {count_params(model_s2):,}')\n",
            "\n",
            "history_s2 = train_model(\n",
            "    model_s2, train_loader2, val_loader2,\n",
            "    optimizer_s2, scheduler_s2, criterion_s2,\n",
            "    DEVICE, EPOCHS,\n",
            "    ckpt_path='/kaggle/working/best_model_s2.pth',\n",
            ")\n",
            "print('Training complete.')",
        ]
        break

# ─────────────────────────────────────────────────────────────────────────────
# Write back
# ─────────────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
