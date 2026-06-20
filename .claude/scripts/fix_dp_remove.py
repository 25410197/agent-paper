import json

NB_PATH = 'E:/UIT/study/claude/learn-claude/papers/13 - GrapeLeafNet/ouputs/grapeleafnet.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# BLOCK 7 (cell 13) — remove DataParallel, plain single-GPU
cells[13]['source'] = "\n".join([
    "# BLOCK 7: Build Model & Summary — Scenario 1",
    "model_s1 = GrapeLeafNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)",
    "",
    "torch_summary(",
    "    model_s1,",
    "    input_size=(1, 3, *IMG_SIZE),",
    "    col_names=['output_size', 'num_params', 'trainable'],",
    "    verbose=1,",
    ")",
    "total_params = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)",
    "print(f'\\nTotal Trainable Parameters: {total_params:,}')",
    "",
])
cells[13]['outputs'] = []

# BLOCK 13 (cell 20) — remove DataParallel, keep empty_cache
cells[20]['source'] = "\n".join([
    "# BLOCK 13: Re-initialize Model — Scenario 2 (prevent data leakage)",
    "torch.cuda.empty_cache()",
    "print('[GPU] Cache cleared between scenarios.')",
    "",
    "model_s2 = GrapeLeafNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)",
    "",
    "total_params_s2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)",
    "print(f'Model re-initialized. Total Trainable Parameters: {total_params_s2:,}')",
    "",
])
cells[20]['outputs'] = []

# Also update run_training to remove DataParallel-aware checkpoint logic
# (no longer needed — simpler state_dict save/load)
old_save = "            _state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()"
new_save = "            torch.save(model.state_dict(), checkpoint_path)"
old_load = (
    "    # Reload best checkpoint before evaluation\n"
    "    _target = model.module if isinstance(model, nn.DataParallel) else model\n"
    "    _target.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))"
)
new_load = (
    "    # Reload best checkpoint before evaluation\n"
    "    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))"
)

src = ''.join(cells[10]['source'])
src = src.replace(old_save, new_save)
src = src.replace(old_load, new_load)
cells[10]['source'] = src
cells[10]['outputs'] = []

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done — DataParallel removed, checkpoint logic simplified.")
