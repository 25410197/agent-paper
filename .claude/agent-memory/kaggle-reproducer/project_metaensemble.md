---
name: project-metaensemble
description: Meta-Ensemble (MLP-Mixer+LSTM+SVM) reproduction 2026-06-19 — TF/Keras stacking ensemble, IEEE Access 2024
metadata:
  type: project
---

**Reproduced:** Meta-Ensemble for Plant Disease Detection (IEEE Access 2024, Maurya et al., DOI 10.1109/ACCESS.2024.3367443) on user's chili/plant dataset `/kaggle/input/datasets/tminhhi/ot-final/dataset_smartcrop_plit_300`.

**Paper folder:** `papers/01_A_Lightweight_Meta-Ensemble_Approach_for_Plant_Disease_Detection_Suitable_for_IoT-Based_Environments/`
**Notebook:** `outputs/MetaEnsemble_Experiment.ipynb` (50 cells)
**Summary:** `outputs/MetaEnsemble_ReproduceSummary.md`

**Architecture:** 2-level STACKING ensemble (NOT end-to-end).
- Level1 Branch A: MLP-Mixer — resize 64->72, reshape 9 patches of 24x24x3=1728, Dense(1728->128), 7 mixer blocks (token-mix Dense9 + channel-mix Dense128 gelu, 2x LN, 2x skip), Dropout0.25, GAP, softmax.
- Level1 Branch B: 4 frozen ImageNet CNNs (MobileNet 1024 + DenseNet121 1024 + DenseNet169 1664 + DenseNet201 1920 = 5632) -> cache features -> LSTM(30, timestep=1) -> softmax.
- Level2: SVM(RBF) GridSearchCV over meta-features concat[P_Mixer,P_LSTM]=(N,2*num_classes).

**User-confirmed params:** TF/Keras; LR=1e-3 Adam; EarlyStopping(patience=10, monitor=val_accuracy, restore_best_weights); CNN normalize pixel/255 [0,1] ONLY (no ImageNet mean/std); SVM grid C=[0.1,1,2,10,100] gamma=[1,0.1,0.01,0.001,0.0001] kernel=[linear,rbf]; sparse_categorical_crossentropy; batch=512; epochs=100; val_split=0.1.

**Why these were assumptions:** paper gave LR<0.001 (range), epochs 50-100 (range), aug types but no ranges, and `_compression` patch step with no text description.

**Key reproduction decisions / risks:**
- Patch embedding Dense(1728->128) gives ~221K params for projection alone -> Mixer total params will be HIGHER than paper's ~810K-915K. Paper's `_compression`=8272 params can't be reproduced 1:1 (no text spec). Flagged HIGH risk in summary §9.
- FEATURE_DIM auto-detected from `feat_train.shape[1]` (not hard-coded 5632) — per prior lesson.
- CNN features cached once to numpy; LSTM trains on static features, NOT re-extracted per epoch. For Scenario 2, features re-extracted on augmented images.
- Augmentation applied OFFLINE: 1 augmented copy per train image concat with originals + shuffle.
- Excel: each scenario has 2 learning curves (Mixer+LSTM); Excel embeds Mixer LC as representative (1 image slot/scenario).

**Notebook build method:** generated via a temporary Python script `_build_notebook.py` that emits nbformat-4 JSON (script deleted after). GOTCHA: inside `r"""..."""` raw strings, `\"\"\"` stays literal backslash-quote and breaks docstrings — use `'''` for docstrings instead. f-string `{suf}` interpolation with `{{}}` for dict literals worked cleanly.
