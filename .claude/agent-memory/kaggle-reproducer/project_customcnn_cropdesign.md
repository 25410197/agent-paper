---
name: project-customcnn-cropdesign
description: Custom CNN reproduction (Crop Design 2025) — TF/Keras, fully self-proposed layer config, chili substitute dataset
metadata:
  type: project
---

Custom CNN (Rahman et al., Crop Design 4 (2025) 100092, DOI 10.1016/j.cropd.2024.100092) reproduced 2026-06-16 into `papers/05 - Crop Design/outputs/` (CustomCNN_Experiment.ipynb + CustomCNN_ReproduceSummary.md).

**Why:** User reproducing this paper's "Custom CNN" onto their own chili pepper dataset. Group C paper — proposes a model but publishes NO layer-level architecture.

**How to apply:**
- Framework **TensorFlow/Keras** (paper-confirmed, Mục 3.7/3.8). Optimizer **Adam lr=1e-4** (lr from Table 14; Adam itself is SELF-PROPOSED — Table 14 lists no optimizer), loss CategoricalCrossentropy, input **256x256**, /255 norm, batch 32, max 50 epochs, EarlyStopping(patience=5, monitor=val_accuracy, restore_best_weights=True).
- Architecture is **almost entirely SELF-PROPOSED** (user-approved 2026-06-16) — paper only states "6 Conv2D + 5 MaxPool", ReLU, SoftMax, 2 Dense (Table 14). Self-proposed config: 6 Conv2D filters **32→64→128→256→512→512** kernel 3x3, each Conv→BatchNorm→ReLU; MaxPool(2x2) after first 5 blocks, **block 6 has NO pool** (this is how "6 conv / 5 pool" is satisfied); then Flatten→Dropout(0.5)→Dense(64,ReLU)→Dropout(0.5)→Dense(5,softmax). L2=1e-4, Dropout=0.5 self-proposed. ONLY paper-confirmed numbers: input 256, lr 1e-4, batch 32, epochs 50, Dense hidden=64 (all from Table 14 pepper/potato/corn/grape row), 6conv/5pool/ReLU/SoftMax/2Dense counts. Fig.4 is a GENERIC borrowed diagram [30], not the real arch — do not mine it.
- Dataset **substitute**: chili pepper at `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300`, pre-split train/val/test, 5 classes (anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf). flow_from_directory directly, no random split. Paper's PlantVillage+rice 35-class set not used.
- Single 5-class model, NOT the per-plant (8 models) setup of the paper.
- Built notebook via a throwaway python script that json.dump's the .ipynb (validated with json.load, then deleted the script) — reliable way to emit valid notebook JSON. Reused [[project-mrwcnn]] notebook as structural template (same TF/Keras + same dataset convention).
- Notebook includes a top-of-notebook PROVENANCE markdown cell explicitly flagging the self-proposed config — required because architecture is not in paper. See [[user-dataset]] convention (pre-split chili dataset).
