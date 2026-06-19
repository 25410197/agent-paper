---
name: project-litemdc
description: Lite-MDC reproduction (Smart Agri Tech 2024) — TF/Keras, MDsConv multi-kernel depthwise module, ~2.29M params, chili substitute dataset
metadata:
  type: project
---

Lite-MDC (Bhagat et al., Smart Agricultural Technology 7 (2024) 100408, DOI 10.1016/j.atech.2024.100408) reproduced 2026-06-18 into `papers/07- Advancing real-time plant disease detection/outputs/` (LiteMDC_Experiment.ipynb + LiteMDC_ReproduceSummary.md).

**Why:** User reproducing this paper's lightweight CNN onto their own chili pepper dataset. Group A paper — full layer table (Table 5, 26 layers) is authoritative.

**How to apply:**
- Framework **TF/Keras** (paper Mục 5.2). Optimizer Adam, loss categorical CE (used sparse_categorical_crossentropy with int labels), batch 16, input 224, ReduceLROnPlateau(factor=0.3, patience=5, monitor=val_loss) — all paper-confirmed.
- **User-confirmed** (not in paper): lr=3e-4, epochs=100, LeakyReLU alpha=0.1, dropout schedule 0.1/0.2/0.3/0.4 after PConv_9/10/11/12, ImageNet mean/std normalization (paper only said per-channel mean subtraction).
- **MDsConv module:** channel split into 2 equal halves via Lambda slice -> DepthwiseConv2D(3x3) on left half + DepthwiseConv2D(5x5) on right half (both depth_multiplier=1, stride from Table 5) -> Concat -> BN -> LeakyReLU -> pointwise Conv2D(1x1, N) -> BN -> LeakyReLU -> optional Dropout. Stride-2 downsampling in MDConv blocks 1,3,6,9,12 (NO MaxPooling).
- **Channel sequence = Table 5 'Output shape' column (authoritative, NOT 'Filters size' col which has internal contradictions e.g. PConv_2):** Conv0=64, PConv outputs 64,128,128,256,256,256,512,512,512,512,512,512 across blocks 1..12. block_cfg (out_ch, stride): (64,2),(128,1),(128,2),(256,1),(256,1),(256,2),(512,1),(512,1),(512,2),(512,1),(512,1),(512,2).
- **Head:** DwConv_13 = DepthwiseConv2D(7, padding='valid') on 7x7 map -> 1x1x512 (keep this, don't replace with GAP — needed for param match) -> PConv_13 (1x1,512) -> Conv_14 (1x1,512, BN, **NO activation** per "LeakyReLU except last one") -> GAP -> Dense softmax.
- **Param count verified analytically = 2,288,965 (~2.29M), matches paper ~2.2M.** All conv use_bias=False (BN follows). Used a pure-python analytic param counter to verify since TF not installed locally — good practice before delivery.
- Dataset **substitute**: chili pepper at `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300`, pre-split train/val/test, NUM_CLASSES auto-detected, used keras image_dataset_from_directory with class_names enforced for consistent ordering across splits. Paper's pigeon-pea dataset not used.
- Aug (Scenario 2) limited to 4 techniques per CLAUDE.md rule 3: RandomRotation(30/360), RandomZoom(0.2), RandomFlip horizontal, RandomBrightness(0.4). Paper used more (shear, shifts, vflip) — trimmed.
- Built notebook via throwaway python script (_build_nb.py) that json.dump's the .ipynb then validates with json.load, then deleted it — same reliable method as [[project-customcnn-cropdesign]] and [[project-mrwcnn]]. See [[user-dataset]] convention.
