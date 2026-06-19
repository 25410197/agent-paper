---
name: project-mrwcnn
description: MRW-CNN reproduction (IEEE Access 2024) — TF/Keras, arch confirmed from Fig.9, chili substitute dataset
metadata:
  type: project
---

MRW-CNN (Joseph et al., IEEE Access 2024, DOI 10.1109/ACCESS.2024.3358333) reproduced 2026-06-16 into `papers/04- Real-Time Plant Disease Dataset Development/outputs/` (MRW-CNN_Experiment.ipynb + MRW-CNN_ReproduceSummary.md).

**Why:** User reproducing this paper's proposed lightweight CNN onto their own chili pepper dataset.

**How to apply:**
- Framework is **TensorFlow/Keras** by user's explicit choice (overrides project PyTorch default). Optimizer = SGD lr=0.001, loss=CategoricalCrossentropy, 100 epochs, batch 32, input 224, /255 norm.
- Architecture (CORRECTED 2026-06-16 after reading Figure 9 directly via vision): 5 conv blocks, each = Conv2D→BatchNorm→ReLU, filters **32→64→64→256→256**, kernel 3x3, then **GAP→Dense(softmax)**. **NO MaxPooling anywhere** (confirmed absent from Fig.9 + Section V text). All of filters/kernel/order/no-pooling/GAP/softmax are now **PAPER-CONFIRMED via Fig.9**. Only remaining **ASSUMED**: stride=2 + padding='same' (downsampling in place of pooling) — user-approved. NOTE: the earlier run wrongly assumed 32→64→128→256→512 + MaxPool2x2 — that is WRONG, do not reuse.
- Lesson: when conv config only appears in a figure, render the PDF page and read it via vision before assuming — Fig.9 was machine-readable via the render_pdf_page.py vision workflow (pdf-extractor now does this automatically).
- Dataset is a **substitute**: chili pepper at `/kaggle/input/datasets/tminhhi/dataset-ot/dataset_smartcrop_plit_300`, pre-split train/val/test, 5 classes (anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf). Paper's maize/rice/wheat dataset is not public. Use flow_from_directory directly, no random split.
- 8 fine-tuned baseline models in paper are NOT required deliverables — only MRW-CNN.
- Structural reference for this user's TF/Keras notebooks: `papers/03 - TrIncNet/outputs/TrIncNet_Experiment.ipynb` (same dataset path/pre-split convention). See [[user-dataset]].
