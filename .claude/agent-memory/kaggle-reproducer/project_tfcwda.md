---
name: project-tfcwda
description: Reproduction 2026-10-20 of Tunio et al. 2024 (CEA 227) TFC + Wasserstein DA — PyTorch, MobileNetV2 local + timm mobilevit_xs global, WGAN-GP critic, single-dataset UDA
metadata:
  type: project
---
Paper folder papers/19_Wasserstein; outputs TFC-WDA_Experiment.ipynb + TFC-WDA_ReproduceSummary.md.
User chose backbone option (b): torchvision MobileNetV2 (local) + timm mobilevit_xs (global), fusion Conv1x1+ADD at 28x28/14x14; accepted all other brainstorm defaults (gamma 10, lambda 1, alpha 1, n_critic 5, critic Adam 1e-4).
UDA with only one pre-split dataset: source = train (labeled), target = train again unlabeled (independent shuffle), TARGET_DIR + USE_WDA toggles; never use val/test as target.
**Why:** user only has a single train/val/test dataset; val/test as target would leak into checkpoint selection.
**How to apply:** For future domain-adaptation papers, propose this same setup up front and flag that WD alignment is near-trivial when target = source. timm mobilevit_xs stage split verified: stem+stages[0:3]->64x28, stages[3]->80x14, stages[4]+final_conv->384x7; MobileNetV2 features[0:7]->32x28, [7:14]->96x14, [14:]->1280x7.
