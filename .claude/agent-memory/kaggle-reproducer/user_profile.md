---
name: user-profile
description: User profile — research focus, dataset domain, preferred workflow for kaggle-reproducer agent
metadata:
  type: user
---

User is working on deep learning reproduction from scientific papers for plant disease detection (specifically maize/corn leaf disease). They upload PDF papers and want Kaggle-ready notebooks.

**Domain:** Agricultural image classification (maize leaf disease, likely other plant diseases)
**Dataset:** User has their own maize leaf disease dataset on Kaggle — NOT the paper's original datasets
**Workflow preference:** User provides PDF path + output directory, expects both `.ipynb` + `_ReproduceSummary.md` simultaneously
**Email:** tominhhai0801@gmail.com

**Why:** User is likely a student or researcher reproducing SOTA models on local/custom datasets for comparison or thesis work.

**How to apply:** Always target user's Kaggle dataset (not paper dataset). Auto-detect NUM_CLASSES. Use 75/15/10 split as default when user dataset structure is unknown.
