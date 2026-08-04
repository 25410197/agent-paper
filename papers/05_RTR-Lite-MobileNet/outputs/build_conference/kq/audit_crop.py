import os
import numpy as np
from PIL import Image

DIR = "ds_cropped/yellow_leaf"
bg = np.array([210,210,210])

results = []
for fname in sorted(os.listdir(DIR)):
    if not fname.lower().endswith('.png'):
        continue
    img = np.array(Image.open(os.path.join(DIR, fname)).convert("RGB")).astype(int)
    dist = np.sqrt(((img - bg)**2).sum(axis=-1))
    near_bg_frac = (dist < 12).mean()  # fraction of pixels "basically background gray"
    results.append((fname, near_bg_frac))

results.sort(key=lambda x: -x[1])
print("Top 15 nghi fail (gan bg nhat):")
for fname, frac in results[:15]:
    print(f"  {fname}: {frac*100:.1f}% pixel gan mau nen")

n_fail = sum(1 for _, f in results if f > 0.85)
print(f"\nTong so anh co >85% pixel gan nen (nghi fail): {n_fail}/{len(results)}")
