import sys, os
import numpy as np
from rembg import remove, new_session
from PIL import Image

CLASS = sys.argv[1]
SOLID_MIN = 0.03  # < 3% dien tich la "chac chan la" -> nghi mask sup, fail
SRC_DIR = os.path.join("ds", CLASS)
OUT_DIR = os.path.join("ds_cropped", CLASS)
FAIL_DIR = os.path.join("ds_cropped", "_failed_" + CLASS)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FAIL_DIR, exist_ok=True)

session = new_session("isnet-general-use")
bg_color = (210, 210, 210, 255)

files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"[{CLASS}] {len(files)} anh")

ok, fail = 0, 0
fail_list = []
for i, fname in enumerate(files):
    src_path = os.path.join(SRC_DIR, fname)
    stem = os.path.splitext(fname)[0]
    try:
        img = Image.open(src_path).convert("RGB")
        cut = remove(img, session=session)
        alpha = np.array(cut)[:, :, 3].astype(float) / 255.0
        solid_frac = (alpha > 0.5).mean()

        bg = Image.new("RGBA", cut.size, bg_color)
        bg.paste(cut, (0, 0), cut)
        result_rgb = bg.convert("RGB")

        if solid_frac < SOLID_MIN:
            result_rgb.save(os.path.join(FAIL_DIR, stem + "_cropped.png"))
            img.save(os.path.join(FAIL_DIR, stem + "_original.jpg"))
            fail += 1
            fail_list.append((fname, solid_frac))
        else:
            result_rgb.save(os.path.join(OUT_DIR, stem + ".png"))
            ok += 1
    except Exception as e:
        print(f"  LOI {fname}: {e}")
        fail += 1
    if (i + 1) % 20 == 0:
        print(f"  ... {i+1}/{len(files)}")

print(f"\n[{CLASS}] xong: {ok} ok -> {OUT_DIR} | {fail} fail -> {FAIL_DIR}")
if fail_list:
    print("Danh sach anh fail (mask sup, can xem lai thu cong):")
    for fname, frac in sorted(fail_list, key=lambda x: x[1]):
        print(f"  {fname}: chi {frac*100:.1f}% dien tich duoc nhan la 'la'")
