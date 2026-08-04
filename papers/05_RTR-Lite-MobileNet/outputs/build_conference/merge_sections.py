# -*- coding: utf-8 -*-
"""
Ghep cac file MD con trong vn/ va en/ lai thanh 1 file MD tong hoan chinh
cho tung ngon ngu (vn/_BaoCao_Full_VN.md, en/_BaoCao_Full_EN.md).

Day la buoc BAT BUOC truoc khi chay build_conference_docx.py /
build_conference_docx_vn.py, vi hai script do doc tu file MD tong.

CACH DUNG:
  1. Sua noi dung trong tung file .md con (00_..., 01_..., ...) - nguon chan
     ly duy nhat cho tung muc.
  2. Chay: python merge_sections.py
  3. Chay tiep build_conference_docx.py / build_conference_docx_vn.py nhu cu.

File _BaoCao_Full_*.md la file SINH RA TU DONG - khong sua tay truc tiep,
moi sua se bi ghi de o lan chay merge tiep theo.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Main manuscript chi gom cac muc 00_ den 09_. Cac file 10_highlights_*.md
# va 11_title_page.md la artifact nop rieng, khong duoc noi sau References.
LANGS = [
    ("vn", "_BaoCao_Full_VN.md"),
    ("en", "_BaoCao_Full_EN.md"),
]


def merge_dir(lang_dir, out_name):
    dir_path = os.path.join(HERE, lang_dir)
    files = sorted(
        f
        for f in os.listdir(dir_path)
        if f.endswith(".md") and re.match(r"^0\d_", f)
    )
    if not files:
        raise SystemExit(f"Khong tim thay file .md nao trong {dir_path}")

    chunks = []
    for fname in files:
        with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
            chunks.append(f.read().rstrip("\n"))

    merged = "\n\n".join(chunks) + "\n"
    out_path = os.path.join(dir_path, out_name)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(merged)
    print(f"{lang_dir}/ ({len(files)} file con) -> {lang_dir}/{out_name}")


if __name__ == "__main__":
    for lang_dir, out_name in LANGS:
        merge_dir(lang_dir, out_name)
