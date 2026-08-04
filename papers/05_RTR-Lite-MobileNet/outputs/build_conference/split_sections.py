# -*- coding: utf-8 -*-
"""
Tach 1 file MD lon (BaoCao_..._VN.md / _EN.md) thanh nhieu file MD con,
moi file = 1 muc lon (## heading), luu trong thu muc vn/ hoac en/.

Chi chay 1 LAN de khoi tao cau truc thu muc tu file MD lon hien co.
Sau khi da co cau truc vn/ + en/, KHONG chay lai script nay - sua truc tiep
trong tung file con roi dung merge_sections.py de tao lai file tong.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# (file MD nguon, thu muc dich, danh sach (heading_prefix_khong_dau_##, ten_file))
JOBS = [
    (
        "BaoCao_RTR_LiteFormer_Conference_VN.md",
        "vn",
        "00_tieu_de.md",
        [
            ("## Tóm tắt", "01_tom_tat.md"),
            ("## 1. Đặt vấn đề", "02_dat_van_de.md"),
            ("## 2. Các nghiên cứu liên quan", "03_nghien_cuu_lien_quan.md"),
            ("## 3. Phương pháp đề xuất", "04_phuong_phap_de_xuat.md"),
            ("## 4. Thực nghiệm và kết quả", "05_thuc_nghiem_ket_qua.md"),
            ("## 5. Diễn giải mô hình", "06_dien_giai_mo_hinh.md"),
            ("## 6. Hạn chế", "07_han_che.md"),
            ("## 7. Kết luận", "08_ket_luan.md"),
            ("## References", "09_tai_lieu_tham_khao.md"),
        ],
    ),
    (
        "BaoCao_RTR_LiteFormer_Conference_EN.md",
        "en",
        "00_title.md",
        [
            ("## Abstract", "01_abstract.md"),
            ("## 1. Introduction", "02_introduction.md"),
            ("## 2. Related Work", "03_related_work.md"),
            ("## 3. Proposed Method", "04_proposed_method.md"),
            ("## 4. Experiments and Results", "05_experiments_results.md"),
            ("## 5. Model Interpretability", "06_model_interpretability.md"),
            ("## 6. Limitations", "07_limitations.md"),
            ("## 7. Conclusion", "08_conclusion.md"),
            ("## References", "09_references.md"),
        ],
    ),
]


def split_file(src_name, out_dir, preamble_name, sections):
    src_path = os.path.join(HERE, src_name)
    with open(src_path, encoding="utf-8") as f:
        text = f.read()

    # tim vi tri bat dau cua tung heading "## " o dau dong
    heading_positions = [m.start() for m in re.finditer(r"(?m)^## ", text)]
    if len(heading_positions) != len(sections):
        raise SystemExit(
            f"[{src_name}] so heading tim thay ({len(heading_positions)}) "
            f"khac so muc khai bao ({len(sections)}). Kiem tra lai JOBS."
        )

    out_path = os.path.join(HERE, out_dir)
    os.makedirs(out_path, exist_ok=True)

    # preamble = tu dau file toi truoc heading dau tien
    preamble = text[: heading_positions[0]].rstrip("\n")
    with open(os.path.join(out_path, preamble_name), "w", encoding="utf-8", newline="\n") as f:
        f.write(preamble + "\n")

    bounds = heading_positions + [len(text)]
    for i, (expected_prefix, fname) in enumerate(sections):
        chunk = text[bounds[i] : bounds[i + 1]].rstrip("\n")
        if not chunk.startswith(expected_prefix):
            raise SystemExit(
                f"[{src_name}] muc #{i} khong khop: mong '{expected_prefix}', "
                f"thuc te bat dau bang '{chunk[:60]!r}'"
            )
        with open(os.path.join(out_path, fname), "w", encoding="utf-8", newline="\n") as f:
            f.write(chunk + "\n")

    print(f"[{src_name}] -> {out_dir}/  ({len(sections)} muc + preamble)")


if __name__ == "__main__":
    for src_name, out_dir, preamble_name, sections in JOBS:
        split_file(src_name, out_dir, preamble_name, sections)
