# -*- coding: utf-8 -*-
"""Convert numbered citations in the Vietnamese source sections to CEA author-year style.

The converter deliberately ignores bracketed integers attached directly to identifiers,
such as ``features[13]``. References are maintained separately in
``vn/09_tai_lieu_tham_khao.md``.
"""
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
VN_DIR = HERE / "vn"

CITATIONS = {
    1: "Maurya et al., 2024",
    2: "Rahman et al., 2026",
    3: "Gole et al., 2023",
    4: "Joseph et al., 2024",
    5: "Duhan et al., 2025",
    6: "Li et al., 2023",
    7: "Bhagat et al., 2024",
    8: "Wang et al., 2023",
    9: "Yang et al., 2024",
    10: "Hemalatha and Jayachandran, 2024",
    11: "Mishra et al., 2025",
    12: "Rahaman et al., 2026",
    13: "Karthik et al., 2024",
    14: "Sujatha et al., 2025",
    15: "Ali et al., 2025",
    16: "Sandler et al., 2018",
    17: "Misra et al., 2021",
    18: "Su et al., 2024",
    19: "Ainslie et al., 2023",
    20: "Zhang and Sennrich, 2019",
    21: "Selvaraju et al., 2017",
    22: "Abnar and Zuidema, 2020",
    23: "Hải, 2026c",
    24: "Zeiler and Fergus, 2014",
    25: "Nirob et al., 2025",
    26: "Erion et al., 2021",
    27: "Hải, 2026a",
    28: "Hải, 2026b",
    29: "Vaswani et al., 2017",
    30: "Snell et al., 2017",
    31: "Pan and Yang, 2010",
}

GROUPS = {
    "[1], [4], [5]": "(Joseph et al., 2024; Maurya et al., 2024; Duhan et al., 2025)",
    "[3], [6]": "(Gole et al., 2023; Li et al., 2023)",
}

SINGLE_CITATION = re.compile(
    r"(?<![A-Za-z0-9_])\[(" + "|".join(map(str, CITATIONS)) + r")\]"
)


def convert_text(text: str) -> str:
    for old, new in GROUPS.items():
        text = text.replace(old, new)

    text = SINGLE_CITATION.sub(lambda m: f"({CITATIONS[int(m.group(1))]})", text)

    # Avoid duplicated author wording after mechanical parenthetical conversion.
    text = text.replace(
        "Mishra và cộng sự (Mishra et al., 2025), Sujatha và cộng sự "
        "(Sujatha et al., 2025), cùng meta-ensemble nhẹ cho IoT của Maurya và "
        "cộng sự (Maurya et al., 2024)",
        "Mishra et al. (2025), Sujatha et al. (2025), cùng meta-ensemble nhẹ cho "
        "IoT của Maurya et al. (2024)",
    )
    text = text.replace(
        "PDLC-ViT (classification-only) (Hemalatha and Jayachandran, 2024)",
        "PDLC-ViT (classification-only; Hemalatha and Jayachandran, 2024)",
    )
    text = text.replace(
        "Đối với nghiên cứu (Sujatha et al., 2025)",
        "Đối với nghiên cứu của Sujatha et al. (2025)",
    )
    text = text.replace(
        "lấy nguyên bản từ (Nirob et al., 2025)",
        "lấy nguyên bản từ bộ dữ liệu của Nirob et al. (2025)",
    )
    text = text.replace(
        "chồng lấn với (Duhan et al., 2025) [96,18; 99,09]",
        "chồng lấn với khoảng [96,18; 99,09] của Duhan et al. (2025)",
    )
    return text


def main() -> None:
    files = sorted(VN_DIR.glob("0[0-8]*.md"))
    changed = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        converted = convert_text(original)
        if converted != original:
            path.write_text(converted, encoding="utf-8", newline="\n")
            changed += 1
            print(f"updated {path.relative_to(HERE)}")
    print(f"updated files: {changed}")


if __name__ == "__main__":
    main()
