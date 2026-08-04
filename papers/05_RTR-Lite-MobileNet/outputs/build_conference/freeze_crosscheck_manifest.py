from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")

MANIFEST_FILE = OUTPUT_ROOT / "CrossCheck_v1.0_manifest.csv"
DUPLICATE_FILE = OUTPUT_ROOT / "CrossCheck_v1.0_duplicate_hashes.csv"
SUMMARY_FILE = OUTPUT_ROOT / "CrossCheck_v1.0_summary.json"

EXPECTED_CLASS_COUNTS = {
    "anthracnose": 166,
    "healthy": 125,
    "leaf_curl": 155,
    "leaf_spot": 160,
    "yellow_leaf": 156,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_crosscheck_root() -> Path:
    expected_classes = set(EXPECTED_CLASS_COUNTS)
    candidates: list[Path] = []

    for directory in INPUT_ROOT.rglob("*"):
        if not directory.is_dir():
            continue

        child_directories = {
            child.name
            for child in directory.iterdir()
            if child.is_dir()
        }

        if expected_classes.issubset(child_directories):
            candidates.append(directory)

    named_candidates = [
        directory
        for directory in candidates
        if "cross" in directory.as_posix().lower()
    ]

    if len(named_candidates) == 1:
        return named_candidates[0]

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one cross-check root containing "
            f"{sorted(expected_classes)}, found {len(candidates)}: "
            f"{candidates}"
        )

    return candidates[0]


def main() -> None:
    crosscheck_root = find_crosscheck_root()
    rows: list[dict[str, object]] = []

    for label in EXPECTED_CLASS_COUNTS:
        class_directory = crosscheck_root / label

        for image_path in sorted(class_directory.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rows.append(
                {
                    "dataset_release": "CrossCheck-v1.0",
                    "relative_path": image_path.relative_to(
                        crosscheck_root
                    ).as_posix(),
                    "role": "independent_cross_check_evaluation",
                    "label": label,
                    "filename": image_path.name,
                    "extension": image_path.suffix.lower(),
                    "size_bytes": image_path.stat().st_size,
                    "sha256": calculate_sha256(image_path),
                }
            )

    columns = [
        "dataset_release",
        "relative_path",
        "role",
        "label",
        "filename",
        "extension",
        "size_bytes",
        "sha256",
    ]

    with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    manifest = pd.read_csv(MANIFEST_FILE)
    actual_class_counts = manifest["label"].value_counts().to_dict()

    assert len(manifest) == 762, (
        f"Expected 762 cross-check images, found {len(manifest)}"
    )
    assert actual_class_counts == EXPECTED_CLASS_COUNTS, {
        "expected": EXPECTED_CLASS_COUNTS,
        "actual": actual_class_counts,
    }
    assert manifest["relative_path"].is_unique, (
        "Duplicate relative paths found"
    )

    duplicates = manifest[
        manifest.duplicated(subset="sha256", keep=False)
    ].copy()
    duplicates = duplicates.sort_values(
        ["sha256", "label", "relative_path"]
    )
    duplicates.to_csv(DUPLICATE_FILE, index=False, encoding="utf-8")

    summary = {
        "dataset_release": "CrossCheck-v1.0",
        "dataset_root": crosscheck_root.name,
        "total_images": int(len(manifest)),
        "class_counts": actual_class_counts,
        "duplicate_hash_rows": int(len(duplicates)),
        "manifest_sha256": calculate_sha256(MANIFEST_FILE),
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCreated files:")
    print(MANIFEST_FILE)
    print(DUPLICATE_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
