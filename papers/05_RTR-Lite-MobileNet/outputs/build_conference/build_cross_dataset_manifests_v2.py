from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path


ORIGINAL_ROOT = Path(
    r"D:\ds\Chili Leaf Disease Original Dataset\Chili Leaf Disease Original Dataset"
)
SUPPORT_ROOT = Path(r"D:\ds\new_ds")
CURRENT_TEST_ROOT = Path(r"D:\ds\test")

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent / "results" / "cross_dataset_manifests_v2"
MODEL_ROOT = HERE / "kq" / "model"

DATASET_TITLE = "Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh"
DATASET_VERSION = "1"
DATASET_DOI = "10.17632/w9mr3vf56s.1"
DATASET_LICENSE = "CC BY 4.0"
DATASET_URL = "https://data.mendeley.com/datasets/w9mr3vf56s/1"

# Thu tu nay cung la thu tu trinh bay trong bai bao.
CLASS_MAPPING = {
    "Healthy Leaf": "healthy",
    "Curl Virus": "leaf_curl",
    "Cercospora Leaf Spot": "leaf_spot",
    "Nutrition Deficiency": "yellow_leaf",
}

GLOBAL_CLASS_INDEX = {
    "anthracnose": 0,
    "healthy": 1,
    "leaf_curl": 2,
    "leaf_spot": 3,
    "yellow_leaf": 4,
}

EXPECTED_SOURCE_COUNTS = {
    "healthy": 458,
    "leaf_curl": 423,
    "leaf_spot": 180,
    "yellow_leaf": 444,
}

EXPECTED_SUPPORT_COUNTS = {
    "healthy": 8,
    "leaf_curl": 9,
    "leaf_spot": 16,
    "yellow_leaf": 9,
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

MANIFEST_COLUMNS = [
    "dataset_title",
    "dataset_version",
    "dataset_doi",
    "dataset_url",
    "license",
    "source_relative_path",
    "filename",
    "extension",
    "size_bytes",
    "sha256",
    "source_label",
    "mapped_label",
    "global_class_index",
    "role",
    "selection_method",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def common_fields(
    path: Path,
    source_label: str,
    mapped_label: str,
    role: str,
    selection_method: str,
) -> dict[str, object]:
    return {
        "dataset_title": DATASET_TITLE,
        "dataset_version": DATASET_VERSION,
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "license": DATASET_LICENSE,
        "source_relative_path": path.relative_to(ORIGINAL_ROOT).as_posix(),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_label": source_label,
        "mapped_label": mapped_label,
        "global_class_index": GLOBAL_CLASS_INDEX[mapped_label],
        "role": role,
        "selection_method": selection_method,
    }


def main() -> None:
    for required in (ORIGINAL_ROOT, SUPPORT_ROOT, CURRENT_TEST_ROOT):
        if not required.is_dir():
            raise FileNotFoundError(required)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source_by_class: dict[str, list[Path]] = {}
    source_by_name: dict[tuple[str, str], Path] = {}
    for source_label, mapped_label in CLASS_MAPPING.items():
        files = image_files(ORIGINAL_ROOT / source_label)
        source_by_class[mapped_label] = files
        for path in files:
            key = (mapped_label, path.name.casefold())
            if key in source_by_name:
                raise RuntimeError(f"Duplicate source filename within class: {key}")
            source_by_name[key] = path

    actual_source_counts = {
        mapped_label: len(files)
        for mapped_label, files in source_by_class.items()
    }
    if actual_source_counts != EXPECTED_SOURCE_COUNTS:
        raise AssertionError(
            {"expected_source": EXPECTED_SOURCE_COUNTS, "actual_source": actual_source_counts}
        )

    support_rows: list[dict[str, object]] = []
    support_hashes: set[str] = set()
    support_names: set[tuple[str, str]] = set()

    reverse_mapping = {mapped: source for source, mapped in CLASS_MAPPING.items()}
    for mapped_label in CLASS_MAPPING.values():
        source_label = reverse_mapping[mapped_label]
        for support_path in image_files(SUPPORT_ROOT / mapped_label):
            key = (mapped_label, support_path.name.casefold())
            source_path = source_by_name.get(key)
            if source_path is None:
                raise AssertionError(f"Support file is absent from the original dataset: {support_path}")

            source_hash = sha256(source_path)
            support_hash = sha256(support_path)
            if source_hash != support_hash:
                raise AssertionError(
                    f"Support file differs from its source file: {support_path} != {source_path}"
                )
            if support_hash in support_hashes:
                raise AssertionError(f"Duplicate SHA-256 inside support: {support_path}")

            row = common_fields(
                source_path,
                source_label,
                mapped_label,
                role="support",
                selection_method=(
                    "manually_selected_by_study_author; "
                    "selection_criterion_not_documented"
                ),
            )
            support_rows.append(row)
            support_hashes.add(support_hash)
            support_names.add(key)

    actual_support_counts = dict(Counter(row["mapped_label"] for row in support_rows))
    if actual_support_counts != EXPECTED_SUPPORT_COUNTS:
        raise AssertionError(
            {"expected_support": EXPECTED_SUPPORT_COUNTS, "actual_support": actual_support_counts}
        )

    query_rows: list[dict[str, object]] = []
    for source_label, mapped_label in CLASS_MAPPING.items():
        for source_path in source_by_class[mapped_label]:
            if sha256(source_path) in support_hashes:
                continue
            query_rows.append(
                common_fields(
                    source_path,
                    source_label,
                    mapped_label,
                    role="query",
                    selection_method="complement_of_support_within_four_mapped_source_classes",
                )
            )

    expected_query_counts = {
        label: EXPECTED_SOURCE_COUNTS[label] - EXPECTED_SUPPORT_COUNTS[label]
        for label in EXPECTED_SOURCE_COUNTS
    }
    actual_query_counts = dict(Counter(row["mapped_label"] for row in query_rows))
    if actual_query_counts != expected_query_counts:
        raise AssertionError(
            {"expected_query": expected_query_counts, "actual_query": actual_query_counts}
        )
    if len(query_rows) != 1463:
        raise AssertionError(f"Expected 1463 query images, found {len(query_rows)}")

    query_hashes = {str(row["sha256"]) for row in query_rows}
    if support_hashes & query_hashes:
        raise AssertionError("Support/query SHA-256 overlap in the intended split")

    current_test_rows: list[dict[str, object]] = []
    for mapped_label in CLASS_MAPPING.values():
        source_label = reverse_mapping[mapped_label]
        for test_path in image_files(CURRENT_TEST_ROOT / mapped_label):
            current_test_rows.append(
                {
                    "mapped_label": mapped_label,
                    "source_label": source_label,
                    "filename": test_path.name,
                    "sha256": sha256(test_path),
                }
            )

    current_test_hashes = {str(row["sha256"]) for row in current_test_rows}
    current_overlap = sorted(current_test_hashes & support_hashes)
    missing_from_current = sorted(query_hashes - current_test_hashes)
    extra_in_current = sorted(current_test_hashes - query_hashes)

    support_path = OUTPUT_ROOT / "fewshot_support_manifest.csv"
    query_path = OUTPUT_ROOT / "cross_dataset_query_manifest.csv"
    summary_path = OUTPUT_ROOT / "cross_dataset_split_summary.csv"
    audit_path = OUTPUT_ROOT / "cross_dataset_split_audit.csv"
    prediction_template_path = OUTPUT_ROOT / "cross_dataset_predictions_template.csv"
    run_metadata_path = OUTPUT_ROOT / "cross_dataset_run_metadata.csv"
    hash_path = OUTPUT_ROOT / "manifest_file_hashes.csv"

    write_csv(support_path, MANIFEST_COLUMNS, support_rows)
    write_csv(query_path, MANIFEST_COLUMNS, query_rows)

    current_counts = Counter(row["mapped_label"] for row in current_test_rows)
    current_overlap_counts = Counter(
        row["mapped_label"]
        for row in current_test_rows
        if str(row["sha256"]) in support_hashes
    )
    summary_rows: list[dict[str, object]] = []
    for mapped_label in CLASS_MAPPING.values():
        summary_rows.append(
            {
                "mapped_label": mapped_label,
                "source_count": EXPECTED_SOURCE_COUNTS[mapped_label],
                "support_count": EXPECTED_SUPPORT_COUNTS[mapped_label],
                "expected_query_count": expected_query_counts[mapped_label],
                "current_test_count": current_counts[mapped_label],
                "current_support_overlap_count": current_overlap_counts[mapped_label],
                "current_test_status": (
                    "PASS"
                    if current_counts[mapped_label] == expected_query_counts[mapped_label]
                    and current_overlap_counts[mapped_label] == 0
                    else "FAIL"
                ),
            }
        )
    summary_rows.append(
        {
            "mapped_label": "TOTAL",
            "source_count": sum(EXPECTED_SOURCE_COUNTS.values()),
            "support_count": sum(EXPECTED_SUPPORT_COUNTS.values()),
            "expected_query_count": sum(expected_query_counts.values()),
            "current_test_count": len(current_test_rows),
            "current_support_overlap_count": len(current_overlap),
            "current_test_status": (
                "PASS"
                if len(current_test_rows) == 1463 and not current_overlap
                else "FAIL"
            ),
        }
    )
    write_csv(
        summary_path,
        [
            "mapped_label",
            "source_count",
            "support_count",
            "expected_query_count",
            "current_test_count",
            "current_support_overlap_count",
            "current_test_status",
        ],
        summary_rows,
    )

    overlap_details = []
    for row in current_test_rows:
        if str(row["sha256"]) in support_hashes:
            overlap_details.append(
                f"{row['mapped_label']}/{row['filename']}|{row['sha256']}"
            )
    audit_rows = [
        {
            "check": "support_count",
            "status": "PASS" if len(support_rows) == 42 else "FAIL",
            "expected": 42,
            "actual": len(support_rows),
            "details": "",
        },
        {
            "check": "intended_query_count",
            "status": "PASS" if len(query_rows) == 1463 else "FAIL",
            "expected": 1463,
            "actual": len(query_rows),
            "details": "Generated as the four-class source complement of support by SHA-256",
        },
        {
            "check": "intended_support_query_hash_overlap",
            "status": "PASS" if not (support_hashes & query_hashes) else "FAIL",
            "expected": 0,
            "actual": len(support_hashes & query_hashes),
            "details": "",
        },
        {
            "check": "current_test_count",
            "status": "PASS" if len(current_test_rows) == 1463 else "FAIL",
            "expected": 1463,
            "actual": len(current_test_rows),
            "details": str(CURRENT_TEST_ROOT),
        },
        {
            "check": "current_test_support_hash_overlap",
            "status": "PASS" if not current_overlap else "FAIL",
            "expected": 0,
            "actual": len(current_overlap),
            "details": "; ".join(overlap_details),
        },
        {
            "check": "expected_query_missing_from_current_test",
            "status": "PASS" if not missing_from_current else "FAIL",
            "expected": 0,
            "actual": len(missing_from_current),
            "details": "; ".join(missing_from_current),
        },
        {
            "check": "unexpected_hashes_in_current_test",
            "status": "PASS" if not extra_in_current else "FAIL",
            "expected": 0,
            "actual": len(extra_in_current),
            "details": "; ".join(extra_in_current),
        },
    ]
    write_csv(
        audit_path,
        ["check", "status", "expected", "actual", "details"],
        audit_rows,
    )

    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(query_rows):
        prediction_rows.append(
            {
                "query_index": index,
                "source_relative_path": row["source_relative_path"],
                "filename": row["filename"],
                "sha256": row["sha256"],
                "source_label": row["source_label"],
                "mapped_label": row["mapped_label"],
                "true_class_index": row["global_class_index"],
                "pred_base": "",
                "pred_fewshot": "",
                "correct_base": "",
                "correct_fewshot": "",
            }
        )
    write_csv(
        prediction_template_path,
        [
            "query_index",
            "source_relative_path",
            "filename",
            "sha256",
            "source_label",
            "mapped_label",
            "true_class_index",
            "pred_base",
            "pred_fewshot",
            "correct_base",
            "correct_fewshot",
        ],
        prediction_rows,
    )

    support_manifest_hash = sha256(support_path)
    query_manifest_hash = sha256(query_path)
    run_metadata_rows: list[dict[str, object]] = []
    for run_id, checkpoint_name, seed, training_data, provenance_status in [
        (
            "base_seed42_current",
            "c14_s2_seed42.pth",
            42,
            "ChiliEdge-DS only",
            "checkpoint load recorded in the executed CrossDataset_Local notebook",
        ),
        (
            "fewshot_seed52_current",
            "c14_s2_seed52.pth",
            52,
            "ChiliEdge-DS + 42 author-selected support images",
            "needs archived executed training notebook/log proving the exact support manifest",
        ),
        (
            "fewshot_seed42_planned",
            "c14_s2_fewshot_seed42.pth",
            42,
            "ChiliEdge-DS + 42 author-selected support images",
            "planned; checkpoint not created yet",
        ),
    ]:
        checkpoint_path = MODEL_ROOT / checkpoint_name
        run_metadata_rows.append(
            {
                "run_id": run_id,
                "checkpoint_filename": checkpoint_name,
                "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path.is_file() else "",
                "training_seed": seed,
                "architecture": "ChiliEdgeNet-C14 S2",
                "training_data": training_data,
                "training_provenance_status": provenance_status,
                "support_manifest_filename": (
                    support_path.name if "support" in training_data else ""
                ),
                "support_manifest_sha256": (
                    support_manifest_hash if "support" in training_data else ""
                ),
                "query_manifest_filename": query_path.name,
                "query_manifest_sha256": query_manifest_hash,
                "query_image_count": 1463,
                "image_size": 224,
                "normalization": "ImageNet mean/std",
                "evaluation_status": (
                    "planned_clean_query_run"
                    if run_id.endswith("planned")
                    else "current_output_used_1464_images; rerun_required"
                ),
            }
        )
    write_csv(
        run_metadata_path,
        [
            "run_id",
            "checkpoint_filename",
            "checkpoint_sha256",
            "training_seed",
            "architecture",
            "training_data",
            "training_provenance_status",
            "support_manifest_filename",
            "support_manifest_sha256",
            "query_manifest_filename",
            "query_manifest_sha256",
            "query_image_count",
            "image_size",
            "normalization",
            "evaluation_status",
        ],
        run_metadata_rows,
    )

    generated = [
        support_path,
        query_path,
        summary_path,
        audit_path,
        prediction_template_path,
        run_metadata_path,
    ]
    hash_rows = [
        {
            "filename": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in generated
    ]
    write_csv(hash_path, ["filename", "sha256", "size_bytes"], hash_rows)

    print(f"Support: {len(support_rows)}")
    print(f"Intended clean query: {len(query_rows)}")
    print(f"Current test: {len(current_test_rows)}")
    print(f"Current support/test hash overlap: {len(current_overlap)}")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
