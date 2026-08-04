#!/usr/bin/env python3
"""Finalize clean-query cross-dataset artifacts from two evaluation runs.

The evaluation notebook exports only misclassified samples.  Because both runs use
the same deterministic ImageFolder ordering, a complete prediction table can be
reconstructed by treating every index absent from a run's error manifest as a
correct prediction.  This script validates every error row against the frozen query
template before writing any result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
RESULTS = OUTPUTS / "results"
CANONICAL = RESULTS / "cross_dataset_manifests_v2"
PACKAGE = RESULTS / "kaggle_upload_ready_v2" / "chili-mendeley-cross-query"
PACKAGE_META = PACKAGE / "metadata"

DEFAULT_BASE_ERRORS = (
    HERE / "kq" / "misclassified_exports" / "20260804_194834_840663" / "misclassified_manifest.csv"
)
DEFAULT_FEWSHOT_ERRORS = (
    HERE / "kq" / "misclassified_exports" / "20260804_194224_476839" / "misclassified_manifest.csv"
)

CLASS_NAMES = ["anthracnose", "healthy", "leaf_curl", "leaf_spot", "yellow_leaf"]
PRESENT_CLASSES = CLASS_NAMES[1:]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar_p(b: int, c: int) -> str:
    n = b + c
    tail = min(b, c)
    getcontext().prec = 60
    p = Decimal(2) * sum(Decimal(math.comb(n, k)) for k in range(tail + 1)) / (Decimal(2) ** n)
    return f"{p:.16E}"


def validated_error_map(
    run_name: str, error_manifest: Path, template: list[dict[str, str]]
) -> dict[int, str]:
    rows = read_csv(error_manifest)
    errors: dict[int, str] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index < 0 or index >= len(template):
            raise ValueError(f"{run_name}: sample_index {index} is outside the query template")
        expected = template[index]
        if Path(row["source_path"]).name != expected["filename"]:
            raise ValueError(f"{run_name}: filename mismatch at query_index {index}")
        if row["true_class"] != expected["mapped_label"]:
            raise ValueError(f"{run_name}: true-label mismatch at query_index {index}")
        if row["predicted_class"] not in CLASS_NAMES:
            raise ValueError(f"{run_name}: unknown prediction {row['predicted_class']!r}")
        if index in errors:
            raise ValueError(f"{run_name}: duplicate sample_index {index}")
        errors[index] = row["predicted_class"]
    return errors


def classification_metrics(rows: list[dict[str, object]], pred_key: str) -> tuple[dict[str, float], list[dict[str, object]]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[str(row["true_label"])][str(row[pred_key])] += 1

    class_rows: list[dict[str, object]] = []
    for label in PRESENT_CLASSES:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[true][label] for true in PRESENT_CLASSES)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append(
            {
                "class_label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    total = len(rows)
    correct = sum(int(row[pred_key] == row["true_label"]) for row in rows)
    macro_precision = sum(float(row["precision"]) for row in class_rows) / len(class_rows)
    macro_recall = sum(float(row["recall"]) for row in class_rows) / len(class_rows)
    macro_f1 = sum(float(row["f1"]) for row in class_rows) / len(class_rows)
    weighted_f1 = sum(float(row["f1"]) * int(row["support"]) for row in class_rows) / total
    summary = {
        "query_image_count": total,
        "correct_count": correct,
        "accuracy": correct / total,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
    }
    return summary, class_rows


def update_run_metadata() -> None:
    path = PACKAGE_META / "cross_dataset_run_metadata.csv"
    rows = read_csv(path)
    for row in rows:
        row["evaluation_status"] = (
            "completed_clean_query_1463_images_on_2026-08-04; "
            "per_image_predictions_saved; paired_comparison_saved"
        )
    fields = list(rows[0])
    for destination in (path, CANONICAL / path.name):
        write_csv(destination, fields, rows)


def update_manifest_hashes(result_names: list[str]) -> None:
    tracked = [
        ("fewshot_support_manifest.csv", CANONICAL / "fewshot_support_manifest.csv"),
        ("cross_dataset_query_manifest.csv", PACKAGE_META / "cross_dataset_query_manifest.csv"),
        ("cross_dataset_split_summary.csv", PACKAGE_META / "cross_dataset_split_summary.csv"),
        ("cross_dataset_split_audit.csv", PACKAGE_META / "cross_dataset_split_audit.csv"),
        ("cross_dataset_predictions_template.csv", PACKAGE_META / "cross_dataset_predictions_template.csv"),
        ("cross_dataset_run_metadata.csv", PACKAGE_META / "cross_dataset_run_metadata.csv"),
    ]
    tracked.extend((name, PACKAGE_META / name) for name in result_names)
    rows = [
        {"filename": name, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for name, path in tracked
    ]
    for destination in (PACKAGE_META / "manifest_file_hashes.csv", CANONICAL / "manifest_file_hashes.csv"):
        write_csv(destination, ["filename", "sha256", "size_bytes"], rows)


def update_package_manifest() -> None:
    files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.name != "PACKAGE_FILE_MANIFEST.csv" and path.suffix.lower() != ".zip"
    )
    rows = [
        {
            "relative_path": path.relative_to(PACKAGE).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    write_csv(PACKAGE / "PACKAGE_FILE_MANIFEST.csv", ["relative_path", "size_bytes", "sha256"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-errors", type=Path, default=DEFAULT_BASE_ERRORS)
    parser.add_argument("--fewshot-errors", type=Path, default=DEFAULT_FEWSHOT_ERRORS)
    args = parser.parse_args()

    template_path = PACKAGE_META / "cross_dataset_predictions_template.csv"
    template = read_csv(template_path)
    if len(template) != 1463:
        raise ValueError(f"Expected 1463 query rows, found {len(template)}")

    base_errors = validated_error_map("base", args.base_errors.resolve(), template)
    fewshot_errors = validated_error_map("fewshot", args.fewshot_errors.resolve(), template)
    if len(base_errors) != 1010 or len(fewshot_errors) != 150:
        raise ValueError(
            f"Unexpected error counts: base={len(base_errors)}, fewshot={len(fewshot_errors)}"
        )

    predictions: list[dict[str, object]] = []
    for row in template:
        index = int(row["query_index"])
        true_label = row["mapped_label"]
        pred_base = base_errors.get(index, true_label)
        pred_fewshot = fewshot_errors.get(index, true_label)
        predictions.append(
            {
                "query_index": index,
                "source_relative_path": row["source_relative_path"],
                "filename": row["filename"],
                "sha256": row["sha256"],
                "source_label": row["source_label"],
                "true_label": true_label,
                "true_class_index": int(row["true_class_index"]),
                "pred_base": pred_base,
                "pred_base_class_index": CLASS_NAMES.index(pred_base),
                "pred_fewshot": pred_fewshot,
                "pred_fewshot_class_index": CLASS_NAMES.index(pred_fewshot),
                "correct_base": int(pred_base == true_label),
                "correct_fewshot": int(pred_fewshot == true_label),
            }
        )

    prediction_fields = list(predictions[0])
    base_summary, base_classes = classification_metrics(predictions, "pred_base")
    few_summary, few_classes = classification_metrics(predictions, "pred_fewshot")

    packaged_artifacts = {
        "results/base/info.txt": RESULTS / "Cross check default" / "infoo.txt",
        "results/base/confusion_matrix.png": RESULTS / "Cross check default" / "cross_dataset_local_cm.png",
        "results/fewshot/info.txt": RESULTS / "cross check them data" / "info.txt",
        "results/fewshot/confusion_matrix.png": RESULTS / "cross check them data" / "cross_dataset_local_cm.png",
    }
    for relative_path, source in packaged_artifacts.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = PACKAGE / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    expected = {"base": (453, 0.1766), "fewshot": (1313, 0.8744)}
    for name, summary in (("base", base_summary), ("fewshot", few_summary)):
        expected_correct, expected_f1 = expected[name]
        if int(summary["correct_count"]) != expected_correct:
            raise ValueError(f"{name}: correct-count check failed")
        if round(float(summary["f1_macro"]), 4) != expected_f1:
            raise ValueError(f"{name}: macro-F1 check failed")

    result_rows = []
    run_specs = [
        (
            "base_seed42_current",
            "Checkpoint goc, khong few-shot",
            "9dee091394ac8e888ab683a6fde3c251df34b8e952c9e693a4ffc6f35dadcea3",
            base_summary,
            "results/base/info.txt",
            "results/base/confusion_matrix.png",
        ),
        (
            "fewshot_seed42_current_misnamed_file",
            "Checkpoint sau few-shot",
            "c31f643e601fd39d6f2e9f910a7fdd2cbeb889e0bd5b2adb2db8ec694766f9c9",
            few_summary,
            "results/fewshot/info.txt",
            "results/fewshot/confusion_matrix.png",
        ),
    ]
    for run_id, configuration, checkpoint_hash, summary, info_file, cm_file in run_specs:
        result_rows.append(
            {
                "run_id": run_id,
                "configuration": configuration,
                "training_seed": 42,
                "query_image_count": summary["query_image_count"],
                "correct_count": summary["correct_count"],
                "accuracy": f"{float(summary['accuracy']):.8f}",
                "precision_macro": f"{float(summary['precision_macro']):.8f}",
                "recall_macro": f"{float(summary['recall_macro']):.8f}",
                "f1_macro": f"{float(summary['f1_macro']):.8f}",
                "f1_weighted": f"{float(summary['f1_weighted']):.8f}",
                "checkpoint_sha256": checkpoint_hash,
                "source_info_file": info_file,
                "confusion_matrix_file": cm_file,
            }
        )

    class_result_rows: list[dict[str, object]] = []
    for run_id, rows in (("base_seed42_current", base_classes), ("fewshot_seed42_current_misnamed_file", few_classes)):
        for row in rows:
            class_result_rows.append(
                {
                    "run_id": run_id,
                    "class_label": row["class_label"],
                    "precision": f"{float(row['precision']):.8f}",
                    "recall": f"{float(row['recall']):.8f}",
                    "f1": f"{float(row['f1']):.8f}",
                    "support": row["support"],
                }
            )

    both_correct = sum(row["correct_base"] == 1 and row["correct_fewshot"] == 1 for row in predictions)
    base_wrong_few_correct = sum(row["correct_base"] == 0 and row["correct_fewshot"] == 1 for row in predictions)
    base_correct_few_wrong = sum(row["correct_base"] == 1 and row["correct_fewshot"] == 0 for row in predictions)
    both_wrong = sum(row["correct_base"] == 0 and row["correct_fewshot"] == 0 for row in predictions)
    paired_rows = [
        {
            "query_image_count": len(predictions),
            "both_correct": both_correct,
            "base_wrong_fewshot_correct": base_wrong_few_correct,
            "base_correct_fewshot_wrong": base_correct_few_wrong,
            "both_wrong": both_wrong,
            "accuracy_difference_percentage_points": f"{(float(few_summary['accuracy']) - float(base_summary['accuracy'])) * 100:.8f}",
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(base_wrong_few_correct, base_correct_few_wrong),
        }
    ]

    outputs = {
        "cross_dataset_predictions.csv": (prediction_fields, predictions),
        "cross_dataset_evaluation_results.csv": (list(result_rows[0]), result_rows),
        "cross_dataset_per_class_metrics.csv": (list(class_result_rows[0]), class_result_rows),
        "cross_dataset_paired_comparison.csv": (list(paired_rows[0]), paired_rows),
    }
    for name, (fields, rows) in outputs.items():
        for destination in (PACKAGE_META / name, CANONICAL / name):
            write_csv(destination, fields, rows)

    update_run_metadata()
    update_manifest_hashes(list(outputs))
    update_package_manifest()

    print(f"query={len(predictions)} base_correct={base_summary['correct_count']} fewshot_correct={few_summary['correct_count']}")
    print(
        "paired="
        f"{both_correct},{base_wrong_few_correct},{base_correct_few_wrong},{both_wrong} "
        f"p={paired_rows[0]['mcnemar_exact_two_sided_p']}"
    )


if __name__ == "__main__":
    main()
