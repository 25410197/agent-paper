from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_ROOT = REPO_ROOT / "papers" / "05_RTR-Lite-MobileNet"
MANIFEST_ROOT = PAPER_ROOT / "outputs" / "results" / "cross_dataset_manifests_v2"
OUTPUT_ROOT = PAPER_ROOT / "outputs" / "results" / "kaggle_upload_ready_v2"
SOURCE_ROOT = Path(
    r"D:\ds\Chili Leaf Disease Original Dataset\Chili Leaf Disease Original Dataset"
)

SOURCE_TITLE = "Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh"
SOURCE_VERSION = "1"
SOURCE_DOI = "10.17632/w9mr3vf56s.1"
SOURCE_URL = "https://data.mendeley.com/datasets/w9mr3vf56s/1"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DERIVED_AUTHOR = "Tô Minh Hải"

SUPPORT_SLUG = "chili-mendeley-fewshot-support"
QUERY_SLUG = "chili-mendeley-cross-query"
SUPPORT_TITLE = "Chili Leaf Disease Few-Shot"
QUERY_TITLE = "Chili Leaf Disease Cross-Dataset Query Set"
SUPPORT_KAGGLE_URL = "https://www.kaggle.com/datasets/tminhhi/chili-leaf-disease-few-shot/version/1"
QUERY_KAGGLE_URL = "https://www.kaggle.com/datasets/tminhhi/chili-leaf-disease-cross-dataset-query-set/version/2"

CLASS_MAP = {
    "Cercospora Leaf Spot": "leaf_spot",
    "Curl Virus": "leaf_curl",
    "Healthy Leaf": "healthy",
    "Nutrition Deficiency": "yellow_leaf",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_metadata(package_root: Path, filenames: list[str]) -> None:
    metadata_root = package_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = MANIFEST_ROOT / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, metadata_root / filename)


def copy_images(
    rows: list[dict[str, str]], package_root: Path, role: str
) -> tuple[int, int]:
    image_root = package_root / role
    copied_bytes = 0
    seen_destinations: set[Path] = set()

    for row in rows:
        source = SOURCE_ROOT / Path(row["source_relative_path"])
        destination = image_root / row["mapped_label"] / row["filename"]
        if destination in seen_destinations:
            raise ValueError(f"Duplicate package path: {destination}")
        seen_destinations.add(destination)

        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Size mismatch: {source}")
        if sha256(source) != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_bytes += destination.stat().st_size

    return len(rows), copied_bytes


def class_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row["mapped_label"] for row in rows)


def count_table(counts: Counter[str]) -> str:
    lines = ["| Class | Images |", "|---|---:|"]
    for label in ("healthy", "leaf_curl", "leaf_spot", "yellow_leaf"):
        lines.append(f"| `{label}` | {counts[label]:,} |")
    lines.append(f"| **Total** | **{sum(counts.values()):,}** |")
    return "\n".join(lines)


def attribution_text() -> str:
    return f"""
# Source attribution

The images in this package are redistributed from the following source dataset:

- **Title:** {SOURCE_TITLE}
- **Authors:** Md Asraful Sharker Nirob; A K M Fazlul Kobir Siam; Prayma Bishshash; Md Assaduzzaman
- **Version:** {SOURCE_VERSION}
- **Published:** 19 February 2025
- **DOI:** [{SOURCE_DOI}](https://doi.org/{SOURCE_DOI})
- **Source page:** [{SOURCE_URL}]({SOURCE_URL})
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)]({LICENSE_URL})

The derived split and accompanying manifests were prepared by {DERIVED_AUTHOR}. This
package does not claim ownership of the source images and does not imply endorsement by
the original authors.

## Changes made

- Retained four source classes and mapped their directory names as follows:
  `Cercospora Leaf Spot` → `leaf_spot`, `Curl Virus` → `leaf_curl`,
  `Healthy Leaf` → `healthy`, and `Nutrition Deficiency` → `yellow_leaf`.
- Excluded source classes not used in this experiment.
- Selected 42 support images manually by the study author. The detailed visual
  selection criterion was not recorded.
- Defined the query split as the SHA-256 complement of those 42 support images within
  the four retained source classes.
- Renamed only the parent class directories in this package. Image pixels and original
  image filenames were not modified.
"""


def license_text() -> str:
    return f"""
SOURCE IMAGE LICENSE NOTICE

The source images in this package come from:
{SOURCE_TITLE}, version {SOURCE_VERSION}
DOI: https://doi.org/{SOURCE_DOI}
Source: {SOURCE_URL}

The source dataset is distributed under the Creative Commons Attribution 4.0
International license (CC BY 4.0):
{LICENSE_URL}

Under CC BY 4.0, reuse and redistribution are permitted provided appropriate credit is
given, a link to the license is supplied, and changes are indicated. The source images
remain attributable to their original authors. The derived split definitions,
manifests, and documentation in this package were prepared by {DERIVED_AUTHOR}.

No additional restriction is imposed by this notice. No endorsement by the original
authors is implied.
"""


def citation_text(derived_title: str, kaggle_url: str) -> str:
    return f"""
# Citation

When using this package, cite the original image dataset and describe this package as a
derived experimental split.

## Original image dataset

Nirob, M. A. S., Siam, A. K. M. F. K., Bishshash, P., & Assaduzzaman, M. (2025).
*{SOURCE_TITLE}* (Version {SOURCE_VERSION}) [Data set]. Mendeley Data.
https://doi.org/{SOURCE_DOI}

## Derived split

{DERIVED_AUTHOR}. (2026). *{derived_title}* (Version 1)
[Derived data split and manifests]. Kaggle.
{kaggle_url}

Derived from Nirob et al. (2025), version {SOURCE_VERSION}, CC BY 4.0. Do not replace
the original-dataset citation with the derived-split citation.
"""


def support_readme(rows: list[dict[str, str]]) -> str:
    return f"""
# {SUPPORT_TITLE}

This package contains the **42 support images actually used for few-shot adaptation** in
the RTR-Lite-MobileNet cross-dataset experiment. It is a derived subset of
[{SOURCE_TITLE}]({SOURCE_URL}), not a newly collected image dataset.

Published as [Kaggle dataset version 1]({SUPPORT_KAGGLE_URL}).

{count_table(class_counts(rows))}

## Directory layout

```text
support/
  healthy/
  leaf_curl/
  leaf_spot/
  yellow_leaf/
metadata/
  fewshot_support_manifest.csv
  cross_dataset_split_summary.csv
  manifest_file_hashes.csv
```

The manifest records the source-relative path, original filename, source and mapped
labels, file size, SHA-256 digest, role, and selection method for every image.

## Selection and limitations

The study author, {DERIVED_AUTHOR}, manually selected these 42 images. The detailed
visual selection criterion was not recorded, so the support set should be treated as a
fixed, reproducible convenience sample rather than a random or statistically balanced
sample. The package contains 8 healthy, 9 leaf-curl, 16 leaf-spot, and 9 yellow-leaf
images.

The source dataset has no source class mapped to the study's `anthracnose` label; that
target class is therefore absent from both support and query data. Source classes not
used by the experiment were excluded.

## Integrity and leakage control

Use `sha256` in `metadata/fewshot_support_manifest.csv` as the stable image identifier.
The companion query package is the hash complement within the same four retained
source classes. The intended support/query intersection is zero.

See `ATTRIBUTION.md`, `LICENSE.txt`, and `CITATION.md` before redistributing or using
the images.
"""


def query_readme(rows: list[dict[str, str]]) -> str:
    return f"""
# {QUERY_TITLE}

This package contains the **1,463-image clean query split** used to evaluate the
cross-dataset experiment. It is a derived subset of
[{SOURCE_TITLE}]({SOURCE_URL}), not a newly collected image dataset.

Published as [Kaggle dataset version 2]({QUERY_KAGGLE_URL}).

{count_table(class_counts(rows))}

## Directory layout

```text
query/
  healthy/
  leaf_curl/
  leaf_spot/
  yellow_leaf/
metadata/
  cross_dataset_query_manifest.csv
  cross_dataset_split_summary.csv
  cross_dataset_split_audit.csv
  cross_dataset_predictions_template.csv
  cross_dataset_run_metadata.csv
  manifest_file_hashes.csv
```

The query split is the SHA-256 complement of the 42-image support split within the
four retained source classes. The intended support/query intersection is zero. In
particular, `Cercospora Leaf Spot00176.JPG` is a support image and is deliberately not
included in this query package.

## Intended use

Use this package only as the held-out query/test split. Do not use these images for
training, support selection, model selection, or hyperparameter tuning. Report results
on all 1,463 images and publish per-image predictions using
`metadata/cross_dataset_predictions_template.csv`.

The two preliminary runs currently recorded in `cross_dataset_run_metadata.csv` used
different seeds and an earlier 1,464-image test folder containing one support overlap.
They are retained for provenance, not as final clean-query results. Final base and
few-shot models should be evaluated on this exact package under the same seed and
protocol.

## Limitations

The source dataset has no source class mapped to the study's `anthracnose` label; that
target class is absent from this query split. Source classes not used by the experiment
were excluded. The class mapping is an experimental label alignment and should not be
interpreted as a new expert relabeling of individual images.

See `ATTRIBUTION.md`, `LICENSE.txt`, and `CITATION.md` before redistributing or using
the images.
"""


def write_common_files(
    package_root: Path,
    title: str,
    kaggle_url: str,
    readme: str,
) -> None:
    write_text(package_root / "README.md", readme)
    write_text(package_root / "ATTRIBUTION.md", attribution_text())
    write_text(package_root / "LICENSE.txt", license_text())
    write_text(package_root / "CITATION.md", citation_text(title, kaggle_url))
    write_json(
        package_root / "dataset-metadata.json",
        {
            "title": title,
            "licenses": [{"name": "CC-BY-4.0"}],
        },
    )


def package_file_manifest(package_root: Path) -> tuple[int, int]:
    rows: list[dict[str, str | int]] = []
    for path in sorted(p for p in package_root.rglob("*") if p.is_file()):
        if path.name == "PACKAGE_FILE_MANIFEST.csv":
            continue
        relative = path.relative_to(package_root).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = package_root / "PACKAGE_FILE_MANIFEST.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows) + 1, sum(int(row["size_bytes"]) for row in rows) + manifest.stat().st_size


def upload_instructions(support_size: int, query_size: int) -> str:
    return f"""
# Hướng dẫn upload Kaggle

## Phiên bản đã công bố

- Query, version 2: {QUERY_KAGGLE_URL}
- Few-shot support, version 1: {SUPPORT_KAGGLE_URL}

Hai thư mục con trong thư mục này là **hai Kaggle Dataset độc lập** đã sẵn sàng để
upload:

1. `{SUPPORT_SLUG}` — 42 ảnh support few-shot, khoảng {support_size / 1024**2:.1f} MiB.
2. `{QUERY_SLUG}` — 1.463 ảnh query sạch, khoảng {query_size / 1024**2:.1f} MiB.

## Trước khi upload

Hai file `dataset-metadata.json` không chứa trường `id`; tên/slug và chủ sở hữu được
chọn trên giao diện Kaggle. Giữ giấy phép là **CC BY 4.0**. Không đổi tên hoặc xóa các file
`README.md`, `ATTRIBUTION.md`, `LICENSE.txt`, `CITATION.md`, hay thư mục `metadata`.

Không upload `D:\\ds\\test`: thư mục đó có 1.464 ảnh và còn trùng một ảnh support. Chỉ
upload hai thư mục đã tạo ở đây.

## Upload bằng giao diện Kaggle

- Tạo một Dataset mới cho từng gói.
- Upload toàn bộ **nội dung** của đúng thư mục gói, gồm ảnh, metadata và tài liệu.
- Để dataset ở chế độ private trong lần upload đầu tiên.
- Kiểm tra số ảnh: support = 42; query = 1.463.
- Kiểm tra README và attribution hiển thị đúng, rồi mới chuyển public.

## Upload bằng Kaggle CLI

Từ PowerShell, sau khi đã cấu hình API token:

```powershell
kaggle datasets create -p "<đường-dẫn>\\{SUPPORT_SLUG}" --dir-mode zip
kaggle datasets create -p "<đường-dẫn>\\{QUERY_SLUG}" --dir-mode zip
```

`--dir-mode zip` là cần thiết để Kaggle không bỏ qua các thư mục ảnh con. Lệnh tạo
dataset mặc định giữ dataset private. Chỉ thêm `--public` sau khi đã kiểm tra đầy đủ.
Nếu cập nhật phiên bản sau này, dùng `kaggle datasets version` cùng `--dir-mode zip` và
ghi rõ thay đổi trong version notes.

## Đường dẫn dự kiến trong Kaggle Notebook

```text
/kaggle/input/{SUPPORT_SLUG}/support
/kaggle/input/{QUERY_SLUG}/query
```

Trong bài báo, luôn cite cả DOI của bộ gốc và URL/version của hai gói derived này.
Vì trường `id` đã được bỏ khỏi metadata theo lựa chọn của tác giả, nên dùng giao diện
Kaggle khi cập nhật version tiếp theo hoặc thêm lại đúng dataset id trước khi dùng CLI.
"""


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists; review or remove it before rebuilding: {OUTPUT_ROOT}"
        )

    support_rows = read_csv(MANIFEST_ROOT / "fewshot_support_manifest.csv")
    query_rows = read_csv(MANIFEST_ROOT / "cross_dataset_query_manifest.csv")
    if len(support_rows) != 42:
        raise ValueError(f"Expected 42 support rows, found {len(support_rows)}")
    if len(query_rows) != 1463:
        raise ValueError(f"Expected 1463 query rows, found {len(query_rows)}")

    support_hashes = {row["sha256"] for row in support_rows}
    query_hashes = {row["sha256"] for row in query_rows}
    overlap = support_hashes & query_hashes
    if overlap:
        raise ValueError(f"Support/query SHA-256 overlap: {len(overlap)}")

    support_root = OUTPUT_ROOT / SUPPORT_SLUG
    query_root = OUTPUT_ROOT / QUERY_SLUG
    support_root.mkdir(parents=True)
    query_root.mkdir(parents=True)

    support_count, support_image_bytes = copy_images(
        support_rows, support_root, "support"
    )
    query_count, query_image_bytes = copy_images(query_rows, query_root, "query")

    copy_metadata(
        support_root,
        [
            "fewshot_support_manifest.csv",
            "cross_dataset_split_summary.csv",
            "manifest_file_hashes.csv",
        ],
    )
    copy_metadata(
        query_root,
        [
            "cross_dataset_query_manifest.csv",
            "cross_dataset_split_summary.csv",
            "cross_dataset_split_audit.csv",
            "cross_dataset_predictions_template.csv",
            "cross_dataset_run_metadata.csv",
            "manifest_file_hashes.csv",
        ],
    )

    write_common_files(
        support_root,
        SUPPORT_TITLE,
        SUPPORT_KAGGLE_URL,
        support_readme(support_rows),
    )
    write_common_files(
        query_root,
        QUERY_TITLE,
        QUERY_KAGGLE_URL,
        query_readme(query_rows),
    )

    support_files, support_total_bytes = package_file_manifest(support_root)
    query_files, query_total_bytes = package_file_manifest(query_root)

    write_text(
        OUTPUT_ROOT / "UPLOAD_INSTRUCTIONS.md",
        upload_instructions(support_total_bytes, query_total_bytes),
    )
    write_json(
        OUTPUT_ROOT / "package_summary.json",
        {
            "source": {
                "title": SOURCE_TITLE,
                "version": SOURCE_VERSION,
                "doi": SOURCE_DOI,
                "url": SOURCE_URL,
                "license": "CC BY 4.0",
            },
            "derived_split_author": DERIVED_AUTHOR,
            "support": {
                "package": SUPPORT_SLUG,
                "kaggle_id": "tminhhi/chili-leaf-disease-few-shot",
                "kaggle_version": 1,
                "kaggle_url": SUPPORT_KAGGLE_URL,
                "images": support_count,
                "image_bytes": support_image_bytes,
                "package_files": support_files,
                "package_bytes": support_total_bytes,
                "class_counts": dict(sorted(class_counts(support_rows).items())),
            },
            "query": {
                "package": QUERY_SLUG,
                "kaggle_id": "tminhhi/chili-leaf-disease-cross-dataset-query-set",
                "kaggle_version": 1,
                "kaggle_url": QUERY_KAGGLE_URL,
                "images": query_count,
                "image_bytes": query_image_bytes,
                "package_files": query_files,
                "package_bytes": query_total_bytes,
                "class_counts": dict(sorted(class_counts(query_rows).items())),
            },
            "support_query_sha256_overlap": 0,
            "excluded_known_overlap": {
                "filename": "Cercospora Leaf Spot00176.JPG",
                "sha256": "3541173176666ad9d5ecd99ed50cb92858d08757c4f9e4fb0b6c955d8b3d288f",
            },
        },
    )

    print(f"Created: {OUTPUT_ROOT}")
    print(f"Support: {support_count} images, {support_total_bytes} bytes")
    print(f"Query: {query_count} images, {query_total_bytes} bytes")
    print("Support/query SHA-256 overlap: 0")


if __name__ == "__main__":
    main()
