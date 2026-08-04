"""Download and extract Plant Pathology 2020 data from Kaggle.

Usage:
    pip install kagglehub
    python download_plant_pathology_2020.py

Kaggle authentication must already be configured (for example, by logging in
through kagglehub or providing a valid Kaggle API token).
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

COMPETITION = "plant-pathology-2020-fgvc7"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DESTINATION = SCRIPT_DIR.parent / "data_moi"


def load_kaggle_token() -> None:
    """Load `token` from the local .env without printing the secret."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().lower() == "token":
            value = value.strip().strip("\"'")
            if value:
                os.environ.setdefault("KAGGLE_API_TOKEN", value)
                print(f"Loaded Kaggle API token from: {env_path}")
            return


def copy_directory(source: Path, destination: Path) -> None:
    """Copy the contents of a downloaded Kaggle directory."""
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def download(destination: Path, force: bool = False) -> None:
    load_kaggle_token()

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'kagglehub'. Install it with: "
            "python -m pip install kagglehub"
        ) from exc

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if any(destination.iterdir()) and not force:
        raise RuntimeError(
            f"Destination is not empty: {destination}\n"
            "Use --force if you want to merge/overwrite files in this folder."
        )

    print(f"Downloading Kaggle competition: {COMPETITION}")
    try:
        downloaded = kagglehub.competition_download(COMPETITION)
    except Exception as exc:
        if exc.__class__.__name__ != "UnauthenticatedError":
            raise

        print("KaggleHub is not authenticated.")
        print("Starting Kaggle login; paste your Kaggle API token when prompted.")
        kagglehub.login()
        downloaded = kagglehub.competition_download(COMPETITION)

    downloaded_path = Path(downloaded).resolve()
    print(f"KaggleHub download path: {downloaded_path}")

    if downloaded_path.is_file() and zipfile.is_zipfile(downloaded_path):
        print(f"Extracting ZIP into: {destination}")
        with zipfile.ZipFile(downloaded_path) as archive:
            archive.extractall(destination)
    elif downloaded_path.is_dir():
        print(f"Copying downloaded files into: {destination}")
        copy_directory(downloaded_path, destination)
    else:
        raise RuntimeError(
            f"Unsupported KaggleHub result (not a ZIP or directory): "
            f"{downloaded_path}"
        )

    files = [path for path in destination.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Completed: {len(files):,} files, {total_bytes / 1e9:.2f} GB")
    print(f"Dataset location: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download Kaggle competition {COMPETITION}."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Extraction directory (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow merging/overwriting files when the destination is non-empty.",
    )
    args = parser.parse_args()
    download(args.destination, args.force)


if __name__ == "__main__":
    main()
