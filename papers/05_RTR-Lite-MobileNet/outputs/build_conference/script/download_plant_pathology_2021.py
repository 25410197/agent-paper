"""Download and extract Plant Pathology 2021 - FGVC8 from Kaggle."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


COMPETITION = "plant-pathology-2021-fgvc8"
SCRIPT_DIR = Path(__file__).resolve().parent
DESTINATION = SCRIPT_DIR.parent / "data_2021"


def load_kaggle_token() -> None:
    """Load `token` from the local .env without displaying the secret."""
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
            return


def copy_directory(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    load_kaggle_token()

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Install KaggleHub first: python -m pip install kagglehub"
        ) from exc

    DESTINATION.mkdir(parents=True, exist_ok=True)
    if any(DESTINATION.iterdir()):
        raise RuntimeError(
            f"Destination is not empty: {DESTINATION}\n"
            "Remove incomplete files or choose an empty folder before retrying."
        )

    print(f"Downloading: {COMPETITION}")
    print("This dataset is much larger than FGVC7; the download may take time.")

    try:
        downloaded = kagglehub.competition_download(COMPETITION)
    except Exception as exc:
        if exc.__class__.__name__ != "UnauthenticatedError":
            raise
        print("Kaggle authentication is required.")
        kagglehub.login()
        downloaded = kagglehub.competition_download(COMPETITION)

    downloaded_path = Path(downloaded).resolve()
    print(f"KaggleHub path: {downloaded_path}")

    if downloaded_path.is_file() and zipfile.is_zipfile(downloaded_path):
        print(f"Extracting into: {DESTINATION}")
        with zipfile.ZipFile(downloaded_path) as archive:
            archive.extractall(DESTINATION)
    elif downloaded_path.is_dir():
        print(f"Copying into: {DESTINATION}")
        copy_directory(downloaded_path, DESTINATION)
    else:
        raise RuntimeError(f"Unsupported downloaded path: {downloaded_path}")

    files = [path for path in DESTINATION.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Completed: {len(files):,} files, {total_bytes / 1e9:.2f} GB")
    print(f"Dataset location: {DESTINATION}")


if __name__ == "__main__":
    main()
