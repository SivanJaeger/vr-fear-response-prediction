"""Download the "Do Virtual Spiders Feel Real?" VR spider exposure dataset from Zenodo.

Dataset: Krumnikl et al. (2026), Zenodo record 20796399, DOI 10.5281/zenodo.20796399,
license CC-BY-4.0. The record contains a single ~467 MB archive (``vr_spider_dataset.zip``)
holding per-participant raw recordings plus a small participant-level ``summary.csv`` with
precomputed physiological features and subjective fear (SUDS) ratings.

This project's analysis only uses ``summary.csv``, so this script downloads the archive
and extracts just that (and its column-description files), a few hundred KB in total.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_RECORD_ID = "20796399"
ZENODO_FILE_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/vr_spider_dataset.zip?download=1"
)
ZENODO_ARCHIVE_MD5 = "607793c117fb02683708bbea31c89072"
ZENODO_ARCHIVE_NAME = "vr_spider_dataset.zip"

SUMMARY_FILES = ["summary.csv", "summary_description.txt", "pX_description.txt"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = DATA_RAW_DIR / "_download_cache"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report(count: int, block_size: int, total_size: int) -> None:
    downloaded = count * block_size
    pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0.0
    sys.stdout.write(f"\r  downloading vr_spider_dataset.zip: {pct:5.1f}%")
    sys.stdout.flush()


def download_archive(cache_dir: Path = CACHE_DIR, force: bool = False) -> Path:
    """Download the Zenodo archive to a local cache, verifying its MD5 checksum."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / ZENODO_ARCHIVE_NAME

    if archive_path.exists() and not force:
        if _md5(archive_path) == ZENODO_ARCHIVE_MD5:
            print(f"Using cached archive at {archive_path}")
            return archive_path
        print("Cached archive checksum mismatch, re-downloading.")

    print(f"Downloading dataset from {ZENODO_FILE_URL}")
    urlretrieve(ZENODO_FILE_URL, archive_path, reporthook=_report)
    print()

    checksum = _md5(archive_path)
    if checksum != ZENODO_ARCHIVE_MD5:
        raise RuntimeError(
            f"Downloaded archive MD5 mismatch: expected {ZENODO_ARCHIVE_MD5}, got {checksum}. "
            "The download may be corrupted or the dataset may have changed on Zenodo."
        )
    print("Checksum verified.")
    return archive_path


def extract_summary_files(archive_path: Path, out_dir: Path = DATA_RAW_DIR) -> None:
    """Extract the small participant-level summary files needed for the main pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        names_in_archive = set(zf.namelist())
        missing = [f for f in SUMMARY_FILES if f not in names_in_archive]
        if missing:
            raise RuntimeError(f"Expected files missing from archive: {missing}")
        for name in SUMMARY_FILES:
            with zf.open(name) as src, (out_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"Extracted {name} -> {out_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true", help="Re-download even if a valid cached archive exists.")
    args = parser.parse_args()

    archive_path = download_archive(force=args.force_download)
    extract_summary_files(archive_path)

    print("\nDone. Summary data available in data/raw/.")


if __name__ == "__main__":
    main()
