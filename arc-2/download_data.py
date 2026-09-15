"""download_data.py — Kaggle ARC-AGI-2 dataset downloader.

Usage:
    python download_data.py
    python download_data.py --dest ./data/arc-agi-2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMPETITION = "arc-prize-2025"
DEFAULT_DIR = "./data/arc-agi-2"


def download(dest: str = DEFAULT_DIR) -> int:
    dest_path = Path(dest)

    # Skip if already present and non-empty
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"[INFO] Dataset already present at {dest_path}, skipping download.")
        return 0

    # Verify kaggle CLI is available
    try:
        subprocess.run(
            ["kaggle", "--version"],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print(
            "[ERROR] kaggle CLI not found.\n"
            "        Install with: pip install kaggle\n"
            "        Then place your API key at ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            f"[ERROR] kaggle CLI returned an error: {exc}\n"
            "        Ensure your API key is configured at ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        return 1

    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {COMPETITION} dataset to {dest_path} ...")
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c", COMPETITION,
            "-p", str(dest_path),
            "--unzip",
        ],
    )
    if result.returncode == 0:
        print(f"[INFO] Download complete: {dest_path}")
    else:
        print(f"[ERROR] kaggle download failed with exit code {result.returncode}",
              file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the ARC-AGI-2 dataset via the Kaggle CLI."
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DIR,
        help=f"Destination directory (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    sys.exit(download(args.dest))
