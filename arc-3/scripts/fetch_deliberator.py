"""Resilient direct downloader for the R11 deliberator weights (Qwen2.5-7B-Instruct).

hf_hub / Xet stalled silently on this connection (socket read hangs with no
exception, so retries never fire). This script streams each safetensors shard
over plain HTTPS with an explicit read timeout and HTTP Range resume, so a
stalled connection raises and we resume from the exact byte already on disk.

Apache-2.0, ungated. Run:  python scripts/fetch_deliberator.py
Idempotent: re-running resumes partial files and skips complete ones.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

REPO = "Qwen/Qwen2.5-7B-Instruct"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
DEST = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-7b-instruct"

SHARDS = [
    ("model-00001-of-00004.safetensors", None),
    ("model-00002-of-00004.safetensors", None),
    ("model-00003-of-00004.safetensors", None),
    ("model-00004-of-00004.safetensors", None),
]

READ_TIMEOUT = 20          # seconds; a stalled read raises after this
CHUNK = 4 * 1024 * 1024    # 4 MiB
MAX_ATTEMPTS = 2000        # ~3 MB/s over 14 GB with stalls needs many resumes


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def download_one(name: str) -> None:
    url = f"{BASE}/{name}"
    dest = DEST / name
    total = _remote_size(url)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have >= total:
            print(f"[ok] {name} complete ({have} bytes)", flush=True)
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r, \
                    open(dest, "ab") as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
            have = dest.stat().st_size
            if total is None or have >= total:
                print(f"[ok] {name} done ({have} bytes)", flush=True)
                return
            print(f"[resume] {name} at {have}/{total} bytes", flush=True)
        except Exception as exc:  # timeout, reset, partial — resume next loop
            have = dest.stat().st_size if dest.exists() else 0
            print(
                f"[retry {attempt}] {name} at {have} bytes after "
                f"{type(exc).__name__}: {str(exc)[:80]}", flush=True,
            )
            time.sleep(3)
    raise SystemExit(f"[fail] {name} did not complete after {MAX_ATTEMPTS} attempts")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, _ in SHARDS:
        download_one(name)
    print("ALL_SHARDS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
