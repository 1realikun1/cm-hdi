"""Build a deterministic SHA-256 manifest for the public repository files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST_SHA256.csv"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append((relative.as_posix(), path.stat().st_size, sha256(path)))

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        writer.writerows(sorted(files))

    print(f"Wrote {len(files)} entries to {OUTPUT.name}")


if __name__ == "__main__":
    main()
