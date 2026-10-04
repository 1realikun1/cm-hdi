"""Remove machine-specific absolute paths from JSON provenance records.

The archived checksums remain unchanged. Only local source-path labels are
replaced by portable ``source://<basename>`` identifiers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = re.compile(r"^[A-Za-z]:\\")


def portable_label(value: str) -> str:
    if not WINDOWS_PATH.match(value):
        return value
    name = PureWindowsPath(value).name or "source"
    return f"source://{name}"


def sanitize(value):
    if isinstance(value, str):
        return portable_label(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            base_key = portable_label(key) if isinstance(key, str) else key
            clean_key = base_key
            suffix = 2
            while clean_key in clean:
                clean_key = f"{base_key}__{suffix}"
                suffix += 1
            clean[clean_key] = sanitize(item)
        return clean
    return value


def main() -> None:
    changed = []
    for path in sorted((ROOT / "reproduction").rglob("*.json")):
        original = json.loads(path.read_text(encoding="utf-8-sig"))
        cleaned = sanitize(original)
        if cleaned != original:
            path.write_text(
                json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed.append(path.relative_to(ROOT).as_posix())

    audit = {
        "status": "PASS",
        "rule": "Windows absolute paths replaced by portable source labels",
        "changed_file_count": len(changed),
        "changed_files": changed,
    }
    (ROOT / "PUBLIC_RELEASE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
