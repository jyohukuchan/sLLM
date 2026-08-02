#!/usr/bin/env python3
"""Check the repository-level license and provenance control files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    errors: list[str] = []
    license_path = ROOT / "LICENSE"
    provenance = ROOT / "docs/provenance/README.md"
    source_lock = ROOT / "docs/references/source-lock.md"
    if not license_path.exists() or "MIT License" not in license_path.read_text(encoding="utf-8"):
        errors.append("LICENSE is missing or is not the declared MIT license")
    if not provenance.exists():
        errors.append("docs/provenance/README.md is missing")
    else:
        text = provenance.read_text(encoding="utf-8")
        for marker in ("THIRD_PARTY_NOTICES.md", "full commit SHA", "imported_sha256", "reuse:"):
            if marker not in text:
                errors.append(f"provenance policy is missing required marker: {marker}")
    if not source_lock.exists():
        errors.append("docs/references/source-lock.md is missing")
    else:
        text = source_lock.read_text(encoding="utf-8")
        if len(re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", text)) < 7:
            errors.append("source-lock does not contain complete source revisions")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("license/provenance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
