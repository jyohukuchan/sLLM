#!/usr/bin/env python3
"""Reject lowp source dependencies that cross back into sLLM/native HIP."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOWP = (ROOT / "native/lowp").resolve()
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[\"<]([^\">]+)[\">]")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc"}


def source_files() -> list[Path]:
    return sorted(path for path in LOWP.rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES)


def resolve_local_include(owner: Path, name: str) -> Path | None:
    # Quoted includes are resolved first relative to the including file.  The
    # lowp tree intentionally has no dependency on native/hip or include/sllm.
    candidates = [owner.parent / name, LOWP / "include" / name, LOWP / "src" / name]
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved
    return None


def main() -> int:
    if not LOWP.is_dir():
        raise SystemExit("native/lowp is missing")
    failures: list[str] = []
    for owner in source_files():
        for line_no, line in enumerate(owner.read_text(encoding="utf-8").splitlines(), 1):
            match = INCLUDE_RE.match(line)
            if match is None:
                continue
            name = match.group(1)
            if name.startswith("sllm/") or "native/hip/src" in name:
                failures.append(f"{owner.relative_to(ROOT)}:{line_no}: forbidden include {name}")
                continue
            angle_include = line.lstrip().startswith("#include <")
            resolved = resolve_local_include(owner, name)
            if resolved is None:
                if angle_include:
                    # A missing angle include is supplied by HIP/ROCm or the
                    # host toolchain.  If a same-named file exists in lowp,
                    # resolve_local_include above checks it rather than
                    # silently treating a local dependency as external.
                    continue
                failures.append(
                    f"{owner.relative_to(ROOT)}:{line_no}: unresolved local include {name}"
                )
            elif LOWP not in (resolved, *resolved.parents):
                failures.append(
                    f"{owner.relative_to(ROOT)}:{line_no}: local include escapes native/lowp: {name}"
                )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"lowp include boundary PASS ({len(source_files())} source files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
