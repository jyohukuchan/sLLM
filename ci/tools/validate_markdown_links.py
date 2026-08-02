#!/usr/bin/env python3
"""Validate relative Markdown links and local heading fragments."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
LINK_RE = re.compile(r"!?(?:\[[^\]]*\])\(([^)<>\s]+)(?:\s+[^)]*)?\)")
REF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def slug(value: str) -> str:
    value = re.sub(r"[`*_~]", "", value).lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    return re.sub(r"\s+", "-", value.strip())


def main() -> int:
    errors: list[str] = []
    for source in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".local-artifacts", "reference", "target"} for part in source.relative_to(ROOT).parts):
            continue
        text = source.read_text(encoding="utf-8")
        links = LINK_RE.findall(text) + REF_RE.findall(text)
        headings = {slug(item) for item in HEADING_RE.findall(text)}
        for raw in links:
            target = unquote(raw).strip()
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith(("//", "mailto:", "tel:")):
                continue
            relative = parsed.path or source.name
            destination = (source.parent / relative).resolve()
            try:
                destination.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{source.relative_to(ROOT)}: link escapes repository: {target}")
                continue
            if not destination.exists():
                errors.append(f"{source.relative_to(ROOT)}: missing local link: {target}")
            if parsed.fragment and destination.suffix.lower() == ".md" and destination.exists():
                destination_headings = {slug(item) for item in HEADING_RE.findall(destination.read_text(encoding="utf-8"))}
                if parsed.fragment.lower() not in destination_headings:
                    errors.append(f"{source.relative_to(ROOT)}: missing heading fragment: {target}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("markdown local links: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
