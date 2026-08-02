#!/usr/bin/env python3
"""Compile and perform deterministic AST checks on local Python sources."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {".git", ".local-artifacts", "reference", "target", ".venv", "venv"}


def python_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if path.is_file() and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compile", "static"), required=True)
    args = parser.parse_args()
    failures: list[str] = []
    for path in python_files():
        try:
            source = path.read_bytes()
            tree = ast.parse(source, filename=str(path))
            if args.mode == "compile":
                compile(tree, str(path), "exec")
            else:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module in {"pytest", "nose"}:
                        failures.append(f"{path}: test runner import is not part of host contract")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                        failures.append(f"{path}:{node.lineno}: dynamic code execution is prohibited")
        except (SyntaxError, UnicodeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"python {args.mode}: checked {len(python_files())} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
