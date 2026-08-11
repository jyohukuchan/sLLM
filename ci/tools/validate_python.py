#!/usr/bin/env python3
"""Compile and perform deterministic AST checks on local Python sources."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {".git", ".local-artifacts", "reference", "target", ".venv", "venv"}
SEMANTIC_G1_CONTROLLER = ROOT / "ci" / "tools" / "orchestrate_rmsnorm_g1_evidence.py"


def python_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if path.is_file() and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts))


def allowed_semantic_controller_exec(
    path: Path, node: ast.Call, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Allow only the sealed controller's exact reviewed-module loader."""

    if path != SEMANTIC_G1_CONTROLLER or node.keywords or len(node.args) != 2:
        return False
    compiled, namespace = node.args
    if not (
        isinstance(node.func, ast.Name)
        and node.func.id == "exec"
        and isinstance(compiled, ast.Call)
        and isinstance(compiled.func, ast.Name)
        and compiled.func.id == "compile"
        and not compiled.keywords
        and len(compiled.args) == 3
        and isinstance(compiled.args[0], ast.Name)
        and compiled.args[0].id == "source"
        and isinstance(compiled.args[1], ast.Attribute)
        and isinstance(compiled.args[1].value, ast.Name)
        and compiled.args[1].value.id == "module"
        and compiled.args[1].attr == "__file__"
        and isinstance(compiled.args[2], ast.Constant)
        and compiled.args[2].value == "exec"
        and isinstance(namespace, ast.Attribute)
        and isinstance(namespace.value, ast.Name)
        and namespace.value.id == "module"
        and namespace.attr == "__dict__"
    ):
        return False
    parent = parents.get(node)
    while parent is not None and not isinstance(
        parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    ):
        parent = parents.get(parent)
    return isinstance(parent, ast.FunctionDef) and parent.name == "_load_reviewed_module"


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
                parents = {
                    child: parent
                    for parent in ast.walk(tree)
                    for child in ast.iter_child_nodes(parent)
                }
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module in {"pytest", "nose"}:
                        failures.append(f"{path}: test runner import is not part of host contract")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                        if not allowed_semantic_controller_exec(path, node, parents):
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
