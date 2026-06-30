#!/usr/bin/env python3
"""Select representative module names for aq logit smoke runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open


LAYER_RE = re.compile(r"(?:^|\.)(?:language_model|model)\.layers\.(\d+)\.")


def load_sampler_module():
    module_path = Path(__file__).with_name("run-aq-tensor-sample.py")
    spec = importlib.util.spec_from_file_location("run_aq_tensor_sample", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def activation_stats_path(path: Path) -> Path:
    if path.is_dir():
        return path / "activation_second_moments.safetensors"
    return path


def module_name_from_stat_key(key: str) -> str | None:
    if key.endswith((".mean_abs", ".max_abs")):
        return None
    if key.startswith("language_model."):
        return "model." + key.removeprefix("language_model.")
    if key.startswith("model."):
        return key
    return None


def layer_index(module_name: str) -> int | None:
    match = LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def load_modules(stats_path: Path) -> list[dict[str, Any]]:
    sampler = load_sampler_module()
    modules: list[dict[str, Any]] = []
    with safe_open(activation_stats_path(stats_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            module_name = module_name_from_stat_key(key)
            if module_name is None:
                continue
            modules.append(
                {
                    "module": module_name,
                    "family": sampler.family_for_tensor(f"{module_name}.weight"),
                    "layer": layer_index(module_name),
                    "activation_elements": int(handle.get_tensor(key).numel()),
                }
            )
    return sorted(modules, key=lambda item: (item["layer"] if item["layer"] is not None else 10**9, item["module"]))


def select_modules(args: argparse.Namespace) -> list[dict[str, Any]]:
    modules = load_modules(args.activation_stats)
    allowed_families = set(args.family) if args.family else None
    allowed_layers = set(args.layer) if args.layer else None
    selected: list[dict[str, Any]] = []
    counts_by_family: defaultdict[str, int] = defaultdict(int)
    counts_by_layer_family: defaultdict[tuple[int | None, str], int] = defaultdict(int)

    for item in modules:
        family = str(item["family"])
        layer = item["layer"]
        if allowed_families is not None and family not in allowed_families:
            continue
        if allowed_layers is not None and layer not in allowed_layers:
            continue
        if args.max_modules_per_family is not None and counts_by_family[family] >= args.max_modules_per_family:
            continue
        layer_family = (layer, family)
        if args.max_modules_per_layer_family is not None:
            if counts_by_layer_family[layer_family] >= args.max_modules_per_layer_family:
                continue
        selected.append(item)
        counts_by_family[family] += 1
        counts_by_layer_family[layer_family] += 1
        if args.max_modules is not None and len(selected) >= args.max_modules:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--layer", type=int, action="append", default=[])
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--max-modules-per-family", type=int, default=None)
    parser.add_argument("--max-modules-per-layer-family", type=int, default=1)
    parser.add_argument("--format", choices=("json", "text", "shell-args"), default="text")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_modules is not None and args.max_modules < 1:
        raise SystemExit("--max-modules must be >= 1")
    if args.max_modules_per_family is not None and args.max_modules_per_family < 1:
        raise SystemExit("--max-modules-per-family must be >= 1")
    if args.max_modules_per_layer_family is not None and args.max_modules_per_layer_family < 1:
        raise SystemExit("--max-modules-per-layer-family must be >= 1")
    args.activation_stats = args.activation_stats.expanduser().resolve()
    selected = select_modules(args)

    if args.format == "json":
        payload = json.dumps({"modules": selected}, indent=2, sort_keys=True)
    elif args.format == "shell-args":
        payload = " ".join(f"--module {shlex.quote(str(item['module']))}" for item in selected)
    else:
        payload = "\n".join(
            f"{item['module']}\tfamily={item['family']}\tlayer={item['layer']}" for item in selected
        )
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
