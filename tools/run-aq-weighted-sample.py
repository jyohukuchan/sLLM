#!/usr/bin/env python3
"""Run aq tensor sampling with activation-stat weighted metrics.

This is a thin entry point over run-aq-tensor-sample.py. Pass
`--activation-stats` to enable weighted metrics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    module_path = Path(__file__).with_name("run-aq-tensor-sample.py")
    spec = importlib.util.spec_from_file_location("run_aq_tensor_sample", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
