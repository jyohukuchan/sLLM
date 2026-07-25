"""Regression tests for the SQ8_0 source-tile performance summary."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "summarize-sq8_0-paged-decode-tile-performance.py"


def load_module():
    specification = importlib.util.spec_from_file_location("sq8_tile_performance", SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_route(root: Path, route: str, tile: int | None, samples: list[float]) -> None:
    tokens = [101, 102, 103]
    result = {
        "passed": True,
        "paged_decode_split_source_tile": tile,
        "paged_decode_split_multi_tile_policy": None if tile is None else "direct-fallback-exact-state.v1",
        "requests": [
            {
                "generated_token_ids": tokens,
                "generated_steps": [
                    {"generated_index": 0, "token_id": tokens[0], "synchronized_seconds": 1.0},
                    {"generated_index": 1, "token_id": tokens[1], "synchronized_seconds": samples[0]},
                    {"generated_index": 2, "token_id": tokens[2], "synchronized_seconds": samples[1]},
                ],
            }
        ],
    }
    path = root / "performance" / route / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")


class SummarizeSq8PagedDecodeTilePerformanceTests(unittest.TestCase):
    def test_calculates_m1_speedup_and_records_guard_policy(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_route(root, "direct", None, [0.050, 0.070])
            write_route(root, "tile128", 128, [0.060, 0.040])
            write_route(root, "tile256", 256, [0.055, 0.065])
            previous_argv = sys.argv
            try:
                sys.argv = [str(SCRIPT), "--result-dir", str(root)]
                with redirect_stdout(io.StringIO()):
                    status = module.main()
            finally:
                sys.argv = previous_argv
            self.assertEqual(status, 0)
            summary = json.loads((root / "performance-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["routes"]["direct"]["m1_sample_count"], 2)
            self.assertEqual(summary["routes"]["tile128"]["multi_tile_policy"], "direct-fallback-exact-state.v1")
            self.assertAlmostEqual(summary["routes"]["tile128"]["speedup_vs_direct"], 1.2)


if __name__ == "__main__":
    unittest.main()
