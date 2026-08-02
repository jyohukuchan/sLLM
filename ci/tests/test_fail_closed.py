#!/usr/bin/env python3
"""Expose the deterministic negative matrix as a normal CI test module."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from self_test import run  # noqa: E402


class FailClosedTests(unittest.TestCase):
    def test_invalid_schema_state_zero_collection_and_artifact_gates_fail(self) -> None:
        # run() asserts invalid schema/state/zero-collection, missing/duplicate/
        # stale/hash-mismatch rows, non-success needs, and prohibited tracked
        # paths are all rejected.
        run()


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("ULLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "ULLM_UNITTEST_COUNTS="
            + json.dumps(
                {
                    "collected": selected,
                    "selected": selected,
                    "passed": selected - failed - skipped,
                    "failed": failed,
                    "skipped": skipped,
                    "deselected": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
