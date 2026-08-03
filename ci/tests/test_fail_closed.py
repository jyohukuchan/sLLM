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

from network_guard import (  # noqa: E402
    NetworkIsolationError,
    _normalize_ipv4_routes,
    _normalize_ipv6_routes,
)
from self_test import run  # noqa: E402


class FailClosedTests(unittest.TestCase):
    def test_invalid_schema_state_zero_collection_and_artifact_gates_fail(self) -> None:
        # run() asserts invalid schema/state/zero-collection, missing/duplicate/
        # stale/hash-mismatch rows, non-success needs, and prohibited tracked
        # paths are all rejected.
        run()


class NetworkRouteNormalizationTests(unittest.TestCase):
    IPV4_HEADER = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT"
    IPV4_FIELDS = [
        "enp7s0",
        "0A0B0C0D",
        "01020304",
        "0003",
        "255",
        "256",
        "257",
        "00FFFFFF",
        "1501",
        "3",
        "65535",
    ]
    IPV6_FIELDS = [
        "20010DB8000000000000000000000001",
        "41",
        "20010DB8000000000000000000000002",
        "07",
        "00000000000000000000000000000003",
        "00000101",
        "000000FF",
        "00000100",
        "00000005",
        "enp7s0",
    ]

    def test_ipv4_counter_changes_are_ignored_but_semantic_changes_are_not(self) -> None:
        baseline = _normalize_ipv4_routes(
            [self.IPV4_HEADER, " ".join(self.IPV4_FIELDS)]
        )

        counters_changed = self.IPV4_FIELDS.copy()
        counters_changed[4] = "256"
        counters_changed[5] = "257"
        self.assertEqual(
            baseline,
            _normalize_ipv4_routes([self.IPV4_HEADER, " ".join(counters_changed)]),
        )

        semantic_changes = {
            0: "enp8s0",
            1: "0A0B0C0E",
            2: "01020305",
            3: "0007",
            6: "256",
            7: "00FFFF00",
            8: "1500",
            9: "4",
            10: "65534",
        }
        for index, value in semantic_changes.items():
            with self.subTest(field=index):
                changed = self.IPV4_FIELDS.copy()
                changed[index] = value
                self.assertNotEqual(
                    baseline,
                    _normalize_ipv4_routes([self.IPV4_HEADER, " ".join(changed)]),
                )

    def test_ipv6_counter_changes_are_ignored_but_semantic_changes_are_not(self) -> None:
        baseline = _normalize_ipv6_routes([" ".join(self.IPV6_FIELDS)])

        counters_changed = self.IPV6_FIELDS.copy()
        counters_changed[6] = "00000100"
        counters_changed[7] = "00000101"
        self.assertEqual(
            baseline,
            _normalize_ipv6_routes([" ".join(counters_changed)]),
        )

        semantic_changes = {
            0: "20010DB8000000000000000000000003",
            1: "42",
            2: "20010DB8000000000000000000000004",
            3: "08",
            4: "00000000000000000000000000000004",
            5: "00000100",
            8: "00000006",
            9: "enp8s0",
        }
        for index, value in semantic_changes.items():
            with self.subTest(field=index):
                changed = self.IPV6_FIELDS.copy()
                changed[index] = value
                self.assertNotEqual(
                    baseline,
                    _normalize_ipv6_routes([" ".join(changed)]),
                )

    def test_malformed_routes_fail_closed(self) -> None:
        ipv4 = self.IPV4_FIELDS.copy()
        ipv6 = self.IPV6_FIELDS.copy()
        invalid_ipv4_cases = [
            [self.IPV4_HEADER.replace("Use", "Uses"), " ".join(ipv4)],
            [self.IPV4_HEADER, " ".join(ipv4[:-1])],
        ]
        invalid_ipv4_hex = ipv4.copy()
        invalid_ipv4_hex[1] = "not-hex!"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_hex)])
        invalid_ipv4_decimal = ipv4.copy()
        invalid_ipv4_decimal[6] = "0x101"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_decimal)])
        invalid_ipv4_range = ipv4.copy()
        invalid_ipv4_range[8] = "4294967296"
        invalid_ipv4_cases.append([self.IPV4_HEADER, " ".join(invalid_ipv4_range)])
        for lines in invalid_ipv4_cases:
            with self.subTest(protocol="IPv4", lines=lines):
                with self.assertRaises(NetworkIsolationError):
                    _normalize_ipv4_routes(lines)

        invalid_ipv6_cases = [ipv6[:-1], ipv6 + ["extra"]]
        invalid_ipv6_hex = ipv6.copy()
        invalid_ipv6_hex[4] = "not-hex"
        invalid_ipv6_cases.append(invalid_ipv6_hex)
        invalid_ipv6_prefix = ipv6.copy()
        invalid_ipv6_prefix[1] = "FF"
        invalid_ipv6_cases.append(invalid_ipv6_prefix)
        invalid_ipv6_counter = ipv6.copy()
        invalid_ipv6_counter[6] = "0x100"
        invalid_ipv6_cases.append(invalid_ipv6_counter)
        for fields in invalid_ipv6_cases:
            with self.subTest(protocol="IPv6", fields=fields):
                with self.assertRaises(NetworkIsolationError):
                    _normalize_ipv6_routes([" ".join(fields)])


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
