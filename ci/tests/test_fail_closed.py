#!/usr/bin/env python3
"""Expose the deterministic negative matrix as a normal CI test module."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from network_guard import (  # noqa: E402
    IsolationPlan,
    NetworkIsolationError,
    assert_isolated,
    _normalize_ipv4_routes,
    _normalize_ipv6_routes,
    prepare_isolation,
    verify_parent_restored,
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


class NetworkNamespaceRestorationTests(unittest.TestCase):
    def _plan(self, parent_netns: str = "net:[4026531840]") -> IsolationPlan:
        return IsolationPlan(
            strategy="test",
            prefix=(),
            parent_netns=parent_netns,
            expected_euid=os.getuid(),
            expected_egid=os.getgid(),
            require_no_capabilities=False,
            execution_environment=(),
        )

    def test_same_parent_netns_with_topology_change_is_accepted(self) -> None:
        plan = self._plan()
        changed_topology = (
            plan.parent_netns,
            ("lo", "eth0"),
            (("eth0", "changed-route"),),
            (("eth0", "changed-ipv6-route"),),
        )
        with (
            patch("network_guard.current_netns", return_value=plan.parent_netns),
            patch("network_guard.parent_connectivity_snapshot", return_value=changed_topology),
        ):
            verify_parent_restored(plan)

    def test_parent_netns_change_fails_closed(self) -> None:
        plan = self._plan()
        with patch("network_guard.current_netns", return_value="net:[4026531841]"):
            with self.assertRaisesRegex(NetworkIsolationError, "parent network namespace"):
                verify_parent_restored(plan)

    def test_parent_netns_change_during_probe_fails_closed(self) -> None:
        plan = self._plan()
        with (
            patch.dict(os.environ, {"ULLM_NETWORK_GUARD_ACTIVE": "0"}),
            patch(
                "network_guard.current_netns",
                side_effect=[plan.parent_netns, "net:[4026531841]"],
            ),
            patch("network_guard._candidate_plans", return_value=[plan]),
            patch("network_guard._probe", return_value=(True, "")),
        ):
            with self.assertRaisesRegex(NetworkIsolationError, "parent network namespace"):
                prepare_isolation()


class ChildIsolationVerificationTests(unittest.TestCase):
    PARENT_NETNS = "net:[4026531840]"
    CHILD_NETNS = "net:[4026531841]"
    ZERO_CAPABILITIES = {
        "CapInh": 0,
        "CapPrm": 0,
        "CapEff": 0,
        "CapBnd": 0,
        "CapAmb": 0,
    }

    def _assert_child_rejected(
        self,
        *,
        child_netns: str = CHILD_NETNS,
        euid: int = 1000,
        egid: int = 1000,
        capabilities: dict[str, int] | None = None,
        no_new_privs: int = 1,
        groups: tuple[int, ...] = (),
        interfaces: tuple[str, ...] = ("lo",),
        routes: tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]] = ((), ()),
    ) -> None:
        with (
            patch("network_guard.current_netns", return_value=child_netns),
            patch("network_guard.os.geteuid", return_value=euid),
            patch("network_guard.os.getegid", return_value=egid),
            patch(
                "network_guard.process_security_state",
                return_value=(capabilities or self.ZERO_CAPABILITIES, no_new_privs),
            ),
            patch("network_guard.os.getgroups", return_value=list(groups)),
            patch("network_guard._interface_names", return_value=interfaces),
            patch("network_guard._route_snapshot", return_value=routes),
            patch("network_guard._assert_external_connect_fails"),
        ):
            with self.assertRaises(NetworkIsolationError):
                assert_isolated(
                    parent_netns=self.PARENT_NETNS,
                    expected_euid=1000,
                    expected_egid=1000,
                    require_no_capabilities=True,
                    address_space_limit_bytes=None,
                )

    def test_same_netns_is_rejected(self) -> None:
        self._assert_child_rejected(child_netns=self.PARENT_NETNS)

    def test_uid_gid_and_privilege_state_mismatches_are_rejected(self) -> None:
        cases = {
            "uid": {"euid": 1001},
            "gid": {"egid": 1001},
            "capability": {"capabilities": {**self.ZERO_CAPABILITIES, "CapEff": 1}},
            "no-new-privs": {"no_new_privs": 0},
            "supplementary-groups": {"groups": (1001,)},
        }
        for label, overrides in cases.items():
            with self.subTest(state=label):
                self._assert_child_rejected(**overrides)

    def test_non_loopback_interface_is_rejected(self) -> None:
        self._assert_child_rejected(interfaces=("lo", "eth0"))

    def test_default_route_is_rejected(self) -> None:
        ipv4_default_route = (("eth0", "00000000"),)
        self._assert_child_rejected(routes=(ipv4_default_route, ()))


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
