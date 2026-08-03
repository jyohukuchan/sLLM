#!/usr/bin/env python3
"""Focused negative tests for H3 runner evidence-mode identity."""

from __future__ import annotations

import sys
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from run_h3_compile import (  # noqa: E402
    H3Error,
    PINNED_IMAGE_CONFIG_DIGEST,
    PINNED_IMAGE_REFERENCE,
    assert_required_network_isolation,
    execution_environment,
    main as runner_main,
)

IPV4_ROUTE_HEADER = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
MEASURED_IPV6_LOOPBACK_ROUTES = (
    "00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
    "00000000000000000000000000000000 ffffffff 00000001 00000000 00200200 lo\n"
    "00000000000000000000000000000001 80 00000000000000000000000000000000 00 "
    "00000000000000000000000000000000 00000000 00000069 00000000 80200001 lo\n"
)


def runner_args(
    *,
    pinned_container: bool = False,
    observed_image_reference: str | None = None,
    observed_image_config_digest: str | None = None,
) -> Namespace:
    return Namespace(
        pinned_container=pinned_container,
        observed_image_reference=observed_image_reference,
        observed_image_config_digest=observed_image_config_digest,
    )


class H3RunnerEvidenceModeTests(unittest.TestCase):
    def test_local_development_is_not_official_container_evidence(self) -> None:
        self.assertEqual(
            execution_environment(runner_args(), "local-development"),
            {
                "mode": "local-development",
                "execution_scope": "local-system",
                "container_image_reference": None,
                "observed_image_config_digest": None,
                "pinned_container": False,
                "identity_verified": False,
                "network_isolated": False,
            },
        )
        for args in (
            runner_args(pinned_container=True),
            runner_args(observed_image_reference=PINNED_IMAGE_REFERENCE),
            runner_args(observed_image_config_digest=PINNED_IMAGE_CONFIG_DIGEST),
        ):
            with self.subTest(args=args):
                with self.assertRaises(H3Error):
                    execution_environment(args, "local-development")

    def test_required_ci_rejects_missing_or_wrong_image_identity(self) -> None:
        invalid = (
            runner_args(),
            runner_args(pinned_container=True),
            runner_args(
                pinned_container=True,
                observed_image_reference="docker.io/rocm/dev-ubuntu-24.04@sha256:" + "0" * 64,
            ),
            runner_args(
                pinned_container=True,
                observed_image_reference=PINNED_IMAGE_REFERENCE,
                observed_image_config_digest="sha256:" + "0" * 64,
            ),
        )
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(H3Error):
                    execution_environment(args, "required-ci")

    def test_required_ci_network_isolation_contract(self) -> None:
        with patch.dict(os.environ, {"SLLM_H3_NETWORK_DISABLED": "0"}, clear=False):
            with self.assertRaises(H3Error):
                assert_required_network_isolation()
        for names in ([], [(1, "eth0")], [(1, "lo"), (2, "eth0")]):
            with self.subTest(names=names), patch.dict(os.environ, {"SLLM_H3_NETWORK_DISABLED": "1"}, clear=False), patch(
                "run_h3_compile.socket.if_nameindex", return_value=names
            ):
                with self.assertRaises(H3Error):
                    assert_required_network_isolation()

        with patch.dict(os.environ, {"SLLM_H3_NETWORK_DISABLED": "1"}, clear=False), patch(
            "run_h3_compile.socket.if_nameindex", return_value=[(1, "lo")]
        ), patch(
            "run_h3_compile.Path.read_text",
            side_effect=[IPV4_ROUTE_HEADER, MEASURED_IPV6_LOOPBACK_ROUTES],
        ):
            self.assertIsNone(assert_required_network_isolation())

        malformed_tables = (
            ("malformed\n", MEASURED_IPV6_LOOPBACK_ROUTES),
            (IPV4_ROUTE_HEADER, "malformed\n"),
        )
        for ipv4_routes, ipv6_routes in malformed_tables:
            with self.subTest(ipv4_routes=ipv4_routes, ipv6_routes=ipv6_routes), patch.dict(
                os.environ, {"SLLM_H3_NETWORK_DISABLED": "1"}, clear=False
            ), patch("run_h3_compile.socket.if_nameindex", return_value=[(1, "lo")]), patch(
                "run_h3_compile.Path.read_text", side_effect=[ipv4_routes, ipv6_routes]
            ):
                with self.assertRaises(H3Error):
                    assert_required_network_isolation()

        ipv4_routes = IPV4_ROUTE_HEADER + "eth0 00000000 00000000 0001 0 0 0 00000000 0 0 0\n"
        with patch.dict(os.environ, {"SLLM_H3_NETWORK_DISABLED": "1"}, clear=False), patch(
            "run_h3_compile.socket.if_nameindex", return_value=[(1, "lo")]
        ), patch(
            "run_h3_compile.Path.read_text", side_effect=[ipv4_routes, MEASURED_IPV6_LOOPBACK_ROUTES]
        ):
            with self.assertRaises(H3Error):
                assert_required_network_isolation()

    def test_failure_report_is_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-h3-runner-failure-") as temporary:
            output = Path(temporary) / "output"
            exit_code = runner_main([
                "--row", "h3-gfx1030",
                "--repo", temporary,
                "--output-dir", str(output),
            ])
            self.assertNotEqual(exit_code, 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["state"], "FAIL")
            from jsonschema import Draft202012Validator, FormatChecker

            schema = json.loads((ROOT / "ci/schema/test-result-v1.schema.json").read_text(encoding="utf-8"))
            errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report))
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
