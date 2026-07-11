from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import sq8_full_campaign_identity as IDENTITY  # noqa: E402
import sq8_full_campaign_resource as RESOURCE  # noqa: E402


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR: Any = load_module(
    "test_sq8_full_campaign_resource_collector",
    TOOLS / "collect-sq8-openwebui-release.py",
)
VALIDATOR: Any = load_module(
    "test_sq8_full_campaign_resource_validator",
    TOOLS / "validate-sq8-openwebui-release.py",
)
IDENTITY_FIXTURES: Any = load_module(
    "test_sq8_full_campaign_resource_identity_fixtures",
    ROOT / "tests" / "test_sq8_full_campaign_identity.py",
)
IDENTITY_FIXTURES.IDENTITY = IDENTITY


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class RecordingIdentity:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def validate_session_header(self, record: dict[str, Any]) -> None:
        self.calls.append("session")
        self.inner.validate_session_header(record)

    def validate_header_source_inputs(self, input_files: Any) -> None:
        self.calls.append("sources")
        self.inner.validate_header_source_inputs(input_files)


class RecordingValidator:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def validate_schedule(self, value: Any, label: str) -> dict[str, Any]:
        self.calls.append("schedule")
        return cast(dict[str, Any], self.inner.validate_schedule(value, label))

    def validate_thresholds(self, value: Any, label: str) -> dict[str, Any]:
        self.calls.append("thresholds")
        return cast(dict[str, Any], self.inner.validate_thresholds(value, label))


class FullCampaignResourceTests(unittest.TestCase):
    identity_fixture: Any
    artifacts: IDENTITY.IdentityArtifacts
    independent_identity: RecordingIdentity
    independent_validator: RecordingValidator

    def setUp(self) -> None:
        self.identity_fixture = IDENTITY_FIXTURES.IdentityFixture()
        self.addCleanup(self.identity_fixture.temporary.cleanup)
        self.artifacts = IDENTITY.build_identity_artifacts(
            self.identity_fixture.inputs, self.identity_fixture.live
        )
        environment = self.artifacts.environment
        model_identity = self.artifacts.model_identity
        identity_data = VALIDATOR.IdentityData(
            environment=environment,
            model_identity=model_identity,
            environment_sha256=sha256(self.artifacts.environment_bytes),
            model_identity_sha256=sha256(self.artifacts.model_identity_bytes),
            expected_commit=environment["git"]["commit"],
            expected_worker_binary_sha256=model_identity["worker"]["binary_sha256"],
            source_by_role={item["role"]: item for item in environment["sources"]},
            source_sets=environment["source_sets"],
            configuration=environment["deployment"]["configuration"],
            service=environment["service"],
            openwebui=environment["openwebui"],
            model_worker=model_identity["worker"],
        )
        self.independent_identity = RecordingIdentity(identity_data)
        self.independent_validator = RecordingValidator(VALIDATOR)

    def build(self) -> RESOURCE.ResourceContract[Any]:
        return RESOURCE.build_resource_contract(
            self.artifacts,
            self.independent_identity,
            self.independent_validator,
            run_id="sq8-full-production-20260712",
            started_utc="2026-07-12T01:00:00Z",
            negative_case_type=COLLECTOR.NegativeCase,
            resource_config_type=COLLECTOR.ResourceSegmentConfig,
            forbidden_values=(b"sk-production-secret-not-present",),
        )

    def validate(self, contract: RESOURCE.ResourceContract[Any]) -> None:
        RESOURCE.validate_resource_contract(
            contract,
            self.artifacts,
            self.independent_identity,
            self.independent_validator,
            forbidden_values=(b"sk-production-secret-not-present",),
        )

    def test_builds_actual_frozen_collector_config_and_canonical_evidence(self) -> None:
        contract = self.build()

        self.assertIsInstance(contract.segment_config, COLLECTOR.ResourceSegmentConfig)
        self.assertEqual(contract.segment_config.target, "/v1/chat/completions")
        self.assertEqual(
            contract.segment_config.resource_body_template["model"],
            "ullm-qwen3-14b-sq8",
        )
        self.assertEqual(
            [
                (case.after_request, case.name, case.expected_status)
                for case in contract.segment_config.negative_cases
            ],
            [
                (25, "context_overflow_1", 400),
                (50, "malformed_json", 400),
                (75, "context_overflow_2", 400),
            ],
        )
        for case in contract.segment_config.negative_cases:
            if case.name == "malformed_json":
                self.assertEqual(case.body, b"{")
            else:
                body = json.loads(case.body)
                self.assertEqual(
                    body["messages"][0]["content"],
                    COLLECTOR.CONTEXT_OVERFLOW_CONTENT[case.name],
                )
                self.assertEqual(
                    body["messages"][0]["content"],
                    VALIDATOR.CONTEXT_OVERFLOW_CONTENT[case.name],
                )
        validation_target = object.__new__(COLLECTOR.ResourceSegmentCollector)
        validation_target.config = contract.segment_config
        COLLECTOR.ResourceSegmentCollector._validate_config(validation_target)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            contract.segment_config.target = "/drift"

        fixture = json.loads(contract.fixture_bytes)
        config = json.loads(contract.config_bytes)
        self.assertEqual(contract.fixture_bytes, RESOURCE._canonical_fixture(fixture))
        self.assertEqual(contract.config_bytes, canonical(config))
        self.assertEqual(config["fixture"]["sha256"], sha256(contract.fixture_bytes))
        self.assertEqual(
            config["positive_request"]["sampled_normal"],
            {
                "request_indices": list(range(5, 101, 5)),
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": "request_index",
            },
        )
        self.assertNotIn(b"sk-production-secret", contract.config_bytes)
        self.assertNotIn(b"sk-production-secret", contract.fixture_bytes)

    def test_header_binds_all_sources_and_both_generated_inputs(self) -> None:
        contract = self.build()
        inputs = contract.session_header_fields["input_files"]
        paths = [item["path"] for item in inputs]

        self.assertEqual(paths, sorted(set(paths), key=lambda path: path.encode()))
        self.assertEqual(len(paths), len(self.artifacts.environment["sources"]) + 2)
        self.assertEqual(
            set(paths),
            {item["path"] for item in self.artifacts.environment["sources"]}
            | {
                "collector/config.json",
                "collector/resource-chat-fixture.json",
            },
        )
        by_path = {item["path"]: item for item in inputs}
        self.assertEqual(
            by_path["collector/config.json"],
            {
                "path": "collector/config.json",
                "bytes": len(contract.config_bytes),
                "sha256": sha256(contract.config_bytes),
            },
        )
        self.assertEqual(
            by_path["collector/resource-chat-fixture.json"]["sha256"],
            sha256(contract.fixture_bytes),
        )

    def test_headers_use_exact_identity_runtime_and_independent_validators(
        self,
    ) -> None:
        contract = self.build()

        VALIDATOR._validate_resource_header(contract.resource_header, "resource header")
        self.assertEqual(contract.session_header_fields["schedule"], COLLECTOR.SCHEDULE)
        self.assertEqual(contract.session_header_fields["schedule"], VALIDATOR.SCHEDULE)
        self.assertEqual(
            contract.session_header_fields["thresholds"], COLLECTOR.THRESHOLDS
        )
        self.assertEqual(
            contract.session_header_fields["thresholds"],
            json.loads(json.dumps(VALIDATOR.THRESHOLDS, default=float)),
        )
        self.assertEqual(
            contract.resource_header["schedule"], COLLECTOR.RESOURCE_SCHEDULE
        )
        self.assertEqual(
            contract.resource_header["schedule"], VALIDATOR.RESOURCE_SCHEDULE
        )
        self.assertEqual(contract.resource_header["commands"], COLLECTOR.COMMANDS)
        self.assertEqual(contract.resource_header["commands"], VALIDATOR.COMMANDS)
        self.assertEqual(
            contract.session_header_fields["identities"]["worker_binary_sha256"],
            self.artifacts.model_identity["worker"]["binary_sha256"],
        )
        self.assertEqual(
            contract.resource_header["tools"]["systemd_version_line"],
            self.artifacts.environment["host"]["tools"]["systemd_version_line"],
        )
        self.assertEqual(
            self.independent_identity.calls,
            ["session", "sources"],
        )
        self.assertEqual(
            self.independent_validator.calls,
            ["schedule", "thresholds"],
        )

    def test_rejects_input_ordering_and_duplicates(self) -> None:
        contract = self.build()
        reversed_inputs = copy.deepcopy(contract.session_header_fields["input_files"])
        reversed_inputs.reverse()
        duplicated_inputs = copy.deepcopy(contract.session_header_fields["input_files"])
        duplicated_inputs.append(copy.deepcopy(duplicated_inputs[-1]))
        for name, input_files in (
            ("ordering", reversed_inputs),
            ("duplicate", duplicated_inputs),
        ):
            with self.subTest(name=name):
                header = copy.deepcopy(contract.session_header_fields)
                header["input_files"] = input_files
                broken = dataclasses.replace(contract, session_header_fields=header)
                with self.assertRaisesRegex(
                    RESOURCE.ResourceContractError, "session header drifted"
                ):
                    self.validate(broken)

    def test_rejects_fixture_and_model_drift(self) -> None:
        contract = self.build()
        with (
            self.subTest("fixture bytes"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "fixture bytes drifted"
            ),
        ):
            self.validate(dataclasses.replace(contract, fixture_bytes=b"{}"))

        template = copy.deepcopy(contract.segment_config.resource_body_template)
        template["model"] = "drifted-model"
        config = COLLECTOR.ResourceSegmentConfig(
            contract.segment_config.target,
            template,
            contract.segment_config.negative_cases,
        )
        with (
            self.subTest("model"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "model, or fixture drifted"
            ),
        ):
            self.validate(dataclasses.replace(contract, segment_config=config))

    def test_rejects_sampling_schedule_and_threshold_drift(self) -> None:
        contract = self.build()
        config_value = json.loads(contract.config_bytes)
        config_value["positive_request"]["sampled_normal"]["temperature"] = 0.7
        with (
            self.subTest("sampling"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "config bytes drifted"
            ),
        ):
            self.validate(
                dataclasses.replace(contract, config_bytes=canonical(config_value))
            )

        header = copy.deepcopy(contract.session_header_fields)
        header["schedule"]["normal_requests"] = 99
        with (
            self.subTest("schedule"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "session header drifted"
            ),
        ):
            self.validate(dataclasses.replace(contract, session_header_fields=header))

        header = copy.deepcopy(contract.session_header_fields)
        header["thresholds"]["final_delta_max_bytes"] += 1
        with (
            self.subTest("threshold"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "session header drifted"
            ),
        ):
            self.validate(dataclasses.replace(contract, session_header_fields=header))

    def test_rejects_negative_schedule_and_source_drift(self) -> None:
        contract = self.build()
        cases = list(contract.segment_config.negative_cases)
        case = cases[0]
        cases[0] = COLLECTOR.NegativeCase(
            24, case.name, case.body, case.expected_status
        )
        config = COLLECTOR.ResourceSegmentConfig(
            contract.segment_config.target,
            contract.segment_config.resource_body_template,
            tuple(cases),
        )
        with (
            self.subTest("negative schedule"),
            self.assertRaisesRegex(RESOURCE.ResourceContractError, "negative schedule"),
        ):
            self.validate(dataclasses.replace(contract, segment_config=config))

        header = copy.deepcopy(contract.session_header_fields)
        header["input_files"][0]["sha256"] = "0" * 64
        with (
            self.subTest("source"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "session header drifted"
            ),
        ):
            self.validate(dataclasses.replace(contract, session_header_fields=header))

    def test_rejects_runtime_identity_and_forbidden_value_drift(self) -> None:
        contract = self.build()
        resource_header = copy.deepcopy(contract.resource_header)
        resource_header["probes"]["gpu_index"] = 1
        with (
            self.subTest("runtime identity"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "runtime header drifted"
            ),
        ):
            self.validate(
                dataclasses.replace(contract, resource_header=resource_header)
            )

        with (
            self.subTest("semantic forbidden value"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "contains a forbidden value"
            ),
        ):
            RESOURCE.validate_resource_contract(
                contract,
                self.artifacts,
                self.independent_identity,
                self.independent_validator,
                forbidden_values=(b"synthetic resource probe",),
            )

        with (
            self.subTest("semantic node bound"),
            self.assertRaisesRegex(
                RESOURCE.ResourceContractError, "semantic secret-scan bound"
            ),
        ):
            RESOURCE._scan_forbidden(
                [None] * (RESOURCE.MAX_SEMANTIC_SCAN_NODES + 1),
                (),
                "oversized semantic value",
            )

    def test_rejects_model_drift_in_identity_artifacts(self) -> None:
        contract = self.build()
        model_identity = copy.deepcopy(self.artifacts.model_identity)
        model_identity["model"]["served_id"] = "drifted-model"
        mutated = IDENTITY.IdentityArtifacts(
            self.artifacts.environment,
            model_identity,
            self.artifacts.environment_bytes,
            self.artifacts.model_identity_bytes,
        )
        with self.assertRaises(IDENTITY.IdentityError):
            RESOURCE.validate_resource_contract(
                contract,
                mutated,
                self.independent_identity,
                self.independent_validator,
            )


if __name__ == "__main__":
    unittest.main()
