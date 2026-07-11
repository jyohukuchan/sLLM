from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_fixture_module(name: str, relative: str) -> ModuleType:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR_FIXTURES = load_fixture_module(
    "full_fake_validator_fixtures", "tests/test_validate_sq8_openwebui_release.py"
)
API_FIXTURES = load_fixture_module(
    "full_fake_api_fixtures", "tests/test_sq8_api_contract_gate_ingest.py"
)
OPENWEBUI_FIXTURES = load_fixture_module(
    "full_fake_openwebui_fixtures", "tests/test_sq8_openwebui_gate_ingest.py"
)
STOP_FIXTURES = load_fixture_module(
    "full_fake_stop_fixtures", "tests/test_sq8_openwebui_stop_gate_ingest.py"
)
FAILURE_FIXTURES = load_fixture_module(
    "full_fake_failure_fixtures", "tests/test_sq8_openwebui_failure_gate_ingest.py"
)
LATENCY_FIXTURES = load_fixture_module(
    "full_fake_latency_fixtures", "tests/test_sq8_http_latency_gate_ingest.py"
)
VIEW_FIXTURES = load_fixture_module(
    "full_fake_view_fixtures", "tests/test_sq8_full_campaign_views.py"
)

from sq8_full_campaign_bundle import AtomicCampaignDirectory  # noqa: E402
import sq8_full_campaign_independent_views as INDEPENDENT  # noqa: E402
from sq8_full_campaign_renderer import FullCampaignRenderer  # noqa: E402


VALIDATOR = VALIDATOR_FIXTURES.VALIDATOR
RUN_ID = "synthetic-full-campaign"
WORKER_SHA256 = VALIDATOR_FIXTURES.WORKER_SHA256
BOOT_ID = VALIDATOR_FIXTURES.BOOT_ID
NORMAL_GATEWAY = 1200
NORMAL_WORKER = 1201
RESTART_GATEWAY = 2200
RESTART_WORKER = 2201
NORMAL_GATEWAY_START = 10_000
NORMAL_WORKER_START = 10_001
RESTART_GATEWAY_START = 20_000
RESTART_WORKER_START = 20_001


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


def shift_absolute(value: Any, offset_ns: int) -> Any:
    result = copy.deepcopy(value)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key.endswith("_monotonic_ns") and type(child) is int:
                    item[key] = child + offset_ns
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(result)
    return result


def maximum_absolute(value: Any) -> int:
    maximum = -1

    def visit(item: Any) -> None:
        nonlocal maximum
        if isinstance(item, dict):
            for key, child in item.items():
                if key.endswith("_monotonic_ns") and type(child) is int:
                    maximum = max(maximum, child)
                else:
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return maximum


class FullResourceBuilder(VALIDATOR_FIXTURES.EvidenceBuilder):
    """Reuse the complete resource fixture while retaining finish SSE evidence."""

    def http_exchange(
        self,
        phase: str,
        case_id: str,
        request_index: int,
        body: bytes,
        response: bytes,
        status: int,
        sent_time: int,
        response_time: int,
    ) -> None:
        if phase in {"resource_normal", "resource_restart"} and status == 200:
            payloads = [
                line.removeprefix(b"data: ")
                for line in response.splitlines()
                if line.startswith(b"data: ") and line != b"data: [DONE]"
            ]
            first = json.loads(payloads[0])
            completion_id = first["id"]
            finish = canonical(
                {
                    "id": completion_id,
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                }
            )[:-1]
            response = (
                b"data: "
                + payloads[0]
                + b"\n\ndata: "
                + finish
                + b"\n\ndata: "
                + payloads[1]
                + b"\n\ndata: [DONE]\n\n"
            )
        super().http_exchange(
            phase,
            case_id,
            request_index,
            body,
            response,
            status,
            sent_time,
            response_time,
        )


class FullFakeCampaign:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.parent = base / "campaigns"
        self.parent.mkdir(mode=0o700)
        self.final_path = self.parent / "sq8-full-fake"
        self.bundle = AtomicCampaignDirectory(
            self.final_path, uid=os.getuid(), gid=os.getgid()
        )
        self.raw_root = base / "raw"
        self.raw_root.mkdir(mode=0o700)
        self.source_root = base / "source"
        self.commit = ""
        self.environment: dict[str, Any]
        self.model_identity: dict[str, Any]
        self.builder = FullResourceBuilder(self.raw_root)
        self.cursor = 0
        self.api_view: dict[str, Any]
        self.combined_view: dict[str, Any]
        self.direct_view: dict[str, Any]
        self.stop_view: dict[str, Any]
        self.failure_view: dict[str, Any]
        self.latency_view: dict[str, Any]
        self.stop_png = b""
        self.failure_png = b""
        self.navigation_sha256 = ""
        self.published = False

    def close(self) -> None:
        if not self.published:
            self.bundle.abort()

    def _next_cursor(self) -> str:
        self.cursor += 1
        return f"full-fake-cursor-{self.cursor:05d}"

    def _session_hook(self, record: dict[str, Any], offset_ns: int) -> None:
        fields = shift_absolute(record["fields"], offset_ns)
        self.builder.session_add(
            record["record_type"], record["phase"], record["case_id"], **fields
        )

    def _normalize_browser_actions(
        self, records: Any, specs: Any, *, preserve_required_text: bool = True
    ) -> None:
        for record, spec in zip(records, specs, strict=True):
            fields = record["fields"]
            fields["selector"] = spec.selector
            if spec.input_sha256 == "navigation":
                if self.navigation_sha256:
                    fields["input_sha256"] = self.navigation_sha256
                else:
                    self.navigation_sha256 = fields["input_sha256"]
            else:
                fields["input_sha256"] = spec.input_sha256
            result = fields["result"]
            result["visible"] = True
            result["enabled"] = spec.enabled
            if spec.text == "none":
                result["text_utf8_bytes"] = None
                result["text_sha256"] = None
            elif spec.text != "required" or not preserve_required_text:
                raw = spec.text.encode("utf-8")
                result["text_utf8_bytes"] = len(raw)
                result["text_sha256"] = sha256(raw)

    def _claim(self, claim: Any, offset_ns: int, gateway_pid: int) -> dict[str, Any]:
        outer = json.loads(claim.raw)
        message = outer["MESSAGE"]
        if message.startswith("INFO:     "):
            message = message[len("INFO:     ") :]
        event = shift_absolute(json.loads(message), offset_ns)
        encoded = json.dumps(
            event,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        cursor = self._next_cursor()
        monotonic_usec = event["observed_monotonic_ns"] // 1000
        self.builder.session_add(
            "gateway_event",
            claim.phase,
            claim.case_id,
            journal_cursor=cursor,
            journal_monotonic_usec=monotonic_usec,
            journal_pid=gateway_pid,
            message=encoded,
            message_sha256=sha256(encoded.encode("utf-8")),
            event=event,
        )
        self.builder.journal_records.append(
            {
                "__CURSOR": cursor,
                "__MONOTONIC_TIMESTAMP": str(monotonic_usec),
                "_BOOT_ID": BOOT_ID,
                "_PID": str(gateway_pid),
                "_SYSTEMD_UNIT": "ullm-openai.service",
                "PRIORITY": "6",
                "MESSAGE": encoded,
            }
        )
        return event

    def _identity_checkout(self) -> None:
        environment, model_identity = VALIDATOR_FIXTURES.build_identity_documents()
        fixed = {
            **VALIDATOR.EXPECTED_ORACLE_FILE_IDENTITIES,
            **VALIDATOR.EXPECTED_TTFT_FIXTURE_IDENTITIES,
        }
        for role, relative in VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS.items():
            path = self.source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (
                (ROOT / relative).read_bytes()
                if role in fixed
                else f"synthetic source {role}\n".encode("ascii")
            )
            path.write_bytes(raw)
        for arguments in (
            ("init", "-q"),
            ("config", "user.email", "full-fake@example.invalid"),
            ("config", "user.name", "Full Fake Campaign"),
            ("add", "."),
            ("commit", "-q", "-m", "full fake source"),
        ):
            subprocess.run(
                ("git", "-C", str(self.source_root), *arguments),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.commit = subprocess.run(
            ("git", "-C", str(self.source_root), "rev-parse", "HEAD"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        environment["git"]["commit"] = self.commit
        self.environment = environment
        self.model_identity = model_identity

    def _header(self) -> None:
        environment_raw = VALIDATOR_FIXTURES.identity_canonical(self.environment)
        model_raw = VALIDATOR_FIXTURES.identity_canonical(self.model_identity)
        source_inputs = [
            {key: source[key] for key in ("path", "bytes", "sha256")}
            for source in self.environment["sources"]
        ]
        fixture_raw = json.dumps(
            self.builder.fixture,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        extra_inputs = (
            {
                "path": "collector/config.json",
                "bytes": 3,
                "sha256": sha256(b"{}\n"),
            },
            {
                "path": VALIDATOR.RESOURCE_FIXTURE_INPUT_PATH,
                "bytes": len(fixture_raw),
                "sha256": sha256(fixture_raw),
            },
        )
        input_files = sorted(
            [*source_inputs, *extra_inputs], key=lambda item: item["path"].encode()
        )
        openwebui = self.environment["openwebui"]
        header_openwebui = {
            key: openwebui[key]
            for key in (
                "version",
                "source_revision",
                "base_image_digest",
                "base_image_id",
                "derived_image_id",
                "Dockerfile_sha256",
                "patch_sha256",
                "patched_middleware_sha256",
            )
        }
        self.builder.session_add(
            "header",
            "preflight",
            None,
            run_id=RUN_ID,
            started_utc="2026-07-11T12:00:01Z",
            clock="python.time.monotonic_ns",
            boot_id=BOOT_ID,
            identities={
                "environment_file": "environment.json",
                "environment_sha256": sha256(environment_raw),
                "model_identity_file": "model-identity.json",
                "model_identity_sha256": sha256(model_raw),
                "openwebui": header_openwebui,
                "docker_network_id": openwebui["network_id"],
                "gateway_source_sha256": self.environment["source_sets"]["gateway"],
                "worker_source_sha256": self.environment["source_sets"]["worker"],
                "worker_binary_sha256": WORKER_SHA256,
            },
            input_files=input_files,
            schedule=copy.deepcopy(VALIDATOR.SCHEDULE),
            thresholds=copy.deepcopy(VALIDATOR_FIXTURES.THRESHOLDS),
        )
        (self.raw_root / "environment.json").write_bytes(environment_raw)
        (self.raw_root / "model-identity.json").write_bytes(model_raw)

    def _api(self) -> None:
        offset = 1_000_000_000
        with API_FIXTURES.GateBundle() as fixture:
            result = API_FIXTURES.ADAPTER.ingest_api_contract_bundle(
                fixture.root, fixture.bindings
            )
            messages = [record["MESSAGE"] for record in fixture.journal_records]
        for record in result.http_records:
            self._session_hook(record, offset)
        cursor_by_sequence: list[tuple[str, int, str]] = []
        journal_base_usec = (offset + 500_000_000) // 1000
        for index, message in enumerate(messages):
            cursor = self._next_cursor()
            usec = journal_base_usec + index
            cursor_by_sequence.append((cursor, usec, message))
            self.builder.journal_records.append(
                {
                    "__CURSOR": cursor,
                    "__MONOTONIC_TIMESTAMP": str(usec),
                    "_BOOT_ID": BOOT_ID,
                    "_PID": str(NORMAL_GATEWAY),
                    "_SYSTEMD_UNIT": "ullm-openai.service",
                    "PRIORITY": "6",
                    "MESSAGE": message,
                }
            )
        for index, record in enumerate(result.journal_records):
            cursor, usec, message = cursor_by_sequence[index]
            fields = copy.deepcopy(record["fields"])
            fields.update(
                {
                    "journal_cursor": cursor,
                    "journal_monotonic_usec": usec,
                    "journal_pid": NORMAL_GATEWAY,
                    "message_utf8_bytes": len(message.encode()),
                    "message_sha256": sha256(message.encode()),
                }
            )
            self.builder.session_add(
                record["record_type"], record["phase"], record["case_id"], **fields
            )
        for index, record in enumerate(result.quiet_check_records):
            cursor, usec, _message = cursor_by_sequence[index]
            fields = copy.deepcopy(record["fields"])
            fields.update(
                {
                    "journal_cursor": cursor,
                    "checked_monotonic_ns": usec * 1000 + 100,
                }
            )
            self.builder.session_add(
                record["record_type"], record["phase"], record["case_id"], **fields
            )
        self.api_view = shift_absolute(result.derived_view, offset)

    def _openwebui(self) -> None:
        offset = 10_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OPENWEBUI_FIXTURES.CombinedBundle(temporary)
            result = OPENWEBUI_FIXTURES.INGEST.ingest_combined_soak_bundle(
                fixture.root, fixture.bindings()
            )
        for index in range(21):
            actions = result.browser_action_records[index * 5 : (index + 1) * 5]
            self._normalize_browser_actions(actions, INDEPENDENT._normal_specs(index))
        for record in result.browser_action_records:
            self._session_hook(record, offset)
        for claim in result.lifecycle_claims:
            self._claim(claim, offset, NORMAL_GATEWAY)
        self.combined_view = shift_absolute(result.derived_view, offset)

    def _cancellation(self) -> None:
        direct_offset = 20_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OPENWEBUI_FIXTURES.DirectCancelBundle(temporary)
            direct = OPENWEBUI_FIXTURES.INGEST.ingest_direct_cancel_bundle(
                fixture.root, fixture.bindings()
            )
        for record in direct.http_records:
            self._session_hook(record, direct_offset)
        for claim in direct.lifecycle_claims:
            self._claim(claim, direct_offset, NORMAL_GATEWAY)
        self.direct_view = shift_absolute(direct.derived_view, direct_offset)

        stop_offset = 30_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            fixture = STOP_FIXTURES.StopBundleFixture(temporary)
            stop = STOP_FIXTURES.INGEST.ingest_stop_gate_bundle(
                fixture.root, fixture.bindings()
            )
            self.stop_png = stop.screenshot_evidence.path.read_bytes()
        self._normalize_browser_actions(
            stop.browser_action_records, INDEPENDENT._stop_specs()
        )
        for record in stop.browser_action_records:
            self._session_hook(record, stop_offset)
        for claim in stop.lifecycle_claims:
            self._claim(claim, stop_offset, NORMAL_GATEWAY)
        self.stop_view = shift_absolute(stop.derived_view, stop_offset)

    def _resource_normal(self) -> None:
        for segment in ("normal", "restart"):
            for boundary in ("before", "after"):
                path = self.raw_root / f"amd-smi-metric-{segment}-{boundary}.json"
                path.write_bytes(
                    canonical([{"segment": segment, "boundary": boundary}])
                )
        self.builder.resource_records.append(self.builder.resource_header())
        self.builder.now = 40_000_000_000
        self.builder.lifecycle_probe(
            "resource_normal", "normal-segment-start", "normal"
        )
        self.builder.now += 1
        self.builder.segment("normal", 100)

    def _failure(self) -> None:
        offset = self.builder.now + 10_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FAILURE_FIXTURES.SyntheticBundle(Path(temporary))
            result = FAILURE_FIXTURES.INGEST.ingest_failure_gate_bundle(
                fixture.build(), fixture.bindings()
            )
            self.failure_png = result.screenshot_evidence.source_path.read_bytes()
        self._normalize_browser_actions(
            result.browser_action_records, INDEPENDENT._failure_specs()
        )
        actions = result.browser_action_records
        claims = result.lifecycle_claims
        for record in actions[:4]:
            self._session_hook(record, offset)
        for claim in claims[:4]:
            self._claim(claim, offset, NORMAL_GATEWAY)
        fault = shift_absolute(result.fault_injection_record, offset)
        fault_fields = fault["fields"]
        fault_fields.update(
            {
                "target_pid": NORMAL_WORKER,
                "target_starttime_ticks": NORMAL_WORKER_START,
                "command": "signal.pidfd_send_signal",
            }
        )
        self.builder.session_add(
            fault["record_type"], fault["phase"], fault["case_id"], **fault_fields
        )
        self._claim(claims[4], offset, NORMAL_GATEWAY)
        self._session_hook(actions[4], offset)
        probe = shift_absolute(result.restart_probe_record, offset)
        probe["fields"].update(
            {
                "control_group": "/system.slice/ullm-openai.service",
                "gateway_pid": RESTART_GATEWAY,
                "gateway_starttime_ticks": RESTART_GATEWAY_START,
                "worker_pid": RESTART_WORKER,
                "worker_starttime_ticks": RESTART_WORKER_START,
                "n_restarts": 3,
            }
        )
        self.builder.session_add(
            probe["record_type"], probe["phase"], probe["case_id"], **probe["fields"]
        )
        for record in actions[5:7]:
            self._session_hook(record, offset)
        for claim in claims[5:]:
            self._claim(claim, offset, RESTART_GATEWAY)
        for record in actions[7:]:
            self._session_hook(record, offset)
        self.failure_view = shift_absolute(result.derived_view, offset)
        self.builder.now = (
            max(
                maximum_absolute(probe),
                maximum_absolute(shift_absolute(result.derived_view, offset)),
            )
            + 1_000_000_000
        )

    def _resource_restart(self) -> None:
        self.builder.lifecycle_probe(
            "resource_restart", "restart-segment-start", "restart"
        )
        self.builder.now += 1
        self.builder.segment("restart", 20)

    def _latency(self) -> None:
        offset = self.builder.now + 10_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LATENCY_FIXTURES.SyntheticLatencyBundle(Path(temporary))
            result = LATENCY_FIXTURES.INGEST.ingest_latency_gate_bundle(
                fixture.bundle, fixture.bindings()
            )
        for record in result.http_records:
            self._session_hook(record, offset)
        maximum = -1
        for claim in result.lifecycle_claims:
            event = self._claim(claim, offset, RESTART_GATEWAY)
            maximum = max(maximum, event["observed_monotonic_ns"])
        self.latency_view = shift_absolute(result.derived_view, offset)
        self.builder.now = maximum + 1_000_000_000

    def _final(self) -> None:
        self.builder.lifecycle_probe("final", "final-service-ready", "restart")
        counts = Counter(
            record["record_type"] for record in self.builder.session_records
        )
        counts["run_end"] += 1
        self.builder.session_add(
            "run_end",
            "final",
            None,
            completed_utc="2026-07-11T13:00:00Z",
            completed_monotonic_ns=self.builder.now + 1_000_000_000,
            final_git_commit=self.commit,
            final_git_status_raw="",
            final_git_status_sha256=sha256(b""),
            record_counts=dict(counts),
            final_journal_cursor=self.builder.journal_records[-1]["__CURSOR"],
        )

    def _write_raw(self) -> None:
        self.builder.write_jsonl(
            "raw-session-results.jsonl", self.builder.session_records
        )
        self.builder.write_jsonl(
            "soak-resources.raw.jsonl", self.builder.resource_records
        )
        self.builder.write_jsonl(
            "service-journal.raw.jsonl", self.builder.journal_records
        )
        existing = {
            "environment.json",
            "model-identity.json",
            "raw-session-results.jsonl",
            "soak-resources.raw.jsonl",
            "service-journal.raw.jsonl",
            "amd-smi-metric-normal-before.json",
            "amd-smi-metric-normal-after.json",
            "amd-smi-metric-restart-before.json",
            "amd-smi-metric-restart-after.json",
        }
        for relative in sorted(existing):
            self.bundle.write_bytes(
                relative,
                (self.raw_root / relative).read_bytes(),
                scan=lambda _raw, _label: None,
            )
        self.bundle.write_bytes(
            "browser/openwebui-stop-before.png",
            self.stop_png,
            scan=lambda _raw, _label: None,
        )
        self.bundle.write_bytes(
            "browser/post-header-failure.png",
            self.failure_png,
            scan=lambda _raw, _label: None,
        )

    def _render(self) -> None:
        normal_resource = types.SimpleNamespace(
            segment="normal",
            identity=types.SimpleNamespace(),
            warmup_requests=10,
            measured_requests=100,
            negative_requests=3,
            resource_samples=505,
            gpu_metrics=2,
            sampling_cases=tuple(VIEW_FIXTURES.sampling_cases()),
        )
        evidence = types.SimpleNamespace(
            preflight=types.SimpleNamespace(
                header_fields={
                    "run_id": RUN_ID,
                    "schedule": copy.deepcopy(VALIDATOR.SCHEDULE),
                    "thresholds": copy.deepcopy(VALIDATOR_FIXTURES.THRESHOLDS),
                }
            ),
            api_contract=types.SimpleNamespace(derived_view=self.api_view),
            combined=types.SimpleNamespace(derived_view=self.combined_view),
            direct_cancel=types.SimpleNamespace(derived_view=self.direct_view),
            stop=types.SimpleNamespace(derived_view=self.stop_view),
            failure=types.SimpleNamespace(derived_view=self.failure_view),
            latency=types.SimpleNamespace(derived_view=self.latency_view),
            resource_normal=normal_resource,
        )
        rendered = FullCampaignRenderer().render(
            types.SimpleNamespace(stage_path=self.bundle.stage_path, evidence=evidence)
        )
        for relative in sorted(rendered):
            self.bundle.write_bytes(
                relative, rendered[relative], scan=lambda _raw, _label: None
            )

    def prepare(self) -> None:
        self._identity_checkout()
        self._header()
        self._api()
        self._openwebui()
        self._cancellation()
        self._resource_normal()
        self._failure()
        self._resource_restart()
        self._latency()
        self._final()
        self._write_raw()
        self._render()

    def validator(self) -> Any:
        return VALIDATOR.FullCampaignIndependentValidator(
            expected_commit=self.commit,
            expected_worker_binary_sha256=WORKER_SHA256,
            repo_root=self.source_root,
            forbidden_values=(b"never-present-full-fake-token",),
        )

    def publish(self) -> Path:
        self.bundle.validate_before_independent_validator()
        evidence = self.validator().validate(self.bundle.stage_path)
        published = self.bundle.publish(evidence)
        self.published = True
        return published

    def refresh_matrix_and_sums(self, relative: str) -> None:
        matrix_path = self.bundle.stage_path / "release-matrix.json"
        matrix = json.loads(matrix_path.read_bytes())
        target = self.bundle.stage_path / relative
        for entry in matrix["files"]:
            if entry["path"] == relative:
                raw = target.read_bytes()
                entry["bytes"] = len(raw)
                entry["sha256"] = sha256(raw)
                break
        matrix_path.write_bytes(canonical(matrix))
        sums = "".join(
            f"{sha256((self.bundle.stage_path / path).read_bytes())}  {path}\n"
            for path in sorted(
                VALIDATOR.BUNDLE_FILES - {"SHA256SUMS"},
                key=lambda value: value.encode(),
            )
        )
        (self.bundle.stage_path / "SHA256SUMS").write_text(sums, encoding="ascii")

    def mutate_jsonl(
        self, relative: str, mutation: Any, *, refresh_matrix: bool = True
    ) -> None:
        path = self.bundle.stage_path / relative
        records = [json.loads(line) for line in path.read_text().splitlines()]
        mutation(records)
        path.write_bytes(b"".join(canonical(record) for record in records))
        if refresh_matrix:
            self.refresh_matrix_and_sums(relative)


class FullFakeCampaignIntegrationTests(unittest.TestCase):
    def fixture(self) -> FullFakeCampaign:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = FullFakeCampaign(Path(temporary.name))
        self.addCleanup(fixture.close)
        fixture.prepare()
        return fixture

    def test_real_renderer_validator_and_atomic_publication(self) -> None:
        fixture = self.fixture()
        published = fixture.publish()
        self.assertEqual(published, fixture.final_path)
        self.assertTrue((published / "release-validation.json").is_file())
        report = json.loads((published / "release-validation.json").read_bytes())
        self.assertEqual(report["release_status"], "complete")
        self.assertTrue(report["full_campaign_validated"])
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o700)
        self.assertEqual(
            set(path.name for path in published.iterdir()),
            set(VALIDATOR.BUNDLE_FILES)
            - {
                "browser/openwebui-stop-before.png",
                "browser/post-header-failure.png",
            }
            | {"browser", "release-validation.json"},
        )

    def test_representative_derived_quiet_journal_source_and_header_mutations_fail(
        self,
    ) -> None:
        mutations = ("derived", "quiet", "journal", "source", "header")
        for kind in mutations:
            with self.subTest(kind=kind):
                fixture = self.fixture()
                if kind == "derived":
                    path = fixture.bundle.stage_path / "sampling-results.json"
                    value = json.loads(path.read_bytes())
                    value["sampled_request_count"] = 19
                    path.write_bytes(canonical(value))
                    fixture.refresh_matrix_and_sums("sampling-results.json")
                elif kind == "quiet":
                    fixture.mutate_jsonl(
                        "raw-session-results.jsonl",
                        lambda records: next(
                            record
                            for record in records
                            if record["record_type"] == "lifecycle_quiet_check"
                        ).__setitem__("observer_event_count", 1),
                    )
                elif kind == "journal":
                    fixture.mutate_jsonl(
                        "service-journal.raw.jsonl",
                        lambda records: records[0].__setitem__(
                            "MESSAGE", records[0]["MESSAGE"] + "-changed"
                        ),
                    )
                elif kind == "source":
                    source = (
                        fixture.source_root
                        / VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS["gate_api_contract"]
                    )
                    source.write_bytes(b"changed source\n")
                else:
                    fixture.mutate_jsonl(
                        "raw-session-results.jsonl",
                        lambda records: records[0]["identities"].__setitem__(
                            "gateway_source_sha256", "0" * 64
                        ),
                    )
                with self.assertRaises(VALIDATOR.ValidationError):
                    fixture.validator().validate(fixture.bundle.stage_path)
                self.assertFalse(
                    (fixture.bundle.stage_path / "release-validation.json").exists()
                )


if __name__ == "__main__":
    unittest.main()
