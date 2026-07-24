from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-generic-reasoning-release-campaign.py"
VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_reasoning_release_campaign", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_reasoning_release_validator_for_campaign", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def test_v2_dispatch_binds_reasoning_release_run_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.json"
    observed: dict[str, object] = {}

    def construct(**arguments: object) -> object:
        observed.update(arguments)
        return type(
            "Binding",
            (),
            {"candidate": type("Candidate", (), {"path": candidate})()},
        )()

    monkeypatch.setattr(TOOL, "optional_v2_binding", construct)
    manifest, binding = TOOL._select_manifest_and_binding(
        active_binding_mode="v2",
        manifest=None,
        candidate_served_model_manifest=candidate,
        active_served_model_manifest=tmp_path / "active.json",
        expected_served_model_manifest_sha256="a" * 64,
        campaign_authorization=tmp_path / "authorization.json",
        run_id="reasoning-release-run",
        output_dir=tmp_path / "release-output",
    )

    assert binding is not None
    assert manifest == candidate
    assert observed["campaign_name"] == "reasoning_release"
    assert observed["run_id"] == "reasoning-release-run"
    assert observed["final_path"] == tmp_path / "release-output"


def _transaction_environment(
    *,
    stage: str,
    staging: Path,
    authorization: Path,
    claim: Path,
    authorization_sha256: str = "a" * 64,
    claim_sha256: str = "b" * 64,
) -> dict[str, str]:
    return {
        TOOL.TRANSACTION_STAGING_OUTPUT_ENV: str(staging),
        TOOL.TRANSACTION_STAGE_ENV: stage,
        TOOL.TRANSACTION_AUTHORIZATION_ENV: str(authorization),
        TOOL.TRANSACTION_CLAIM_ENV: str(claim),
        TOOL.TRANSACTION_AUTHORIZATION_SHA256_ENV: authorization_sha256,
        TOOL.TRANSACTION_CLAIM_SHA256_ENV: claim_sha256,
    }


def _v2_transaction_binding(
    *,
    final: Path,
    authorization: Path,
    claim: Path,
    run_id: str = "release-run",
) -> SimpleNamespace:
    return SimpleNamespace(
        campaign_name="reasoning_release",
        run_id=run_id,
        final_path=final,
        claim=SimpleNamespace(
            authorization_path=authorization,
            authorization_sha256="a" * 64,
            path=claim,
            sha256="b" * 64,
        ),
    )


def test_transaction_staging_is_opt_in_and_preserves_legacy_default() -> None:
    output = Path("relative-legacy-output")

    selected, transaction_run_id = TOOL._transaction_publication_output(
        authorized_output=output,
        active_binding_mode="legacy",
        campaign_authorization_path=None,
        run_id=None,
        active_binding=None,
        environment={
            TOOL.TRANSACTION_STAGE_ENV: "forged-but-not-opted-in",
            TOOL.TRANSACTION_CLAIM_SHA256_ENV: "not-a-hash",
        },
    )

    assert selected == output
    assert transaction_run_id is None


def test_v2_transaction_staging_is_mandatory(tmp_path: Path) -> None:
    with pytest.raises(TOOL.CampaignError, match="requires locked transaction"):
        TOOL._transaction_publication_output(
            authorized_output=tmp_path / "authorized-release",
            active_binding_mode="v2",
            campaign_authorization_path=tmp_path / "authorization.json",
            run_id="release-run",
            active_binding=SimpleNamespace(),
            environment={},
        )


def test_sq8_transaction_staging_binds_claim_stage_run_and_final_without_leak(
    tmp_path: Path,
) -> None:
    final = tmp_path / "authorized-release"
    staging = tmp_path / "private-release-stage"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    binding = _v2_transaction_binding(
        final=final,
        authorization=authorization,
        claim=claim,
    )

    selected, transaction_run_id = TOOL._transaction_publication_output(
        authorized_output=final,
        active_binding_mode="v2",
        campaign_authorization_path=authorization,
        run_id="release-run",
        active_binding=binding,
        environment=_transaction_environment(
            stage="reasoning_release",
            staging=staging,
            authorization=authorization,
            claim=claim,
        ),
    )

    assert selected == staging
    assert transaction_run_id == "release-run"
    public_result = {"output_dir": str(final), "run_id": transaction_run_id}
    assert str(staging) not in json.dumps(public_result, sort_keys=True)
    assert binding.final_path == final


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-stage",
        "wrong-claim",
        "wrong-claim-hash",
        "wrong-final",
        "double-slash",
        "equal",
        "overlap",
        "existing",
    ],
)
def test_sq8_transaction_staging_rejects_forged_environment_and_paths(
    tmp_path: Path,
    mutation: str,
) -> None:
    final = tmp_path / "authorized-release"
    staging = tmp_path / "private-release-stage"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    binding = _v2_transaction_binding(
        final=final,
        authorization=authorization,
        claim=claim,
    )
    environment = _transaction_environment(
        stage="reasoning_release",
        staging=staging,
        authorization=authorization,
        claim=claim,
    )
    if mutation == "wrong-stage":
        environment[TOOL.TRANSACTION_STAGE_ENV] = "reasoning_browser"
    elif mutation == "wrong-claim":
        environment[TOOL.TRANSACTION_CLAIM_ENV] = str(
            tmp_path / "forged-claim.json"
        )
    elif mutation == "wrong-claim-hash":
        environment[TOOL.TRANSACTION_CLAIM_SHA256_ENV] = "c" * 64
    elif mutation == "wrong-final":
        binding.final_path = tmp_path / "different-final"
    elif mutation == "double-slash":
        environment[TOOL.TRANSACTION_STAGING_OUTPUT_ENV] = (
            f"{tmp_path}//private-release-stage"
        )
    elif mutation == "equal":
        environment[TOOL.TRANSACTION_STAGING_OUTPUT_ENV] = str(final)
    elif mutation == "overlap":
        environment[TOOL.TRANSACTION_STAGING_OUTPUT_ENV] = str(
            final / "nested-stage"
        )
    elif mutation == "existing":
        staging.mkdir()

    with pytest.raises(TOOL.CampaignError, match="transaction"):
        TOOL._transaction_publication_output(
            authorized_output=final,
            active_binding_mode="v2",
            campaign_authorization_path=authorization,
            run_id="release-run",
            active_binding=binding,
            environment=environment,
        )


def test_fresh_aq4_transaction_staging_loads_consumed_claim_and_derives_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "authorized-aq4-release"
    staging = tmp_path / "private-aq4-release-stage"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    record = SimpleNamespace(
        snapshot=SimpleNamespace(path=claim, sha256="b" * 64),
        authorization=SimpleNamespace(
            snapshot=SimpleNamespace(
                path=authorization,
                sha256="a" * 64,
            ),
            document={
                "campaigns": {
                    "aq4_reasoning_release": {
                        "run_id": "aq4-release-run",
                        "final_path": str(final),
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        TOOL.campaign_authorization,
        "load_live_claim",
        lambda path, **_kwargs: record
        if path == authorization
        else pytest.fail("unexpected authorization path"),
    )
    environment = _transaction_environment(
        stage="aq4_reasoning_release",
        staging=staging,
        authorization=authorization,
        claim=claim,
    )

    selected, transaction_run_id = TOOL._transaction_publication_output(
        authorized_output=final,
        active_binding_mode="legacy",
        campaign_authorization_path=None,
        run_id=None,
        active_binding=None,
        environment=environment,
    )

    assert selected == staging
    assert transaction_run_id == "aq4-release-run"
    with pytest.raises(TOOL.CampaignError, match="authorization claim or output"):
        TOOL._transaction_publication_output(
            authorized_output=tmp_path / "forged-final",
            active_binding_mode="legacy",
            campaign_authorization_path=None,
            run_id=None,
            active_binding=None,
            environment=environment,
        )


def test_modes_and_request_body_are_explicit_and_bounded() -> None:
    fixture = TOOL.Fixture("fixture", "hello", "ok")

    assert TOOL._mode_fields("disabled") == {"reasoning_effort": "none"}
    assert TOOL._mode_fields("budget-32") == {"thinking_budget_tokens": 32}
    assert TOOL._mode_fields("budget-128") == {"thinking_budget_tokens": 128}
    assert TOOL._mode_fields("budget-256") == {"thinking_budget_tokens": 256}
    assert TOOL._mode_fields("unbounded") == {"thinking_budget_tokens": -1}

    body = json.loads(TOOL._request_body("model", "budget-128", fixture))
    assert body["model"] == "model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_completion_tokens"] == 512
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["thinking_budget_tokens"] == 128

    with pytest.raises(TOOL.CampaignError, match="unknown release mode"):
        TOOL._mode_fields("invalid")


def test_nonstream_response_is_bounded_and_matches_release_case_contract() -> None:
    payload = {
        "id": "completion-nonstream",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "timings": {"prompt_per_second": 100.0, "predicted_per_second": 80.0},
    }
    output = json.dumps(payload, separators=(",", ":")).encode("ascii")
    marker = b"\n__ULLM_HTTP_STATUS__200\n"
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(" + repr(output + marker) + ")",
    ]
    result = TOOL._nonstream_request(b"request", command=command, timeout_seconds=2.0)

    assert result.stream is False
    assert result.status == 200
    assert result.completion_id == "completion-nonstream"
    assert result.sse_chunk_count == 0
    assert result.answer_text == "ok"
    assert result.reasoning_tokens == 0

    fixture = TOOL.Fixture("fixture", "hello", "ok")
    release = {
        "completion_id": result.completion_id,
        "outcome": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "reset_complete": True,
        "admit_to_start_ns": 1,
        "start_to_release_ns": 2,
        "admit_to_release_ns": 3,
    }
    sample = TOOL.ResourceSample(100, 200, 50.0, 100.0)
    case, lifecycle, _ = TOOL._case_and_lifecycle(
        mode="disabled", fixture=fixture, result=result, release=release,
        before=sample, after=sample,
    )
    assert case["id"] == "generic-reasoning-disabled-nonstream"
    assert case["stream"] is False
    assert case["sse_chunk_count"] == 0
    assert case["timing"]["answer_decode_tokens_per_second"] == 80.0
    assert VALIDATOR._validate_case(case) == "disabled"
    assert VALIDATOR._validate_lifecycle({
        "schema_version": VALIDATOR.LIFECYCLE_SCHEMA_VERSION,
        "events": [lifecycle],
    }, {case["id"]: case})["case_ids"] == {case["id"]}


def test_stream_and_nonstream_semantics_are_compared_without_persisting_text() -> None:
    fields = {
        "status": 200,
        "completion_id": "different-id",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "reasoning_tokens": 1,
        "usage_timings": {},
        "answer_text": "ok",
        "reasoning_text": "step",
        "sse_chunk_count": 3,
        "first_reasoning_ms": 1.0,
        "first_answer_ms": 2.0,
        "latency_ms": 3.0,
    }
    stream = TOOL.StreamResult(**fields, stream=True)
    nonstream = TOOL.StreamResult(**fields, stream=False)

    TOOL._assert_transport_match("budget-32", stream, nonstream)
    nonstream_mismatch = TOOL.StreamResult(**{**fields, "answer_text": "different"}, stream=False)
    with pytest.raises(TOOL.CampaignError, match="stream/non-stream contract differs"):
        TOOL._assert_transport_match("budget-32", stream, nonstream_mismatch)


def test_paired_campaign_cases_form_a_validator_compatible_hash_only_artifact(tmp_path: Path) -> None:
    budgets = {
        "disabled": 0,
        "budget-32": 8,
        "budget-128": 20,
        "budget-256": 24,
        "unbounded": 30,
    }
    cases: list[dict] = []
    events: list[dict] = []
    fixtures: dict[str, TOOL.Fixture] = {}
    for mode, reasoning in budgets.items():
        fixture = TOOL.Fixture(f"fixture-{mode}", f"prompt-{mode}", f"answer-{mode}")
        fixtures[mode] = fixture
        forced = 0 if mode == "disabled" else 1
        for stream_enabled in (True, False):
            completion_tokens = reasoning + forced + 2
            result = TOOL.StreamResult(
                status=200,
                completion_id=f"id-{mode}-{stream_enabled}",
                finish_reason="stop",
                prompt_tokens=16,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning,
                usage_timings={"prompt_per_second": 100.0, "predicted_per_second": 80.0},
                answer_text=fixture.expected_answer,
                reasoning_text="" if mode == "disabled" else "internal",
                sse_chunk_count=3 if stream_enabled else 0,
                first_reasoning_ms=2.0 if reasoning else None,
                first_answer_ms=4.0,
                latency_ms=10.0,
                stream=stream_enabled,
            )
            release = {
                "completion_id": result.completion_id,
                "outcome": "stop",
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reset_complete": True,
                "admit_to_start_ns": 1,
                "start_to_release_ns": 2,
                "admit_to_release_ns": 3,
            }
            if mode != "disabled":
                release.update({"reasoning_tokens": reasoning, "forced_end_tokens": forced})
            sample = TOOL.ResourceSample(100, 200, 50.0, 100.0)
            case, event, _ = TOOL._case_and_lifecycle(
                mode=mode, fixture=fixture, result=result, release=release,
                before=sample, after=sample,
            )
            cases.append(case)
            events.append(event)

    document = {
        "schema_version": VALIDATOR.SCHEMA_VERSION,
        "status": "incomplete",
        "production_activation_performed": False,
        "source_commit": "1" * 40,
        "active_promotion_source_commit": "2" * 40,
        "source_commit_aligned": False,
        "git_worktree_clean": True,
        "git_worktree_status_sha256": "f" * 64,
        "identity": {
            "manifest_sha256": "b" * 64,
            "worker_binary_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "openwebui_image": "ullm/open-webui@sha256:" + "e" * 64,
        },
        "cases": cases,
        "lifecycle": {
            "schema_version": VALIDATOR.LIFECYCLE_SCHEMA_VERSION,
            "events": events,
        },
    }
    path = tmp_path / "paired-release.json"
    raw = json.dumps(document, ensure_ascii=True)
    path.write_text(raw, encoding="ascii")
    report = VALIDATOR.validate(path)

    assert report["structurally_valid"] is True
    assert report["case_count"] == 10
    assert report["lifecycle_event_count"] == 10
    assert report["gate_eligible"] is False
    assert all(fixture.expected_answer not in raw for fixture in fixtures.values())


def test_manifest_preflight_rejects_v1_before_external_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "ullm.served_model.v1"}), encoding="ascii")

    with pytest.raises(TOOL.CampaignError, match="not v2"):
        TOOL._validate_manifest(manifest)


def test_immutable_http_image_is_required(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_bytes(b"opaque-token")

    with pytest.raises(TOOL.CampaignError, match="immutable Docker"):
        TOOL.execute(
            output_dir=tmp_path / "out",
            manifest=tmp_path / "missing-manifest",
            fixture_suite=TOOL.DEFAULT_FIXTURES,
            token_file=token,
            http_image="curl:latest",
        )


def test_fresh_aq4_runner_publishes_only_to_transaction_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = tmp_path / "token"
    token.write_bytes(b"opaque-token")
    final = tmp_path / "authorized-release"
    staging = tmp_path / "private-stage" / "release"
    authorization = tmp_path / "authorization.json"
    claim = tmp_path / "claim.json"
    fixtures = {
        f"generic-reasoning-{mode}": TOOL.Fixture(
            f"generic-reasoning-{mode}",
            f"prompt-{mode}",
            "ok",
        )
        for mode in TOOL.MODES
    }

    def result(mode: str, *, stream: bool) -> TOOL.StreamResult:
        reasoning_tokens = 0 if mode == "disabled" else 1
        return TOOL.StreamResult(
            status=200,
            completion_id=f"{mode}-{'stream' if stream else 'nonstream'}",
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=reasoning_tokens + 2,
            reasoning_tokens=reasoning_tokens,
            usage_timings={
                "prompt_per_second": 100.0,
                "predicted_per_second": 80.0,
            },
            answer_text="ok",
            reasoning_text="" if mode == "disabled" else "thought",
            sse_chunk_count=2 if stream else 0,
            first_reasoning_ms=1.0 if reasoning_tokens else None,
            first_answer_ms=2.0,
            latency_ms=10.0,
            stream=stream,
        )

    stream_modes = iter(TOOL.MODES)
    nonstream_modes = iter(TOOL.MODES)
    monkeypatch.setattr(
        TOOL,
        "_stream_request",
        lambda *_args, **_kwargs: result(next(stream_modes), stream=True),
    )
    monkeypatch.setattr(
        TOOL,
        "_nonstream_request",
        lambda *_args, **_kwargs: result(
            next(nonstream_modes),
            stream=False,
        ),
    )
    monkeypatch.setattr(TOOL, "_load_fixtures", lambda _path: fixtures)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest",
        lambda _path: {
            "manifest_sha256": "d" * 64,
            "model_id": "ullm-qwen3.5-9b-aq4",
            "format_id": "AQ4_0",
            "worker": {
                "binary": "/opt/ullm/bin/ullm-aq4-worker",
                "binary_sha256": "e" * 64,
            },
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_read_gpu_processes",
        lambda *_args, **_kwargs: {"positive_vram_processes": []},
    )
    monkeypatch.setattr(TOOL, "_bind_gpu_processes", lambda *_args: None)
    monkeypatch.setattr(TOOL, "_docker_command", lambda **_kwargs: [])
    sample = TOOL.ResourceSample(100, 200, 50.0, 100.0)
    monkeypatch.setattr(
        TOOL,
        "_resource_sample",
        lambda *_args, **_kwargs: sample,
    )

    class Observer:
        def __init__(self, _path: Path) -> None:
            pass

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        def wait_release(
            self,
            completion_id: str,
            _timeout_seconds: float,
        ) -> dict[str, object]:
            mode = completion_id.rsplit("-", 1)[0]
            reasoning_tokens = 0 if mode == "disabled" else 1
            release: dict[str, object] = {
                "completion_id": completion_id,
                "outcome": "stop",
                "prompt_tokens": 8,
                "completion_tokens": reasoning_tokens + 2,
                "reset_complete": True,
                "admit_to_start_ns": 1,
                "start_to_release_ns": 2,
                "admit_to_release_ns": 3,
            }
            if mode != "disabled":
                release.update(
                    {"reasoning_tokens": reasoning_tokens, "forced_end_tokens": 0}
                )
            return release

    monkeypatch.setattr(TOOL, "LifecycleObserver", Observer)
    record = SimpleNamespace(
        snapshot=SimpleNamespace(path=claim, sha256="b" * 64),
        authorization=SimpleNamespace(
            snapshot=SimpleNamespace(
                path=authorization,
                sha256="a" * 64,
            ),
            document={
                "campaigns": {
                    "aq4_reasoning_release": {
                        "run_id": "fresh-aq4-release",
                        "final_path": str(final),
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        TOOL.campaign_authorization,
        "load_live_claim",
        lambda _path, **_kwargs: record,
    )
    for name, value in _transaction_environment(
        stage="aq4_reasoning_release",
        staging=staging,
        authorization=authorization,
        claim=claim,
    ).items():
        monkeypatch.setenv(name, value)

    execution = TOOL.execute(
        output_dir=final,
        manifest=tmp_path / "manifest.json",
        fixture_suite=tmp_path / "fixtures.json",
        token_file=token,
        http_image="sha256:" + "f" * 64,
    )

    assert not final.exists()
    assert staging.is_dir()
    assert {
        entry.name for entry in staging.iterdir()
    } == TOOL.CAMPAIGN_OUTPUT_FILES
    assert execution["output_dir"] == str(final)
    assert execution["run_id"] == "fresh-aq4-release"
    assert str(staging) not in json.dumps(execution, sort_keys=True)
    summary = json.loads(
        (staging / "summary.json").read_text(encoding="ascii")
    )
    assert summary["transaction_campaign"] == {
        "name": "aq4_reasoning_release",
        "run_id": "fresh-aq4-release",
        "final_path": str(final),
    }
    assert all(
        str(staging) not in entry.read_text(encoding="ascii")
        for entry in staging.iterdir()
    )


def test_campaign_directory_publication_is_atomic_no_replace(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".campaign.incomplete"
    stage.mkdir()
    target = tmp_path / "campaign"
    target.mkdir()
    marker = target / "belongs-to-racer"
    marker.write_bytes(b"preserve")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(TOOL.CampaignError, match="already exists"):
            TOOL._rename_directory_noreplace(
                descriptor,
                stage.name,
                target.name,
            )
    finally:
        os.close(descriptor)

    assert marker.read_bytes() == b"preserve"
    assert stage.is_dir()


@pytest.mark.parametrize("basename", ["ullm-aq4-worker", "ullm-sq8-worker"])
def test_gpu_process_identity_is_bound_to_manifest_binary_and_name(
    basename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = {
        "positive_vram_processes": [{"pid": "123", "process": basename}]
    }
    monkeypatch.setattr(TOOL, "_hash_process_executable", lambda _pid: "a" * 64)

    TOOL._bind_gpu_processes(preflight, "a" * 64, basename)
    assert preflight["positive_vram_processes"][0]["binary_sha256"] == "a" * 64

    monkeypatch.setattr(TOOL, "_hash_process_executable", lambda _pid: "b" * 64)
    with pytest.raises(TOOL.CampaignError, match="differs from the v2 manifest"):
        TOOL._bind_gpu_processes(preflight, "a" * 64, basename)


@pytest.mark.parametrize(
    ("format_id", "binary", "expected"),
    [
        ("AQ4_0", "/opt/ullm/bin/ullm-aq4-worker", "ullm-aq4-worker"),
        ("SQ8_0", "/opt/ullm/bin/ullm-sq8-worker", "ullm-sq8-worker"),
    ],
)
def test_worker_process_basename_is_derived_from_validated_manifest(
    format_id: str, binary: str, expected: str
) -> None:
    assert (
        TOOL._worker_process_basename(
            {"format_id": format_id, "worker": {"binary": binary}}
        )
        == expected
    )


@pytest.mark.parametrize(
    ("format_id", "binary"),
    [
        ("SQ8_0", "/opt/ullm/bin/ullm-worker"),
        ("SQ8_0", "ullm-sq8-worker"),
        ("SQ8_0", "/opt/ullm/bin/llama-server"),
        ("SQ8_0", ""),
        ("SQ8_0", "/opt/ullm/bin/ullm-aq4-worker"),
        ("AQ4_0", "/opt/ullm/bin/ullm-sq8-worker"),
        ("UNKNOWN", "/opt/ullm/bin/ullm-sq8-worker"),
    ],
)
def test_worker_process_basename_rejects_unbound_names(
    format_id: str, binary: str
) -> None:
    with pytest.raises(TOOL.CampaignError, match="worker executable"):
        TOOL._worker_process_basename(
            {"format_id": format_id, "worker": {"binary": binary}}
        )


def test_gpu_preflight_requires_the_manifest_derived_process_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "system": {
            "PID123": "ullm-sq8-worker, 1, 4096",
        }
    }
    monkeypatch.setattr(
        TOOL.subprocess,
        "run",
        lambda *args, **kwargs: TOOL.subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    preflight = TOOL._read_gpu_processes(
        expected_process_basename="ullm-sq8-worker"
    )
    assert preflight["positive_vram_processes"][0]["process"] == "ullm-sq8-worker"
    with pytest.raises(TOOL.CampaignError, match="unexpected process"):
        TOOL._read_gpu_processes(expected_process_basename="ullm-aq4-worker")
    with pytest.raises(TOOL.CampaignError, match="not bound"):
        TOOL._read_gpu_processes()


def test_gpu_preflight_accepts_rocm_no_process_output(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[str]] = []

    def run(args, **_kwargs):
        observed.append(args)
        return TOOL.subprocess.CompletedProcess(
            args, 0, stdout="", stderr="WARNING: No JSON data to report.\n"
        )

    monkeypatch.setattr(
        TOOL.subprocess,
        "run",
        run,
    )

    assert TOOL._read_gpu_processes(
        TOOL.CANONICAL_ROCM_SMI
    )["positive_vram_processes"] == []
    assert observed == [
        [
            TOOL.CANONICAL_PYTHON,
            *TOOL.ROCM_PYTHON_ARGUMENTS,
            TOOL.CANONICAL_ROCM_SMI,
            "--showpids",
            "--json",
        ]
    ]


def test_lifecycle_observer_correlates_release_and_removes_socket(tmp_path: Path) -> None:
    path = tmp_path / "observer.sock"
    observer = TOOL.LifecycleObserver(path)
    observer.open()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def send() -> None:
        sender.sendto(
            json.dumps(
                {
                    "schema_version": TOOL.LIFECYCLE_SCHEMA,
                    "event": "request_released",
                    "completion_id": "matching",
                }
            ).encode("ascii"),
            str(path),
        )

    thread = threading.Thread(target=send)
    thread.start()
    assert observer.wait_release("matching", 2.0)["completion_id"] == "matching"
    thread.join()
    sender.close()
    observer.close()
    assert not path.exists()
