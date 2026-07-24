from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXED_ACTIVE_MANIFEST_PATH = "/etc/ullm/served-models/active.json"
TOOL_PATH = ROOT / "tools/prepare-generic-reasoning-release-evidence.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_release_preparer", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def measured_case(case_id: str, mode: str, reasoning: int = 0) -> dict:
    answer = 2
    return {
        "id": case_id,
        "mode": mode,
        "prompt_fixture_id": f"fixture-{case_id}",
        "prompt_sha256": "a" * 64,
        "stream": True,
        "http_status": 200,
        "sse_chunk_count": 3,
        "finish_reason": "stop",
        "raw": {
            "prompt_tokens": 8,
            "completion_tokens": reasoning + answer,
            "reasoning_tokens": reasoning,
            "forced_end_tokens": 0,
            "answer_tokens": answer,
            "budget_overshoot": 0,
            "empty_answer": False,
            "usage_completion_tokens": reasoning + answer,
        },
        "timing": {
            "prefill_tokens_per_second": 100.0,
            "first_reasoning_token_ms": 10.0 if reasoning else None,
            "first_answer_token_ms": 20.0,
            "reasoning_decode_tokens_per_second": 50.0 if reasoning else None,
            "answer_decode_tokens_per_second": 80.0,
            "decode_tokens_per_second": 70.0,
            "latency_ms": 100.0,
        },
        "resource": {
            "rss_delta_bytes": 0,
            "vram_delta_bytes": 0,
            "gpu_temperature_c": 60.0,
            "power_w": 200.0,
        },
        "quality": {"correct": True, "score": 1.0},
    }


def cases() -> list[dict]:
    return [
        measured_case("disabled", "disabled"),
        measured_case("budget-32", "budget-32", 8),
        measured_case("budget-128", "budget-128", 20),
        measured_case("budget-256", "budget-256", 24),
        measured_case("unbounded", "unbounded", 30),
    ]


def lifecycle_events(measured: list[dict]) -> list[dict]:
    events = []
    for item in measured:
        raw = item["raw"]
        enabled = item["mode"] != "disabled"
        events.append(
            {
                "case_id": item["id"],
                "stream": item["stream"],
                "outcome": item["finish_reason"],
                "prompt_tokens": raw["prompt_tokens"],
                "completion_tokens": raw["completion_tokens"],
                "reset_complete": True,
                "reasoning_tokens": raw["reasoning_tokens"] if enabled else None,
                "forced_end_tokens": raw["forced_end_tokens"] if enabled else None,
                "admit_to_start_ns": 1,
                "start_to_release_ns": 2,
                "admit_to_release_ns": 3,
            }
        )
    return events


def write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    fixture = ROOT / "services/openai-gateway/tests/fixtures/served-model/aq4"
    candidate = root / "served-model"
    shutil.copytree(fixture, candidate)
    manifest = candidate / "served-model.json"
    worker = candidate / "worker"
    cases_path = root / "cases.json"
    measured = cases()
    cases_path.write_text(json.dumps(measured), encoding="ascii")
    lifecycle_path = root / "lifecycle.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "schema_version": "ullm.generic_reasoning_lifecycle_evidence.v1",
                "events": lifecycle_events(measured),
            }
        ),
        encoding="ascii",
    )
    return cases_path, manifest, worker, lifecycle_path


def write_v2_campaign(
    root: Path,
    *,
    manifest: Path,
    cases_path: Path,
    lifecycle_path: Path,
    run_id: str = "reasoning-release-run",
) -> tuple[Path, Path, Path]:
    campaign = root / "reasoning-campaign"
    campaign.mkdir()
    candidate_raw = manifest.read_bytes()
    candidate_sha256 = hashlib.sha256(candidate_raw).hexdigest()
    authorization = root / "authorization.json"
    authorization.write_bytes(b'{"authorization":true}\n')
    authorization.chmod(0o444)
    claim_path = root / "authorization.claimed.json"
    claim_path.write_bytes(b'{"claim":true}\n')
    claim_path.chmod(0o444)
    claim = {
        "path": str(claim_path),
        "sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest(),
        "bytes": len(claim_path.read_bytes()),
        "authorization_path": str(authorization),
        "authorization_sha256": hashlib.sha256(
            authorization.read_bytes()
        ).hexdigest(),
    }
    identity = {
        "device": 1,
        "inode": 2,
        "mode": 0o444,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "bytes": len(candidate_raw),
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    rows = [
        {
            "schema_version": "ullm.served_model.active_manifest_observation.v1",
            "sequence": sequence,
            "stage": stage,
            "observed_unix_ns": sequence,
            "observed_monotonic_ns": sequence,
            "candidate": {
                "path": str(manifest),
                "sha256": candidate_sha256,
                "identity": identity,
            },
            "active": {
                "path": FIXED_ACTIVE_MANIFEST_PATH,
                "sha256": candidate_sha256,
                "identity": identity,
            },
            "bytes_equal": True,
            "claim": claim,
        }
        for sequence, stage in enumerate(TOOL.REASONING_CAMPAIGN_STAGES)
    ]
    observations_raw = b"".join(
        (
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("ascii")
        for row in rows
    )
    binding = {
        "schema_version": "ullm.served_model.active_binding.v1",
        "status": "complete",
        "candidate": {
            "artifact": "candidate-served-model.json",
            "source_path": str(manifest),
            "sha256": candidate_sha256,
            "bytes": len(candidate_raw),
        },
        "actual_active_path": FIXED_ACTIVE_MANIFEST_PATH,
        "expected_stages": list(TOOL.REASONING_CAMPAIGN_STAGES),
        "observation_count": len(rows),
        "observations": {
            "artifact": "active-manifest-observations.jsonl",
            "sha256": hashlib.sha256(observations_raw).hexdigest(),
            "bytes": len(observations_raw),
        },
        "claim": claim,
        "campaign": {
            "name": "reasoning_release",
            "run_id": run_id,
            "final_path": str(campaign),
        },
    }
    summary = {
        "schema_version": TOOL.REASONING_CAMPAIGN_SCHEMA_V2,
        "status": "incomplete",
        "raw_bodies_stored": False,
        "case_count": len(json.loads(cases_path.read_text(encoding="ascii"))),
        "stream_case_count": len(json.loads(cases_path.read_text(encoding="ascii"))),
        "nonstream_case_count": 0,
        "modes": ["disabled", "budget-32", "budget-128", "budget-256", "unbounded"],
        "manifest_sha256": candidate_sha256,
        "model_id": "fixture",
        "worker_binary_sha256": "1" * 64,
        "gpu_exclusive_preflight": {},
        "active_manifest_binding": binding,
        "run_id": run_id,
    }
    values = {
        "cases.json": cases_path.read_bytes(),
        "lifecycle.json": lifecycle_path.read_bytes(),
        "resource-samples.jsonl": b"{}\n",
        "summary.json": (
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n"
        ).encode("ascii"),
        "candidate-served-model.json": candidate_raw,
        "active-manifest-observations.jsonl": observations_raw,
        "active-manifest-binding.json": (
            json.dumps(binding, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("ascii"),
    }
    for name, raw in values.items():
        path = campaign / name
        path.write_bytes(raw)
        path.chmod(0o444)
    campaign.chmod(0o555)
    return campaign, campaign / "cases.json", campaign / "lifecycle.json"


def test_prepare_writes_valid_complete_hash_only_evidence(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    commit = "1" * 40
    monkeypatch.setattr(TOOL, "_git_commit", lambda: commit)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")
    output = tmp_path / "release.json"

    document = TOOL.prepare(
        cases_path,
        manifest,
        worker,
        "ullm/open-webui@sha256:" + "b" * 64,
        commit,
        output,
        lifecycle_path=lifecycle_path,
        status="complete",
    )

    assert json.loads(output.read_text(encoding="ascii")) == document
    assert document["git_worktree_clean"] is True
    assert document["git_worktree_status_sha256"] == hashlib.sha256(b"").hexdigest()
    assert TOOL._load_validator().validate(output)["gate_eligible"] is True


def test_prepare_v2_binds_exact_claim_run_output_and_every_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    campaign, cases_path, lifecycle_path = write_v2_campaign(
        tmp_path,
        manifest=manifest,
        cases_path=cases_path,
        lifecycle_path=lifecycle_path,
    )
    commit = "1" * 40
    monkeypatch.setattr(TOOL, "_git_commit", lambda: commit)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")
    output = tmp_path / "release-v2.json"

    document = TOOL.prepare(
        cases_path,
        manifest,
        worker,
        "ullm/open-webui@sha256:" + "b" * 64,
        commit,
        output,
        lifecycle_path=lifecycle_path,
        campaign_output_dir=campaign,
        status="complete",
    )

    assert document["schema_version"] == TOOL.EVIDENCE_SCHEMA_V2
    lineage = document["campaign_lineage"]
    assert lineage["claim"]["sha256"] == hashlib.sha256(
        (tmp_path / "authorization.claimed.json").read_bytes()
    ).hexdigest()
    assert lineage["campaign"]["run_id"] == "reasoning-release-run"
    assert len(lineage["observations"]["stages"]) == 12
    report = TOOL._load_validator().validate(output)
    assert report["gate_eligible"] is True
    assert report["campaign_lineage"]["observation_count"] == 12
    assert output.stat().st_mode & 0o777 == 0o444
    assert output.stat().st_nlink == 1


def test_prepare_keeps_dirty_incomplete_evidence_but_rejects_complete(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    commit = "1" * 40
    monkeypatch.setattr(TOOL, "_git_commit", lambda: commit)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b" M source.py\n")

    incomplete = tmp_path / "incomplete.json"
    document = TOOL.prepare(
        cases_path,
        manifest,
        worker,
        "ullm/open-webui@sha256:" + "b" * 64,
        commit,
        incomplete,
        lifecycle_path=lifecycle_path,
    )
    assert document["git_worktree_clean"] is False
    assert TOOL._load_validator().validate(incomplete)["gate_eligible"] is False

    with pytest.raises(TOOL.EvidenceError, match="clean Git worktree"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            commit,
            tmp_path / "complete.json",
            lifecycle_path=lifecycle_path,
            status="complete",
        )


def test_prepare_complete_rejects_unaligned_active_promotion(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")

    with pytest.raises(TOOL.EvidenceError, match="not production-gate eligible"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            "2" * 40,
            tmp_path / "unaligned.json",
            lifecycle_path=lifecycle_path,
            status="complete",
        )


def test_prepare_rejects_cleartext_case_fields(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    value = json.loads(cases_path.read_text(encoding="ascii"))
    value[0]["response"] = "secret"
    cases_path.write_text(json.dumps(value), encoding="ascii")
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")

    with pytest.raises(TOOL.EvidenceError, match="forbidden field"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            "1" * 40,
            tmp_path / "release.json",
            lifecycle_path=lifecycle_path,
        )


def test_prepare_rejects_invalid_served_model_manifest(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    document = json.loads(manifest.read_text(encoding="ascii"))
    document["worker"]["protocol"] = "ullm.worker.v2"
    manifest.write_text(json.dumps(document), encoding="ascii")
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")

    with pytest.raises(TOOL.EvidenceError, match="served-model manifest failed validation"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            "1" * 40,
            tmp_path / "release.json",
            lifecycle_path=lifecycle_path,
        )


def test_evidence_publication_never_replaces_a_raced_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release.json"
    output.write_bytes(b"racer-owned\n")

    with pytest.raises(TOOL.EvidenceError, match="already exists"):
        TOOL._atomic_write(output, {"schema_version": "fixture"})

    assert output.read_bytes() == b"racer-owned\n"


def test_v2_evidence_rejects_replayed_observation_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    campaign, cases_path, lifecycle_path = write_v2_campaign(
        tmp_path,
        manifest=manifest,
        cases_path=cases_path,
        lifecycle_path=lifecycle_path,
    )
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")
    output = tmp_path / "release-v2.json"
    TOOL.prepare(
        cases_path,
        manifest,
        worker,
        "ullm/open-webui@sha256:" + "b" * 64,
        "1" * 40,
        output,
        lifecycle_path=lifecycle_path,
        campaign_output_dir=campaign,
    )
    observations = campaign / "active-manifest-observations.jsonl"
    observations.chmod(0o644)
    lines = observations.read_bytes().splitlines(keepends=True)
    observations.write_bytes(b"".join([lines[0], *lines[:-1]]))
    observations.chmod(0o444)

    validator = TOOL._load_validator()
    with pytest.raises(validator.ValidationError):
        validator.validate(output)
