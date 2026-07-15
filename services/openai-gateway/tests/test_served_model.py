from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from ullm_openai_gateway.served_model import (
    MAX_MANIFEST_BYTES,
    MAX_STRING_BYTES,
    ServedModelError,
    load_served_model,
)


FIXTURES = Path(__file__).parent / "fixtures/served-model"


@pytest.mark.parametrize(
    ("name", "model_id", "format_id", "vocab_size", "has_artifact"),
    [
        ("sq8", "ullm-qwen3-14b-sq8", "SQ8_0", 151_936, True),
        ("aq4", "ullm-qwen3.5-9b-aq4", "AQ4_0", 248_320, False),
        (
            "sq8/served-model-fq6.json",
            "ullm-qwen3-14b-fq6-fixture",
            "FQ6_0",
            151_936,
            True,
        ),
    ],
)
def test_quantization_format_fixtures_use_the_same_loader(
    name: str,
    model_id: str,
    format_id: str,
    vocab_size: int,
    has_artifact: bool,
) -> None:
    path = (
        FIXTURES / name
        if name.endswith(".json")
        else FIXTURES / name / "served-model.json"
    )
    loaded = load_served_model(path)

    assert loaded.manifest_path == path.resolve()
    assert len(loaded.manifest_sha256) == 64
    assert loaded.public.id == model_id
    assert loaded.format.format_id == format_id
    assert loaded.generation.vocab_size == vocab_size
    assert (loaded.product.artifact is not None) is has_artifact
    assert loaded.worker.arguments == ("--served-model-manifest", "{manifest}")
    assert loaded.worker.binary.is_absolute()
    assert loaded.tokenizer.root.is_absolute()


def test_promotion_authorization_audit_is_optional_and_typed(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    loaded = load_served_model(path)
    assert loaded.promotion.authorization_audit is None

    value = _document(path)
    value["promotion"]["authorization_audit"] = None
    _write(path, value)
    loaded = load_served_model(path)
    assert loaded.promotion.authorization_audit is None

    audit = path.parent / "authorization-audit.json"
    audit.write_text("{\"verdict\":\"implementation_ready\"}\n", encoding="ascii")
    audit.chmod(0o444)
    value["promotion"]["authorization_audit"] = {
        "path": str(audit.resolve()),
        "sha256": _sha256(audit),
    }
    _write(path, value)
    loaded = load_served_model(path)
    assert loaded.promotion.authorization_audit is not None
    assert loaded.promotion.authorization_audit.path == audit.resolve()
    assert loaded.promotion.authorization_audit.sha256 == _sha256(audit)


def test_promotion_authorization_lineage_is_optional_typed_and_rehashed(
    tmp_path: Path,
) -> None:
    path = _copy_fixture(tmp_path)
    assert load_served_model(path).promotion.authorization_lineage is None
    value = _document(path)
    lineage = {
        "schema_version": "ullm.sq8_authorization_lineage_input.v1",
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": {
            "commit": "a" * 40, "tree_oid": "b" * 40,
            "archive_sha256": "c" * 64,
        },
        "entries": [{"index": index} for index in range(6)],
    }
    input_path = (tmp_path / "lineage-input.json").resolve()
    runtime_path = (tmp_path / "lineage-runtime.json").resolve()
    for lineage_path in (input_path, runtime_path):
        lineage_path.write_text(json.dumps(lineage) + "\n", encoding="ascii")
        lineage_path.chmod(0o444)
    entries_sha256 = hashlib.sha256(
        json.dumps(
            lineage["entries"], ensure_ascii=True, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    value["promotion"]["authorization_lineage"] = {
        "schema_version": "ullm.sq8_authorization_lineage_ref.v1",
        "input_path": str(input_path),
        "runtime_path": str(runtime_path),
        "sha256": _sha256(input_path),
        "entries_sha256": entries_sha256,
    }
    _write(path, value)
    loaded = load_served_model(path)
    assert loaded.promotion.authorization_lineage is not None
    assert loaded.promotion.authorization_lineage.entries_sha256 == entries_sha256

    runtime_path.chmod(0o644)
    runtime_path.write_text("{}\n", encoding="ascii")
    runtime_path.chmod(0o444)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def _publish_lineage(path: Path, value: Any) -> str:
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")
    path.chmod(0o444)
    return _sha256(path)


def _entries_sha(entries: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            entries, ensure_ascii=True, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _first_v2_lineage(tmp_path: Path) -> dict[str, Any]:
    request = "sq8-promotion-" + "9" * 64
    v1_source = {
        "commit": "a" * 40, "tree_oid": "2" * 40,
        "archive_sha256": "3" * 64,
    }
    source_receipts = [
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "verdict": "implementation_ready", "actual": "not_executed",
            "audited_source": {"commit": "0" * 40},
            "authorization": {"eligible_for_fresh_authorization_builder": True},
        },
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "verdict": "implementation_no_go", "actual": "not_executed",
            "audited_source": {"commit": "1" * 40}, "reason_codes": ["first"],
        },
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
            "verdict": "implementation_no_go", "actual": "not_executed",
            "audited_source": {"commit": "2" * 40}, "reason_codes": ["second"],
        },
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "actual_failed", "request_id": request,
            "source_commit": "3" * 40,
            "actual": {"status": "failed", "request_id": request},
        },
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "status": "actual_failed", "request_id": request,
            "source_commit": "4" * 40,
            "actual": {"status": "failed", "request_id": request},
        },
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
            "verdict": "implementation_no_go", "actual": "not_executed",
            "audited_source": {"commit": "5" * 40},
            "fixed_request_id": request,
            "reason_code": "restore_retry_terminal_identity_not_fail_closed",
        },
    ]
    relations = (
        "implementation_go_eligible_for_fresh_runtime_audit",
        "superseded_capture_implementation_no_go",
        "superseded_capture_implementation_no_go",
        "consumed_actual_failure_latest",
        "consumed_actual_failure_predecessor",
        "superseded_restore_implementation_no_go",
    )
    migrated_relations = (
        "historical_implementation_audit", "capture_implementation_no_go",
        "capture_implementation_no_go", "actual_failure", "actual_failure",
        "restore_implementation_no_go",
    )
    v1_entries = []
    migrated = []
    for sequence, (receipt, relation, migrated_relation) in enumerate(
        zip(source_receipts, relations, migrated_relations, strict=True)
    ):
        receipt_path = (tmp_path / f"v1-entry-{sequence}.json").resolve()
        digest = _publish_lineage(receipt_path, receipt)
        entry = {
            "relation": relation, "path": str(receipt_path), "sha256": digest,
            "schema_version": receipt["schema_version"],
            "consumed": sequence != 0, "reusable_as_runtime_authorization": False,
        }
        if sequence == 0:
            entry.update(verdict="implementation_ready", actual="not_executed")
        elif sequence in {1, 2}:
            entry.update(
                verdict="implementation_no_go", actual="not_executed",
                reason_codes=receipt["reason_codes"],
            )
        elif sequence in {3, 4}:
            entry.update(status="actual_failed", actual_status="failed", request_id=request)
        else:
            entry.update(
                verdict="implementation_no_go", actual="not_executed",
                reason_code="restore_retry_terminal_identity_not_fail_closed",
            )
        v1_entries.append(entry)
        migrated.append(
            {
                "sequence": sequence, "relation": migrated_relation,
                "path": str(receipt_path), "sha256": digest,
                "schema_version": receipt["schema_version"],
                "status": receipt.get("status", receipt.get("verdict")),
                "request_id": request if sequence in {3, 4, 5} else None,
                "source_commit": receipt.get(
                    "source_commit", receipt.get("audited_source", {}).get("commit")
                ),
            }
        )
    v1 = {
        "schema_version": "ullm.sq8_authorization_lineage_input.v1",
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": v1_source, "entries": v1_entries,
    }
    v1_path = (tmp_path / "lineage-v1.json").resolve()
    v1_sha = _publish_lineage(v1_path, v1)
    latest_request = "sq8-promotion-" + "6" * 64
    latest = {
        "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
        "status": "actual_failed", "request_id": latest_request,
        "source_commit": v1_source["commit"],
        "source_provenance": {
            "tree_sha256": v1_source["tree_oid"],
            "archive_sha256": v1_source["archive_sha256"],
        },
        "actual": {"status": "failed", "request_id": latest_request},
    }
    latest_path = (tmp_path / "latest-failure.json").resolve()
    latest_sha = _publish_lineage(latest_path, latest)
    migrated.append(
        {
            "sequence": 6, "relation": "actual_failure", "path": str(latest_path),
            "sha256": latest_sha, "schema_version": latest["schema_version"],
            "status": "actual_failed", "request_id": latest_request,
            "source_commit": v1_source["commit"],
        }
    )
    current_commit = "b" * 40
    current = {
        "schema_version": "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
        "verdict": "implementation_ready", "actual": "not_executed",
        "audited_source": {"commit": current_commit},
        "authorization": {"eligible_for_fresh_authorization_builder": True},
    }
    current_path = (tmp_path / "current-go.json").resolve()
    current_sha = _publish_lineage(current_path, current)
    migrated.append(
        {
            "sequence": 7, "relation": "implementation_ready_current",
            "path": str(current_path), "sha256": current_sha,
            "schema_version": current["schema_version"],
            "status": "implementation_ready", "request_id": None,
            "source_commit": current_commit,
        }
    )
    lineage = {
        "schema_version": "ullm.sq8_authorization_lineage_input.v2",
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": {
            "commit": current_commit, "tree_oid": "c" * 40,
            "archive_sha256": "d" * 64,
        },
        "predecessor": {
            "schema_version": "ullm.sq8_authorization_lineage_input.v1",
            "path": str(v1_path), "sha256": v1_sha,
            "migrated_prefix_sha256": _entries_sha(migrated[:6]),
            "migrated_prefix_count": 6,
        },
        "entries": migrated,
    }
    input_path = (tmp_path / "lineage-v2-input.json").resolve()
    runtime_path = (tmp_path / "lineage-v2-runtime.json").resolve()
    lineage_sha = _publish_lineage(input_path, lineage)
    assert _publish_lineage(runtime_path, lineage) == lineage_sha
    reference = {
        "schema_version": "ullm.sq8_authorization_lineage_ref.v2",
        "input_path": str(input_path), "runtime_path": str(runtime_path),
        "sha256": lineage_sha, "entries_sha256": _entries_sha(migrated),
        "entry_count": 8,
        "current_implementation_audit": {
            "path": str(current_path), "sha256": current_sha,
        },
    }
    return {
        "lineage": lineage, "reference": reference, "v1": v1,
        "v1_path": v1_path, "input_path": input_path, "runtime_path": runtime_path,
    }


def _load_with_lineage(tmp_path: Path, fixture: dict[str, Any]) -> Any:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    value["promotion"]["authorization_lineage"] = fixture["reference"]
    _write(path, value)
    return load_served_model(path)


def _refresh_lineage_fixture(fixture: dict[str, Any]) -> None:
    lineage = fixture["lineage"]
    digest = _publish_lineage(fixture["input_path"], lineage)
    assert _publish_lineage(fixture["runtime_path"], lineage) == digest
    fixture["reference"]["sha256"] = digest
    fixture["reference"]["entries_sha256"] = _entries_sha(lineage["entries"])
    fixture["reference"]["entry_count"] = len(lineage["entries"])


def test_promotion_authorization_lineage_first_v2_migration_is_typed(
    tmp_path: Path,
) -> None:
    fixture = _first_v2_lineage(tmp_path)
    identity = _load_with_lineage(tmp_path, fixture).promotion.authorization_lineage
    assert identity is not None
    assert identity.schema_version == "ullm.sq8_authorization_lineage_ref.v2"
    assert identity.entry_count == 8


@pytest.mark.parametrize(
    "mutation", ["unknown", "missing", "type", "prefix_digest", "prefix_count"]
)
def test_first_v2_migration_predecessor_shape_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _first_v2_lineage(tmp_path)
    predecessor = fixture["lineage"]["predecessor"]
    if mutation == "unknown":
        predecessor["unknown"] = True
    elif mutation == "missing":
        predecessor.pop("schema_version")
    elif mutation == "type":
        predecessor["migrated_prefix_count"] = "6"
    elif mutation == "prefix_digest":
        predecessor["migrated_prefix_sha256"] = "0" * 64
    else:
        predecessor["migrated_prefix_count"] = 5
    _refresh_lineage_fixture(fixture)
    with pytest.raises(ServedModelError):
        _load_with_lineage(tmp_path, fixture)


@pytest.mark.parametrize("field", ["commit", "tree_oid", "archive_sha256"])
def test_first_v2_migration_rejects_v1_source_spoof(
    tmp_path: Path, field: str
) -> None:
    fixture = _first_v2_lineage(tmp_path)
    fixture["v1"]["source"][field] = "e" * (64 if field == "archive_sha256" else 40)
    predecessor_sha = _publish_lineage(fixture["v1_path"], fixture["v1"])
    fixture["lineage"]["predecessor"]["sha256"] = predecessor_sha
    _refresh_lineage_fixture(fixture)
    with pytest.raises(ServedModelError):
        _load_with_lineage(tmp_path, fixture)


def test_first_v2_migration_cannot_be_reused_for_ninth_entry(
    tmp_path: Path,
) -> None:
    fixture = _first_v2_lineage(tmp_path)
    request_id = "sq8-promotion-" + "7" * 64
    receipt = {
        "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
        "status": "actual_failed", "request_id": request_id,
        "source_commit": "7" * 40,
        "actual": {"status": "failed", "request_id": request_id},
    }
    receipt_path = (tmp_path / "ninth-failure.json").resolve()
    receipt_sha = _publish_lineage(receipt_path, receipt)
    fixture["lineage"]["entries"].append(
        {
            "sequence": 8, "relation": "actual_failure",
            "path": str(receipt_path), "sha256": receipt_sha,
            "schema_version": receipt["schema_version"],
            "status": "actual_failed", "request_id": request_id,
            "source_commit": "7" * 40,
        }
    )
    _refresh_lineage_fixture(fixture)
    with pytest.raises(ServedModelError):
        _load_with_lineage(tmp_path, fixture)


def _subsequent_v2_lineage(tmp_path: Path) -> dict[str, Any]:
    fixture = _first_v2_lineage(tmp_path)
    previous = fixture["lineage"]
    request_id = "sq8-promotion-" + "8" * 64
    receipt = {
        "schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
        "status": "actual_failed", "request_id": request_id,
        "source_commit": "8" * 40,
        "actual": {"status": "failed", "request_id": request_id},
    }
    receipt_path = (tmp_path / "appended-failure.json").resolve()
    receipt_sha = _publish_lineage(receipt_path, receipt)
    appended = json.loads(json.dumps(previous))
    appended["predecessor"] = {
        "schema_version": "ullm.sq8_authorization_lineage_input.v2",
        "path": str(fixture["input_path"]),
        "sha256": fixture["reference"]["sha256"],
        "entries_sha256": fixture["reference"]["entries_sha256"],
        "entry_count": 8,
    }
    appended["entries"].append(
        {
            "sequence": 8, "relation": "actual_failure",
            "path": str(receipt_path), "sha256": receipt_sha,
            "schema_version": receipt["schema_version"],
            "status": "actual_failed", "request_id": request_id,
            "source_commit": "8" * 40,
        }
    )
    fixture["lineage"] = appended
    fixture["input_path"] = (tmp_path / "lineage-v2-appended-input.json").resolve()
    fixture["runtime_path"] = (
        tmp_path / "lineage-v2-appended-runtime.json"
    ).resolve()
    fixture["reference"]["input_path"] = str(fixture["input_path"])
    fixture["reference"]["runtime_path"] = str(fixture["runtime_path"])
    _refresh_lineage_fixture(fixture)
    return fixture


def test_subsequent_v2_predecessor_append_is_accepted(tmp_path: Path) -> None:
    fixture = _subsequent_v2_lineage(tmp_path)
    identity = _load_with_lineage(tmp_path, fixture).promotion.authorization_lineage
    assert identity is not None
    assert identity.entry_count == 9


@pytest.mark.parametrize("mutation", ["unknown", "missing", "type"])
def test_subsequent_v2_predecessor_shape_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _subsequent_v2_lineage(tmp_path)
    predecessor = fixture["lineage"]["predecessor"]
    if mutation == "unknown":
        predecessor["migrated_prefix_count"] = 8
    elif mutation == "missing":
        predecessor.pop("entries_sha256")
    else:
        predecessor["entry_count"] = "8"
    _refresh_lineage_fixture(fixture)
    with pytest.raises(ServedModelError):
        _load_with_lineage(tmp_path, fixture)


def test_first_v2_external_runtime_copy_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _first_v2_lineage(tmp_path)
    fixture["runtime_path"].chmod(0o644)
    fixture["runtime_path"].write_text("{}\n", encoding="ascii")
    fixture["runtime_path"].chmod(0o444)
    with pytest.raises(ServedModelError):
        _load_with_lineage(tmp_path, fixture)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["promotion"].__setitem__(
            "authorization_audit", {"path": "audit.json", "sha256": "0" * 64}
        ),
        lambda value: value["promotion"].__setitem__(
            "authorization_audit", {"path": "/tmp/audit.json", "sha256": "A" * 64}
        ),
        lambda value: value["promotion"].__setitem__(
            "authorization_audit", {"path": "/tmp/audit.json", "sha256": "0" * 64}
        ),
        lambda value: value["promotion"].__setitem__(
            "authorization_audit", {"path": "/tmp/audit.json", "sha256": "0" * 64, "extra": 1}
        ),
    ],
)
def test_promotion_authorization_audit_rejects_weak_or_mismatched_refs(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    mutate(value)
    _write(path, value)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def _readiness() -> dict[str, Any]:
    body = '{"status":"ready"}'
    return {
        "schema": "ullm.bridge_container_readiness.v1",
        "container": {
            "name": "open-webui", "id": "4" * 64,
            "image_id": "sha256:" + "5" * 64,
            "config_image": "ullm/open-webui:test",
        },
        "network": {
            "name": "open-webui-network", "id": "6" * 64,
            "driver": "bridge", "bridge_interface": "br-" + "6" * 12,
        },
        "endpoint": {
            "url": "http://172.20.0.1:8000/readyz", "path": "/readyz",
            "expected_status": 200, "expected_body": body,
            "expected_body_sha256": hashlib.sha256(body.encode("ascii")).hexdigest(),
            "timeout_seconds": 5,
        },
    }


def test_promotion_readiness_is_optional_and_typed(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    assert load_served_model(path).promotion.readiness is None
    value = _document(path)
    value["promotion"]["readiness"] = _readiness()
    _write(path, value)

    readiness = load_served_model(path).promotion.readiness

    assert readiness is not None
    assert readiness.container_id == "4" * 64
    assert readiness.network_id == "6" * 64
    assert readiness.bridge_interface == "br-" + "6" * 12
    assert readiness.expected_status == 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["container"].__setitem__("id", "A" * 64),
        lambda value: value["container"].__setitem__("image_id", "5" * 64),
        lambda value: value["network"].__setitem__("driver", "host"),
        lambda value: value["network"].__setitem__("bridge_interface", "docker0"),
        lambda value: value["endpoint"].__setitem__("url", "http://127.0.0.1:8000/readyz"),
        lambda value: value["endpoint"].__setitem__("expected_status", 204),
        lambda value: value["endpoint"].__setitem__("expected_body", '{"status":"ready"}\n'),
        lambda value: value["endpoint"].__setitem__("timeout_seconds", 6),
        lambda value: value["endpoint"].__setitem__("extra", True),
    ],
)
def test_promotion_readiness_rejects_identity_weakening(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    readiness = _readiness()
    mutate(readiness)
    value["promotion"]["readiness"] = readiness
    _write(path, value)

    with pytest.raises(ServedModelError):
        load_served_model(path)


def _copy_fixture(tmp_path: Path, name: str = "sq8") -> Path:
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    return target / "served-model.json"


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_virtual_format_changes_only_public_and_format_contracts() -> None:
    existing = _document(FIXTURES / "sq8/served-model.json")
    virtual = _document(FIXTURES / "sq8/served-model-fq6.json")

    assert existing["public"] != virtual["public"]
    assert existing["format"] != virtual["format"]
    for section in (
        "schema_version",
        "generation",
        "tokenizer",
        "worker",
        "product",
        "promotion",
    ):
        assert virtual[section] == existing[section]


def test_v2_manifest_loads_reasoning_dialect_without_changing_v1_loader() -> None:
    path = FIXTURES / "aq4/served-model.json"
    value = _document(path)
    value["schema_version"] = "ullm.served_model.v2"
    value["worker"]["protocol"] = "ullm.worker.v2"
    value["reasoning"] = {
        "enabled_by_default": False,
        "dialect_id": "synthetic.multi-token.v1",
        "start_token_ids": [248068, 12],
        "end_token_ids": [248069, 13],
        "forced_end_token_ids": [248069, 13],
        "initial_phase": "reasoning",
        "eos_policy": "close",
        "effort_budgets": {"low": 32, "medium": 64, "high": 128},
        "max_budget_tokens": 128,
        "reserved_answer_tokens": 1,
        "history_reasoning_policy": "omit",
    }
    # Exercise the parser directly with a temporary copy so resource roots and
    # hashes remain exactly the same as the fixture.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "aq4"
        shutil.copytree(path.parent, root)
        copied = root / "served-model.json"
        _write(copied, value)
        loaded = load_served_model(copied)

    assert loaded.reasoning_dialect is not None
    assert loaded.reasoning_dialect.identity == "synthetic.multi-token.v1"
    assert loaded.reasoning_dialect.effort_budgets[-1] == ("high", 128)


@pytest.mark.parametrize("schema,worker_protocol", [("ullm.served_model.v1", "ullm.worker.v2"), ("ullm.served_model.v2", "ullm.worker.v1")])
def test_manifest_schema_and_worker_protocol_must_be_version_aligned(
    tmp_path: Path, schema: str, worker_protocol: str
) -> None:
    root = tmp_path / "aq4"
    shutil.copytree(FIXTURES / "aq4", root)
    path = root / "served-model.json"
    value = _document(path)
    value["schema_version"] = schema
    value["worker"]["protocol"] = worker_protocol
    if schema == "ullm.served_model.v2":
        value["reasoning"] = {
            "enabled_by_default": False,
            "dialect_id": "synthetic.multi-token.v1",
            "start_token_ids": [248068, 12],
            "end_token_ids": [248069, 13],
            "forced_end_token_ids": [248069, 13],
            "initial_phase": "reasoning",
            "eos_policy": "close",
            "effort_budgets": {"low": 32, "medium": 64, "high": 128},
            "max_budget_tokens": 128,
            "reserved_answer_tokens": 1,
            "history_reasoning_policy": "omit",
        }
    _write(path, value)
    with pytest.raises(ServedModelError, match="version aligned"):
        load_served_model(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("unknown", 1),
        lambda value: value.pop("format"),
        lambda value: value["public"].__setitem__("context_length", True),
        lambda value: value["generation"].__setitem__("eos_token_ids", "151645"),
        lambda value: value["worker"]["identity"].__setitem__("extra", "x"),
        lambda value: value.__setitem__("schema_version", "ullm.served_model.v2"),
    ],
)
def test_unknown_missing_wrong_type_and_schema_are_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    mutate(value)
    _write(path, value)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace(
            '{\n  "schema_version"',
            '{\n  "schema_version":"duplicate",\n  "schema_version"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ServedModelError, match="strict JSON"):
        load_served_model(path)


@pytest.mark.parametrize("payload", [b"\xff", b'{"schema_version":NaN}', b"[]"])
def test_non_utf8_nonfinite_and_nonobject_json_are_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def test_manifest_size_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ServedModelError, match="size limit"):
        load_served_model(path)


@pytest.mark.parametrize(
    "payload",
    [
        ("[" * 17 + "0" + "]" * 17).encode("ascii"),
        json.dumps({"value": "x" * (MAX_STRING_BYTES + 1)}).encode("ascii"),
        json.dumps({"value": [0] * 16_385}).encode("ascii"),
    ],
)
def test_json_depth_string_and_node_counts_are_bounded(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ServedModelError, match="bounds"):
        load_served_model(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["generation"].__setitem__("eos_token_ids", [151936]),
        lambda value: value["generation"].__setitem__(
            "eos_token_ids", [151645, 151645]
        ),
        lambda value: value["generation"].__setitem__("max_completion_tokens", 4097),
        lambda value: value["generation"]["sampling"].__setitem__("top_k", 151937),
        lambda value: value["generation"]["sampling"].__setitem__("temperature", False),
    ],
)
def test_generation_cross_contract_violations_are_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    mutate(value)
    _write(path, value)
    with pytest.raises(ServedModelError):
        load_served_model(path)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("worker", "binary_sha256", "A" * 64),
        ("worker", "binary_sha256", "0" * 64),
        ("promotion", "receipt_sha256", "0" * 64),
    ],
)
def test_malformed_or_mismatched_sha_is_rejected(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    path = _copy_fixture(tmp_path)
    document = _document(path)
    document[section][field] = value
    _write(path, document)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def test_tokenizer_and_product_payload_hashes_are_verified(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    (path.parent / "tokenizer/tokenizer.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ServedModelError, match="SHA-256"):
        load_served_model(path)

    path = _copy_fixture(tmp_path, "aq4")
    (path.parent / "product/package/manifest.json").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ServedModelError, match="SHA-256"):
        load_served_model(path)


@pytest.mark.parametrize(
    ("field_path", "unsafe"),
    [
        (("tokenizer", "root"), "../sq8-tokenizer"),
        (("worker", "binary"), "../worker"),
        (("product", "root"), "../product"),
        (("promotion", "receipt"), "../promotion.json"),
    ],
)
def test_relative_roots_cannot_escape_manifest_directory(
    tmp_path: Path, field_path: tuple[str, str], unsafe: str
) -> None:
    path = _copy_fixture(tmp_path)
    document = _document(path)
    document[field_path[0]][field_path[1]] = unsafe
    _write(path, document)
    with pytest.raises(ServedModelError, match="relative path"):
        load_served_model(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("tokenizer", "files"),
        ("product", "package"),
    ],
)
def test_child_paths_cannot_escape_declared_root(
    tmp_path: Path, section: str, field: str
) -> None:
    path = _copy_fixture(tmp_path)
    document = _document(path)
    if section == "tokenizer":
        document[section][field] = {"../promotion.json": "0" * 64}
    else:
        document[section][field]["manifest_path"] = "../promotion.json"
    _write(path, document)
    with pytest.raises(ServedModelError, match="relative path"):
        load_served_model(path)


def test_manifest_and_resource_symlinks_are_rejected(tmp_path: Path) -> None:
    real = _copy_fixture(tmp_path)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(real)
    with pytest.raises(ServedModelError, match="symlink"):
        load_served_model(link)

    tokenizer_file = real.parent / "tokenizer/tokenizer.json"
    replacement = real.parent / "tokenizer/replacement.json"
    replacement.write_bytes(tokenizer_file.read_bytes())
    tokenizer_file.unlink()
    tokenizer_file.symlink_to(replacement.name)
    with pytest.raises(ServedModelError, match="symlink"):
        load_served_model(real)


@pytest.mark.parametrize(
    "target", ["manifest", "worker", "tokenizer", "tokenizer_root", "package"]
)
def test_world_writable_manifest_and_resources_are_rejected(
    tmp_path: Path, target: str
) -> None:
    path = _copy_fixture(tmp_path)
    targets = {
        "manifest": path,
        "worker": path.parent / "worker",
        "tokenizer": path.parent / "tokenizer/tokenizer.json",
        "tokenizer_root": path.parent / "tokenizer",
        "package": path.parent / "product/package/manifest.json",
    }
    selected = targets[target]
    selected.chmod(selected.stat().st_mode | 0o002)
    with pytest.raises(ServedModelError, match="safe"):
        load_served_model(path)


def test_worker_launch_contract_is_strict(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    value["worker"]["arguments"] = ["--manifest", "missing-placeholder"]
    _write(path, value)
    with pytest.raises(ServedModelError, match="manifest"):
        load_served_model(path)

    path = _copy_fixture(tmp_path, "aq4")
    value = _document(path)
    value["worker"]["required_environment"] = ["invalid-name"]
    _write(path, value)
    with pytest.raises(ServedModelError, match="required_environment"):
        load_served_model(path)


def test_worker_binary_must_be_executable(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    binary = path.parent / "worker"
    binary.chmod(0o644)
    with pytest.raises(ServedModelError, match="executable"):
        load_served_model(path)


def test_fixture_permissions_are_not_world_writable() -> None:
    for path in FIXTURES.rglob("*"):
        assert not path.stat().st_mode & 0o002
        if path.name == "worker":
            assert os.access(path, os.X_OK)
