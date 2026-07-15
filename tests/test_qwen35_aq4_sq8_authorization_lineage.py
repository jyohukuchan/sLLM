from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/qwen35_aq4_sq8_authorization_lineage.py"
)
SPEC = importlib.util.spec_from_file_location("sq8_authorization_lineage", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def publish(path: Path, value: object) -> str:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode("ascii")
    path.write_bytes(raw)
    path.chmod(0o444)
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    request = "sq8-promotion-" + "a" * 64
    sources = [
        {
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "verdict": "implementation_ready",
            "actual": "not_executed",
            "audited_source": {"commit": "0" * 40},
            "authorization": {"eligible_for_fresh_authorization_builder": True},
        },
        {
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "verdict": "implementation_no_go",
            "actual": "not_executed",
            "audited_source": {"commit": "1" * 40},
            "reason_codes": ["first"],
        },
        {
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "verdict": "implementation_no_go",
            "actual": "not_executed",
            "audited_source": {"commit": "2" * 40},
            "reason_codes": ["second"],
        },
        {
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request,
            "source_commit": "3" * 40,
            "actual": {"status": "failed", "request_id": request},
        },
        {
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request,
            "source_commit": "4" * 40,
            "actual": {"status": "failed", "request_id": request},
        },
        {
            "schema_version": TOOL.RUNTIME_AUDIT_SCHEMA,
            "verdict": "implementation_no_go",
            "actual": "not_executed",
            "audited_source": {"commit": "5" * 40},
            "fixed_request_id": request,
            "reason_code": "restore_retry_terminal_identity_not_fail_closed",
        },
    ]
    entries = []
    for index, source in enumerate(sources):
        path = (tmp_path / f"source-{index}.json").resolve()
        digest = publish(path, source)
        common = {
            "relation": TOOL.V1_RELATIONS[index],
            "path": str(path),
            "sha256": digest,
            "schema_version": source["schema_version"],
            "consumed": index != 0,
            "reusable_as_runtime_authorization": False,
        }
        if index == 0:
            common.update(verdict=source["verdict"], actual=source["actual"])
        elif index in {1, 2}:
            common.update(
                verdict=source["verdict"],
                actual=source["actual"],
                reason_codes=source["reason_codes"],
            )
        elif index in {3, 4}:
            common.update(
                status=source["status"],
                actual_status="failed",
                request_id=request,
            )
        else:
            common.update(
                verdict=source["verdict"],
                actual=source["actual"],
                reason_code=source["reason_code"],
            )
        entries.append(common)
    document = {
        "schema_version": TOOL.MANIFEST_SCHEMA_V1,
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": {
            "commit": "b" * 40,
            "tree_oid": "c" * 40,
            "archive_sha256": "d" * 64,
        },
        "entries": entries,
    }
    manifest = (tmp_path / "lineage.json").resolve()
    publish(manifest, document)
    return manifest, document


def republish(path: Path, document: dict) -> None:
    path.chmod(0o644)
    publish(path, document)


def test_exact_manifest_and_runtime_reference_are_accepted(tmp_path: Path) -> None:
    manifest, document = fixture(tmp_path)
    validated = TOOL.validate_manifest(manifest, expected_source=document["source"])
    runtime = (tmp_path / "runtime.json").resolve()
    runtime.write_bytes(validated["raw"])
    runtime.chmod(0o444)
    reference = TOOL.make_reference(validated, runtime)
    assert (
        TOOL.validate_reference(reference, expected_runtime_path=runtime) == reference
    )


@pytest.mark.parametrize(
    "mutation", ["unknown", "missing", "reorder", "duplicate", "hash"]
)
def test_manifest_shape_order_duplicate_and_hash_drift_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    manifest, document = fixture(tmp_path)
    if mutation == "unknown":
        document["unknown"] = True
    elif mutation == "missing":
        document.pop("disposition")
    elif mutation == "reorder":
        document["entries"][0], document["entries"][1] = (
            document["entries"][1],
            document["entries"][0],
        )
    elif mutation == "duplicate":
        document["entries"][1]["path"] = document["entries"][0]["path"]
        document["entries"][1]["sha256"] = document["entries"][0]["sha256"]
    else:
        document["entries"][0]["sha256"] = "0" * 64
    republish(manifest, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    manifest, _document = fixture(tmp_path)
    manifest.chmod(0o644)
    manifest.write_text(
        '{"schema_version":"%s","schema_version":"%s"}\n'
        % (TOOL.MANIFEST_SCHEMA_V1, TOOL.MANIFEST_SCHEMA_V1),
        encoding="ascii",
    )
    manifest.chmod(0o444)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_path_alias_symlink_hardlink_and_mode_fail_closed(tmp_path: Path) -> None:
    manifest, document = fixture(tmp_path)
    aliased = Path(str(manifest.parent / "child" / ".." / manifest.name))
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(aliased)

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(manifest)
    with pytest.raises((TOOL.LineageError, OSError)):
        TOOL.validate_manifest(symlink)

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(manifest)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)
    hardlink.unlink()

    manifest.chmod(0o644)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)
    manifest.chmod(0o444)
    assert TOOL.validate_manifest(manifest)["document"] == document


def v2_fixture(tmp_path: Path) -> tuple[Path, dict]:
    v1_path, _v1_document = fixture(tmp_path)
    v1 = TOOL.validate_manifest(v1_path)
    entries = TOOL._migrated_v1_entries(v1)
    source_identity = {
        "commit": "b" * 40,
        "tree_oid": "c" * 40,
        "archive_sha256": "d" * 64,
    }
    latest_request = "sq8-promotion-" + "6" * 64
    latest_source = {
        "schema_version": TOOL.PROMOTION_SCHEMA,
        "status": "actual_failed",
        "request_id": latest_request,
        "source_commit": v1["document"]["source"]["commit"],
        "source_provenance": {
            "tree_sha256": v1["document"]["source"]["tree_oid"],
            "archive_sha256": v1["document"]["source"]["archive_sha256"],
        },
        "actual": {"status": "failed", "request_id": latest_request},
    }
    latest_path = (tmp_path / "v2-latest-failure.json").resolve()
    latest_sha = publish(latest_path, latest_source)
    entries.append(
        {
            "sequence": 6,
            "relation": "actual_failure",
            "path": str(latest_path),
            "sha256": latest_sha,
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": latest_request,
            "source_commit": v1["document"]["source"]["commit"],
        }
    )
    current_source = {
        "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
        "verdict": "implementation_ready",
        "actual": "not_executed",
        "audited_source": {"commit": source_identity["commit"]},
        "authorization": {"eligible_for_fresh_authorization_builder": True},
    }
    current_path = (tmp_path / "v2-current-go.json").resolve()
    current_sha = publish(current_path, current_source)
    entries.append(
        {
            "sequence": 7,
            "relation": "implementation_ready_current",
            "path": str(current_path),
            "sha256": current_sha,
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "status": "implementation_ready",
            "request_id": None,
            "source_commit": source_identity["commit"],
        }
    )
    document = {
        "schema_version": TOOL.MANIFEST_SCHEMA,
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": source_identity,
        "predecessor": {
            "schema_version": TOOL.MANIFEST_SCHEMA_V1,
            "path": str(v1_path),
            "sha256": v1["sha256"],
            "migrated_prefix_sha256": TOOL.canonical_sha(entries[:6]),
            "migrated_prefix_count": 6,
        },
        "entries": entries,
    }
    manifest = (tmp_path / "lineage-v2.json").resolve()
    publish(manifest, document)
    return manifest, document


def test_v2_manifest_and_typed_reference_are_accepted(tmp_path: Path) -> None:
    manifest, document = v2_fixture(tmp_path)
    current = document["entries"][7]
    validated = TOOL.validate_manifest(
        manifest,
        expected_source=document["source"],
        expected_current_implementation_audit={
            "path": current["path"],
            "sha256": current["sha256"],
        },
    )
    assert validated["authorization_eligible"] is True
    assert validated["entry_count"] == 8
    runtime = (tmp_path / "runtime-v2.json").resolve()
    runtime.write_bytes(validated["raw"])
    runtime.chmod(0o444)
    reference = TOOL.make_reference(validated, runtime)
    assert reference["entry_count"] == 8
    assert (
        TOOL.validate_reference(reference, expected_runtime_path=runtime) == reference
    )


def test_v1_is_diagnostic_only(tmp_path: Path) -> None:
    manifest, _document = fixture(tmp_path)
    validated = TOOL.validate_manifest(manifest)
    assert validated["authorization_eligible"] is False
    assert (
        TOOL.make_reference(validated, manifest)["schema_version"]
        == TOOL.REFERENCE_SCHEMA_V1
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "delete",
        "reorder",
        "relation",
        "hash",
        "source_commit",
        "source_tree",
        "source_archive",
    ],
)
def test_v1_migration_rejects_predecessor_spoof(tmp_path: Path, mutation: str) -> None:
    manifest, document = v2_fixture(tmp_path)
    predecessor_path = Path(document["predecessor"]["path"])
    predecessor = json.loads(predecessor_path.read_text(encoding="ascii"))
    if mutation == "delete":
        predecessor["entries"].pop()
    elif mutation == "reorder":
        predecessor["entries"][1], predecessor["entries"][2] = (
            predecessor["entries"][2],
            predecessor["entries"][1],
        )
    elif mutation == "relation":
        predecessor["entries"][1]["relation"] = (
            "implementation_go_eligible_for_fresh_runtime_audit"
        )
    elif mutation == "hash":
        predecessor["entries"][1]["sha256"] = "0" * 64
    elif mutation == "source_commit":
        predecessor["source"]["commit"] = "e" * 40
    elif mutation == "source_tree":
        predecessor["source"]["tree_oid"] = "e" * 40
    else:
        predecessor["source"]["archive_sha256"] = "e" * 64
    republish(predecessor_path, predecessor)
    document["predecessor"]["sha256"] = hashlib.sha256(
        predecessor_path.read_bytes()
    ).hexdigest()
    republish(manifest, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_v1_migration_can_only_be_used_for_exact_first_v2(tmp_path: Path) -> None:
    manifest, document = v2_fixture(tmp_path)
    request_id = "sq8-promotion-" + "9" * 64
    source = {
        "schema_version": TOOL.PROMOTION_SCHEMA,
        "status": "actual_failed",
        "request_id": request_id,
        "source_commit": "9" * 40,
        "actual": {"status": "failed", "request_id": request_id},
    }
    source_path = (tmp_path / "second-migration-failure.json").resolve()
    source_sha = publish(source_path, source)
    document["entries"].append(
        {
            "sequence": 8,
            "relation": "actual_failure",
            "path": str(source_path),
            "sha256": source_sha,
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request_id,
            "source_commit": "9" * 40,
        }
    )
    republish(manifest, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def successor_v2_fixture(
    tmp_path: Path,
    previous_path: Path,
    previous_document: dict,
    *,
    suffix: str,
    current_commit: str,
) -> tuple[Path, dict]:
    previous = TOOL.validate_manifest(previous_path)
    request_id = "sq8-promotion-" + suffix * 64
    source = {
        "schema_version": TOOL.PROMOTION_SCHEMA,
        "status": "actual_failed",
        "request_id": request_id,
        "source_commit": previous_document["source"]["commit"],
        "actual": {"status": "failed", "request_id": request_id},
    }
    source_path = (tmp_path / f"successor-{suffix}-failure.json").resolve()
    digest = publish(source_path, source)
    document = json.loads(json.dumps(previous_document))
    document["source"] = {
        "commit": current_commit,
        "tree_oid": suffix * 40,
        "archive_sha256": suffix * 64,
    }
    document["predecessor"] = {
        "schema_version": TOOL.MANIFEST_SCHEMA,
        "path": str(previous_path),
        "sha256": previous["sha256"],
        "entries_sha256": previous["entries_sha256"],
        "entry_count": previous["entry_count"],
    }
    document["entries"].append(
        {
            "sequence": len(document["entries"]),
            "relation": "actual_failure",
            "path": str(source_path),
            "sha256": digest,
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request_id,
            "source_commit": previous_document["source"]["commit"],
        }
    )
    current_source = {
        "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
        "verdict": "implementation_ready",
        "actual": "not_executed",
        "audited_source": {"commit": current_commit},
        "authorization": {"eligible_for_fresh_authorization_builder": True},
    }
    current_path = (tmp_path / f"successor-{suffix}-current-go.json").resolve()
    current_sha = publish(current_path, current_source)
    document["entries"].append(
        {
            "sequence": len(document["entries"]),
            "relation": "implementation_ready_current",
            "path": str(current_path),
            "sha256": current_sha,
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "status": "implementation_ready",
            "request_id": None,
            "source_commit": current_commit,
        }
    )
    appended_path = (tmp_path / f"lineage-v2-successor-{suffix}.json").resolve()
    publish(appended_path, document)
    return appended_path, document


def test_v2_successor_appends_failure_and_last_current_go(tmp_path: Path) -> None:
    previous_path, previous_document = v2_fixture(tmp_path)
    appended_path, document = successor_v2_fixture(
        tmp_path,
        previous_path,
        previous_document,
        suffix="7",
        current_commit="e" * 40,
    )
    appended = TOOL.validate_manifest(appended_path)
    assert appended["document"]["schema_version"] == TOOL.MANIFEST_SCHEMA
    assert appended["entry_count"] == 10
    assert appended["document"]["entries"][:8] == previous_document["entries"]
    assert appended["current_implementation_audit"] == {
        "path": document["entries"][9]["path"],
        "sha256": document["entries"][9]["sha256"],
    }

    next_path, next_document = successor_v2_fixture(
        tmp_path,
        appended_path,
        document,
        suffix="8",
        current_commit="f" * 40,
    )
    next_validated = TOOL.validate_manifest(next_path)
    assert next_validated["entry_count"] == 12
    assert next_document["entries"][:10] == document["entries"]


@pytest.mark.parametrize(
    "mutation",
    [
        "old_current_ref",
        "go_not_final",
        "go_only",
        "failure_only",
        "fake_go_source",
        "duplicate_source",
    ],
)
def test_v2_successor_current_selection_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    previous_path, previous_document = v2_fixture(tmp_path)
    successor_path, document = successor_v2_fixture(
        tmp_path,
        previous_path,
        previous_document,
        suffix="7",
        current_commit="e" * 40,
    )
    if mutation == "old_current_ref":
        old = previous_document["entries"][7]
        with pytest.raises(TOOL.LineageError):
            TOOL.validate_manifest(
                successor_path,
                expected_current_implementation_audit={
                    "path": old["path"],
                    "sha256": old["sha256"],
                },
            )
        return
    if mutation == "go_not_final":
        document["entries"][8], document["entries"][9] = (
            document["entries"][9],
            document["entries"][8],
        )
        document["entries"][8]["sequence"] = 8
        document["entries"][9]["sequence"] = 9
    elif mutation == "go_only":
        document["entries"].pop(8)
        document["entries"][8]["sequence"] = 8
    elif mutation == "failure_only":
        request_id = "sq8-promotion-" + "9" * 64
        failure_receipt = {
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request_id,
            "source_commit": document["source"]["commit"],
            "actual": {"status": "failed", "request_id": request_id},
        }
        failure_path = (tmp_path / "failure-after-current.json").resolve()
        failure_sha = publish(failure_path, failure_receipt)
        document["entries"].append(
            {
                "sequence": 10,
                "relation": "actual_failure",
                "path": str(failure_path),
                "sha256": failure_sha,
                "schema_version": TOOL.PROMOTION_SCHEMA,
                "status": "actual_failed",
                "request_id": request_id,
                "source_commit": document["source"]["commit"],
            }
        )
    elif mutation == "fake_go_source":
        document["entries"][9]["source_commit"] = "f" * 40
    else:
        document["source"]["commit"] = previous_document["source"]["commit"]
        current_path = Path(document["entries"][9]["path"])
        current_receipt = json.loads(current_path.read_text(encoding="ascii"))
        current_receipt["audited_source"]["commit"] = previous_document["source"][
            "commit"
        ]
        republish(current_path, current_receipt)
        document["entries"][9]["sha256"] = hashlib.sha256(
            current_path.read_bytes()
        ).hexdigest()
        document["entries"][9]["source_commit"] = previous_document["source"]["commit"]
    republish(successor_path, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(successor_path)


@pytest.mark.parametrize("mutation", ["delete", "replace", "reorder", "duplicate"])
def test_v2_append_only_history_mutation_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    previous_path, previous_document = v2_fixture(tmp_path)
    previous = TOOL.validate_manifest(previous_path)
    document = json.loads(json.dumps(previous_document))
    document["predecessor"] = {
        "schema_version": TOOL.MANIFEST_SCHEMA,
        "path": str(previous_path),
        "sha256": previous["sha256"],
        "entries_sha256": previous["entries_sha256"],
        "entry_count": previous["entry_count"],
    }
    document["entries"].append(json.loads(json.dumps(document["entries"][-1])))
    document["entries"][-1]["sequence"] = 8
    if mutation == "delete":
        document["entries"].pop(1)
        for index, entry in enumerate(document["entries"]):
            entry["sequence"] = index
    elif mutation == "replace":
        document["entries"][1]["relation"] = "historical_implementation_audit"
    elif mutation == "reorder":
        document["entries"][1], document["entries"][2] = (
            document["entries"][2],
            document["entries"][1],
        )
        document["entries"][1]["sequence"] = 1
        document["entries"][2]["sequence"] = 2
    else:
        document["entries"][-1]["sequence"] = 8
    candidate = (tmp_path / f"mutated-{mutation}.json").resolve()
    publish(candidate, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(candidate)


def test_v2_fake_current_go_and_live_entry_drift_fail_closed(tmp_path: Path) -> None:
    manifest, document = v2_fixture(tmp_path)
    current = document["entries"][0]
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(
            manifest,
            expected_current_implementation_audit={
                "path": current["path"],
                "sha256": "0" * 64,
            },
        )

    entry_path = Path(current["path"])
    entry_path.chmod(0o644)
    source = json.loads(entry_path.read_text(encoding="ascii"))
    source["audited_source"]["commit"] = "e" * 40
    publish(entry_path, source)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_live_file_toctou_identity_change_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _document = v2_fixture(tmp_path)
    real_fstat = TOOL.os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        result = real_fstat(descriptor)
        if calls != 2:
            return result
        fields = {
            name: getattr(result, name)
            for name in (
                "st_mode",
                "st_nlink",
                "st_size",
                "st_dev",
                "st_ino",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        fields["st_mtime_ns"] += 1
        return SimpleNamespace(**fields)

    monkeypatch.setattr(TOOL.os, "fstat", changing_fstat)
    with pytest.raises(TOOL.LineageError, match="changed while reading"):
        TOOL.validate_manifest(manifest)
