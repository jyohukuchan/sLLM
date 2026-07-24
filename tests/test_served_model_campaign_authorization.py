from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/served_model_campaign_authorization.py"
SPEC = importlib.util.spec_from_file_location(
    "test_served_model_campaign_authorization_module", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
AUTH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUTH
SPEC.loader.exec_module(AUTH)

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


def policy(tmp_path: Path) -> object:
    claims = tmp_path / "claims"
    outcomes = tmp_path / "outcomes"
    claims.mkdir(mode=0o700)
    outcomes.mkdir(mode=0o700)
    return AUTH.RegistryPolicy(
        claim_registry=claims,
        outcome_registry=outcomes,
        required_uid=os.geteuid(),
    )


def document(tmp_path: Path) -> dict[str, object]:
    outputs = tmp_path / "outputs"
    outputs.mkdir(mode=0o700)
    aq4_source = tmp_path / "aq4-source"
    aq4_source.mkdir(mode=0o700)
    aq4_worker = tmp_path / "aq4-worker"
    aq4_worker.write_bytes(b"fixture AQ4 worker\n")
    bundle_root = outputs / "aq4-bundle-components"
    bundle_root.mkdir(mode=0o700)
    promotion_source = tmp_path / "aq4-promotion-source"
    promotion_source.mkdir(mode=0o700)
    promotion_evidence = promotion_source / "promotion-evidence.json"
    promotion_evidence.write_bytes(b'{"fixture":"evidence"}\n')
    promotion_receipt = promotion_source / "promotion-receipt.json"
    promotion_receipt.write_bytes(b'{"fixture":"receipt"}\n')
    return {
        "schema_version": AUTH.AUTHORIZATION_SCHEMA,
        "authorization_id": "sq8-v2-window-20260724-001",
        "issued_at": AUTH.utc_timestamp(NOW - timedelta(minutes=1)),
        "expires_at": AUTH.utc_timestamp(NOW + timedelta(hours=2)),
        "max_attempts": 1,
        "authorization_note": "Reviewed candidate-only evidence window.",
        "purpose": "temporary_candidate_active_evidence_collection_only",
        "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
        "source": {"commit": "a" * 40, "tree": "b" * 40},
        "before": {
            "model_id": "ullm-qwen3.5-9b-aq4",
            "format_id": "AQ4_0",
            "manifest_sha256": "1" * 64,
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_path": str(aq4_worker),
            "worker_binary_sha256": "2" * 64,
            "promotion_source_commit": "c" * 40,
            "promotion_receipt_path": str(promotion_receipt),
            "promotion_receipt_sha256": "d" * 64,
        },
        "aq4_release": {
            "source": {
                "root": str(aq4_source),
                "commit": "c" * 40,
                "tree": "d" * 40,
            },
            "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
            "promotion_evidence": {
                "source_path": str(promotion_evidence),
                "path": str(bundle_root / promotion_evidence.name),
                "sha256": "f" * 64,
            },
            "promotion_receipt": {
                "source_path": str(promotion_receipt),
                "path": str(bundle_root / promotion_receipt.name),
                "sha256": "d" * 64,
            },
            "release_evidence_path": str(bundle_root / "release-evidence.json"),
            "release_validator_path": str(bundle_root / "release-validator.json"),
            "browser_validator_path": str(bundle_root / "browser-validator.json"),
        },
        "candidate": {
            "model_id": "ullm-qwen3-14b-sq8",
            "format_id": "SQ8_0",
            "manifest_sha256": "3" * 64,
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_sha256": "4" * 64,
            "promotion_source_commit": "a" * 40,
            "promotion_receipt_sha256": "5" * 64,
        },
        "campaigns": {
            "aq4_reasoning_release": {
                "run_id": "aq4-reasoning-release-20260724-001",
                "final_path": str(outputs / "aq4-reasoning-release"),
            },
            "aq4_reasoning_browser": {
                "run_id": "aq4-reasoning-browser-20260724-001",
                "final_path": str(bundle_root / "browser-evidence.json"),
            },
            "aq4_bundle": {
                "run_id": "aq4-bundle-20260724-001",
                "final_path": str(bundle_root / "bundle.json"),
            },
            "sq8_full": {
                "run_id": "sq8-full-20260724-001",
                "final_path": str(outputs / "sq8-full"),
            },
            "reasoning_release": {
                "run_id": "reasoning-release-20260724-001",
                "final_path": str(outputs / "reasoning-release"),
            },
            "reasoning_browser": {
                "run_id": "reasoning-browser-20260724-001",
                "final_path": str(outputs / "reasoning-browser"),
            },
        },
        "rollback": {
            "backup_path": str(outputs / "aq4-exact-backup.json"),
            "systemd_unit_sha256": "6" * 64,
            "environment_sha256": "7" * 64,
        },
        "prior_outcome": None,
    }


def issue(tmp_path: Path) -> tuple[object, Path, dict[str, object]]:
    selected_policy = policy(tmp_path)
    value = document(tmp_path)
    authorization_dir = tmp_path / "authorizations"
    authorization_dir.mkdir(mode=0o700)
    path = authorization_dir / "authorization.json"
    AUTH.issue_authorization(
        value,
        path,
        now=NOW,
        policy=selected_policy,
    )
    return selected_policy, path, value


def outcome_document(
    claim: object,
    *,
    status: str = "succeeded_restored",
    failure_stage: str | None = None,
) -> dict[str, object]:
    authorization = claim.authorization.document
    stages = {name: "passed" for name in AUTH.OUTCOME_STAGE_FIELDS}
    restoration = {
        "expected_manifest_sha256": authorization["before"]["manifest_sha256"],
        "displaced_manifest_sha256": authorization["candidate"]["manifest_sha256"],
        "observed_manifest_sha256": authorization["before"]["manifest_sha256"],
        "bytes_equal": True,
        "reverse_reconciliation_passed": True,
        "final_checks_passed": True,
        "model_id": "ullm-qwen3.5-9b-aq4",
        "format_id": "AQ4_0",
        "worker_binary_sha256": authorization["before"]["worker_binary_sha256"],
        "proof": {
            "schema_version": AUTH.restoration_proof.SCHEMA_VERSION,
            "authorization_sha256": claim.authorization.snapshot.sha256,
            "claim_sha256": claim.snapshot.sha256,
            "captured_at": AUTH.utc_timestamp(NOW),
            "active_manifest": {
                "path": str(AUTH.FIXED_ACTIVE_MANIFEST),
                "expected_sha256": authorization["before"]["manifest_sha256"],
                "observed_sha256": authorization["before"]["manifest_sha256"],
                "bytes_equal": True,
            },
            "service": {
                "unit": "ullm-openai.service",
                "active_state": "active",
                "sub_state": "running",
                "boot_id": "11111111-2222-3333-4444-555555555555",
                "n_restarts": 0,
            },
            "gateway": {
                "pid": 100,
                "ppid": 1,
                "starttime_ticks": 10,
                "executable_sha256": "7" * 64,
            },
            "worker": {
                "pid": 101,
                "ppid": 100,
                "starttime_ticks": 11,
                "executable_sha256": authorization["before"][
                    "worker_binary_sha256"
                ],
            },
            "endpoints": {
                "gateway_healthz": {"status": 200},
                "gateway_readyz": {"status": 200},
                "gateway_models": {
                    "status": 200,
                    "model_ids": ["ullm-qwen3.5-9b-aq4"],
                },
                "openwebui_health": {"status": 200},
                "openwebui_models": {
                    "status": 200,
                    "model_ids": ["ullm-qwen3.5-9b-aq4"],
                },
            },
            "epoch_stable": True,
            "passed": True,
        },
    }
    if failure_stage is not None:
        stages[failure_stage] = "failed"
    if status == "failed_restore":
        stages["aq4_restore"] = "failed"
        stages["reverse_reconciliation"] = "skipped"
        stages["final_checks"] = "skipped"
        restoration.update(
            observed_manifest_sha256=None,
            bytes_equal=False,
            reverse_reconciliation_passed=False,
            final_checks_passed=False,
            model_id=None,
            format_id=None,
            worker_binary_sha256=None,
            proof=None,
        )
    campaigns = {}
    for name, value in authorization["campaigns"].items():
        campaigns[name] = (
            None
            if stages[name] != "passed"
            else {
                "run_id": value["run_id"],
                "path": value["final_path"],
                "kind": (
                    "file"
                    if name in {"aq4_reasoning_browser", "aq4_bundle"}
                    else "directory"
                ),
                "sha256": "8" * 64,
                "artifact_count": 1,
                "total_bytes": 2,
                "selected_artifacts": {"SHA256SUMS": "9" * 64},
            }
        )
    return {
        "schema_version": AUTH.OUTCOME_SCHEMA,
        "authorization_id": authorization["authorization_id"],
        "authorization_path": str(claim.authorization.snapshot.path),
        "authorization_sha256": claim.authorization.snapshot.sha256,
        "claim_path": str(claim.snapshot.path),
        "claim_sha256": claim.snapshot.sha256,
        "started_at": claim.document["claimed_at"],
        "completed_at": AUTH.utc_timestamp(NOW + timedelta(minutes=1)),
        "status": status,
        "failure_stage": failure_stage,
        "stages": stages,
        "aq4_observations": [
            {
                "stage": stage,
                "active_manifest_sha256": authorization["before"][
                    "manifest_sha256"
                ],
                "bytes_equal": True,
            }
            for stage in AUTH.AQ4_OBSERVATION_STAGES
        ],
        "candidate_observations": [
            {
                "stage": stage,
                "active_manifest_sha256": authorization["candidate"][
                    "manifest_sha256"
                ],
                "bytes_equal": True,
            }
            for stage in AUTH.CANDIDATE_OBSERVATION_STAGES
        ],
        "campaigns": campaigns,
        "restoration": restoration,
    }


def test_issue_and_claim_are_canonical_immutable_and_replay_safe(
    tmp_path: Path,
) -> None:
    selected_policy, path, value = issue(tmp_path)
    assert path.read_bytes() == AUTH.canonical_json_bytes(value)
    metadata = path.stat()
    assert metadata.st_mode & 0o777 == 0o444
    assert metadata.st_nlink == 1

    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    assert claim.document["attempt"] == 1
    assert claim.document["max_attempts"] == 1
    assert claim.snapshot.path == AUTH.claim_path(
        claim.authorization.snapshot.sha256,
        policy=selected_policy,
    ).resolve()
    assert claim.snapshot.mode == 0o444
    assert claim.snapshot.nlink == 1
    loaded = AUTH.load_claim(path, now=NOW, policy=selected_policy)
    assert loaded.snapshot.sha256 == claim.snapshot.sha256
    with pytest.raises(AUTH.AuthorizationConsumed):
        AUTH.claim_authorization(path, now=NOW, policy=selected_policy)


def test_concurrent_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    selected_policy, path, _ = issue(tmp_path)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    lock = threading.Lock()

    def compete() -> None:
        barrier.wait()
        try:
            AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
        except AUTH.AuthorizationConsumed:
            result = "consumed"
        else:
            result = "claimed"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=compete) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("claimed") == 1
    assert outcomes.count("consumed") == 7


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(max_attempts=2), "max_attempts"),
        (
            lambda value: value["candidate"].update(format_id="AQ4_0"),
            "candidate identity",
        ),
        (
            lambda value: value["candidate"].update(worker_protocol="ullm.worker.v1"),
            "candidate identity",
        ),
        (
            lambda value: value["source"].update(commit="8" * 40),
            "source/candidate commit",
        ),
        (
            lambda value: value["aq4_release"]["source"].update(
                commit="8" * 40
            ),
            "AQ4 release source",
        ),
        (
            lambda value: value["before"].update(
                worker_protocol="ullm.worker.v1"
            ),
            "before identity",
        ),
        (
            lambda value: value["aq4_release"]["promotion_receipt"].update(
                sha256="8" * 64
            ),
            "promotion receipt differs",
        ),
        (
            lambda value: value["aq4_release"].update(
                openwebui_image="fixture/openwebui@sha256:" + "8" * 64
            ),
            "OpenWebUI image differs",
        ),
        (
            lambda value: value["campaigns"]["sq8_full"].update(
                final_path=value["campaigns"]["reasoning_release"]["final_path"]
            ),
            "distinct",
        ),
        (
            lambda value: value.update(expires_at="2026-07-24T11:59:59Z"),
            "expired",
        ),
    ],
)
def test_semantic_mutations_are_rejected(
    tmp_path: Path, mutate: object, match: str
) -> None:
    value = document(tmp_path)
    mutate(value)
    with pytest.raises(AUTH.AuthorizationError, match=match):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
        )


def test_authorization_file_requires_canonical_bytes_mode_owner_and_nlink(
    tmp_path: Path,
) -> None:
    selected_policy = policy(tmp_path)
    value = document(tmp_path)
    authorization_dir = tmp_path / "authorizations"
    authorization_dir.mkdir(mode=0o700)
    path = authorization_dir / "authorization.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(AUTH.AuthorizationError, match="not canonical"):
        AUTH.load_authorization(path, now=NOW, policy=selected_policy)

    path.chmod(0o644)
    with pytest.raises(AUTH.AuthorizationError, match="metadata"):
        AUTH.load_authorization(path, now=NOW, policy=selected_policy)
    path.chmod(0o444)
    sibling = authorization_dir / "second-link.json"
    os.link(path, sibling)
    with pytest.raises(AUTH.AuthorizationError, match="metadata"):
        AUTH.load_authorization(path, now=NOW, policy=selected_policy)


def test_claim_remains_valid_after_authorized_outputs_are_created(
    tmp_path: Path,
) -> None:
    selected_policy, path, value = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    for campaign in value["campaigns"].values():
        Path(campaign["final_path"]).mkdir()
    Path(value["rollback"]["backup_path"]).write_bytes(b"aq4\n")
    loaded = AUTH.load_claim(path, now=NOW, policy=selected_policy)
    live = AUTH.load_live_claim(path, now=NOW, policy=selected_policy)
    assert loaded.snapshot.sha256 == claim.snapshot.sha256
    assert live.snapshot.sha256 == claim.snapshot.sha256


def test_consumed_claim_and_outcome_remain_loadable_after_expiry(
    tmp_path: Path,
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    AUTH.publish_outcome(
        claim,
        outcome_document(claim),
        policy=selected_policy,
    )
    after_expiry = NOW + timedelta(days=1)

    loaded_claim = AUTH.load_claim(
        path,
        now=after_expiry,
        policy=selected_policy,
    )
    with pytest.raises(AUTH.AuthorizationError, match="expired"):
        AUTH.load_live_claim(
            path,
            now=after_expiry,
            policy=selected_policy,
        )
    _snapshot, loaded_outcome = AUTH.load_outcome(
        path,
        now=after_expiry,
        policy=selected_policy,
    )

    assert loaded_claim.snapshot.sha256 == claim.snapshot.sha256
    assert loaded_outcome["authorization_sha256"] == (
        claim.authorization.snapshot.sha256
    )
    with pytest.raises(AUTH.AuthorizationError, match="expired"):
        AUTH.load_authorization(
            path,
            now=after_expiry,
            policy=selected_policy,
        )


def test_window_and_campaign_bindings_are_exact(tmp_path: Path) -> None:
    selected_policy, path, value = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    AUTH.require_window_binding(
        claim,
        source_commit=value["source"]["commit"],
        source_tree=value["source"]["tree"],
        aq4_source_root=Path(value["aq4_release"]["source"]["root"]),
        aq4_source_commit=value["aq4_release"]["source"]["commit"],
        aq4_source_tree=value["aq4_release"]["source"]["tree"],
        before_manifest_sha256=value["before"]["manifest_sha256"],
        before_worker_protocol=value["before"]["worker_protocol"],
        before_worker_binary_path=Path(value["before"]["worker_binary_path"]),
        before_promotion_receipt_path=Path(
            value["before"]["promotion_receipt_path"]
        ),
        before_promotion_receipt_sha256=value["before"][
            "promotion_receipt_sha256"
        ],
        aq4_promotion_evidence_path=Path(
            value["aq4_release"]["promotion_evidence"]["source_path"]
        ),
        aq4_promotion_evidence_sha256=value["aq4_release"][
            "promotion_evidence"
        ]["sha256"],
        candidate_manifest_sha256=value["candidate"]["manifest_sha256"],
        candidate_worker_binary_sha256=value["candidate"]["worker_binary_sha256"],
        candidate_promotion_receipt_sha256=value["candidate"][
            "promotion_receipt_sha256"
        ],
        rollback_backup_path=Path(value["rollback"]["backup_path"]),
    )
    campaign = value["campaigns"]["sq8_full"]
    AUTH.require_campaign_binding(
        claim,
        campaign_name="sq8_full",
        run_id=campaign["run_id"],
        final_path=Path(campaign["final_path"]),
    )
    with pytest.raises(AUTH.AuthorizationError, match="window identity"):
        AUTH.require_window_binding(
            claim,
            source_commit=value["source"]["commit"],
            source_tree=value["source"]["tree"],
            aq4_source_root=Path(value["aq4_release"]["source"]["root"]),
            aq4_source_commit=value["aq4_release"]["source"]["commit"],
            aq4_source_tree=value["aq4_release"]["source"]["tree"],
            before_manifest_sha256=value["before"]["manifest_sha256"],
            before_worker_protocol=value["before"]["worker_protocol"],
            before_worker_binary_path=Path(
                value["before"]["worker_binary_path"]
            ),
            before_promotion_receipt_path=Path(
                value["before"]["promotion_receipt_path"]
            ),
            before_promotion_receipt_sha256=value["before"][
                "promotion_receipt_sha256"
            ],
            aq4_promotion_evidence_path=Path(
                value["aq4_release"]["promotion_evidence"]["source_path"]
            ),
            aq4_promotion_evidence_sha256=value["aq4_release"][
                "promotion_evidence"
            ]["sha256"],
            candidate_manifest_sha256="0" * 64,
            candidate_worker_binary_sha256=value["candidate"][
                "worker_binary_sha256"
            ],
            candidate_promotion_receipt_sha256=value["candidate"][
                "promotion_receipt_sha256"
            ],
            rollback_backup_path=Path(value["rollback"]["backup_path"]),
        )
    with pytest.raises(AUTH.AuthorizationError, match="run/output"):
        AUTH.require_campaign_binding(
            claim,
            campaign_name="sq8_full",
            run_id="wrong",
            final_path=Path(campaign["final_path"]),
        )


def test_prior_outcome_must_be_live_immutable_and_hash_bound(
    tmp_path: Path,
) -> None:
    prior_policy, prior_authorization, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(
        prior_authorization,
        now=NOW,
        policy=prior_policy,
    )
    outcome_value = outcome_document(
        claim,
        status="failed_restored",
        failure_stage="sq8_full",
    )
    outcome_snapshot = AUTH.publish_outcome(
        claim,
        outcome_value,
        policy=prior_policy,
    )
    outcome = outcome_snapshot.path
    successor_root = tmp_path / "successor"
    successor_root.mkdir()
    value = document(successor_root)
    value["prior_outcome"] = {
        "path": str(outcome),
        "sha256": AUTH.hashlib.sha256(outcome.read_bytes()).hexdigest(),
    }
    AUTH.validate_authorization_document(
        value,
        now=NOW,
        required_uid=os.geteuid(),
        policy=prior_policy,
    )
    value["prior_outcome"]["sha256"] = "0" * 64
    with pytest.raises(AUTH.AuthorizationError, match="SHA-256 differs"):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
            policy=prior_policy,
        )


def test_prior_outcome_rejects_success_external_copy_and_unrelated_lineage(
    tmp_path: Path,
) -> None:
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    prior_policy, prior_authorization, _ = issue(prior_root)
    claim = AUTH.claim_authorization(
        prior_authorization,
        now=NOW,
        policy=prior_policy,
    )
    successful = AUTH.publish_outcome(
        claim,
        outcome_document(claim),
        policy=prior_policy,
    )
    successor_root = tmp_path / "successor-success"
    successor_root.mkdir()
    successor = document(successor_root)
    successor["prior_outcome"] = {
        "path": str(successful.path),
        "sha256": successful.sha256,
    }
    with pytest.raises(AUTH.AuthorizationError, match="not a failed"):
        AUTH.validate_authorization_document(
            successor,
            now=NOW,
            required_uid=os.geteuid(),
            policy=prior_policy,
        )

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_policy, failed_authorization, _ = issue(failed_root)
    failed_claim = AUTH.claim_authorization(
        failed_authorization,
        now=NOW,
        policy=failed_policy,
    )
    failed = AUTH.publish_outcome(
        failed_claim,
        outcome_document(
            failed_claim,
            status="failed_restored",
            failure_stage="sq8_full",
        ),
        policy=failed_policy,
    )
    external_dir = tmp_path / "external"
    external_dir.mkdir(mode=0o700)
    external = external_dir / "copied.outcome.json"
    external.write_bytes(failed.raw)
    external.chmod(0o444)
    successor_root = tmp_path / "successor-copy"
    successor_root.mkdir()
    successor = document(successor_root)
    successor["prior_outcome"] = {
        "path": str(external),
        "sha256": failed.sha256,
    }
    with pytest.raises(AUTH.AuthorizationError, match="outside"):
        AUTH.validate_authorization_document(
            successor,
            now=NOW,
            required_uid=os.geteuid(),
            policy=failed_policy,
        )

    successor["prior_outcome"]["path"] = str(failed.path)
    successor["candidate"]["worker_binary_sha256"] = "e" * 64
    with pytest.raises(AUTH.AuthorizationError, match="lineage differs"):
        AUTH.validate_authorization_document(
            successor,
            now=NOW,
            required_uid=os.geteuid(),
            policy=failed_policy,
        )


def test_publish_and_load_outcome_are_exact_claim_bound_and_no_replace(
    tmp_path: Path,
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    value = outcome_document(claim)

    snapshot = AUTH.publish_outcome(
        claim,
        value,
        policy=selected_policy,
    )

    assert snapshot.path == AUTH.outcome_path(
        claim.authorization.snapshot.sha256,
        policy=selected_policy,
    ).resolve()
    assert snapshot.mode == 0o444
    assert snapshot.nlink == 1
    loaded_snapshot, loaded = AUTH.load_outcome(
        path,
        now=NOW,
        policy=selected_policy,
    )
    assert loaded_snapshot.sha256 == snapshot.sha256
    assert loaded == value
    with pytest.raises(AUTH.AuthorizationConsumed, match="already exists"):
        AUTH.publish_outcome(claim, value, policy=selected_policy)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.update(authorization_sha256="0" * 64),
            "claim identity",
        ),
        (
            lambda value: value["campaigns"]["sq8_full"].update(run_id="wrong"),
            "run/output",
        ),
        (
            lambda value: value["restoration"].update(bytes_equal=False),
            "byte result",
        ),
        (
            lambda value: value["stages"].update(sq8_full="pending"),
            "pending",
        ),
        (
            lambda value: value["candidate_observations"].pop(0),
            "observations differ",
        ),
        (
            lambda value: value["restoration"]["proof"]["active_manifest"].update(
                path="/tmp/untrusted-active.json"
            ),
            "live restoration proof differs",
        ),
        (
            lambda value: value["restoration"]["proof"]["service"].update(
                unit="other.service"
            ),
            "live restoration proof differs",
        ),
        (
            lambda value: value["restoration"]["proof"]["worker"].update(
                executable_sha256="f" * 64
            ),
            "live restoration proof differs",
        ),
    ],
)
def test_outcome_mutations_are_rejected(
    tmp_path: Path, mutate: object, match: str
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    value = outcome_document(claim)
    mutate(value)
    with pytest.raises(AUTH.AuthorizationError, match=match):
        AUTH.validate_outcome_document(value, claim=claim)


def test_failed_restored_and_failed_restore_outcomes_are_distinct(
    tmp_path: Path,
) -> None:
    selected_policy, path, _ = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    failed_campaign = outcome_document(
        claim,
        status="failed_restored",
        failure_stage="sq8_full",
    )
    AUTH.validate_outcome_document(failed_campaign, claim=claim)

    failed_restore = outcome_document(
        claim,
        status="failed_restore",
        failure_stage="aq4_restore",
    )
    AUTH.validate_outcome_document(failed_restore, claim=claim)


def test_authorization_rejects_campaign_outputs_inside_source_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    document_root = tmp_path / "document"
    document_root.mkdir()
    value = document(document_root)
    inside = source / "campaign-output"
    value["campaigns"]["sq8_full"]["final_path"] = str(inside)

    with pytest.raises(AUTH.AuthorizationError, match="outside the source"):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
            source_root=source,
        )


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_output", "source_overlap"),
)
def test_authorization_rejects_double_slash_posix_path_aliases(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    document_root = tmp_path / "document"
    document_root.mkdir()
    value = document(document_root)
    if mutation == "duplicate_output":
        canonical = value["campaigns"]["sq8_full"]["final_path"]
        value["campaigns"]["reasoning_release"]["final_path"] = f"/{canonical}"
    else:
        inside = source / "campaign-output"
        value["campaigns"]["sq8_full"]["final_path"] = f"/{inside}"

    with pytest.raises(AUTH.AuthorizationError, match="canonical"):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
            source_root=source,
        )


def test_archival_claim_load_survives_disappeared_aq4_source(
    tmp_path: Path,
) -> None:
    selected_policy, path, value = issue(tmp_path)
    claim = AUTH.claim_authorization(path, now=NOW, policy=selected_policy)
    Path(value["aq4_release"]["source"]["root"]).rmdir()

    loaded = AUTH.load_claim(
        path,
        now=NOW + timedelta(hours=3),
        policy=selected_policy,
    )
    assert loaded.snapshot.sha256 == claim.snapshot.sha256


def test_aq4_promotion_copy_destination_is_fresh_and_below_bundle_root(
    tmp_path: Path,
) -> None:
    value = document(tmp_path)
    value["aq4_release"]["promotion_evidence"]["path"] = str(
        tmp_path / "outside-promotion-copy.json"
    )
    with pytest.raises(AUTH.AuthorizationError, match="below its output parent"):
        AUTH.validate_authorization_document(
            value,
            now=NOW,
            required_uid=os.geteuid(),
        )

    second = tmp_path / "second"
    second.mkdir()
    other = document(second)
    destination = Path(
        other["aq4_release"]["promotion_evidence"]["path"]
    )
    destination.write_bytes(b"already present\n")
    with pytest.raises(AUTH.AuthorizationError, match="fresh output"):
        AUTH.validate_authorization_document(
            other,
            now=NOW,
            required_uid=os.geteuid(),
        )


def test_stable_read_detects_parent_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved"
    parent.mkdir(mode=0o700)
    replacement.mkdir(mode=0o700)
    target = parent / "document.json"
    target.write_bytes(b"{}\n")
    target.chmod(0o444)
    (replacement / target.name).write_bytes(b"{}\n")
    (replacement / target.name).chmod(0o444)
    real_open_directory = AUTH._open_directory
    calls = 0

    def replace_on_verification(path: Path, label: str) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            parent.rename(moved)
            replacement.rename(parent)
        return real_open_directory(path, label)

    monkeypatch.setattr(AUTH, "_open_directory", replace_on_verification)
    with pytest.raises(AUTH.AuthorizationError, match="changed"):
        AUTH._stable_read(
            target,
            "race fixture",
            required_mode=0o444,
            required_uid=os.geteuid(),
            required_nlink=1,
        )


def test_no_replace_publication_detects_parent_registry_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved"
    registry.mkdir(mode=0o700)
    replacement.mkdir(mode=0o700)
    registry.chmod(0o700)
    replacement.chmod(0o700)
    destination = registry / "receipt.json"
    real_open_directory = AUTH._open_directory
    calls = 0

    def replace_before_verification(path: Path, label: str) -> int:
        nonlocal calls
        calls += 1
        if calls == 3:
            registry.rename(moved)
            replacement.rename(registry)
        return real_open_directory(path, label)

    monkeypatch.setattr(
        AUTH,
        "_open_directory",
        replace_before_verification,
    )
    with pytest.raises(AUTH.AuthorizationError, match="directory changed"):
        AUTH._publish_no_replace(
            destination,
            b"{}\n",
            mode=0o444,
            required_uid=os.geteuid(),
            label="publication race fixture",
        )
    assert not destination.exists()
    assert (moved / destination.name).exists()


def test_no_replace_publication_is_single_link_after_rename_boundary_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    destination = registry / "claim.json"
    raw = b'{"claim":"fixture"}\n'
    real_rename = AUTH._rename_noreplace_at

    def rename_then_fail(*args: object, **kwargs: object) -> None:
        real_rename(*args, **kwargs)
        raise RuntimeError("injected fault immediately after rename")

    monkeypatch.setattr(AUTH, "_rename_noreplace_at", rename_then_fail)
    with pytest.raises(RuntimeError, match="immediately after rename"):
        AUTH._publish_no_replace(
            destination,
            raw,
            mode=0o444,
            required_uid=os.geteuid(),
            label="claim fault fixture",
        )

    metadata = destination.stat()
    assert destination.read_bytes() == raw
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert metadata.st_nlink == 1
    assert not tuple(registry.glob(".*.tmp"))
    assert AUTH._stable_read(
        destination,
        "claim fault fixture",
        required_mode=0o444,
        required_uid=os.geteuid(),
        required_nlink=1,
    ).raw == raw


def test_no_replace_publication_is_single_link_at_parent_fsync_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    destination = registry / "outcome.json"
    raw = b'{"outcome":"fixture"}\n'
    real_fsync = AUTH.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected parent fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr(AUTH.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="parent fsync fault"):
        AUTH._publish_no_replace(
            destination,
            raw,
            mode=0o444,
            required_uid=os.geteuid(),
            label="outcome fault fixture",
        )

    metadata = destination.stat()
    assert destination.read_bytes() == raw
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert metadata.st_nlink == 1
    assert not tuple(registry.glob(".*.tmp"))
