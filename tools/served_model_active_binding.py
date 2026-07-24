#!/usr/bin/env python3
"""Stable byte binding between a frozen candidate and the actual active manifest.

This module is intentionally service-agnostic.  It never activates a model or
invokes systemd/GPU tooling.  Campaign runners use it to:

* pin the exact candidate bytes and expected SHA-256;
* stable-read the actual active path without following symbolic links;
* compare bytes, not merely parsed JSON or a caller-provided digest; and
* retain an ordered observation stream plus an exact candidate copy.

The legacy campaign paths do not import any schema meaning from this module.
Callers must select the v2 binding mode explicitly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import served_model_campaign_authorization as campaign_authorization  # noqa: E402


MAX_MANIFEST_BYTES = 1_048_576
MAX_CLAIM_BYTES = 1_048_576
READ_CHUNK_BYTES = 64 << 10
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
STAGE_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}\Z")

BINDING_SCHEMA = "ullm.served_model.active_binding.v1"
OBSERVATION_SCHEMA = "ullm.served_model.active_manifest_observation.v1"
CANDIDATE_ARTIFACT_NAME = "candidate-served-model.json"
OBSERVATIONS_ARTIFACT_NAME = "active-manifest-observations.jsonl"
BINDING_ARTIFACT_NAME = "active-manifest-binding.json"


class ActiveBindingError(RuntimeError):
    """A candidate/active byte binding could not be proven."""


def fail(message: str) -> NoReturn:
    raise ActiveBindingError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def public_document(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": stat.S_IMODE(self.mode),
            "links": self.links,
            "uid": self.uid,
            "gid": self.gid,
            "bytes": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class StableFileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    identity: FileIdentity


@dataclasses.dataclass(frozen=True, slots=True)
class ClaimReference:
    path: Path
    sha256: str
    bytes: int
    authorization_path: Path
    authorization_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "path": os.fspath(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "authorization_path": os.fspath(self.authorization_path),
            "authorization_sha256": self.authorization_sha256,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ActiveBindingArtifacts:
    candidate_manifest: bytes
    observations_jsonl: bytes
    binding_json: bytes

    def by_name(self) -> dict[str, bytes]:
        return {
            CANDIDATE_ARTIFACT_NAME: self.candidate_manifest,
            OBSERVATIONS_ARTIFACT_NAME: self.observations_jsonl,
            BINDING_ARTIFACT_NAME: self.binding_json,
        }


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        fail("O_DIRECTORY and O_NOFOLLOW are required for active-manifest binding")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for active-manifest binding")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _lexical_absolute(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        fail(f"{label} path must be absolute")
    absolute = Path(os.path.abspath(path))
    if absolute != path or path.name in {"", ".", ".."} or ".." in path.parts:
        fail(f"{label} path must be lexically canonical")
    return absolute


def _open_parent(path: Path, label: str) -> int:
    """Open every parent component with O_NOFOLLOW and return the final fd."""

    absolute = _lexical_absolute(path, label)
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, _directory_flags())
        for component in absolute.parent.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                fail(f"{label} parent component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except ActiveBindingError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        fail(f"{label} parent path is unavailable or traverses a symlink")


def _read_all(descriptor: int, maximum: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(
                descriptor,
                min(READ_CHUNK_BYTES, maximum - total + 1),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                fail(f"{label} exceeds its byte bound")
            chunks.append(chunk)
    except ActiveBindingError:
        raise
    except OSError:
        fail(f"{label} cannot be read")
    return b"".join(chunks)


def stable_read_regular(
    path: Path,
    label: str,
    *,
    maximum: int,
    require_single_link: bool = False,
    require_read_only: bool = False,
) -> StableFileSnapshot:
    """Read a bounded path while binding both the opened fd and pathname entry."""

    if type(maximum) is not int or maximum < 1:
        fail(f"{label} byte bound is invalid")
    absolute = _lexical_absolute(path, label)
    parent_descriptor = _open_parent(absolute, label)
    descriptor = -1
    verification_parent = -1
    try:
        parent_before = FileIdentity.from_stat(os.fstat(parent_descriptor))
        try:
            entry_before = FileIdentity.from_stat(
                os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
            descriptor = os.open(
                absolute.name,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError:
            fail(f"{label} is unavailable or is a symlink")
        opened = FileIdentity.from_stat(os.fstat(descriptor))
        if entry_before != opened:
            fail(f"{label} changed while it was opened")
        if (
            not stat.S_ISREG(opened.mode)
            or opened.mode & stat.S_IWOTH
            or opened.size < 1
            or opened.size > maximum
            or (require_single_link and opened.links != 1)
            or (
                require_read_only
                and stat.S_IMODE(opened.mode)
                & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
        ):
            fail(f"{label} file identity, mode, links, or size is unsafe")
        raw = _read_all(descriptor, maximum, label)
        opened_after = FileIdentity.from_stat(os.fstat(descriptor))
        try:
            entry_after = FileIdentity.from_stat(
                os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError:
            fail(f"{label} pathname disappeared while it was read")
        parent_after = FileIdentity.from_stat(os.fstat(parent_descriptor))
        verification_parent = _open_parent(absolute, label)
        parent_by_path = FileIdentity.from_stat(os.fstat(verification_parent))
        if (
            opened != opened_after
            or opened != entry_after
            or parent_before != parent_after
            or parent_before != parent_by_path
            or len(raw) != opened.size
        ):
            fail(f"{label} changed while it was read")
        return StableFileSnapshot(
            absolute,
            raw,
            hashlib.sha256(raw).hexdigest(),
            opened,
        )
    except ActiveBindingError:
        raise
    except OSError:
        fail(f"{label} cannot be inspected safely")
    finally:
        for value in (verification_parent, descriptor, parent_descriptor):
            if value >= 0:
                try:
                    os.close(value)
                except OSError:
                    pass


def _canonical_json(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("active-manifest binding evidence cannot be serialized")


def _validate_stages(stages: Sequence[str]) -> tuple[str, ...]:
    result = tuple(stages)
    if (
        not result
        or len(result) > 256
        or len(set(result)) != len(result)
        or any(type(stage) is not str or STAGE_RE.fullmatch(stage) is None for stage in result)
    ):
        fail("active-manifest observation stages are invalid")
    return result


class ActiveManifestBinding:
    """Pin one candidate and record exact active-byte observations in order."""

    def __init__(
        self,
        *,
        candidate_path: Path,
        active_path: Path,
        expected_sha256: str,
        expected_stages: Sequence[str],
        authorization_path: Path | None = None,
        campaign_name: str | None = None,
        run_id: str | None = None,
        final_path: Path | None = None,
        expected_source_commit: str | None = None,
        authorization_policy: campaign_authorization.RegistryPolicy | None = None,
        authorization_now: Callable[[], datetime] = (
            lambda: datetime.now(timezone.utc)
        ),
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(expected_sha256) is not str or SHA256_RE.fullmatch(expected_sha256) is None:
            fail("expected candidate manifest SHA-256 is invalid")
        self.expected_stages = _validate_stages(expected_stages)
        self.candidate = stable_read_regular(
            candidate_path,
            "candidate served-model manifest",
            maximum=MAX_MANIFEST_BYTES,
        )
        if self.candidate.sha256 != expected_sha256:
            fail("candidate served-model manifest SHA-256 differs")
        self.active_path = _lexical_absolute(
            active_path, "actual active served-model manifest"
        )
        if self.active_path == self.candidate.path:
            fail("candidate and actual active manifest paths must differ")
        authorization_values = (
            authorization_path,
            campaign_name,
            run_id,
            final_path,
        )
        if any(value is not None for value in authorization_values) and any(
            value is None for value in authorization_values
        ):
            fail("campaign authorization binding is incomplete")
        if not callable(authorization_now):
            fail("campaign authorization clock differs")
        self.authorization_path = authorization_path
        self.campaign_name = campaign_name
        self.run_id = run_id
        self.final_path = final_path
        self.expected_source_commit = expected_source_commit
        self.authorization_policy = (
            campaign_authorization.RegistryPolicy()
            if authorization_policy is None
            else authorization_policy
        )
        if (
            authorization_path is not None
            and self.active_path != self.authorization_policy.active_manifest_path
        ):
            fail(
                "actual active served-model manifest path differs from "
                "the authorization policy"
            )
        self._authorization_now = authorization_now
        self.claim = (
            self._load_authorized_claim()
            if authorization_path is not None
            else None
        )
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns
        self._observations: list[dict[str, Any]] = []

    @staticmethod
    def _manifest_identity(raw: bytes) -> dict[str, str]:
        def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, child in pairs:
                if key in value:
                    fail("candidate served-model manifest contains duplicate keys")
                value[key] = child
            return value

        try:
            document = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicate,
                parse_constant=lambda _value: fail(
                    "candidate served-model manifest contains a non-finite number"
                ),
            )
        except ActiveBindingError:
            raise
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            fail("candidate served-model manifest is not strict JSON")
        if type(document) is not dict:
            fail("candidate served-model manifest root differs")
        public = document.get("public")
        format_value = document.get("format")
        worker = document.get("worker")
        promotion = document.get("promotion")
        if (
            document.get("schema_version") != "ullm.served_model.v2"
            or type(public) is not dict
            or type(format_value) is not dict
            or type(worker) is not dict
            or type(promotion) is not dict
        ):
            fail("candidate served-model v2 identity differs")
        identity = {
            "model_id": public.get("id"),
            "format_id": format_value.get("format_id"),
            "worker_protocol": worker.get("protocol"),
            "worker_binary_sha256": worker.get("binary_sha256"),
            "promotion_source_commit": promotion.get("source_commit"),
            "promotion_receipt_sha256": promotion.get("receipt_sha256"),
        }
        if (
            identity["model_id"] != "ullm-qwen3-14b-sq8"
            or identity["format_id"] != "SQ8_0"
            or identity["worker_protocol"] != "ullm.worker.v2"
            or any(type(value) is not str for value in identity.values())
            or SHA256_RE.fullmatch(identity["worker_binary_sha256"]) is None
            or SHA256_RE.fullmatch(identity["promotion_receipt_sha256"]) is None
            or not identity["promotion_source_commit"]
        ):
            fail("candidate SQ8_0 manifest identity differs")
        return identity

    def _load_authorized_claim(self) -> ClaimReference:
        assert self.authorization_path is not None
        assert self.campaign_name is not None
        assert self.run_id is not None
        assert self.final_path is not None
        now = self._authorization_now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            fail("campaign authorization clock result differs")
        try:
            record = campaign_authorization.load_live_claim(
                self.authorization_path,
                now=now,
                policy=self.authorization_policy,
            )
            campaign_authorization.require_campaign_binding(
                record,
                campaign_name=self.campaign_name,
                run_id=self.run_id,
                final_path=self.final_path,
            )
        except campaign_authorization.AuthorizationError as error:
            raise ActiveBindingError(
                "campaign authorization claim or run/output binding differs"
            ) from error
        candidate = record.authorization.document["candidate"]
        manifest = self._manifest_identity(self.candidate.raw)
        if (
            candidate["manifest_sha256"] != self.candidate.sha256
            or any(
                manifest[key] != candidate[key]
                for key in (
                    "model_id",
                    "format_id",
                    "worker_protocol",
                    "worker_binary_sha256",
                    "promotion_source_commit",
                    "promotion_receipt_sha256",
                )
            )
            or (
                self.expected_source_commit is not None
                and record.authorization.document["source"]["commit"]
                != self.expected_source_commit
            )
        ):
            fail("candidate/source identity differs from campaign authorization")
        return ClaimReference(
            record.snapshot.path,
            record.snapshot.sha256,
            len(record.snapshot.raw),
            record.authorization.snapshot.path,
            record.authorization.snapshot.sha256,
        )

    @property
    def complete(self) -> bool:
        return len(self._observations) == len(self.expected_stages)

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def revalidate_sources(self) -> None:
        """Re-read the pinned candidate and immutable claim without observing active."""

        candidate_now = stable_read_regular(
            self.candidate.path,
            "candidate served-model manifest",
            maximum=MAX_MANIFEST_BYTES,
        )
        if (
            candidate_now.raw != self.candidate.raw
            or candidate_now.sha256 != self.candidate.sha256
        ):
            fail("candidate served-model manifest changed after it was pinned")
        if self.claim is not None:
            claim_now = self._load_authorized_claim()
            if claim_now != self.claim:
                fail("campaign authorization claim changed after it was pinned")

    def observe(self, stage: str) -> dict[str, Any]:
        sequence = len(self._observations)
        if sequence >= len(self.expected_stages):
            fail("active-manifest observation count exceeds the fixed stage set")
        expected_stage = self.expected_stages[sequence]
        if stage != expected_stage:
            fail("active-manifest observation stage order differs")

        self.revalidate_sources()
        candidate_now = stable_read_regular(
            self.candidate.path,
            "candidate served-model manifest",
            maximum=MAX_MANIFEST_BYTES,
        )
        active = stable_read_regular(
            self.active_path,
            "actual active served-model manifest",
            maximum=MAX_MANIFEST_BYTES,
        )
        if (
            active.sha256 != self.candidate.sha256
            or active.raw != self.candidate.raw
        ):
            fail("actual active served-model manifest bytes differ from candidate")

        wall_ns = self._wall_clock_ns()
        monotonic_ns = self._monotonic_clock_ns()
        if (
            type(wall_ns) is not int
            or wall_ns < 0
            or type(monotonic_ns) is not int
            or monotonic_ns < 0
        ):
            fail("active-manifest observation clock differs")
        observation = {
            "schema_version": OBSERVATION_SCHEMA,
            "sequence": sequence,
            "stage": stage,
            "observed_unix_ns": wall_ns,
            "observed_monotonic_ns": monotonic_ns,
            "candidate": {
                "path": os.fspath(self.candidate.path),
                "sha256": self.candidate.sha256,
                "identity": candidate_now.identity.public_document(),
            },
            "active": {
                "path": os.fspath(self.active_path),
                "sha256": active.sha256,
                "identity": active.identity.public_document(),
            },
            "bytes_equal": True,
            "claim": None if self.claim is None else self.claim.document(),
        }
        self._observations.append(observation)
        return observation

    def artifacts(self, *, require_complete: bool = True) -> ActiveBindingArtifacts:
        if require_complete and not self.complete:
            fail("active-manifest observations are incomplete")
        observations = b"".join(_canonical_json(item) for item in self._observations)
        if not observations:
            fail("active-manifest observations are empty")
        observations_sha256 = hashlib.sha256(observations).hexdigest()
        binding = {
            "schema_version": BINDING_SCHEMA,
            "status": "complete" if self.complete else "incomplete",
            "candidate": {
                "artifact": CANDIDATE_ARTIFACT_NAME,
                "source_path": os.fspath(self.candidate.path),
                "sha256": self.candidate.sha256,
                "bytes": len(self.candidate.raw),
            },
            "actual_active_path": os.fspath(self.active_path),
            "expected_stages": list(self.expected_stages),
            "observation_count": len(self._observations),
            "observations": {
                "artifact": OBSERVATIONS_ARTIFACT_NAME,
                "sha256": observations_sha256,
                "bytes": len(observations),
            },
            "claim": None if self.claim is None else self.claim.document(),
            "campaign": (
                None
                if self.claim is None
                else {
                    "name": self.campaign_name,
                    "run_id": self.run_id,
                    "final_path": os.fspath(self.final_path),
                }
            ),
        }
        return ActiveBindingArtifacts(
            self.candidate.raw,
            observations,
            _canonical_json(binding),
        )


def optional_v2_binding(
    *,
    mode: str,
    candidate_path: Path | None,
    active_path: Path | None,
    expected_sha256: str | None,
    expected_stages: Sequence[str],
    authorization_path: Path | None = None,
    campaign_name: str | None = None,
    run_id: str | None = None,
    final_path: Path | None = None,
    expected_source_commit: str | None = None,
    authorization_policy: campaign_authorization.RegistryPolicy | None = None,
    authorization_now: Callable[[], datetime] = (
        lambda: datetime.now(timezone.utc)
    ),
) -> ActiveManifestBinding | None:
    """Explicitly dispatch legacy inputs or construct the complete v2 binding."""

    if mode not in {"legacy", "v2"}:
        fail("active-manifest binding mode differs")
    values = (
        candidate_path,
        active_path,
        expected_sha256,
        authorization_path,
        campaign_name,
        run_id,
        final_path,
    )
    if mode == "legacy":
        if any(value is not None for value in values) or expected_source_commit is not None:
            fail("v2 active-manifest inputs require active-binding mode v2")
        return None
    if (
        candidate_path is None
        or active_path is None
        or expected_sha256 is None
        or authorization_path is None
        or campaign_name is None
        or run_id is None
        or final_path is None
    ):
        fail(
            "v2 active binding requires candidate, active, SHA-256, and "
            "authorization campaign identity"
        )
    return ActiveManifestBinding(
        candidate_path=candidate_path,
        active_path=active_path,
        expected_sha256=expected_sha256,
        expected_stages=expected_stages,
        authorization_path=authorization_path,
        campaign_name=campaign_name,
        run_id=run_id,
        final_path=final_path,
        expected_source_commit=expected_source_commit,
        authorization_policy=authorization_policy,
        authorization_now=authorization_now,
    )


__all__ = [
    "ActiveBindingArtifacts",
    "ActiveBindingError",
    "ActiveManifestBinding",
    "BINDING_ARTIFACT_NAME",
    "BINDING_SCHEMA",
    "CANDIDATE_ARTIFACT_NAME",
    "ClaimReference",
    "FileIdentity",
    "MAX_MANIFEST_BYTES",
    "OBSERVATIONS_ARTIFACT_NAME",
    "OBSERVATION_SCHEMA",
    "StableFileSnapshot",
    "optional_v2_binding",
    "stable_read_regular",
]
