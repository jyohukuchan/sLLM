#!/usr/bin/env python3
"""Render the deterministic prevalidation views for one full SQ8 campaign."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence, cast

from sq8_full_campaign_views import (
    FullCampaignViewError,
    build_full_campaign_views,
)


MATRIX_SCHEMA = "ullm.sq8.openwebui_release.matrix.v1"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
HASH_CHUNK_BYTES = 64 << 10

DERIVED_VIEW_PATHS = (
    "sampling-results.json",
    "cancel-results.json",
    "prefill-latency-results.json",
    "api-contract-results.json",
    "openwebui-smoke.json",
    "soak-results.json",
)

MATRIX_ROLES = {
    "environment.json": "environment",
    "model-identity.json": "model_identity",
    "raw-session-results.jsonl": "session_raw",
    "soak-resources.raw.jsonl": "resource_raw",
    "service-journal.raw.jsonl": "service_journal_raw",
    "amd-smi-metric-normal-before.json": "gpu_metric_raw",
    "amd-smi-metric-normal-after.json": "gpu_metric_raw",
    "amd-smi-metric-restart-before.json": "gpu_metric_raw",
    "amd-smi-metric-restart-after.json": "gpu_metric_raw",
    "sampling-results.json": "derived_view",
    "cancel-results.json": "derived_view",
    "prefill-latency-results.json": "derived_view",
    "api-contract-results.json": "derived_view",
    "openwebui-smoke.json": "derived_view",
    "soak-results.json": "derived_view",
    "browser/openwebui-stop-before.png": "browser_screenshot",
    "browser/post-header-failure.png": "browser_screenshot",
}

EXISTING_ROOT_PATHS = frozenset(
    {
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
)
EXISTING_BROWSER_PATHS = frozenset(
    {"openwebui-stop-before.png", "post-header-failure.png"}
)
EXISTING_PATHS = EXISTING_ROOT_PATHS | frozenset(
    f"browser/{name}" for name in EXISTING_BROWSER_PATHS
)
MATRIX_INPUT_PATHS = tuple(sorted(MATRIX_ROLES, key=lambda item: item.encode("utf-8")))
PREVALIDATION_PATHS = frozenset(MATRIX_ROLES) | {
    "release-matrix.json",
    "summary.md",
    "SHA256SUMS",
}
SHA256SUM_INPUT_PATHS = tuple(
    sorted(
        PREVALIDATION_PATHS - {"SHA256SUMS"},
        key=lambda item: item.encode("utf-8"),
    )
)
RENDERED_PATHS = frozenset(DERIVED_VIEW_PATHS) | {
    "release-matrix.json",
    "summary.md",
    "SHA256SUMS",
}

SCHEDULE = {
    "openwebui_chats": 20,
    "cancel_phases": [
        "after_started_before_progress",
        "prefill_after_128",
        "prefill_after_2048",
        "decode_after_first_content",
        "openwebui_stop_after_visible_content",
    ],
    "normal_warmups": 10,
    "normal_requests": 100,
    "sampled_normal_indices": list(range(5, 101, 5)),
    "restart_warmups": 10,
    "restart_requests": 20,
    "ttft_fixture_ids": [
        "exact-p0032",
        "exact-p0128",
        "exact-p0512",
        "exact-p2048",
        "exact-p3584",
    ],
    "latency_warmups_per_case": 2,
    "latency_measured_per_case": 10,
    "decode_warmups": 2,
    "decode_measured": 10,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
THRESHOLDS = {
    "ttft_seconds_maximum": {
        "exact-p0032": {"p50": 2.5, "p95": 3},
        "exact-p0128": {"p50": 4, "p95": 5},
        "exact-p0512": {"p50": 10, "p95": 12},
        "exact-p2048": {"p50": 30, "p95": 35},
        "exact-p3584": {"p50": 50, "p95": 60},
    },
    "decode_p50_tokens_per_second_minimum": 15,
    "decode_p95_inter_content_seconds_maximum": 0.1,
    "cancel_release_max_ns": 5_000_000_000,
    "final_delta_max_bytes": 67_108_864,
    "theil_sen_max_bytes_per_request": 262_144,
}

MAXIMUM_BYTES = {
    "environment.json": 16 << 20,
    "model-identity.json": 16 << 20,
    "raw-session-results.jsonl": 512 << 20,
    "soak-resources.raw.jsonl": 512 << 20,
    "service-journal.raw.jsonl": 512 << 20,
    "amd-smi-metric-normal-before.json": 16 << 20,
    "amd-smi-metric-normal-after.json": 16 << 20,
    "amd-smi-metric-restart-before.json": 16 << 20,
    "amd-smi-metric-restart-after.json": 16 << 20,
    "browser/openwebui-stop-before.png": 128 << 20,
    "browser/post-header-failure.png": 128 << 20,
}


class FullCampaignRendererError(RuntimeError):
    """A fail-closed full-campaign rendering error."""


def fail(message: str) -> NoReturn:
    raise FullCampaignRendererError(message)


@dataclasses.dataclass(frozen=True)
class FileSeal:
    bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class _Identity:
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
    def from_stat(cls, value: os.stat_result) -> _Identity:
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


def _canonical_json(value: Mapping[str, Any], label: str) -> bytes:
    if type(value) is not dict:
        fail(f"{label} root differs")

    def reject_passed(item: Any) -> None:
        if type(item) is dict:
            if "passed" in item:
                fail(f"{label} contains forbidden key passed")
            for key, child in item.items():
                if type(key) is not str:
                    fail(f"{label} contains a non-text key")
                reject_passed(child)
        elif type(item) is list:
            for child in item:
                reject_passed(child)

    reject_passed(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii", errors="strict")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise FullCampaignRendererError(
            f"{label} is not canonical JSON data"
        ) from error


def _canonical_copy(
    value: Any, expected: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if type(value) is not dict:
        fail(f"{label} differs from the frozen campaign value")
    raw = _canonical_json(cast(dict[str, Any], value), label)
    if raw != _canonical_json(expected, f"frozen {label}"):
        fail(f"{label} differs from the frozen campaign value")
    decoded = json.loads(raw)
    if type(decoded) is not dict:
        fail(f"{label} canonical copy differs")
    return cast(dict[str, Any], decoded)


def _attribute(value: object, name: str, label: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise FullCampaignRendererError(f"{label} lacks {name}") from error


def _mapping_attribute(value: object, name: str, label: str) -> dict[str, Any]:
    result = _attribute(value, name, label)
    if type(result) is not dict:
        fail(f"{label}.{name} is not an exact object")
    return cast(dict[str, Any], result)


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for campaign rendering")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for campaign rendering")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


class _StageSnapshot:
    def __init__(self, stage_path: Path):
        if not isinstance(stage_path, os.PathLike):
            fail("campaign stage path type differs")
        self.path = Path(stage_path)
        if not self.path.is_absolute() or ".." in self.path.parts:
            fail("campaign stage path is not an absolute normalized path")
        self.stage_fd = -1
        self.browser_fd = -1
        self.stage_identity: _Identity | None = None
        self.browser_identity: _Identity | None = None
        self.file_identities: dict[str, _Identity] = {}
        self.uid = -1
        self.gid = -1

    def __enter__(self) -> _StageSnapshot:
        try:
            path_before = _Identity.from_stat(os.stat(self.path, follow_symlinks=False))
            self.stage_fd = os.open(self.path, _directory_flags())
            stage = _Identity.from_stat(os.fstat(self.stage_fd))
            if path_before != stage:
                fail("campaign stage path changed while opening")
            try:
                descriptor_path = Path(os.readlink(f"/proc/self/fd/{self.stage_fd}"))
            except OSError as error:
                raise FullCampaignRendererError(
                    "failed to resolve the campaign stage descriptor"
                ) from error
            if descriptor_path != self.path:
                fail("campaign stage path contains a symbolic link")
            if (
                not stat.S_ISDIR(stage.mode)
                or stat.S_IMODE(stage.mode) != 0o700
                or stage.links < 2
                or stage.uid != os.geteuid()
                or stage.gid != os.getegid()
            ):
                fail("campaign stage mode, owner, or links differ")
            self.uid = stage.uid
            self.gid = stage.gid
            self.stage_identity = stage
            self._validate_root_entries()
            browser_before = self._entry_identity(self.stage_fd, "browser")
            self.browser_fd = os.open(
                "browser", _directory_flags(), dir_fd=self.stage_fd
            )
            browser = _Identity.from_stat(os.fstat(self.browser_fd))
            if browser != browser_before:
                fail("campaign browser directory changed while opening")
            if (
                not stat.S_ISDIR(browser.mode)
                or stat.S_IMODE(browser.mode) != 0o700
                or browser.links < 2
                or browser.uid != self.uid
                or browser.gid != self.gid
            ):
                fail("campaign browser mode, owner, or links differ")
            self.browser_identity = browser
            self._validate_browser_entries()
            return self
        except FullCampaignRendererError as error:
            try:
                self.close()
            except FullCampaignRendererError:
                error.add_note("campaign stage descriptor cleanup also failed")
            raise
        except OSError as error:
            try:
                self.close()
            except FullCampaignRendererError:
                error.add_note("campaign stage descriptor cleanup also failed")
            raise FullCampaignRendererError("failed to open campaign stage") from error

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _entry_identity(parent_fd: int, name: str) -> _Identity:
        try:
            return _Identity.from_stat(
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            )
        except OSError as error:
            raise FullCampaignRendererError(
                f"campaign entry is unavailable: {name}"
            ) from error

    @staticmethod
    def _names(parent_fd: int, label: str) -> set[str]:
        try:
            names = os.listdir(parent_fd)
        except OSError as error:
            raise FullCampaignRendererError(f"failed to enumerate {label}") from error
        if any(type(name) is not str or not name or "/" in name for name in names):
            fail(f"{label} contains an invalid entry name")
        return set(names)

    def _validate_root_entries(self) -> None:
        expected = set(EXISTING_ROOT_PATHS) | {"browser"}
        actual = self._names(self.stage_fd, "campaign stage")
        if actual != expected:
            fail(
                "campaign stage file set differs: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )

    def _validate_browser_entries(self) -> None:
        actual = self._names(self.browser_fd, "campaign browser directory")
        expected = set(EXISTING_BROWSER_PATHS)
        if actual != expected:
            fail(
                "campaign browser file set differs: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )

    def _parent_and_name(self, relative: str) -> tuple[int, str]:
        pure = PurePosixPath(relative)
        if relative in EXISTING_ROOT_PATHS and len(pure.parts) == 1:
            return self.stage_fd, relative
        if (
            len(pure.parts) == 2
            and pure.parts[0] == "browser"
            and pure.parts[1] in EXISTING_BROWSER_PATHS
        ):
            return self.browser_fd, pure.parts[1]
        fail(f"campaign existing path is outside the fixed layout: {relative}")

    def hash_file(self, relative: str, *, expected: FileSeal | None = None) -> FileSeal:
        parent_fd, name = self._parent_and_name(relative)
        before = self._entry_identity(parent_fd, name)
        prior_identity = self.file_identities.get(relative)
        if prior_identity is not None and before != prior_identity:
            fail(f"campaign input identity changed between passes: {relative}")
        maximum = MAXIMUM_BYTES[relative]
        if (
            not stat.S_ISREG(before.mode)
            or stat.S_IMODE(before.mode) != 0o600
            or before.links != 1
            or before.uid != self.uid
            or before.gid != self.gid
            or before.size < 1
            or before.size > maximum
        ):
            fail(f"campaign input mode, owner, links, or size differ: {relative}")
        descriptor = -1
        try:
            descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
            opened = _Identity.from_stat(os.fstat(descriptor))
            if opened != before:
                fail(f"campaign input changed while opening: {relative}")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, HASH_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    fail(f"campaign input exceeds its size limit: {relative}")
                digest.update(chunk)
            after = _Identity.from_stat(os.fstat(descriptor))
            entry_after = self._entry_identity(parent_fd, name)
            if before != after or after != entry_after or total != before.size:
                fail(f"campaign input changed while streaming: {relative}")
            seal = FileSeal(total, digest.hexdigest())
            if expected is not None and seal != expected:
                fail(f"campaign input changed between rendering passes: {relative}")
            if prior_identity is None:
                self.file_identities[relative] = before
            return seal
        except FullCampaignRendererError:
            raise
        except OSError as error:
            raise FullCampaignRendererError(
                f"failed to stream campaign input: {relative}"
            ) from error
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as error:
                    raise FullCampaignRendererError(
                        f"failed to close campaign input: {relative}"
                    ) from error

    def resource_path(self) -> Path:
        if self.stage_fd < 0:
            fail("campaign stage descriptor is closed")
        return Path(f"/proc/self/fd/{self.stage_fd}/soak-resources.raw.jsonl")

    def verify_layout(self) -> None:
        if self.stage_identity is None or self.browser_identity is None:
            fail("campaign stage snapshot is incomplete")
        self._validate_root_entries()
        self._validate_browser_entries()
        current_stage = _Identity.from_stat(os.fstat(self.stage_fd))
        current_browser = _Identity.from_stat(os.fstat(self.browser_fd))
        try:
            path_stage = _Identity.from_stat(os.stat(self.path, follow_symlinks=False))
        except OSError as error:
            raise FullCampaignRendererError(
                "campaign stage path was replaced"
            ) from error
        if (
            current_stage != self.stage_identity
            or path_stage != self.stage_identity
            or current_browser != self.browser_identity
            or self._entry_identity(self.stage_fd, "browser") != self.browser_identity
        ):
            fail("campaign stage changed while rendering")
        if set(self.file_identities) != set(EXISTING_PATHS):
            fail("campaign input identity snapshot is incomplete")
        for relative, expected in self.file_identities.items():
            parent_fd, name = self._parent_and_name(relative)
            if self._entry_identity(parent_fd, name) != expected:
                fail(f"campaign input changed before render completion: {relative}")

    def close(self) -> None:
        errors: list[OSError] = []
        for attribute in ("browser_fd", "stage_fd"):
            descriptor = cast(int, getattr(self, attribute))
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as error:
                    errors.append(error)
                setattr(self, attribute, -1)
        if errors:
            raise FullCampaignRendererError(
                "failed to close campaign stage descriptors"
            ) from errors[0]


def _byte_seal(raw: bytes, label: str) -> FileSeal:
    if type(raw) is not bytes or not raw:
        fail(f"{label} bytes differ")
    return FileSeal(len(raw), hashlib.sha256(raw).hexdigest())


def _summary(run_id: str, schedule: Mapping[str, Any]) -> bytes:
    schedule_line = _canonical_json(schedule, "summary schedule")[:-1].decode("ascii")
    paths = sorted(PREVALIDATION_PATHS, key=lambda item: item.encode("utf-8"))
    lines = [
        "# SQ8 OpenWebUI full campaign",
        "",
        f"Run ID: `{run_id}`",
        "",
        f"Schedule: `{schedule_line}`",
        "",
        "Artifacts:",
        *(f"- `{path}`" for path in paths),
        "",
    ]
    raw = "\n".join(lines).encode("ascii", errors="strict")
    if b"passed" in raw.lower() or b"verdict" in raw.lower():
        fail("summary contains a producer decision")
    return raw


class FullCampaignRenderer:
    """Production renderer satisfying the orchestrator's structural protocol."""

    def render(self, context: object) -> dict[str, bytes]:
        stage_value = _attribute(context, "stage_path", "render context")
        if not isinstance(stage_value, os.PathLike):
            fail("render context stage_path type differs")
        evidence = _attribute(context, "evidence", "render context")
        preflight = _attribute(evidence, "preflight", "campaign evidence")
        header = _mapping_attribute(preflight, "header_fields", "preflight evidence")
        run_id = header.get("run_id")
        if type(run_id) is not str or RUN_ID_RE.fullmatch(run_id) is None:
            fail("campaign run_id differs")
        schedule = _canonical_copy(
            header.get("schedule"), SCHEDULE, "campaign schedule"
        )
        thresholds = _canonical_copy(
            header.get("thresholds"), THRESHOLDS, "campaign thresholds"
        )

        api = _mapping_attribute(
            _attribute(evidence, "api_contract", "campaign evidence"),
            "derived_view",
            "API evidence",
        )
        combined = _mapping_attribute(
            _attribute(evidence, "combined", "campaign evidence"),
            "derived_view",
            "combined evidence",
        )
        direct = _mapping_attribute(
            _attribute(evidence, "direct_cancel", "campaign evidence"),
            "derived_view",
            "direct cancellation evidence",
        )
        stop = _mapping_attribute(
            _attribute(evidence, "stop", "campaign evidence"),
            "derived_view",
            "Stop evidence",
        )
        failure = _mapping_attribute(
            _attribute(evidence, "failure", "campaign evidence"),
            "derived_view",
            "failure evidence",
        )
        latency = _mapping_attribute(
            _attribute(evidence, "latency", "campaign evidence"),
            "derived_view",
            "latency evidence",
        )
        normal_resource = _attribute(evidence, "resource_normal", "campaign evidence")
        sampling_cases = _attribute(
            normal_resource, "sampling_cases", "normal resource evidence"
        )
        if not isinstance(sampling_cases, Sequence) or isinstance(
            sampling_cases, (str, bytes, bytearray)
        ):
            fail("normal resource sampling cases differ")

        with _StageSnapshot(Path(stage_value)) as stage:
            existing = {
                relative: stage.hash_file(relative)
                for relative in sorted(
                    EXISTING_PATHS, key=lambda item: item.encode("utf-8")
                )
            }
            try:
                views = build_full_campaign_views(
                    api,
                    combined,
                    direct,
                    stop,
                    failure,
                    latency,
                    sampling_cases,
                    stage.resource_path(),
                ).serialized()
            except FullCampaignViewError as error:
                raise FullCampaignRendererError(
                    "failed to build canonical full campaign views"
                ) from error
            if tuple(views) != DERIVED_VIEW_PATHS:
                fail("full campaign producer view set differs")

            generated: dict[str, bytes] = dict(views)
            generated_seals = {
                relative: _byte_seal(raw, relative) for relative, raw in views.items()
            }
            entries: list[dict[str, Any]] = []
            for relative in MATRIX_INPUT_PATHS:
                seal = generated_seals.get(relative, existing.get(relative))
                if seal is None:
                    fail(f"matrix input lacks a seal: {relative}")
                entries.append(
                    {
                        "role": MATRIX_ROLES[relative],
                        "path": relative,
                        "bytes": seal.bytes,
                        "sha256": seal.sha256,
                    }
                )
            matrix = {
                "schema_version": MATRIX_SCHEMA,
                "run_id": run_id,
                "files": entries,
                "schedule": schedule,
                "thresholds": thresholds,
            }
            generated["release-matrix.json"] = _canonical_json(matrix, "release matrix")
            generated["summary.md"] = _summary(run_id, schedule)

            lines: list[str] = []
            for relative in SHA256SUM_INPUT_PATHS:
                raw = generated.get(relative)
                if raw is not None:
                    seal = _byte_seal(raw, f"checksum input {relative}")
                else:
                    expected = existing.get(relative)
                    if expected is None:
                        fail(f"checksum input lacks a seal: {relative}")
                    seal = stage.hash_file(relative, expected=expected)
                lines.append(f"{seal.sha256}  {relative}\n")
            generated["SHA256SUMS"] = "".join(lines).encode("ascii")
            stage.verify_layout()

        if set(generated) != RENDERED_PATHS or any(
            type(raw) is not bytes or not raw for raw in generated.values()
        ):
            fail("rendered full campaign artifact set differs")
        return generated


__all__ = [
    "DERIVED_VIEW_PATHS",
    "EXISTING_PATHS",
    "FullCampaignRenderer",
    "FullCampaignRendererError",
    "MATRIX_INPUT_PATHS",
    "MATRIX_ROLES",
    "MATRIX_SCHEMA",
    "PREVALIDATION_PATHS",
    "RENDERED_PATHS",
    "SCHEDULE",
    "SHA256SUM_INPUT_PATHS",
    "THRESHOLDS",
]
