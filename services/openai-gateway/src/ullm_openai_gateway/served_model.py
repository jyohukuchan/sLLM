"""Bounded, fail-closed loader for the served-model deployment contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .reasoning import ReasoningDialect


SCHEMA_VERSION = "ullm.served_model.v1"
SCHEMA_VERSION_V2 = "ullm.served_model.v2"
MAX_MANIFEST_BYTES = 1_048_576
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 16_384
MAX_STRING_BYTES = 65_536
MAX_TOKENIZER_FILES = 128
MAX_ARGUMENTS = 128
MAX_REQUIRED_ENVIRONMENT = 128

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Z_][A-Z0-9_]*\Z")


class ServedModelError(RuntimeError):
    """Raised when a served-model manifest or one of its resources is unsafe."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublicModel:
    id: str
    name: str
    description: str
    upstream_id: str
    revision: str
    context_length: int


@dataclass(frozen=True, slots=True)
class SamplingContract:
    top_k: int
    temperature: bool
    top_p: bool


@dataclass(frozen=True, slots=True)
class GenerationContract:
    max_completion_tokens: int
    vocab_size: int
    eos_token_ids: tuple[int, ...]
    sampling: SamplingContract


@dataclass(frozen=True, slots=True)
class FormatContract:
    format_id: str
    implementation_id: str


@dataclass(frozen=True, slots=True)
class TokenizerFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TokenizerContract:
    root: Path
    transformers_version: str
    class_name: str
    chat_template_sha256: str
    files: tuple[TokenizerFile, ...]
    add_generation_prompt: bool
    enable_thinking: bool


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    device: str
    execution_profile: str


@dataclass(frozen=True, slots=True)
class WorkerContract:
    protocol: str
    binary: Path
    binary_sha256: str
    arguments: tuple[str, ...]
    required_environment: tuple[str, ...]
    identity: WorkerIdentity


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    manifest_path: str
    manifest_sha256: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class PackageIdentity:
    manifest_path: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class AuthorizationAuditIdentity:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ReadinessIdentity:
    container_name: str
    container_id: str
    image_id: str
    config_image: str
    network_name: str
    network_id: str
    network_driver: str
    bridge_interface: str
    url: str
    path: str
    expected_status: int
    expected_body: str
    expected_body_sha256: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class ProductContract:
    root: Path
    artifact: ArtifactIdentity | None
    package: PackageIdentity


@dataclass(frozen=True, slots=True)
class AuthorizationLineageIdentity:
    input_path: Path
    runtime_path: Path
    sha256: str
    entries_sha256: str
    schema_version: str = "ullm.sq8_authorization_lineage_ref.v1"
    entry_count: int | None = None
    current_implementation_audit: AuthorizationAuditIdentity | None = None


@dataclass(frozen=True, slots=True)
class PromotionContract:
    source_commit: str
    receipt: Path
    receipt_sha256: str
    authorization_audit: AuthorizationAuditIdentity | None = None
    authorization_lineage: AuthorizationLineageIdentity | None = None
    readiness: ReadinessIdentity | None = None


@dataclass(frozen=True, slots=True)
class ServedModel:
    manifest_path: Path
    manifest_sha256: str
    public: PublicModel
    generation: GenerationContract
    format: FormatContract
    tokenizer: TokenizerContract
    worker: WorkerContract
    product: ProductContract
    promotion: PromotionContract
    reasoning_dialect: ReasoningDialect | None = None


def load_served_model(path: Path) -> ServedModel:
    """Load and validate one immutable ``ullm.served_model.v1`` document."""

    manifest_path = _safe_regular_file(path, "served-model manifest")
    raw = _bounded_read(manifest_path, MAX_MANIFEST_BYTES, "served-model manifest")
    document = _decode_document(raw)
    schema_version = document.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        expected_keys = {
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "promotion",
        }
    elif schema_version == SCHEMA_VERSION_V2:
        expected_keys = {
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "promotion",
            "reasoning",
        }
    else:
        raise ServedModelError("manifest schema_version is unsupported")
    _exact_keys(document, expected_keys, "manifest")

    public = _parse_public(document["public"])
    generation = _parse_generation(document["generation"], public)
    format_contract = _parse_format(document["format"])
    tokenizer = _parse_tokenizer(document["tokenizer"], manifest_path.parent)
    worker = _parse_worker(document["worker"], manifest_path.parent)
    expected_worker_schema = (
        "ullm.worker.v2" if schema_version == SCHEMA_VERSION_V2 else "ullm.worker.v1"
    )
    if worker.protocol != expected_worker_schema:
        raise ServedModelError(
            "manifest schema_version and worker.protocol must be version aligned"
        )
    product = _parse_product(document["product"], manifest_path.parent)
    promotion = _parse_promotion(document["promotion"], manifest_path.parent)
    reasoning_dialect = (
        _parse_reasoning(document["reasoning"], generation.vocab_size)
        if schema_version == SCHEMA_VERSION_V2
        else None
    )
    if reasoning_dialect is not None:
        reserved_for_max_budget = (
            reasoning_dialect.max_budget_tokens
            + len(reasoning_dialect.forced_end_sequence)
            + reasoning_dialect.reserved_answer_tokens
        )
        if reserved_for_max_budget > generation.max_completion_tokens:
            raise ServedModelError(
                "reasoning maximum budget exceeds the generation reservation"
            )

    return ServedModel(
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(raw),
        public=public,
        generation=generation,
        format=format_contract,
        tokenizer=tokenizer,
        worker=worker,
        product=product,
        promotion=promotion,
        reasoning_dialect=reasoning_dialect,
    )


def _parse_reasoning(value: Any, vocab_size: int) -> ReasoningDialect:
    item = _mapping(value, "reasoning")
    _exact_keys(
        item,
        {
            "enabled_by_default",
            "dialect_id",
            "start_token_ids",
            "end_token_ids",
            "forced_end_token_ids",
            "initial_phase",
            "eos_policy",
            "effort_budgets",
            "max_budget_tokens",
            "reserved_answer_tokens",
            "history_reasoning_policy",
        },
        "reasoning",
    )
    raw_effort = _mapping(item["effort_budgets"], "reasoning.effort_budgets")
    _exact_keys(raw_effort, {"low", "medium", "high"}, "reasoning.effort_budgets")
    effort_budgets = tuple(
        (name, _positive_integer(raw_effort[name], f"reasoning.effort_budgets.{name}"))
        for name in ("low", "medium", "high")
    )
    def token_sequence(name: str) -> tuple[int, ...]:
        raw = item[name]
        if not isinstance(raw, list) or not raw:
            raise ServedModelError(f"reasoning.{name} must be a nonempty array")
        values = tuple(
            _nonnegative_integer(token, f"reasoning.{name}[{index}]")
            for index, token in enumerate(raw)
        )
        if len(values) != len(set(values)):
            raise ServedModelError(f"reasoning.{name} contains duplicates")
        if any(token >= vocab_size for token in values):
            raise ServedModelError(f"reasoning.{name} exceeds vocabulary")
        return values

    start = token_sequence("start_token_ids")
    end = token_sequence("end_token_ids")
    forced = token_sequence("forced_end_token_ids")
    dialect = ReasoningDialect(
        identity=_text(item["dialect_id"], "reasoning.dialect_id", maximum=256),
        start_sequence=start,
        end_sequence=end,
        forced_end_sequence=forced,
        max_budget_tokens=_positive_integer(
            item["max_budget_tokens"], "reasoning.max_budget_tokens"
        ),
        reserved_answer_tokens=_positive_integer(
            item["reserved_answer_tokens"], "reasoning.reserved_answer_tokens"
        ),
        enabled_by_default=_boolean(
            item["enabled_by_default"], "reasoning.enabled_by_default"
        ),
        effort_budgets=effort_budgets,
        history_reasoning_policy=_text(
            item["history_reasoning_policy"],
            "reasoning.history_reasoning_policy",
            maximum=32,
        ),
        initial_phase=_text(item["initial_phase"], "reasoning.initial_phase", maximum=32),
        eos_policy=_text(item["eos_policy"], "reasoning.eos_policy", maximum=32),
    )
    if dialect.end_sequence != dialect.forced_end_sequence:
        raise ServedModelError("reasoning end sequences must match")
    try:
        dialect.validate(vocab_size=vocab_size)
    except ValueError as error:
        raise ServedModelError("reasoning dialect is invalid") from error
    if any(budget > dialect.max_budget_tokens for _, budget in effort_budgets):
        raise ServedModelError("reasoning effort budget exceeds max_budget_tokens")
    return dialect


def _parse_public(value: Any) -> PublicModel:
    item = _mapping(value, "public")
    _exact_keys(
        item,
        {"id", "name", "description", "upstream_id", "revision", "context_length"},
        "public",
    )
    return PublicModel(
        id=_text(item["id"], "public.id", maximum=256),
        name=_text(item["name"], "public.name", maximum=512),
        description=_text(item["description"], "public.description", maximum=4096),
        upstream_id=_text(item["upstream_id"], "public.upstream_id", maximum=512),
        revision=_text(item["revision"], "public.revision", maximum=256),
        context_length=_positive_integer(
            item["context_length"], "public.context_length"
        ),
    )


def _parse_generation(value: Any, public: PublicModel) -> GenerationContract:
    item = _mapping(value, "generation")
    _exact_keys(
        item,
        {"max_completion_tokens", "vocab_size", "eos_token_ids", "sampling"},
        "generation",
    )
    maximum = _positive_integer(
        item["max_completion_tokens"], "generation.max_completion_tokens"
    )
    vocabulary = _positive_integer(item["vocab_size"], "generation.vocab_size")
    raw_eos = item["eos_token_ids"]
    if not isinstance(raw_eos, list) or not raw_eos:
        raise ServedModelError("generation.eos_token_ids must be a nonempty array")
    eos = tuple(
        _nonnegative_integer(token_id, f"generation.eos_token_ids[{index}]")
        for index, token_id in enumerate(raw_eos)
    )
    if len(eos) != len(set(eos)):
        raise ServedModelError("generation.eos_token_ids contains duplicates")
    if any(token_id >= vocabulary for token_id in eos):
        raise ServedModelError("an EOS token ID is outside generation.vocab_size")
    if maximum > public.context_length:
        raise ServedModelError(
            "generation.max_completion_tokens exceeds public.context_length"
        )

    sampling_item = _mapping(item["sampling"], "generation.sampling")
    _exact_keys(sampling_item, {"top_k", "temperature", "top_p"}, "generation.sampling")
    top_k = _positive_integer(sampling_item["top_k"], "generation.sampling.top_k")
    if top_k > vocabulary:
        raise ServedModelError("generation.sampling.top_k exceeds vocab_size")
    temperature = _boolean(
        sampling_item["temperature"], "generation.sampling.temperature"
    )
    top_p = _boolean(sampling_item["top_p"], "generation.sampling.top_p")
    if (not temperature or not top_p) and top_k != 1:
        raise ServedModelError(
            "disabled temperature or top_p requires deterministic top_k=1"
        )
    return GenerationContract(
        max_completion_tokens=maximum,
        vocab_size=vocabulary,
        eos_token_ids=eos,
        sampling=SamplingContract(top_k, temperature, top_p),
    )


def _parse_format(value: Any) -> FormatContract:
    item = _mapping(value, "format")
    _exact_keys(item, {"format_id", "implementation_id"}, "format")
    return FormatContract(
        format_id=_text(item["format_id"], "format.format_id", maximum=128),
        implementation_id=_text(
            item["implementation_id"], "format.implementation_id", maximum=256
        ),
    )


def _parse_tokenizer(value: Any, base: Path) -> TokenizerContract:
    item = _mapping(value, "tokenizer")
    _exact_keys(
        item,
        {
            "root",
            "transformers_version",
            "class",
            "chat_template_sha256",
            "files",
            "template_options",
        },
        "tokenizer",
    )
    root = _safe_directory(
        _resolve_root(base, _text(item["root"], "tokenizer.root", maximum=4096)),
        "tokenizer.root",
    )
    files_item = _mapping(item["files"], "tokenizer.files")
    if not files_item or len(files_item) > MAX_TOKENIZER_FILES:
        raise ServedModelError("tokenizer.files size is outside the supported range")
    files: list[TokenizerFile] = []
    for raw_path, raw_sha256 in files_item.items():
        relative = _relative_path(raw_path, "tokenizer.files path")
        digest = _sha256(raw_sha256, f"tokenizer.files[{raw_path!r}]")
        target = _contained_regular_file(root, relative, "tokenizer file")
        _verify_file_sha256(target, digest, "tokenizer file")
        files.append(TokenizerFile(relative, digest))
    files.sort(key=lambda entry: entry.path.encode("utf-8"))

    options = _mapping(item["template_options"], "tokenizer.template_options")
    _exact_keys(
        options,
        {"add_generation_prompt", "enable_thinking"},
        "tokenizer.template_options",
    )
    return TokenizerContract(
        root=root,
        transformers_version=_text(
            item["transformers_version"], "tokenizer.transformers_version", maximum=64
        ),
        class_name=_text(item["class"], "tokenizer.class", maximum=128),
        chat_template_sha256=_sha256(
            item["chat_template_sha256"], "tokenizer.chat_template_sha256"
        ),
        files=tuple(files),
        add_generation_prompt=_boolean(
            options["add_generation_prompt"],
            "tokenizer.template_options.add_generation_prompt",
        ),
        enable_thinking=_boolean(
            options["enable_thinking"], "tokenizer.template_options.enable_thinking"
        ),
    )


def _parse_worker(value: Any, base: Path) -> WorkerContract:
    item = _mapping(value, "worker")
    _exact_keys(
        item,
        {
            "protocol",
            "binary",
            "binary_sha256",
            "arguments",
            "required_environment",
            "identity",
        },
        "worker",
    )
    binary = _safe_regular_file(
        _resolve_root(base, _text(item["binary"], "worker.binary", maximum=4096)),
        "worker.binary",
    )
    if not os.access(binary, os.X_OK):
        raise ServedModelError("worker.binary is not executable")
    binary_digest = _sha256(item["binary_sha256"], "worker.binary_sha256")
    _verify_file_sha256(binary, binary_digest, "worker.binary")

    raw_arguments = item["arguments"]
    if not isinstance(raw_arguments, list) or len(raw_arguments) > MAX_ARGUMENTS:
        raise ServedModelError("worker.arguments must be a bounded array")
    arguments = tuple(
        _text(argument, f"worker.arguments[{index}]", maximum=4096)
        for index, argument in enumerate(raw_arguments)
    )
    if arguments.count("{manifest}") != 1:
        raise ServedModelError("worker.arguments must contain {manifest} exactly once")

    raw_environment = item["required_environment"]
    if (
        not isinstance(raw_environment, list)
        or len(raw_environment) > MAX_REQUIRED_ENVIRONMENT
    ):
        raise ServedModelError("worker.required_environment must be a bounded array")
    environment = tuple(
        _text(name, f"worker.required_environment[{index}]", maximum=256)
        for index, name in enumerate(raw_environment)
    )
    if len(environment) != len(set(environment)) or any(
        _ENVIRONMENT_NAME.fullmatch(name) is None for name in environment
    ):
        raise ServedModelError("worker.required_environment is invalid")

    identity_item = _mapping(item["identity"], "worker.identity")
    _exact_keys(identity_item, {"device", "execution_profile"}, "worker.identity")
    return WorkerContract(
        protocol=_text(item["protocol"], "worker.protocol", maximum=128),
        binary=binary,
        binary_sha256=binary_digest,
        arguments=arguments,
        required_environment=environment,
        identity=WorkerIdentity(
            device=_text(
                identity_item["device"], "worker.identity.device", maximum=128
            ),
            execution_profile=_text(
                identity_item["execution_profile"],
                "worker.identity.execution_profile",
                maximum=256,
            ),
        ),
    )


def _parse_product(value: Any, base: Path) -> ProductContract:
    item = _mapping(value, "product")
    _exact_keys(item, {"root", "artifact", "package"}, "product")
    root = _safe_directory(
        _resolve_root(base, _text(item["root"], "product.root", maximum=4096)),
        "product.root",
    )

    artifact: ArtifactIdentity | None
    if item["artifact"] is None:
        artifact = None
    else:
        artifact_item = _mapping(item["artifact"], "product.artifact")
        _exact_keys(
            artifact_item,
            {"manifest_path", "manifest_sha256", "content_sha256"},
            "product.artifact",
        )
        artifact_path = _relative_path(
            artifact_item["manifest_path"], "product.artifact.manifest_path"
        )
        artifact_digest = _sha256(
            artifact_item["manifest_sha256"], "product.artifact.manifest_sha256"
        )
        artifact_file = _contained_regular_file(
            root, artifact_path, "product artifact manifest"
        )
        _verify_file_sha256(artifact_file, artifact_digest, "product artifact manifest")
        artifact = ArtifactIdentity(
            artifact_path,
            artifact_digest,
            _sha256(artifact_item["content_sha256"], "product.artifact.content_sha256"),
        )

    package_item = _mapping(item["package"], "product.package")
    _exact_keys(package_item, {"manifest_path", "manifest_sha256"}, "product.package")
    package_path = _relative_path(
        package_item["manifest_path"], "product.package.manifest_path"
    )
    package_digest = _sha256(
        package_item["manifest_sha256"], "product.package.manifest_sha256"
    )
    package_file = _contained_regular_file(
        root, package_path, "product package manifest"
    )
    _verify_file_sha256(package_file, package_digest, "product package manifest")
    return ProductContract(
        root=root,
        artifact=artifact,
        package=PackageIdentity(package_path, package_digest),
    )


def _lineage_json(raw: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise _DuplicateKeyError(key)
            result[key] = item
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (_DuplicateKeyError, UnicodeError, json.JSONDecodeError) as error:
        raise ServedModelError(f"{label} JSON differs") from error
    if not isinstance(value, dict):
        raise ServedModelError(f"{label} must be an object")
    return value


def _live_lineage_file(path: Path, digest: str, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.resolve() != path:
        raise ServedModelError(f"{label} path must be canonical absolute")
    resolved = _safe_regular_file(path, label)
    metadata = resolved.stat(follow_symlinks=False)
    if stat.S_IMODE(metadata.st_mode) != 0o444 or metadata.st_nlink != 1:
        raise ServedModelError(f"{label} must be immutable single-link")
    _verify_file_sha256(resolved, digest, label)
    return _lineage_json(_bounded_read(resolved, MAX_MANIFEST_BYTES, label), label)


def _lineage_entries_sha(entries: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            entries, ensure_ascii=True, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _lineage_source(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "commit", "tree_oid", "archive_sha256"
    }:
        raise ServedModelError(f"{label} source differs")
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(value.get("commit", ""))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("tree_oid", ""))) is None
        or _SHA256.fullmatch(str(value.get("archive_sha256", ""))) is None
    ):
        raise ServedModelError(f"{label} source differs")
    return value


def _validate_lineage_v1_migration(
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    label = "promotion.authorization_lineage v1 predecessor"
    if (
        set(document) != {"schema_version", "disposition", "source", "entries"}
        or document.get("schema_version")
        != "ullm.sq8_authorization_lineage_input.v1"
        or document.get("disposition")
        != "authorization_input_not_yet_runtime_bound"
    ):
        raise ServedModelError(f"{label} differs")
    source_identity = _lineage_source(document.get("source"), label)
    entries = document.get("entries")
    relations = (
        "implementation_go_eligible_for_fresh_runtime_audit",
        "superseded_capture_implementation_no_go",
        "superseded_capture_implementation_no_go",
        "consumed_actual_failure_latest",
        "consumed_actual_failure_predecessor",
        "superseded_restore_implementation_no_go",
    )
    if not isinstance(entries, list) or len(entries) != len(relations):
        raise ServedModelError(f"{label} entry count differs")
    migrated_relations = (
        "historical_implementation_audit",
        "capture_implementation_no_go",
        "capture_implementation_no_go",
        "actual_failure",
        "actual_failure",
        "restore_implementation_no_go",
    )
    common = {
        "relation", "path", "sha256", "schema_version", "consumed",
        "reusable_as_runtime_authorization",
    }
    paths: set[str] = set()
    migrated: list[dict[str, Any]] = []
    for sequence, (entry, relation, migrated_relation) in enumerate(
        zip(entries, relations, migrated_relations, strict=True)
    ):
        if not isinstance(entry, dict) or entry.get("relation") != relation:
            raise ServedModelError(f"{label} relation/order differs")
        if sequence == 0:
            expected = common | {"verdict", "actual"}
        elif sequence in {1, 2}:
            expected = common | {"verdict", "actual", "reason_codes"}
        elif sequence in {3, 4}:
            expected = common | {"status", "actual_status", "request_id"}
        else:
            expected = common | {"verdict", "actual", "reason_code"}
        if (
            set(entry) != expected
            or entry.get("reusable_as_runtime_authorization") is not False
            or entry.get("consumed") is not (sequence != 0)
        ):
            raise ServedModelError(f"{label} entry shape differs")
        path_text = entry.get("path")
        digest = entry.get("sha256")
        if (
            not isinstance(path_text, str)
            or path_text in paths
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ServedModelError(f"{label} entry identity differs")
        paths.add(path_text)
        receipt = _live_lineage_file(Path(path_text), digest, f"{label} entry")
        schema = entry.get("schema_version")
        if receipt.get("schema_version") != schema:
            raise ServedModelError(f"{label} entry schema differs")
        if sequence == 0:
            authorization = receipt.get("authorization")
            if (
                schema
                != "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
                or entry.get("verdict") != "implementation_ready"
                or entry.get("actual") != "not_executed"
                or receipt.get("verdict") != entry["verdict"]
                or receipt.get("actual") != entry["actual"]
                or not isinstance(authorization, dict)
                or authorization.get("eligible_for_fresh_authorization_builder")
                is not True
            ):
                raise ServedModelError(f"{label} implementation GO differs")
        elif sequence in {1, 2}:
            reason_codes = entry.get("reason_codes")
            if (
                schema
                != "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
                or entry.get("verdict") != "implementation_no_go"
                or entry.get("actual") != "not_executed"
                or not isinstance(reason_codes, list)
                or not reason_codes
                or len(reason_codes) != len(set(reason_codes))
                or not all(isinstance(code, str) and code for code in reason_codes)
                or receipt.get("verdict") != entry["verdict"]
                or receipt.get("actual") != entry["actual"]
                or receipt.get("reason_codes") != reason_codes
            ):
                raise ServedModelError(f"{label} capture No-Go differs")
        elif sequence in {3, 4}:
            request_id = entry.get("request_id")
            actual = receipt.get("actual")
            if (
                schema != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
                or entry.get("status") != "actual_failed"
                or entry.get("actual_status") != "failed"
                or not isinstance(request_id, str)
                or re.fullmatch(r"sq8-promotion-[0-9a-f]{64}", request_id) is None
                or receipt.get("status") != entry["status"]
                or receipt.get("request_id") != request_id
                or not isinstance(actual, dict)
                or actual.get("status") != entry["actual_status"]
                or actual.get("request_id") != request_id
            ):
                raise ServedModelError(f"{label} actual failure differs")
        elif (
            schema != "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
            or entry.get("verdict") != "implementation_no_go"
            or entry.get("actual") != "not_executed"
            or entry.get("reason_code")
            != "restore_retry_terminal_identity_not_fail_closed"
            or receipt.get("verdict") != entry["verdict"]
            or receipt.get("actual") != entry["actual"]
            or receipt.get("reason_code") != entry["reason_code"]
        ):
            raise ServedModelError(f"{label} restore No-Go differs")
        audited = receipt.get("audited_source")
        source_commit = (
            receipt.get("source_commit")
            if schema == "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
            else audited.get("commit") if isinstance(audited, dict) else None
        )
        request_id = entry.get("request_id")
        if request_id is None:
            candidate = receipt.get("fixed_request_id")
            request_id = candidate if isinstance(candidate, str) else None
        status = entry.get("status", entry.get("verdict"))
        if (
            re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) is None
            or request_id is not None
            and re.fullmatch(r"sq8-promotion-[0-9a-f]{64}", request_id) is None
        ):
            raise ServedModelError(f"{label} migrated identity differs")
        migrated.append(
            {
                "sequence": sequence,
                "relation": migrated_relation,
                "path": path_text,
                "sha256": digest,
                "schema_version": schema,
                "status": status,
                "request_id": request_id,
                "source_commit": source_commit,
            }
        )
    return migrated, source_identity


def _validate_lineage_v2_document(
    document: dict[str, Any], *, seen: frozenset[Path] = frozenset()
) -> tuple[str, int, dict[str, str]]:
    if set(document) != {
        "schema_version", "disposition", "source", "predecessor", "entries"
    } or document.get("schema_version") != "ullm.sq8_authorization_lineage_input.v2":
        raise ServedModelError("promotion.authorization_lineage v2 manifest differs")
    source_identity = _lineage_source(
        document.get("source"), "promotion.authorization_lineage v2"
    )
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ServedModelError("promotion.authorization_lineage v2 entries differ")
    entry_keys = {
        "sequence", "relation", "path", "sha256", "schema_version", "status",
        "request_id", "source_commit",
    }
    allowed = {
        "implementation_ready_current", "capture_implementation_no_go",
        "restore_implementation_no_go", "actual_failure",
        "historical_implementation_audit", "historical_runtime_audit",
    }
    paths: set[str] = set()
    digests: set[str] = set()
    counts = {
        "implementation_ready_current": 0,
        "capture_implementation_no_go": 0,
        "restore_implementation_no_go": 0,
        "actual_failure": 0,
    }
    current: dict[str, str] | None = None
    entry_documents: list[dict[str, Any]] = []
    for sequence, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or set(entry) != entry_keys
            or entry.get("sequence") != sequence
            or entry.get("relation") not in allowed
        ):
            raise ServedModelError("promotion.authorization_lineage v2 entry differs")
        path_text = entry.get("path")
        digest = entry.get("sha256")
        if (
            not isinstance(path_text, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or path_text in paths
            or digest in digests
            or re.fullmatch(r"[0-9a-f]{40}", str(entry.get("source_commit", ""))) is None
        ):
            raise ServedModelError("promotion.authorization_lineage v2 entry identity differs")
        paths.add(path_text)
        digests.add(digest)
        receipt = _live_lineage_file(
            Path(path_text), digest, "promotion.authorization_lineage entry"
        )
        entry_documents.append(receipt)
        if receipt.get("schema_version") != entry.get("schema_version"):
            raise ServedModelError("promotion.authorization_lineage entry schema differs")
        relation = entry["relation"]
        status = entry.get("status")
        request_id = entry.get("request_id")
        if relation == "actual_failure":
            actual = receipt.get("actual")
            observed_commit = receipt.get("source_commit")
            if (
                entry["schema_version"] != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
                or status != "actual_failed"
                or not isinstance(request_id, str)
                or re.fullmatch(r"sq8-promotion-[0-9a-f]{64}", request_id) is None
                or receipt.get("status") != status
                or receipt.get("request_id") != request_id
                or not isinstance(actual, dict)
                or actual.get("status") != "failed"
                or actual.get("request_id") != request_id
            ):
                raise ServedModelError("promotion.authorization_lineage actual failure differs")
        else:
            audited = receipt.get("audited_source")
            observed_commit = audited.get("commit") if isinstance(audited, dict) else None
            if (
                request_id is not None
                and (
                    not isinstance(request_id, str)
                    or re.fullmatch(r"sq8-promotion-[0-9a-f]{64}", request_id)
                    is None
                    or receipt.get("fixed_request_id") != request_id
                )
            ) or receipt.get("verdict") != status or receipt.get("actual") != "not_executed":
                raise ServedModelError("promotion.authorization_lineage audit entry differs")
        if observed_commit != entry["source_commit"]:
            raise ServedModelError("promotion.authorization_lineage entry source differs")
        if relation in counts:
            counts[relation] += 1
        if relation == "implementation_ready_current":
            authorization = receipt.get("authorization")
            if (
                status != "implementation_ready"
                or entry["source_commit"] != source_identity["commit"]
                or entry["schema_version"]
                not in {
                    "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1",
                    "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
                }
                or (
                    entry["schema_version"]
                    == "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
                    and (
                        not isinstance(authorization, dict)
                        or authorization.get(
                            "eligible_for_fresh_authorization_builder"
                        )
                        is not True
                    )
                )
            ):
                raise ServedModelError("promotion.authorization_lineage current GO differs")
            current = {"path": path_text, "sha256": digest}
        elif relation == "capture_implementation_no_go" and (
            entry["schema_version"]
            != "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
            or status != "implementation_no_go"
        ):
            raise ServedModelError("promotion.authorization_lineage capture No-Go differs")
        elif relation == "restore_implementation_no_go" and (
            entry["schema_version"]
            != "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
            or status != "implementation_no_go"
            or receipt.get("reason_code")
            != "restore_retry_terminal_identity_not_fail_closed"
        ):
            raise ServedModelError("promotion.authorization_lineage restore No-Go differs")
        elif relation == "historical_implementation_audit" and (
            entry["schema_version"]
            != "ullm.qwen35_aq4_sq8_overlay_capture_failure_independent_audit.v1"
            or status not in {"implementation_ready", "implementation_no_go"}
        ):
            raise ServedModelError(
                "promotion.authorization_lineage historical implementation audit differs"
            )
        elif relation == "historical_runtime_audit" and (
            entry["schema_version"]
            != "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
            or status not in {"implementation_ready", "implementation_no_go"}
        ):
            raise ServedModelError(
                "promotion.authorization_lineage historical runtime audit differs"
            )
    if (
        counts["implementation_ready_current"] != 1
        or counts["capture_implementation_no_go"] < 2
        or counts["restore_implementation_no_go"] < 1
        or counts["actual_failure"] < 3
        or current is None
    ):
        raise ServedModelError("promotion.authorization_lineage minimum history differs")
    predecessor = document.get("predecessor")
    if not isinstance(predecessor, dict) or predecessor.get("schema_version") not in {
        "ullm.sq8_authorization_lineage_input.v1",
        "ullm.sq8_authorization_lineage_input.v2",
    }:
        raise ServedModelError("promotion.authorization_lineage predecessor differs")
    predecessor_schema = predecessor["schema_version"]
    if predecessor_schema == "ullm.sq8_authorization_lineage_input.v1":
        if set(predecessor) != {
            "schema_version", "path", "sha256", "migrated_prefix_sha256",
            "migrated_prefix_count",
        }:
            raise ServedModelError("promotion.authorization_lineage predecessor differs")
        predecessor_path = Path(str(predecessor.get("path", "")))
        if predecessor_path in seen:
            raise ServedModelError("promotion.authorization_lineage predecessor cycle differs")
        previous_document = _live_lineage_file(
            predecessor_path,
            _sha256(predecessor.get("sha256"), "lineage predecessor SHA"),
            "promotion.authorization_lineage predecessor",
        )
        migrated, previous_source = _validate_lineage_v1_migration(
            previous_document
        )
        if (
            predecessor.get("migrated_prefix_sha256")
            != _lineage_entries_sha(migrated)
            or predecessor.get("migrated_prefix_count") != len(migrated)
            or len(entries) != len(migrated) + 2
            or entries[:len(migrated)] != migrated
            or entries[6]["relation"] != "actual_failure"
            or entries[6]["source_commit"] != previous_source["commit"]
            or entry_documents[6].get("source_provenance")
            != {
                "tree_sha256": previous_source["tree_oid"],
                "archive_sha256": previous_source["archive_sha256"],
            }
            or entries[7]["relation"] != "implementation_ready_current"
        ):
            raise ServedModelError(
                "promotion.authorization_lineage v1 migration differs"
            )
    else:
        if set(predecessor) != {
            "schema_version", "path", "sha256", "entries_sha256", "entry_count"
        }:
            raise ServedModelError("promotion.authorization_lineage predecessor differs")
        predecessor_path = Path(str(predecessor.get("path", "")))
        if predecessor_path in seen:
            raise ServedModelError("promotion.authorization_lineage predecessor cycle differs")
        previous_document = _live_lineage_file(
            predecessor_path,
            _sha256(predecessor.get("sha256"), "lineage predecessor SHA"),
            "promotion.authorization_lineage predecessor",
        )
        previous_entries_sha, previous_count, _ = _validate_lineage_v2_document(
            previous_document, seen=seen | {predecessor_path}
        )
        if (
            predecessor.get("entries_sha256") != previous_entries_sha
            or predecessor.get("entry_count") != previous_count
            or len(entries) <= previous_count
            or entries[:previous_count] != previous_document["entries"]
        ):
            raise ServedModelError("promotion.authorization_lineage is not append-only")
    entries_sha = _lineage_entries_sha(entries)
    return entries_sha, len(entries), current


def _parse_promotion(value: Any, base: Path) -> PromotionContract:
    item = _mapping(value, "promotion")
    expected = {"source_commit", "receipt", "receipt_sha256"}
    optional = {"authorization_audit", "authorization_lineage", "readiness"}
    if not expected.issubset(item) or not set(item).issubset(expected | optional):
        raise ServedModelError("promotion field set differs")
    receipt = _safe_regular_file(
        _resolve_root(base, _text(item["receipt"], "promotion.receipt", maximum=4096)),
        "promotion.receipt",
    )
    digest = _sha256(item["receipt_sha256"], "promotion.receipt_sha256")
    _verify_file_sha256(receipt, digest, "promotion.receipt")
    authorization_audit: AuthorizationAuditIdentity | None = None
    if "authorization_audit" in item and item["authorization_audit"] is not None:
        audit_item = _mapping(item["authorization_audit"], "promotion.authorization_audit")
        _exact_keys(audit_item, {"path", "sha256"}, "promotion.authorization_audit")
        raw_audit_path = _text(
            audit_item["path"], "promotion.authorization_audit.path", maximum=4096
        )
        audit_path = Path(raw_audit_path)
        if not audit_path.is_absolute() or audit_path.resolve() != audit_path:
            raise ServedModelError(
                "promotion.authorization_audit.path must be a canonical absolute path"
            )
        audit_path = _safe_regular_file(
            audit_path, "promotion.authorization_audit.path"
        )
        audit_digest = _sha256(
            audit_item["sha256"], "promotion.authorization_audit.sha256"
        )
        _verify_file_sha256(
            audit_path, audit_digest, "promotion.authorization_audit"
        )
        authorization_audit = AuthorizationAuditIdentity(audit_path, audit_digest)
    authorization_lineage: AuthorizationLineageIdentity | None = None
    if "authorization_lineage" in item and item["authorization_lineage"] is not None:
        lineage_item = _mapping(
            item["authorization_lineage"], "promotion.authorization_lineage"
        )
        lineage_schema = lineage_item.get("schema_version")
        lineage_keys = {
            "schema_version", "input_path", "runtime_path", "sha256", "entries_sha256"
        }
        if lineage_schema == "ullm.sq8_authorization_lineage_ref.v2":
            lineage_keys |= {"entry_count", "current_implementation_audit"}
        _exact_keys(
            lineage_item,
            lineage_keys,
            "promotion.authorization_lineage",
        )
        if lineage_schema not in {
            "ullm.sq8_authorization_lineage_ref.v1",
            "ullm.sq8_authorization_lineage_ref.v2",
        }:
            raise ServedModelError("promotion.authorization_lineage schema differs")
        lineage_paths = []
        for name in ("input_path", "runtime_path"):
            raw = _text(
                lineage_item[name], f"promotion.authorization_lineage.{name}",
                maximum=4096,
            )
            path = Path(raw)
            if not path.is_absolute() or path.resolve() != path:
                raise ServedModelError(
                    f"promotion.authorization_lineage.{name} must be a canonical absolute path"
                )
            resolved = _safe_regular_file(
                path, f"promotion.authorization_lineage.{name}"
            )
            metadata = resolved.stat(follow_symlinks=False)
            if stat.S_IMODE(metadata.st_mode) != 0o444 or metadata.st_nlink != 1:
                raise ServedModelError(
                    f"promotion.authorization_lineage.{name} must be immutable single-link"
                )
            lineage_paths.append(resolved)
        lineage_digest = _sha256(
            lineage_item["sha256"], "promotion.authorization_lineage.sha256"
        )
        entries_digest = _sha256(
            lineage_item["entries_sha256"],
            "promotion.authorization_lineage.entries_sha256",
        )
        raw_manifest = None
        for path in lineage_paths:
            _verify_file_sha256(path, lineage_digest, "promotion.authorization_lineage")
            raw_manifest = _bounded_read(
                path, MAX_MANIFEST_BYTES, "promotion.authorization_lineage"
            )
        try:
            assert raw_manifest is not None
            lineage_document = _lineage_json(
                raw_manifest, "promotion.authorization_lineage"
            )
            if lineage_schema == "ullm.sq8_authorization_lineage_ref.v1":
                if (
                    set(lineage_document) != {
                        "schema_version", "disposition", "source", "entries"
                    }
                    or lineage_document["schema_version"]
                    != "ullm.sq8_authorization_lineage_input.v1"
                    or not isinstance(lineage_document["entries"], list)
                    or len(lineage_document["entries"]) != 6
                ):
                    raise ValueError("manifest schema")
                observed_entries_digest = hashlib.sha256(
                    json.dumps(
                        lineage_document["entries"], ensure_ascii=True,
                        allow_nan=False, separators=(",", ":"), sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
                entry_count = None
                current_identity = None
            else:
                observed_entries_digest, observed_count, observed_current = (
                    _validate_lineage_v2_document(lineage_document)
                )
                entry_count = lineage_item.get("entry_count")
                if (
                    not isinstance(entry_count, int)
                    or isinstance(entry_count, bool)
                    or entry_count != observed_count
                ):
                    raise ValueError("entry count")
                current_item = _mapping(
                    lineage_item.get("current_implementation_audit"),
                    "promotion.authorization_lineage.current_implementation_audit",
                )
                _exact_keys(
                    current_item, {"path", "sha256"},
                    "promotion.authorization_lineage.current_implementation_audit",
                )
                if current_item != observed_current:
                    raise ValueError("current implementation audit")
                current_identity = AuthorizationAuditIdentity(
                    Path(current_item["path"]), current_item["sha256"]
                )
        except (KeyError, TypeError, ValueError, UnicodeError, ServedModelError) as error:
            raise ServedModelError(
                "promotion.authorization_lineage manifest differs"
            ) from error
        if observed_entries_digest != entries_digest:
            raise ServedModelError(
                "promotion.authorization_lineage entries SHA-256 differs"
            )
        authorization_lineage = AuthorizationLineageIdentity(
            lineage_paths[0], lineage_paths[1], lineage_digest, entries_digest,
            schema_version=str(lineage_schema), entry_count=entry_count,
            current_implementation_audit=current_identity,
        )
    readiness: ReadinessIdentity | None = None
    if "readiness" in item and item["readiness"] is not None:
        readiness = _parse_readiness(item["readiness"])
    return PromotionContract(
        source_commit=_text(
            item["source_commit"], "promotion.source_commit", maximum=256
        ),
        receipt=receipt,
        receipt_sha256=digest,
        authorization_audit=authorization_audit,
        authorization_lineage=authorization_lineage,
        readiness=readiness,
    )


def _parse_readiness(value: Any) -> ReadinessIdentity:
    item = _mapping(value, "promotion.readiness")
    _exact_keys(
        item, {"schema", "container", "network", "endpoint"},
        "promotion.readiness",
    )
    if item["schema"] != "ullm.bridge_container_readiness.v1":
        raise ServedModelError("promotion.readiness schema differs")
    container = _mapping(item["container"], "promotion.readiness.container")
    _exact_keys(
        container, {"name", "id", "image_id", "config_image"},
        "promotion.readiness.container",
    )
    network = _mapping(item["network"], "promotion.readiness.network")
    _exact_keys(
        network, {"name", "id", "driver", "bridge_interface"},
        "promotion.readiness.network",
    )
    endpoint = _mapping(item["endpoint"], "promotion.readiness.endpoint")
    _exact_keys(
        endpoint,
        {"url", "path", "expected_status", "expected_body", "expected_body_sha256", "timeout_seconds"},
        "promotion.readiness.endpoint",
    )
    container_id = _text(container["id"], "promotion.readiness.container.id", maximum=64)
    image_id = _text(container["image_id"], "promotion.readiness.container.image_id", maximum=71)
    network_id = _text(network["id"], "promotion.readiness.network.id", maximum=64)
    body = _text(endpoint["expected_body"], "promotion.readiness.endpoint.expected_body", maximum=256)
    body_sha256 = _sha256(
        endpoint["expected_body_sha256"], "promotion.readiness.endpoint.expected_body_sha256"
    )
    expected = {
        "container_name": "open-webui",
        "network_driver": "bridge",
        "url": "http://172.20.0.1:8000/readyz",
        "path": "/readyz",
        "expected_status": 200,
        "expected_body": '{"status":"ready"}',
        "timeout_seconds": 5,
    }
    if (
        _text(container["name"], "promotion.readiness.container.name", maximum=256) != expected["container_name"]
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        or not _text(container["config_image"], "promotion.readiness.container.config_image", maximum=512)
        or re.fullmatch(r"[0-9a-f]{64}", network_id) is None
        or _text(network["driver"], "promotion.readiness.network.driver", maximum=64) != expected["network_driver"]
        or _text(network["bridge_interface"], "promotion.readiness.network.bridge_interface", maximum=64) != f"br-{network_id[:12]}"
        or _text(endpoint["url"], "promotion.readiness.endpoint.url", maximum=512) != expected["url"]
        or _text(endpoint["path"], "promotion.readiness.endpoint.path", maximum=256) != expected["path"]
        or endpoint["expected_status"] != expected["expected_status"]
        or body != expected["expected_body"]
        or hashlib.sha256(body.encode("ascii")).hexdigest() != body_sha256
        or endpoint["timeout_seconds"] != expected["timeout_seconds"]
    ):
        raise ServedModelError("promotion.readiness identity differs")
    return ReadinessIdentity(
        container_name=expected["container_name"],
        container_id=container_id,
        image_id=image_id,
        config_image=container["config_image"],
        network_name=_text(network["name"], "promotion.readiness.network.name", maximum=256),
        network_id=network_id,
        network_driver=expected["network_driver"],
        bridge_interface=network["bridge_interface"],
        url=expected["url"],
        path=expected["path"],
        expected_status=expected["expected_status"],
        expected_body=body,
        expected_body_sha256=body_sha256,
        timeout_seconds=expected["timeout_seconds"],
    )


def _decode_document(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ServedModelError("manifest is not valid UTF-8") from error

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKeyError(key)
            result[key] = value
        return result

    def reject_constant(_: str) -> Any:
        raise ValueError("non-finite JSON number")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        _DuplicateKeyError,
        ValueError,
        RecursionError,
    ) as error:
        raise ServedModelError("manifest is not strict JSON") from error
    _validate_json_bounds(value)
    return _mapping(value, "manifest")


def _validate_json_bounds(root: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ServedModelError("manifest JSON structure exceeds bounds")
        if isinstance(value, str):
            if len(value.encode("utf-8")) > MAX_STRING_BYTES:
                raise ServedModelError("manifest JSON string exceeds bounds")
        elif isinstance(value, dict):
            if len(value) > MAX_JSON_NODES:
                raise ServedModelError("manifest JSON object exceeds bounds")
            for key, item in value.items():
                if len(key.encode("utf-8")) > MAX_STRING_BYTES:
                    raise ServedModelError("manifest JSON key exceeds bounds")
                stack.append((item, depth + 1))
        elif isinstance(value, list):
            if len(value) > MAX_JSON_NODES:
                raise ServedModelError("manifest JSON array exceeds bounds")
            stack.extend((item, depth + 1) for item in value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ServedModelError(f"{label} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ServedModelError(f"{label} field set differs")


def _text(value: Any, label: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ServedModelError(f"{label} must be bounded nonempty text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ServedModelError(f"{label} must be a boolean")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ServedModelError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ServedModelError(f"{label} must be a nonnegative integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ServedModelError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolve_root(base: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return base / _relative_path(raw, "resource root")


def _relative_path(value: Any, label: str) -> str:
    raw = _text(value, label, maximum=4096)
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ServedModelError(f"{label} must be a contained relative path")
    return path.as_posix()


def _safe_directory(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    try:
        metadata = path.stat()
    except OSError as error:
        raise ServedModelError(f"{label} is absent or unreadable") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & stat.S_IWOTH:
        raise ServedModelError(f"{label} is not a safe directory")
    return path.resolve(strict=True)


def _safe_regular_file(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    try:
        metadata = path.stat()
    except OSError as error:
        raise ServedModelError(f"{label} is absent or unreadable") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & stat.S_IWOTH:
        raise ServedModelError(f"{label} is not a safe regular file")
    return path.resolve(strict=True)


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ServedModelError(f"{label} traverses a symlink")
        except FileNotFoundError as error:
            raise ServedModelError(f"{label} is absent") from error
        except OSError as error:
            raise ServedModelError(f"{label} is unreadable") from error


def _contained_regular_file(root: Path, relative: str, label: str) -> Path:
    target = _safe_regular_file(root / relative, label)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ServedModelError(f"{label} escapes its root") from error
    return target


def _bounded_read(path: Path, maximum: int, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            value = handle.read(maximum + 1)
    except OSError as error:
        raise ServedModelError(f"{label} is unreadable") from error
    if len(value) > maximum:
        raise ServedModelError(f"{label} exceeds its size limit")
    return value


def _verify_file_sha256(path: Path, expected: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ServedModelError(f"{label} is unreadable") from error
    if digest.hexdigest() != expected:
        raise ServedModelError(f"{label} SHA-256 differs")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
