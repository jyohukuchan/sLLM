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
class ProductContract:
    root: Path
    artifact: ArtifactIdentity | None
    package: PackageIdentity


@dataclass(frozen=True, slots=True)
class PromotionContract:
    source_commit: str
    receipt: Path
    receipt_sha256: str


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
    product = _parse_product(document["product"], manifest_path.parent)
    promotion = _parse_promotion(document["promotion"], manifest_path.parent)
    reasoning_dialect = (
        _parse_reasoning(document["reasoning"], generation.vocab_size)
        if schema_version == SCHEMA_VERSION_V2
        else None
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


def _parse_promotion(value: Any, base: Path) -> PromotionContract:
    item = _mapping(value, "promotion")
    _exact_keys(item, {"source_commit", "receipt", "receipt_sha256"}, "promotion")
    receipt = _safe_regular_file(
        _resolve_root(base, _text(item["receipt"], "promotion.receipt", maximum=4096)),
        "promotion.receipt",
    )
    digest = _sha256(item["receipt_sha256"], "promotion.receipt_sha256")
    _verify_file_sha256(receipt, digest, "promotion.receipt")
    return PromotionContract(
        source_commit=_text(
            item["source_commit"], "promotion.source_commit", maximum=256
        ),
        receipt=receipt,
        receipt_sha256=digest,
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
