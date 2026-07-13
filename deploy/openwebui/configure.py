#!/usr/bin/env python3
"""Add the uLLM provider to an existing OpenWebUI SQLite database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_URL = "http://172.20.0.1:8000/v1"
MODEL_ID = "ullm-qwen3-14b-sq8"
MODEL_NAME = "uLLM Qwen3 14B SQ8"
CONTEXT_LENGTH = 4_096
MODEL_DESCRIPTION = "Qwen3 14B served locally by uLLM SQ8_0."
MAX_KEY_BYTES = 65_536
MAX_MANIFEST_BYTES = 1_048_576
SERVED_MODEL_SCHEMA_V1 = "ullm.served_model.v1"
SERVED_MODEL_SCHEMA_V2 = "ullm.served_model.v2"
SERVED_MODEL_KEYS_V1 = {
    "schema_version",
    "public",
    "generation",
    "format",
    "tokenizer",
    "worker",
    "product",
    "promotion",
}
SERVED_MODEL_KEYS_V2 = SERVED_MODEL_KEYS_V1 | {"reasoning"}
REASONING_KEYS = {
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
}
PUBLIC_MODEL_KEYS = {
    "id",
    "name",
    "description",
    "upstream_id",
    "revision",
    "context_length",
}


class ConfigurationError(RuntimeError):
    """Raised when the existing OpenWebUI state is unsafe to update."""


class DuplicateKeyError(ValueError):
    """Raised when a manifest JSON object repeats a key."""


def read_served_model_manifest(path: Path) -> tuple[str, str, int, str, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ConfigurationError(
            "served-model manifest is absent or unreadable"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ConfigurationError("served-model manifest path is not a regular file")
        if metadata.st_mode & stat.S_IWOTH:
            raise ConfigurationError("served-model manifest is world-writable")
        if metadata.st_size > MAX_MANIFEST_BYTES:
            raise ConfigurationError("served-model manifest has an invalid size")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    finally:
        os.close(descriptor)
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise ConfigurationError("served-model manifest has an invalid size")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateKeyError) as error:
        raise ConfigurationError("served-model manifest is not valid JSON") from error
    document = require_mapping(document, "served-model manifest")
    schema_version = document.get("schema_version")
    if schema_version == SERVED_MODEL_SCHEMA_V1:
        require_exact_keys(document, SERVED_MODEL_KEYS_V1, "served-model manifest")
    elif schema_version == SERVED_MODEL_SCHEMA_V2:
        require_exact_keys(document, SERVED_MODEL_KEYS_V2, "served-model manifest")
        validate_reasoning_manifest(document.get("reasoning"))
    else:
        raise ConfigurationError("served-model manifest schema differs")
    public = require_mapping(document.get("public"), "served-model manifest public")
    require_exact_keys(public, PUBLIC_MODEL_KEYS, "served-model manifest public")
    model_id = require_text_field(public, "id", 256)
    model_name = require_text_field(public, "name", 512)
    description = require_text_field(public, "description", 4_096)
    context_length = public.get("context_length")
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise ConfigurationError(
            "served-model manifest public.context_length must be an integer"
        )
    require_context_length(context_length, "served-model manifest context length")
    return (
        model_id,
        model_name,
        context_length,
        description,
        hashlib.sha256(raw).hexdigest(),
    )


def validate_reasoning_manifest(value: Any) -> None:
    reasoning = require_mapping(value, "served-model manifest reasoning")
    require_exact_keys(reasoning, REASONING_KEYS, "served-model manifest reasoning")
    dialect_id = reasoning.get("dialect_id")
    if not isinstance(dialect_id, str) or not dialect_id.strip():
        raise ConfigurationError("reasoning dialect id must be nonempty text")
    for name in ("start_token_ids", "end_token_ids", "forced_end_token_ids"):
        values = reasoning.get(name)
        if not isinstance(values, list) or not values or any(
            type(token) is not int or token < 0 for token in values
        ):
            raise ConfigurationError(f"reasoning {name} must be a nonempty integer list")
    effort_budgets = require_mapping(
        reasoning.get("effort_budgets"), "reasoning effort budgets"
    )
    require_exact_keys(
        effort_budgets, {"low", "medium", "high"}, "reasoning effort budgets"
    )
    if any(
        type(budget) is not int or budget < 1 for budget in effort_budgets.values()
    ):
        raise ConfigurationError("reasoning effort budgets must be positive integers")
    for name in ("max_budget_tokens", "reserved_answer_tokens"):
        value = reasoning.get(name)
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"reasoning {name} must be a positive integer")
    if reasoning.get("enabled_by_default") not in (True, False):
        raise ConfigurationError("reasoning enabled_by_default must be boolean")
    if reasoning.get("initial_phase") not in {"reasoning", "answer"}:
        raise ConfigurationError("reasoning initial_phase is invalid")
    if reasoning.get("eos_policy") not in {"close", "finish", "continue"}:
        raise ConfigurationError("reasoning eos_policy is invalid")
    if reasoning.get("history_reasoning_policy") not in {"omit", "preserve"}:
        raise ConfigurationError("reasoning history policy is invalid")


def parse_model_id_list(raw: str, label: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{label} must be a JSON string array") from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{label} must be a JSON string array")
    return [require_model_id(item, label) for item in value]


def read_api_key(path: Path) -> str:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ConfigurationError("API key path is not a regular file")
    value = path.read_bytes()
    if len(value) > MAX_KEY_BYTES:
        raise ConfigurationError("API key is too large")
    if value.endswith(b"\r\n"):
        value = value[:-2]
    elif value.endswith(b"\n"):
        value = value[:-1]
    if not value or b"\n" in value or b"\r" in value:
        raise ConfigurationError("API key file must contain one nonempty line")
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as error:
        raise ConfigurationError("API key must be ASCII") from error


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object")
    return value


def require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ConfigurationError(f"{label} fields differ")


def configure_provider(
    data: dict[str, Any], api_key: str, base_url: str = BASE_URL
) -> int:
    base_url = require_nonempty(base_url, "OpenAI base URL")
    openai = require_mapping(data.setdefault("openai", {}), "openai")
    base_urls = openai.setdefault("api_base_urls", [])
    api_keys = openai.setdefault("api_keys", [])
    api_configs = require_mapping(
        openai.setdefault("api_configs", {}), "openai.api_configs"
    )
    if not isinstance(base_urls, list) or not all(
        isinstance(value, str) for value in base_urls
    ):
        raise ConfigurationError("openai.api_base_urls must be a string list")
    if not isinstance(api_keys, list) or not all(
        isinstance(value, str) for value in api_keys
    ):
        raise ConfigurationError("openai.api_keys must be a string list")

    try:
        provider_index = base_urls.index(base_url)
    except ValueError:
        provider_index = len(base_urls)
        base_urls.append(base_url)

    while len(api_keys) <= provider_index:
        api_keys.append("")
    api_keys[provider_index] = api_key
    api_configs[str(provider_index)] = {
        "enable": True,
        "connection_type": "local",
        "auth_type": "bearer",
        "model_ids": [],
        "prefix_id": "",
        "tags": [],
    }
    openai["enable"] = True

    task = require_mapping(data.setdefault("task", {}), "task")
    for task_name in ("follow_up", "tags", "title"):
        task_config = require_mapping(
            task.setdefault(task_name, {}), f"task.{task_name}"
        )
        task_config["enable"] = False

    return provider_index


def parse_json_object(raw: Any, label: str) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"{label} is not valid JSON") from error
    return require_mapping(raw, label)


def require_nonempty(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ConfigurationError(f"{label} must not be empty")
    return value


def require_model_id(value: str, label: str = "model id") -> str:
    value = require_nonempty(value, label)
    if len(value.encode("utf-8")) > 256 or any(
        ord(character) < 0x20 for character in value
    ):
        raise ConfigurationError(f"{label} must be bounded text")
    return value


def require_text_field(value: Mapping[str, Any], name: str, maximum_bytes: int) -> str:
    field = value.get(name)
    if not isinstance(field, str):
        raise ConfigurationError(
            f"served-model manifest public.{name} must be a string"
        )
    if (
        not field
        or len(field.encode("utf-8")) > maximum_bytes
        or any(ord(character) < 0x20 for character in field)
    ):
        raise ConfigurationError(
            f"served-model manifest public.{name} must be bounded nonempty text"
        )
    return field


def require_context_length(value: int, label: str = "context length") -> int:
    if value <= 0:
        raise ConfigurationError(f"{label} must be greater than zero")
    return value


def model_values(
    connection: sqlite3.Connection,
    model_id: str = MODEL_ID,
    context_length: int = CONTEXT_LENGTH,
    description: str = MODEL_DESCRIPTION,
    base_url: str = BASE_URL,
    manifest_sha256: str | None = None,
) -> tuple[str, str, str]:
    context_length = require_context_length(context_length)
    base_url = require_nonempty(base_url, "OpenAI base URL")
    owner = connection.execute(
        "SELECT id FROM user ORDER BY CASE role WHEN 'admin' THEN 0 ELSE 1 END, created_at LIMIT 1"
    ).fetchone()
    if owner is None or not isinstance(owner[0], str) or not owner[0]:
        raise ConfigurationError("OpenWebUI has no model owner")

    existing = connection.execute(
        "SELECT meta, params FROM model WHERE id = ?", (model_id,)
    ).fetchone()
    meta = parse_json_object(existing[0], "model.meta") if existing else {}
    params = parse_json_object(existing[1], "model.params") if existing else {}

    capabilities = require_mapping(
        meta.setdefault("capabilities", {}), "model capabilities"
    )
    for capability in (
        "builtin_tools",
        "citations",
        "code_interpreter",
        "file_context",
        "file_upload",
        "image_generation",
        "status_updates",
        "vision",
        "web_search",
    ):
        capabilities[capability] = False
    capabilities["usage"] = True
    ullm = require_mapping(meta.setdefault("ullm", {}), "model uLLM metadata")
    ullm.update(
        {
            "managed": True,
            "base_url": base_url,
            "served_model_manifest_sha256": manifest_sha256,
        }
    )
    meta.update(
        {
            "profile_image_url": "/static/favicon.png",
            "description": description,
            "n_ctx_train": context_length,
            "context_length": context_length,
        }
    )
    # Context length is metadata only in OpenWebUI v0.9.4. Do not set num_ctx,
    # which its OpenAI adapter forwards as an unsupported upstream field.
    params.pop("num_ctx", None)
    params.pop("max_tokens", None)
    return (
        owner[0],
        json.dumps(meta, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        json.dumps(params, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    )


def managed_models_at_base_url(
    connection: sqlite3.Connection, base_url: str, active_model_id: str
) -> dict[str, str | None]:
    managed: dict[str, str | None] = {}
    for model_id, raw_meta in connection.execute(
        "SELECT id, meta FROM model WHERE id != ? AND is_active = 1",
        (active_model_id,),
    ):
        try:
            meta = parse_json_object(raw_meta, "managed model.meta")
        except ConfigurationError:
            continue
        marker = meta.get("ullm")
        if (
            isinstance(marker, dict)
            and marker.get("managed") is True
            and marker.get("base_url") == base_url
        ):
            manifest_sha256 = marker.get("served_model_manifest_sha256")
            managed[model_id] = (
                manifest_sha256
                if isinstance(manifest_sha256, str) and len(manifest_sha256) == 64
                else None
            )
    return managed


def previous_managed_model_updates(
    connection: sqlite3.Connection,
    explicit_previous_models: Mapping[str, str | None],
    base_url: str,
    active_model_id: str,
) -> list[tuple[str, str]]:
    if active_model_id in explicit_previous_models:
        raise ConfigurationError(
            "active model id cannot also be a previous managed model id"
        )
    previous_models = managed_models_at_base_url(connection, base_url, active_model_id)
    previous_models.update(explicit_previous_models)
    updates: list[tuple[str, str]] = []
    for model_id, manifest_sha256 in sorted(previous_models.items()):
        row = connection.execute(
            "SELECT meta FROM model WHERE id = ?", (model_id,)
        ).fetchone()
        if row is None:
            continue
        meta = parse_json_object(row[0], f"previous managed model {model_id!r} meta")
        marker = require_mapping(
            meta.setdefault("ullm", {}),
            f"previous managed model {model_id!r} uLLM metadata",
        )
        if manifest_sha256 is None:
            existing_sha256 = marker.get("served_model_manifest_sha256")
            if not isinstance(existing_sha256, str) or len(existing_sha256) != 64:
                existing_sha256 = None
            manifest_sha256 = existing_sha256
        marker.update(
            {
                "managed": True,
                "base_url": base_url,
                "served_model_manifest_sha256": manifest_sha256,
            }
        )
        updates.append(
            (
                model_id,
                json.dumps(
                    meta,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )
    return updates


def collect_previous_models(
    previous_managed_model_ids: Sequence[str],
    previous_served_model_manifests: Sequence[Path],
) -> dict[str, str | None]:
    previous_models: dict[str, str | None] = {}

    def add(model_id: str, manifest_sha256: str | None) -> None:
        if model_id in previous_models:
            raise ConfigurationError(f"previous model id {model_id!r} is duplicated")
        previous_models[model_id] = manifest_sha256

    for previous_model_id in previous_managed_model_ids:
        add(require_model_id(previous_model_id, "previous model id"), None)
    for previous_manifest in previous_served_model_manifests:
        previous_model_id, _, _, _, previous_manifest_sha256 = (
            read_served_model_manifest(previous_manifest)
        )
        add(previous_model_id, previous_manifest_sha256)
    return previous_models


def backup_database(connection: sqlite3.Connection, backup_dir: Path) -> Path:
    backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(backup_dir, 0o700)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_path = backup_dir / f"webui-before-ullm-{stamp}-{time.time_ns()}.db"
    destination = sqlite3.connect(backup_path)
    try:
        connection.backup(destination)
    finally:
        destination.close()
    os.chmod(backup_path, 0o600)
    return backup_path


def configure(
    database: Path,
    key_file: Path,
    backup_dir: Path,
    model_id: str | None = None,
    model_name: str | None = None,
    context_length: int | None = None,
    description: str | None = None,
    base_url: str = BASE_URL,
    served_model_manifest: Path | None = None,
    previous_managed_model_ids: Sequence[str] = (),
    previous_served_model_manifests: Sequence[Path] = (),
) -> tuple[int, Path]:
    if not database.is_file():
        raise ConfigurationError("OpenWebUI database does not exist")
    legacy_values = (model_id, model_name, context_length, description)
    manifest_sha256 = None
    if served_model_manifest is not None:
        if any(value is not None for value in legacy_values):
            raise ConfigurationError(
                "served-model manifest mode cannot be mixed with legacy model settings"
            )
        (
            model_id,
            model_name,
            context_length,
            description,
            manifest_sha256,
        ) = read_served_model_manifest(served_model_manifest)
    else:
        if previous_managed_model_ids or previous_served_model_manifests:
            raise ConfigurationError(
                "previous managed models require served-model manifest mode"
            )
        model_id = MODEL_ID if model_id is None else model_id
        model_name = MODEL_NAME if model_name is None else model_name
        context_length = CONTEXT_LENGTH if context_length is None else context_length
        description = MODEL_DESCRIPTION if description is None else description
        model_id = require_model_id(model_id)
        model_name = require_nonempty(model_name, "model name")
        description = require_nonempty(description, "model description")

    context_length = require_context_length(context_length)
    base_url = require_nonempty(base_url, "OpenAI base URL")
    previous_models = collect_previous_models(
        previous_managed_model_ids, previous_served_model_manifests
    )
    api_key = read_api_key(key_file)
    connection = sqlite3.connect(database, timeout=30)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        config_row = connection.execute(
            "SELECT id, data FROM config ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if config_row is None:
            raise ConfigurationError("OpenWebUI config row does not exist")
        data = parse_json_object(config_row[1], "config.data")
        provider_index = configure_provider(data, api_key, base_url)
        owner_id, meta, params = model_values(
            connection,
            model_id,
            context_length,
            description,
            base_url,
            manifest_sha256,
        )
        previous_updates = previous_managed_model_updates(
            connection, previous_models, base_url, model_id
        )
        backup_path = backup_database(connection, backup_dir)

        now = int(time.time())
        encoded_config = json.dumps(
            data, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        with connection:
            connection.executemany(
                "UPDATE model SET meta = ?, is_active = 0, updated_at = ? WHERE id = ?",
                ((meta, now, previous_id) for previous_id, meta in previous_updates),
            )
            connection.execute(
                "UPDATE config SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (encoded_config, config_row[0]),
            )
            connection.execute(
                """
                INSERT INTO model (
                    id, user_id, base_model_id, name, meta, params,
                    created_at, updated_at, is_active
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    base_model_id = NULL,
                    name = excluded.name,
                    meta = excluded.meta,
                    params = excluded.params,
                    updated_at = excluded.updated_at,
                    is_active = 1
                """,
                (model_id, owner_id, model_name, meta, params, now, now),
            )
        return provider_index, backup_path
    finally:
        connection.close()


def parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def environment_default(environ: Mapping[str, str], name: str, fallback: str) -> str:
    value = environ.get(name, fallback)
    if not value.strip():
        raise ConfigurationError(f"{name} must not be empty")
    return value


def parse_args(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    if environ is None:
        environ = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("/data/webui.db"))
    parser.add_argument(
        "--api-key-file", type=Path, default=Path("/run/secrets/ullm-api-key")
    )
    parser.add_argument("--backup-dir", type=Path, default=Path("/data/backups"))
    manifest_environment = environ.get("ULLM_SERVED_MODEL_MANIFEST")
    if manifest_environment is not None and not manifest_environment.strip():
        raise ConfigurationError("ULLM_SERVED_MODEL_MANIFEST must not be empty")
    parser.add_argument(
        "--served-model-manifest",
        type=Path,
        default=Path(manifest_environment) if manifest_environment else None,
    )
    previous_ids_environment = environ.get("ULLM_PREVIOUS_MANAGED_MODEL_IDS")
    previous_ids = (
        parse_model_id_list(previous_ids_environment, "ULLM_PREVIOUS_MANAGED_MODEL_IDS")
        if previous_ids_environment is not None
        else []
    )
    parser.add_argument(
        "--previous-managed-model-id",
        action="append",
        default=previous_ids,
        help="explicit prior uLLM model id to mark managed and inactive; repeatable",
    )
    previous_manifest_environment = environ.get("ULLM_PREVIOUS_SERVED_MODEL_MANIFEST")
    if (
        previous_manifest_environment is not None
        and not previous_manifest_environment.strip()
    ):
        raise ConfigurationError(
            "ULLM_PREVIOUS_SERVED_MODEL_MANIFEST must not be empty"
        )
    parser.add_argument(
        "--previous-served-model-manifest",
        action="append",
        type=Path,
        default=(
            [Path(previous_manifest_environment)]
            if previous_manifest_environment is not None
            else []
        ),
        help="prior served-model manifest whose model row must be retired; repeatable",
    )
    parser.add_argument(
        "--model-id",
        default=environ.get("ULLM_MODEL_ID"),
    )
    parser.add_argument(
        "--model-name",
        default=environ.get("ULLM_MODEL_NAME"),
    )
    parser.add_argument(
        "--context-length",
        type=parse_positive_int,
        default=(
            parse_positive_int(environ["ULLM_MODEL_CONTEXT_LENGTH"])
            if "ULLM_MODEL_CONTEXT_LENGTH" in environ
            else None
        ),
    )
    parser.add_argument(
        "--description",
        default=environ.get("ULLM_MODEL_DESCRIPTION"),
    )
    parser.add_argument(
        "--base-url",
        default=environment_default(environ, "ULLM_OPENAI_BASE_URL", BASE_URL),
    )
    args = parser.parse_args(argv)
    legacy_values = (
        args.model_id,
        args.model_name,
        args.context_length,
        args.description,
    )
    if args.served_model_manifest is not None:
        if any(value is not None for value in legacy_values):
            raise ConfigurationError(
                "served-model manifest mode cannot be mixed with legacy model settings"
            )
    else:
        if args.previous_managed_model_id or args.previous_served_model_manifest:
            raise ConfigurationError(
                "previous managed models require served-model manifest mode"
            )
        args.model_id = MODEL_ID if args.model_id is None else args.model_id
        args.model_name = MODEL_NAME if args.model_name is None else args.model_name
        args.context_length = (
            CONTEXT_LENGTH if args.context_length is None else args.context_length
        )
        args.description = (
            MODEL_DESCRIPTION if args.description is None else args.description
        )
        args.model_id = require_model_id(args.model_id)
        args.model_name = require_nonempty(args.model_name, "model name")
        args.description = require_nonempty(args.description, "model description")
    args.base_url = require_nonempty(args.base_url, "OpenAI base URL")
    return args


def main() -> None:
    args = parse_args()
    provider_index, backup_path = configure(
        database=args.database,
        key_file=args.api_key_file,
        backup_dir=args.backup_dir,
        model_id=args.model_id,
        model_name=args.model_name,
        context_length=args.context_length,
        description=args.description,
        base_url=args.base_url,
        served_model_manifest=args.served_model_manifest,
        previous_managed_model_ids=args.previous_managed_model_id,
        previous_served_model_manifests=args.previous_served_model_manifest,
    )
    configured_model = (
        f"manifest {args.served_model_manifest}"
        if args.served_model_manifest is not None
        else f"model {args.model_id}"
    )
    print(
        f"Configured provider index {provider_index} and {configured_model}; "
        f"backup={backup_path}"
    )


if __name__ == "__main__":
    main()
