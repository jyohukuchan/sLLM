from __future__ import annotations

import importlib.util
import json
import sqlite3
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_PATH = ROOT / "deploy/openwebui/configure.py"
SPEC = importlib.util.spec_from_file_location("openwebui_configure", CONFIGURE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONFIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIGURE)


def _create_database(path: Path) -> tuple[dict, dict, dict]:
    existing_config = {
        "unrelated": {"retained": True},
        "openai": {
            "enable": False,
            "api_base_urls": ["https://existing.example/v1"],
            "api_keys": ["existing-key"],
            "api_configs": {
                "0": {
                    "enable": True,
                    "connection_type": "external",
                    "retained": "provider-value",
                }
            },
            "retained": "openai-value",
        },
        "task": {
            "follow_up": {"enable": True, "retained": "follow-up"},
            "tags": {"enable": True},
            "title": {"enable": True},
        },
    }
    existing_meta = {
        "capabilities": {
            "usage": False,
            "vision": True,
            "retained_capability": True,
        },
        "retained_meta": {"value": 7},
    }
    existing_params = {
        "temperature": 0.25,
        "retained_params": {"value": 9},
        "num_ctx": 8_192,
        "max_tokens": 128,
    }

    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE user (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE config (
                id INTEGER PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT
            );
            CREATE TABLE model (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                base_model_id TEXT,
                name TEXT NOT NULL,
                meta TEXT NOT NULL,
                params TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                is_active INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user(id)
            );
            """
        )
        connection.execute(
            "INSERT INTO user (id, role, created_at) VALUES (?, ?, ?)",
            ("owner", "admin", 1),
        )
        connection.execute(
            "INSERT INTO config (id, data, updated_at) VALUES (?, ?, ?)",
            (1, json.dumps(existing_config), "before"),
        )
        connection.execute(
            """
            INSERT INTO model (
                id, user_id, base_model_id, name, meta, params,
                created_at, updated_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                CONFIGURE.MODEL_ID,
                "owner",
                "existing-base",
                "Existing uLLM model",
                json.dumps(existing_meta),
                json.dumps(existing_params),
                1,
                1,
                0,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return existing_config, existing_meta, existing_params


def test_configure_enables_usage_and_preserves_existing_state(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    key_file = tmp_path / "api-key"
    backup_dir = tmp_path / "backups"
    existing_config, existing_meta, existing_params = _create_database(database)
    key_file.write_text("new-api-key\n", encoding="ascii")

    provider_index, backup_path = CONFIGURE.configure(
        database, key_file, backup_dir
    )

    assert provider_index == 1
    with sqlite3.connect(database) as connection:
        config = json.loads(
            connection.execute("SELECT data FROM config WHERE id = 1").fetchone()[0]
        )
        model_row = connection.execute(
            "SELECT user_id, base_model_id, name, meta, params, is_active "
            "FROM model WHERE id = ?",
            (CONFIGURE.MODEL_ID,),
        ).fetchone()

    assert config["unrelated"] == existing_config["unrelated"]
    assert config["openai"]["api_base_urls"] == [
        "https://existing.example/v1",
        CONFIGURE.BASE_URL,
    ]
    assert config["openai"]["api_keys"] == ["existing-key", "new-api-key"]
    assert config["openai"]["api_configs"]["0"] == existing_config["openai"][
        "api_configs"
    ]["0"]
    assert config["openai"]["retained"] == "openai-value"
    assert config["task"]["follow_up"]["retained"] == "follow-up"

    assert model_row is not None
    owner_id, base_model_id, name, meta_raw, params_raw, is_active = model_row
    meta = json.loads(meta_raw)
    params = json.loads(params_raw)
    assert (owner_id, base_model_id, name, is_active) == (
        "owner",
        None,
        CONFIGURE.MODEL_NAME,
        1,
    )
    assert meta["capabilities"]["usage"] is True
    assert meta["capabilities"]["retained_capability"] is True
    assert meta["retained_meta"] == existing_meta["retained_meta"]
    assert params["temperature"] == existing_params["temperature"]
    assert params["retained_params"] == existing_params["retained_params"]
    assert "num_ctx" not in params
    assert "max_tokens" not in params

    assert backup_path.parent == backup_dir
    assert backup_path.is_file()
    assert stat.S_IMODE(backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
    with sqlite3.connect(backup_path) as connection:
        backup_config = json.loads(
            connection.execute("SELECT data FROM config WHERE id = 1").fetchone()[0]
        )
        backup_meta = json.loads(
            connection.execute(
                "SELECT meta FROM model WHERE id = ?", (CONFIGURE.MODEL_ID,)
            ).fetchone()[0]
        )
    assert backup_config == existing_config
    assert backup_meta == existing_meta


def test_configure_adds_custom_model_without_replacing_sq8(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    key_file = tmp_path / "api-key"
    backup_dir = tmp_path / "backups"
    _create_database(database)
    key_file.write_text("new-api-key\n", encoding="ascii")

    model_id = "ullm-qwen3.5-9b-aq4"
    model_name = "uLLM Qwen3.5 9B AQ4"
    description = "Qwen3.5 9B served locally by uLLM AQ4_0."
    base_url = "http://172.20.0.1:18000/v1"
    CONFIGURE.configure(
        database,
        key_file,
        backup_dir,
        model_id=model_id,
        model_name=model_name,
        context_length=32_768,
        description=description,
        base_url=base_url,
    )

    with sqlite3.connect(database) as connection:
        config = json.loads(
            connection.execute("SELECT data FROM config WHERE id = 1").fetchone()[0]
        )
        sq8_row = connection.execute(
            "SELECT name FROM model WHERE id = ?", (CONFIGURE.MODEL_ID,)
        ).fetchone()
        aq4_row = connection.execute(
            "SELECT name, meta, params, is_active FROM model WHERE id = ?",
            (model_id,),
        ).fetchone()

    assert sq8_row == ("Existing uLLM model",)
    assert config["openai"]["api_base_urls"][-1] == base_url
    assert aq4_row is not None
    name, meta_raw, params_raw, is_active = aq4_row
    meta = json.loads(meta_raw)
    params = json.loads(params_raw)
    assert (name, is_active) == (model_name, 1)
    assert meta["description"] == description
    assert meta["n_ctx_train"] == 32_768
    assert meta["context_length"] == 32_768
    assert meta["capabilities"]["usage"] is True
    assert "num_ctx" not in params
    assert "max_tokens" not in params


def test_parse_args_reads_model_environment_and_allows_cli_override() -> None:
    environ = {
        "ULLM_OPENAI_BASE_URL": "http://127.0.0.1:18000/v1",
        "ULLM_MODEL_ID": "ullm-qwen3.5-9b-aq4",
        "ULLM_MODEL_NAME": "uLLM Qwen3.5 9B AQ4",
        "ULLM_MODEL_CONTEXT_LENGTH": "32768",
        "ULLM_MODEL_DESCRIPTION": "AQ4 environment description.",
    }

    environment_args = CONFIGURE.parse_args([], environ)
    cli_args = CONFIGURE.parse_args(
        [
            "--model-id",
            "custom-model",
            "--model-name",
            "Custom name",
            "--context-length",
            "8192",
            "--description",
            "Custom description.",
            "--base-url",
            "http://127.0.0.1:28000/v1",
        ],
        environ,
    )

    assert environment_args.model_id == environ["ULLM_MODEL_ID"]
    assert environment_args.base_url == environ["ULLM_OPENAI_BASE_URL"]
    assert environment_args.model_name == environ["ULLM_MODEL_NAME"]
    assert environment_args.context_length == 32_768
    assert environment_args.description == environ["ULLM_MODEL_DESCRIPTION"]
    assert cli_args.model_id == "custom-model"
    assert cli_args.model_name == "Custom name"
    assert cli_args.context_length == 8_192
    assert cli_args.description == "Custom description."
    assert cli_args.base_url == "http://127.0.0.1:28000/v1"
