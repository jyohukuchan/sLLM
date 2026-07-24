from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import time
from typing import Any
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/openwebui-session-jwt.py"
SPEC = importlib.util.spec_from_file_location("openwebui_session_jwt", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

UID = os.getuid()
GID = os.getgid()
KEY_MODE = 0o600
TOKEN_MODE = 0o640
ADMIN_ID = "private-admin-identity"
SECRET = b"K" * 64


def _b64url(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _decode_segment(segment: bytes) -> Any:
    return json.loads(base64.urlsafe_b64decode(segment + b"=" * (-len(segment) % 4)))


def _reference_token(
    key: bytes,
    *,
    header: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> bytes:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"} if header is None else header
    payload = (
        {
            "id": ADMIN_ID,
            "exp": now + 3600,
            "jti": str(uuid.uuid4()),
            "iat": now,
        }
        if payload is None
        else payload
    )
    first = _b64url(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    second = _b64url(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = first + b"." + second
    signature = hmac.new(key, signing_input, hashlib.sha256).digest()
    return signing_input + b"." + _b64url(signature)


def _create_database(
    path: Path,
    *,
    admins: tuple[str, ...] = (ADMIN_ID,),
    expiry: object = "4w",
    include_expiry: bool = True,
) -> None:
    config: dict[str, Any] = {}
    if include_expiry:
        config["auth"] = {"jwt_expiry": expiry}
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE user (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE config (
                id INTEGER PRIMARY KEY,
                data TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO user (id, role, created_at) VALUES (?, 'admin', ?)",
            ((admin, index) for index, admin in enumerate(admins)),
        )
        connection.execute(
            "INSERT INTO config (id, data) VALUES (1, ?)",
            (json.dumps(config),),
        )
        connection.commit()


def _write_key(path: Path, value: bytes = SECRET) -> None:
    path.write_bytes(value)
    path.chmod(KEY_MODE)


def _write_token(path: Path, token: bytes) -> None:
    path.write_bytes(token + b"\n")
    path.chmod(TOKEN_MODE)


def _metadata_args(prefix: str, uid: int, gid: int, mode: int) -> list[str]:
    return [
        f"--{prefix}-uid",
        str(uid),
        f"--{prefix}-gid",
        str(gid),
        f"--{prefix}-mode",
        f"{mode:04o}",
    ]


def _common_args(database: Path, key: Path) -> list[str]:
    return [
        "--database",
        str(database),
        "--secret-key-file",
        str(key),
        *_metadata_args("secret-key", UID, GID, KEY_MODE),
    ]


def _mint(
    database: Path,
    key: Path,
    output: Path,
    *extra: str,
    parent_mode: int | None = None,
) -> subprocess.CompletedProcess[str]:
    actual_parent_mode = stat.S_IMODE(output.parent.stat().st_mode)
    if parent_mode is not None:
        actual_parent_mode = parent_mode
    return subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "mint",
            *_common_args(database, key),
            "--output",
            str(output),
            *_metadata_args(
                "output-parent", UID, GID, actual_parent_mode
            ),
            *_metadata_args("output", UID, GID, TOKEN_MODE),
            *extra,
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def _verify(
    database: Path,
    key: Path,
    token_file: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "verify",
            *_common_args(database, key),
            "--token-file",
            str(token_file),
            *_metadata_args("token", UID, GID, TOKEN_MODE),
            "--minimum-validity",
            "1s",
            *extra,
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def test_mint_trims_all_trailing_lf_and_has_reference_hs256_signature(
    tmp_path: Path,
) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    output = tmp_path / "session.jwt"
    _create_database(database, expiry="24h")
    _write_key(key, SECRET + b"\n\n\n")

    result = _mint(database, key, output)

    assert result.returncode == 0, result.stderr
    token = output.read_bytes().rstrip(b"\n")
    header_part, payload_part, signature_part = token.split(b".")
    signature = base64.urlsafe_b64decode(
        signature_part + b"=" * (-len(signature_part) % 4)
    )
    assert hmac.compare_digest(
        signature,
        hmac.new(
            SECRET,
            header_part + b"." + payload_part,
            hashlib.sha256,
        ).digest(),
    )
    assert _decode_segment(header_part) == {"alg": "HS256", "typ": "JWT"}
    payload = _decode_segment(payload_part)
    assert set(payload) == {"id", "exp", "jti", "iat"}
    assert (
        base64.urlsafe_b64decode(
            payload_part + b"=" * (-len(payload_part) % 4)
        ).decode("ascii")
    ).startswith('{"id":')
    assert payload["id"] == ADMIN_ID
    assert uuid.UUID(payload["jti"]).version == 4
    assert payload["exp"] - payload["iat"] == 24 * 60 * 60

    summary = json.loads(result.stdout)
    assert summary["algorithm"] == "HS256"
    assert summary["expiry_source"] == "configured"
    assert summary["token_written"] is True
    assert token.decode("ascii") not in result.stdout
    assert SECRET.decode("ascii") not in result.stdout + result.stderr
    assert ADMIN_ID not in result.stdout + result.stderr


def test_verify_success_emits_only_nonsecret_summary(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    token_file = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    _write_token(token_file, _reference_token(SECRET))

    result = _verify(database, key, token_file)

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["signature_valid"] is True
    assert summary["user_binding_valid"] is True
    assert summary["role"] == "admin"
    assert summary["claims"] == ["exp", "iat", "id", "jti"]
    assert summary["remaining_seconds"] > 0
    assert ADMIN_ID not in result.stdout + result.stderr
    assert SECRET.decode("ascii") not in result.stdout + result.stderr
    assert token_file.read_text(encoding="ascii").strip() not in result.stdout


@pytest.mark.parametrize("failure", ["tamper", "wrong-key"])
def test_verify_rejects_tamper_and_wrong_key(
    tmp_path: Path, failure: str
) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    token_file = tmp_path / "session.jwt"
    _create_database(database)
    token = _reference_token(SECRET)
    if failure == "tamper":
        replacement = b"A" if token[-1:] != b"A" else b"B"
        token = token[:-1] + replacement
        _write_key(key)
    else:
        _write_key(key, b"W" * 64)
    _write_token(token_file, token)

    result = _verify(database, key, token_file)

    assert result.returncode != 0
    assert not result.stdout
    assert ADMIN_ID not in result.stderr
    assert SECRET.decode("ascii") not in result.stderr


@pytest.mark.parametrize(
    ("header", "payload_change"),
    [
        ({"alg": "none", "typ": "JWT"}, {}),
        ({"alg": "HS256", "typ": "JWT", "kid": "unexpected"}, {}),
        (None, {"extra": True}),
        (None, {"jti": "not-a-uuid"}),
        (None, {"exp": "9999999999"}),
    ],
)
def test_verify_rejects_bad_algorithm_header_or_claim(
    tmp_path: Path,
    header: dict[str, Any] | None,
    payload_change: dict[str, Any],
) -> None:
    now = int(time.time())
    payload = {
        "id": ADMIN_ID,
        "exp": now + 3600,
        "jti": str(uuid.uuid4()),
        "iat": now,
    }
    payload.update(payload_change)
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    token_file = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    _write_token(
        token_file,
        _reference_token(
            SECRET,
            header={"alg": "HS256", "typ": "JWT"} if header is None else header,
            payload=payload,
        ),
    )

    result = _verify(database, key, token_file)

    assert result.returncode != 0
    assert not result.stdout
    assert ADMIN_ID not in result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        {
            "id": ADMIN_ID,
            "exp": 2,
            "jti": str(uuid.uuid4()),
            "iat": 1,
        },
        {
            "id": ADMIN_ID,
            "exp": 100,
            "jti": str(uuid.uuid4()),
            "iat": 100,
        },
        {
            "id": ADMIN_ID,
            "exp": int(time.time()) + 7200,
            "jti": str(uuid.uuid4()),
            "iat": int(time.time()) + 3600,
        },
    ],
)
def test_verify_rejects_expired_invalid_relation_and_future_iat(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    token_file = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    _write_token(token_file, _reference_token(SECRET, payload=payload))

    result = _verify(database, key, token_file)

    assert result.returncode != 0
    assert not result.stdout


def test_verify_checks_minimum_validity_and_current_admin_role(
    tmp_path: Path,
) -> None:
    now = int(time.time())
    token = _reference_token(
        SECRET,
        payload={
            "id": ADMIN_ID,
            "exp": now + 30,
            "jti": str(uuid.uuid4()),
            "iat": now,
        },
    )
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    token_file = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    _write_token(token_file, token)

    validity_result = _verify(
        database, key, token_file, "--minimum-validity", "1m"
    )
    assert validity_result.returncode != 0

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE user SET role = 'user' WHERE id = ?", (ADMIN_ID,)
        )
        connection.commit()
    binding_result = _verify(database, key, token_file)
    assert binding_result.returncode != 0
    assert ADMIN_ID not in binding_result.stderr


def test_verify_rejects_validity_equal_to_minimum() -> None:
    now = int(time.time())
    token = _reference_token(
        SECRET,
        payload={
            "id": ADMIN_ID,
            "exp": now + 60,
            "jti": str(uuid.uuid4()),
            "iat": now,
        },
    )
    with pytest.raises(TOOL.JwtToolError):
        TOOL.verify_token(token, SECRET, now, 60)


def test_mint_requires_exactly_one_admin(tmp_path: Path) -> None:
    key = tmp_path / "secret"
    _write_key(key)
    for count in (0, 2):
        database = tmp_path / f"webui-{count}.db"
        output = tmp_path / f"session-{count}.jwt"
        admins = () if count == 0 else ("first-private-id", "second-private-id")
        _create_database(database, admins=admins)

        result = _mint(database, key, output)

        assert result.returncode != 0
        assert not output.exists()
        assert all(admin not in result.stderr for admin in admins)


def test_duration_is_openwebui_compatible_but_consumes_all_input() -> None:
    assert TOOL.parse_duration("4w") == 2_419_200
    assert TOOL.parse_duration("1h30m") == 5_400
    assert TOOL.parse_duration("1.5s") == 1
    assert TOOL.parse_duration("2h-30m") == 5_400
    for invalid in ("0", "-1", "500ms", "1h junk", "junk1h", "1h 30m"):
        with pytest.raises(TOOL.JwtToolError):
            TOOL.parse_duration(invalid)


def test_latest_config_default_and_expiry_override(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    _create_database(database, include_expiry=False)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO config (id, data) VALUES (2, ?)",
            (json.dumps({"auth": {"jwt_expiry": "2h30m"}}),),
        )
        connection.commit()

    user_id, seconds, source = TOOL.load_mint_context(database)
    assert user_id == ADMIN_ID
    assert seconds == 9_000
    assert source == "configured"

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE config SET data = ? WHERE id = 2",
            (json.dumps({"auth": {"jwt_expiry": "0"}}),),
        )
        connection.commit()
    with pytest.raises(TOOL.JwtToolError):
        TOOL.load_mint_context(database)
    _, seconds, source = TOOL.load_mint_context(database, "24h")
    assert seconds == 86_400
    assert source == "override"

    default_database = tmp_path / "default.db"
    _create_database(default_database, include_expiry=False)
    _, seconds, source = TOOL.load_mint_context(default_database)
    assert seconds == 2_419_200
    assert source == "default"


@pytest.mark.parametrize(
    "value",
    [
        b"abc\ninternal\n",
        b"abc\r\n",
        b"abc\0def\n",
        b"\n\n",
        b"\xff\n",
    ],
)
def test_secret_key_rejects_forbidden_or_empty_values(
    tmp_path: Path, value: bytes
) -> None:
    key = tmp_path / "secret"
    _write_key(key, value)
    policy = TOOL.FilePolicy(UID, GID, KEY_MODE)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.read_secret_key(key, policy)


def test_secret_key_rejects_symlink_hardlink_and_wrong_mode(
    tmp_path: Path,
) -> None:
    key = tmp_path / "secret"
    _write_key(key)
    policy = TOOL.FilePolicy(UID, GID, KEY_MODE)

    symlink = tmp_path / "secret-link"
    symlink.symlink_to(key)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.read_secret_key(symlink, policy)

    hardlink = tmp_path / "secret-hardlink"
    os.link(key, hardlink)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.read_secret_key(key, policy)
    hardlink.unlink()

    key.chmod(0o644)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.read_secret_key(key, policy)


def test_stable_read_rejects_named_entry_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "secret"
    replacement = tmp_path / "replacement"
    _write_key(key)
    _write_key(replacement, b"R" * 64)
    original_read = TOOL.os.read
    rotated = False

    def rotating_read(descriptor: int, size: int) -> bytes:
        nonlocal rotated
        result = original_read(descriptor, size)
        if result and not rotated:
            rotated = True
            os.replace(replacement, key)
        return result

    monkeypatch.setattr(TOOL.os, "read", rotating_read)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.read_secret_key(key, TOOL.FilePolicy(UID, GID, KEY_MODE))


def test_atomic_output_refuses_overwrite_and_replace_is_explicit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    output = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    output.write_bytes(b"existing-data")
    output.chmod(TOKEN_MODE)
    original_inode = output.stat().st_ino

    refused = _mint(database, key, output, "--expires-in", "24h")
    assert refused.returncode != 0
    assert output.read_bytes() == b"existing-data"
    assert output.stat().st_ino == original_inode

    replaced = _mint(
        database, key, output, "--expires-in", "24h", "--replace"
    )
    assert replaced.returncode == 0, replaced.stderr
    assert output.read_bytes() != b"existing-data"
    assert output.stat().st_ino != original_inode
    assert stat.S_IMODE(output.stat().st_mode) == TOKEN_MODE
    assert output.stat().st_uid == UID
    assert output.stat().st_gid == GID
    assert output.stat().st_nlink == 1
    assert json.loads(replaced.stdout)["expiry_source"] == "override"


def test_replace_rejects_existing_target_with_unexpected_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    output = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    output.write_bytes(b"preserve-me")
    output.chmod(0o600)

    result = _mint(database, key, output, "--replace")

    assert result.returncode != 0
    assert output.read_bytes() == b"preserve-me"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_atomic_write_removes_published_file_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "session.jwt"
    parent_policy = TOOL.FilePolicy(
        UID, GID, stat.S_IMODE(tmp_path.stat().st_mode)
    )
    output_policy = TOOL.FilePolicy(UID, GID, TOKEN_MODE)
    original_fsync = TOOL.os.fsync
    calls = 0

    def failing_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(TOOL.os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        TOOL.atomic_write(
            output,
            b"header.payload.signature\n",
            parent_policy,
            output_policy,
            False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".openwebui-jwt-*.tmp"))


def test_atomic_write_cleans_both_links_when_temp_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "session.jwt"
    parent_policy = TOOL.FilePolicy(
        UID, GID, stat.S_IMODE(tmp_path.stat().st_mode)
    )
    output_policy = TOOL.FilePolicy(UID, GID, TOKEN_MODE)
    original_unlink = TOOL.os.unlink
    injected = False

    def failing_unlink(
        path: str | bytes,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal injected
        if (
            not injected
            and isinstance(path, str)
            and path.startswith(".openwebui-jwt-")
        ):
            injected = True
            raise OSError("injected unlink failure")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(TOOL.os, "unlink", failing_unlink)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.atomic_write(
            output,
            b"header.payload.signature\n",
            parent_policy,
            output_policy,
            False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".openwebui-jwt-*.tmp"))


def test_atomic_write_rejects_parent_named_entry_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "session.jwt"
    parent_policy = TOOL.FilePolicy(
        UID, GID, stat.S_IMODE(tmp_path.stat().st_mode)
    )
    output_policy = TOOL.FilePolicy(UID, GID, TOKEN_MODE)
    original_lstat = TOOL.os.lstat
    real = original_lstat(tmp_path)
    changed_values = list(real)
    changed_values[1] += 1
    changed = os.stat_result(changed_values)
    calls = 0

    def rotating_lstat(path: str | bytes | os.PathLike[str]) -> os.stat_result:
        nonlocal calls
        if Path(path) == tmp_path:
            calls += 1
            if calls > 1:
                return changed
        return original_lstat(path)

    monkeypatch.setattr(TOOL.os, "lstat", rotating_lstat)
    with pytest.raises(TOOL.JwtToolError):
        TOOL.atomic_write(
            output,
            b"header.payload.signature\n",
            parent_policy,
            output_policy,
            False,
        )
    assert not output.exists()


def test_output_parent_metadata_is_enforced(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    key = tmp_path / "secret"
    output = tmp_path / "session.jwt"
    _create_database(database)
    _write_key(key)
    actual = stat.S_IMODE(tmp_path.stat().st_mode)
    wrong = 0o755 if actual != 0o755 else 0o700

    result = _mint(database, key, output, parent_mode=wrong)

    assert result.returncode != 0
    assert not output.exists()
    assert SECRET.decode("ascii") not in result.stderr


def test_database_symlink_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "webui.db"
    database_link = tmp_path / "webui-link.db"
    key = tmp_path / "secret"
    output = tmp_path / "session.jwt"
    _create_database(database)
    database_link.symlink_to(database)
    _write_key(key)

    result = _mint(database_link, key, output)

    assert result.returncode != 0
    assert not output.exists()
