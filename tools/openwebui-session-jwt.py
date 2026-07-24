#!/usr/bin/env python3
"""Safely mint and verify OpenWebUI 0.9.4 session JWTs.

Only the Python standard library is used.  Secret and token values are never
written to stdout or included in diagnostics.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import decimal
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote
import uuid


ALGORITHM = "HS256"
CLAIMS = frozenset({"id", "exp", "jti", "iat"})
DEFAULT_EXPIRY = "4w"
MAX_SECRET_BYTES = 16 * 1024
MAX_TOKEN_BYTES = 64 * 1024
MAX_JSON_BYTES = 32 * 1024
MAX_ID_BYTES = 4 * 1024

SECRET_UID = 0
SECRET_GID = 0
SECRET_MODE = 0o600
OUTPUT_PARENT_UID = 0
OUTPUT_PARENT_GID = 1000
OUTPUT_PARENT_MODE = 0o750
OUTPUT_UID = 0
OUTPUT_GID = 1000
OUTPUT_MODE = 0o640

_DURATION_PART = re.compile(r"(-?\d+(?:\.\d+)?)(ms|s|m|h|d|w)")
_DURATION_FACTORS = {
    "ms": decimal.Decimal("0.001"),
    "s": decimal.Decimal(1),
    "m": decimal.Decimal(60),
    "h": decimal.Decimal(3600),
    "d": decimal.Decimal(86400),
    "w": decimal.Decimal(604800),
}
_B64URL = re.compile(rb"[A-Za-z0-9_-]+")


class JwtToolError(Exception):
    """An expected, secret-free validation failure."""


class FilePolicy:
    __slots__ = ("uid", "gid", "mode")

    def __init__(self, uid: int, gid: int, mode: int) -> None:
        self.uid = uid
        self.gid = gid
        self.mode = mode


def _metadata_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
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


def _node_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
    )


def _validate_regular_metadata(
    value: os.stat_result, policy: FilePolicy, label: str
) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise JwtToolError(f"{label} must be a regular file")
    if value.st_nlink != 1:
        raise JwtToolError(f"{label} must have exactly one link")
    if value.st_uid != policy.uid or value.st_gid != policy.gid:
        raise JwtToolError(f"{label} ownership differs from the required metadata")
    if stat.S_IMODE(value.st_mode) != policy.mode:
        raise JwtToolError(f"{label} mode differs from the required metadata")


def _open_readonly_nofollow(path: Path) -> int:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_CLOEXEC")
        or not hasattr(os, "O_NONBLOCK")
    ):
        raise JwtToolError("required secure file-open flags are unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        return os.open(path, flags)
    except OSError as error:
        if error.errno in {
            errno.ELOOP,
            errno.ENOENT,
            errno.ENOTDIR,
            errno.EACCES,
            errno.EPERM,
        }:
            raise JwtToolError("secure input file open failed") from None
        raise


def read_stable_file(
    path: Path, policy: FilePolicy, maximum: int, label: str
) -> bytes:
    """Read a bounded, stable, single-link regular file without following it."""

    try:
        named_before = os.lstat(path)
    except OSError:
        raise JwtToolError(f"{label} path validation failed") from None
    _validate_regular_metadata(named_before, policy, label)

    descriptor = _open_readonly_nofollow(path)
    try:
        before = os.fstat(descriptor)
        _validate_regular_metadata(before, policy, label)
        if _metadata_tuple(named_before) != _metadata_tuple(before):
            raise JwtToolError(f"{label} changed while it was opened")
        if before.st_size > maximum:
            raise JwtToolError(f"{label} exceeds the size limit")

        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum:
            raise JwtToolError(f"{label} exceeds the size limit")

        after = os.fstat(descriptor)
        try:
            named_after = os.lstat(path)
        except OSError:
            raise JwtToolError(f"{label} changed while it was read") from None
        if not (
            _metadata_tuple(named_before)
            == _metadata_tuple(before)
            == _metadata_tuple(after)
            == _metadata_tuple(named_after)
        ):
            raise JwtToolError(f"{label} changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def read_secret_key(path: Path, policy: FilePolicy) -> bytes:
    """Apply start.sh command-substitution trimming to a validated key file."""

    raw = read_stable_file(path, policy, MAX_SECRET_BYTES, "secret key file")
    key = raw.rstrip(b"\n")
    if not key:
        raise JwtToolError("secret key file contains no key")
    if b"\n" in key or b"\r" in key or b"\0" in key:
        raise JwtToolError("secret key file contains a forbidden byte")
    try:
        key.decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise JwtToolError("secret key file is not valid UTF-8") from None
    return key


def _decode_json(raw: bytes, label: str) -> Mapping[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        raise JwtToolError(f"{label} exceeds the size limit")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JwtToolError(f"{label} has duplicate fields")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise JwtToolError(f"{label} contains a non-finite number")

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except JwtToolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JwtToolError(f"{label} is not valid JSON") from None
    if not isinstance(value, dict):
        raise JwtToolError(f"{label} must be a JSON object")
    return value


def parse_duration(value: str) -> int:
    """Parse OpenWebUI duration syntax, strictly, into positive whole seconds."""

    if not isinstance(value, str) or not value or len(value) > 128:
        raise JwtToolError("JWT expiry duration is invalid")
    if value in {"0", "-1"}:
        raise JwtToolError("JWT expiry without an expiration is gate-incompatible")

    position = 0
    total = decimal.Decimal(0)
    while position < len(value):
        match = _DURATION_PART.match(value, position)
        if match is None:
            raise JwtToolError("JWT expiry duration is invalid")
        try:
            total += decimal.Decimal(match.group(1)) * _DURATION_FACTORS[
                match.group(2)
            ]
        except decimal.InvalidOperation:
            raise JwtToolError("JWT expiry duration is invalid") from None
        position = match.end()

    # NumericDate claims have one-second resolution.  Truncation is conservative
    # for positive fractional OpenWebUI durations.
    seconds = int(total)
    if seconds <= 0:
        raise JwtToolError("JWT expiry duration must resolve to positive seconds")
    if seconds >= 2**63:
        raise JwtToolError("JWT expiry duration is too large")
    return seconds


def _database_identity(path: Path) -> tuple[int, ...]:
    try:
        value = os.lstat(path)
    except OSError:
        raise JwtToolError("OpenWebUI database path validation failed") from None
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise JwtToolError("OpenWebUI database must be a non-symlink regular file")
    return _metadata_tuple(value)


def _open_database(path: Path) -> tuple[sqlite3.Connection, tuple[int, ...]]:
    identity = _database_identity(path)
    absolute = os.path.abspath(os.fspath(path))
    uri = f"file:{quote(absolute, safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5, isolation_level=None)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN")
        return connection, identity
    except sqlite3.Error:
        raise JwtToolError("read-only OpenWebUI database open failed") from None


def _close_database(
    connection: sqlite3.Connection, path: Path, identity: tuple[int, ...]
) -> None:
    try:
        if _database_identity(path) != identity:
            raise JwtToolError("OpenWebUI database changed during validation")
    finally:
        connection.close()


def _validate_user_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_ID_BYTES
        or any(ord(character) < 0x20 for character in value)
    ):
        raise JwtToolError("OpenWebUI user identity is invalid")
    return value


def load_mint_context(
    database: Path, expires_in: str | None = None
) -> tuple[str, int, str]:
    connection, identity = _open_database(database)
    try:
        try:
            admins = connection.execute(
                'SELECT id FROM "user" WHERE role = ? LIMIT 2', ("admin",)
            ).fetchall()
            if len(admins) != 1:
                raise JwtToolError(
                    "database must contain exactly one administrator"
                )
            admin_id = _validate_user_id(admins[0][0])
        except sqlite3.Error:
            raise JwtToolError("OpenWebUI database validation failed") from None

        if expires_in is not None:
            return admin_id, parse_duration(expires_in), "override"

        try:
            row = connection.execute(
                "SELECT data FROM config ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            raise JwtToolError("OpenWebUI database validation failed") from None
        expiry = DEFAULT_EXPIRY
        source = "default"
        if row is not None:
            raw = row[0]
            if not isinstance(raw, str):
                raise JwtToolError("latest OpenWebUI config data is invalid")
            config = _decode_json(raw.encode("utf-8"), "latest config data")
            auth = config.get("auth")
            if auth is not None:
                if not isinstance(auth, dict):
                    raise JwtToolError("latest OpenWebUI auth config is invalid")
                configured = auth.get("jwt_expiry")
                if configured is not None:
                    if not isinstance(configured, str):
                        raise JwtToolError("configured JWT expiry is invalid")
                    expiry = configured
                    source = "configured"
        return admin_id, parse_duration(expiry), source
    finally:
        _close_database(connection, database, identity)


def validate_user_binding(database: Path, user_id: str) -> None:
    connection, identity = _open_database(database)
    try:
        try:
            rows = connection.execute(
                'SELECT role FROM "user" WHERE id = ? LIMIT 2', (user_id,)
            ).fetchall()
        except sqlite3.Error:
            raise JwtToolError("OpenWebUI database validation failed") from None
        if len(rows) != 1 or rows[0][0] != "admin":
            raise JwtToolError("token user binding is invalid")
    finally:
        _close_database(connection, database, identity)


def _b64url_encode(value: bytes) -> bytes:
    return base64.urlsafe_b64encode(value).rstrip(b"=")


def _b64url_decode(value: bytes, label: str) -> bytes:
    if not value or _B64URL.fullmatch(value) is None or len(value) % 4 == 1:
        raise JwtToolError(f"{label} is not canonical base64url")
    try:
        decoded = base64.b64decode(
            value + b"=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError):
        raise JwtToolError(f"{label} is not canonical base64url") from None
    if _b64url_encode(decoded) != value:
        raise JwtToolError(f"{label} is not canonical base64url")
    return decoded


def mint_token(user_id: str, key: bytes, issued_at: int, lifetime: int) -> bytes:
    payload = {
        "id": _validate_user_id(user_id),
        "exp": issued_at + lifetime,
        "jti": str(uuid.uuid4()),
        "iat": issued_at,
    }
    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_segment = _b64url_encode(
        json.dumps(
            header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    )
    payload_segment = _b64url_encode(
        json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=False
        ).encode("ascii")
    )
    signing_input = header_segment + b"." + payload_segment
    signature = hmac.new(key, signing_input, hashlib.sha256).digest()
    return signing_input + b"." + _b64url_encode(signature)


def _validate_claims(
    payload: Mapping[str, Any], now: int, minimum_validity: int
) -> tuple[str, int, int, int]:
    if set(payload) != CLAIMS:
        raise JwtToolError("token claim set is invalid")
    user_id = _validate_user_id(payload.get("id"))
    issued_at = payload.get("iat")
    expires_at = payload.get("exp")
    if (
        isinstance(issued_at, bool)
        or not isinstance(issued_at, int)
        or isinstance(expires_at, bool)
        or not isinstance(expires_at, int)
        or issued_at <= 0
        or expires_at <= 0
        or issued_at >= 2**63
        or expires_at >= 2**63
    ):
        raise JwtToolError("token time claims are invalid")
    if expires_at <= issued_at:
        raise JwtToolError("token expiration does not follow issuance")
    if issued_at > now:
        raise JwtToolError("token issuance time is in the future")
    if expires_at <= now:
        raise JwtToolError("token has expired")
    remaining = expires_at - now
    if remaining <= minimum_validity:
        raise JwtToolError("token has insufficient remaining validity")

    jti = payload.get("jti")
    if not isinstance(jti, str):
        raise JwtToolError("token identifier is invalid")
    try:
        parsed = uuid.UUID(jti)
    except (ValueError, AttributeError):
        raise JwtToolError("token identifier is invalid") from None
    if (
        str(parsed) != jti
        or parsed.version != 4
        or parsed.variant != uuid.RFC_4122
    ):
        raise JwtToolError("token identifier is invalid")
    return user_id, issued_at, expires_at, remaining


def verify_token(
    token: bytes, key: bytes, now: int, minimum_validity: int
) -> tuple[str, int, int, int]:
    if not token or len(token) > MAX_TOKEN_BYTES or any(byte > 0x7F for byte in token):
        raise JwtToolError("token encoding is invalid")
    parts = token.split(b".")
    if len(parts) != 3:
        raise JwtToolError("token compact serialization is invalid")

    header_raw = _b64url_decode(parts[0], "token header")
    payload_raw = _b64url_decode(parts[1], "token payload")
    signature = _b64url_decode(parts[2], "token signature")
    header = _decode_json(header_raw, "token header")
    payload = _decode_json(payload_raw, "token payload")
    if set(header) != {"alg", "typ"}:
        raise JwtToolError("token header fields are invalid")
    if header.get("alg") != ALGORITHM or header.get("typ") != "JWT":
        raise JwtToolError("token algorithm or type is invalid")
    if len(signature) != hashlib.sha256().digest_size:
        raise JwtToolError("token signature is invalid")

    expected = hmac.new(key, parts[0] + b"." + parts[1], hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise JwtToolError("token signature is invalid")
    return _validate_claims(payload, now, minimum_validity)


def read_token(path: Path, policy: FilePolicy) -> bytes:
    raw = read_stable_file(path, policy, MAX_TOKEN_BYTES + 8, "token file")
    token = raw.rstrip(b"\n")
    if not token or b"\n" in token or b"\r" in token or b"\0" in token:
        raise JwtToolError("token file encoding is invalid")
    return token


def _validate_parent_metadata(
    value: os.stat_result, policy: FilePolicy
) -> None:
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise JwtToolError("output parent must be a non-symlink directory")
    if value.st_uid != policy.uid or value.st_gid != policy.gid:
        raise JwtToolError("output parent ownership differs")
    if stat.S_IMODE(value.st_mode) != policy.mode:
        raise JwtToolError("output parent mode differs")


def _open_parent(
    path: Path, policy: FilePolicy
) -> tuple[int, str, Path, tuple[int, ...]]:
    name = path.name
    if not name or name in {".", ".."}:
        raise JwtToolError("output filename is invalid")
    parent = path.parent if os.fspath(path.parent) else Path(".")
    try:
        named = os.lstat(parent)
    except OSError:
        raise JwtToolError("secure output parent open failed") from None
    _validate_parent_metadata(named, policy)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(parent, flags)
    except OSError:
        raise JwtToolError("secure output parent open failed") from None
    try:
        value = os.fstat(descriptor)
        _validate_parent_metadata(value, policy)
        if _node_identity(named) != _node_identity(value):
            raise JwtToolError("output parent changed while it was opened")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, name, parent, _node_identity(value)


def _validate_parent_stable(
    parent: Path,
    descriptor: int,
    expected: tuple[int, ...],
    policy: FilePolicy,
) -> None:
    try:
        named = os.lstat(parent)
    except OSError:
        raise JwtToolError("output parent changed during publication") from None
    opened = os.fstat(descriptor)
    _validate_parent_metadata(named, policy)
    _validate_parent_metadata(opened, policy)
    if _node_identity(named) != expected or _node_identity(opened) != expected:
        raise JwtToolError("output parent changed during publication")


def _unlink_if_inode(
    parent_fd: int, name: str, expected_device: int, expected_inode: int
) -> None:
    try:
        value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(value.st_mode)
        and value.st_dev == expected_device
        and value.st_ino == expected_inode
    ):
        os.unlink(name, dir_fd=parent_fd)


def atomic_write(
    path: Path,
    content: bytes,
    parent_policy: FilePolicy,
    output_policy: FilePolicy,
    replace: bool,
) -> None:
    parent_fd, name, parent_path, parent_identity = _open_parent(
        path, parent_policy
    )
    temporary = f".openwebui-jwt-{uuid.uuid4().hex}.tmp"
    temp_fd = -1
    published = False
    completed = False
    created_device = -1
    created_inode = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
        )
        temp_fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        os.fchown(temp_fd, output_policy.uid, output_policy.gid)
        os.fchmod(temp_fd, output_policy.mode)
        view = memoryview(content)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise JwtToolError("token output write failed")
            view = view[written:]
        os.fsync(temp_fd)
        created = os.fstat(temp_fd)
        _validate_regular_metadata(created, output_policy, "token output")
        if created.st_size != len(content):
            raise JwtToolError("token output size verification failed")
        created_device = created.st_dev
        created_inode = created.st_ino
        os.close(temp_fd)
        temp_fd = -1

        try:
            if replace:
                try:
                    existing_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    existing_fd = -1
                except OSError:
                    raise JwtToolError(
                        "existing token output validation failed"
                    ) from None
                if existing_fd >= 0:
                    try:
                        _validate_regular_metadata(
                            os.fstat(existing_fd),
                            output_policy,
                            "existing token output",
                        )
                    finally:
                        os.close(existing_fd)
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                published = True
            else:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                published = True
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except OSError:
                    raise JwtToolError(
                        "temporary token link cleanup failed"
                    ) from None
        except FileExistsError:
            raise JwtToolError("token output already exists") from None

        result_fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            result = os.fstat(result_fd)
            _validate_regular_metadata(result, output_policy, "token output")
            if (
                result.st_dev != created_device
                or result.st_ino != created_inode
                or result.st_size != len(content)
            ):
                raise JwtToolError("token output size verification failed")
        finally:
            os.close(result_fd)
        _validate_parent_stable(
            parent_path, parent_fd, parent_identity, parent_policy
        )
        os.fsync(parent_fd)
        completed = True
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if published and not completed:
            try:
                _unlink_if_inode(
                    parent_fd, name, created_device, created_inode
                )
            except OSError:
                pass
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        if not completed:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        os.close(parent_fd)


def _parse_nonnegative(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from None
    if result < 0 or result >= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _parse_mode(value: str) -> int:
    text = value[2:] if value.lower().startswith("0o") else value
    if not text or any(character not in "01234567" for character in text):
        raise argparse.ArgumentTypeError("must be an octal mode")
    result = int(text, 8)
    if result > 0o7777:
        raise argparse.ArgumentTypeError("must be an octal mode")
    return result


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--secret-key-file", required=True, type=Path)
    parser.add_argument("--secret-key-uid", type=_parse_nonnegative, default=SECRET_UID)
    parser.add_argument("--secret-key-gid", type=_parse_nonnegative, default=SECRET_GID)
    parser.add_argument("--secret-key-mode", type=_parse_mode, default=SECRET_MODE)


def _add_file_metadata(
    parser: argparse.ArgumentParser, prefix: str, defaults: tuple[int, int, int]
) -> None:
    uid, gid, mode = defaults
    parser.add_argument(f"--{prefix}-uid", type=_parse_nonnegative, default=uid)
    parser.add_argument(f"--{prefix}-gid", type=_parse_nonnegative, default=gid)
    parser.add_argument(f"--{prefix}-mode", type=_parse_mode, default=mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mint or verify an OpenWebUI 0.9.4 HS256 session JWT"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mint = subparsers.add_parser("mint")
    _add_common(mint)
    mint.add_argument("--output", required=True, type=Path)
    mint.add_argument("--expires-in")
    mint.add_argument("--replace", action="store_true")
    _add_file_metadata(
        mint,
        "output-parent",
        (OUTPUT_PARENT_UID, OUTPUT_PARENT_GID, OUTPUT_PARENT_MODE),
    )
    _add_file_metadata(mint, "output", (OUTPUT_UID, OUTPUT_GID, OUTPUT_MODE))

    verify = subparsers.add_parser("verify")
    _add_common(verify)
    verify.add_argument("--token-file", required=True, type=Path)
    verify.add_argument("--minimum-validity", default="5m")
    _add_file_metadata(verify, "token", (OUTPUT_UID, OUTPUT_GID, OUTPUT_MODE))
    return parser


def _secret_policy(arguments: argparse.Namespace) -> FilePolicy:
    return FilePolicy(
        arguments.secret_key_uid,
        arguments.secret_key_gid,
        arguments.secret_key_mode,
    )


def _emit(value: Mapping[str, Any], stream: Any = sys.stdout) -> None:
    print(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        file=stream,
        flush=True,
    )


def run(arguments: argparse.Namespace) -> None:
    key = read_secret_key(arguments.secret_key_file, _secret_policy(arguments))
    if arguments.command == "mint":
        user_id, lifetime, expiry_source = load_mint_context(
            arguments.database, arguments.expires_in
        )
        now = int(time.time())
        token = mint_token(user_id, key, now, lifetime)
        atomic_write(
            arguments.output,
            token + b"\n",
            FilePolicy(
                arguments.output_parent_uid,
                arguments.output_parent_gid,
                arguments.output_parent_mode,
            ),
            FilePolicy(
                arguments.output_uid, arguments.output_gid, arguments.output_mode
            ),
            arguments.replace,
        )
        _emit(
            {
                "algorithm": ALGORITHM,
                "claims": sorted(CLAIMS),
                "exp": now + lifetime,
                "expiry_source": expiry_source,
                "iat": now,
                "role": "admin",
                "token_written": True,
                "user_binding_valid": True,
                "validity_seconds": lifetime,
            }
        )
        return

    minimum_validity = parse_duration(arguments.minimum_validity)
    token = read_token(
        arguments.token_file,
        FilePolicy(arguments.token_uid, arguments.token_gid, arguments.token_mode),
    )
    now = int(time.time())
    user_id, issued_at, expires_at, remaining = verify_token(
        token, key, now, minimum_validity
    )
    validate_user_binding(arguments.database, user_id)
    _emit(
        {
            "algorithm": ALGORITHM,
            "claims": sorted(CLAIMS),
            "exp": expires_at,
            "iat": issued_at,
            "minimum_validity_seconds": minimum_validity,
            "remaining_seconds": remaining,
            "role": "admin",
            "signature_valid": True,
            "user_binding_valid": True,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        run(arguments)
    except JwtToolError as error:
        _emit({"error": str(error)}, stream=sys.stderr)
        return 2
    except (OSError, sqlite3.Error):
        _emit({"error": "operation failed safely"}, stream=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
