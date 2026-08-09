#!/usr/bin/env python3
"""Fail-closed per-command network isolation for required host CI."""

from __future__ import annotations

import argparse
import math
import os
import resource
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class NetworkIsolationError(RuntimeError):
    """The required no-network boundary could not be established."""


SCRIPT = Path(__file__).resolve()
Route = tuple[str, ...]
RouteSnapshot = tuple[Route, ...]

_IPV4_ROUTE_HEADER = (
    "Iface", "Destination", "Gateway", "Flags", "RefCnt", "Use", "Metric",
    "Mask", "MTU", "Window", "IRTT",
)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class IsolationPlan:
    strategy: str
    prefix: tuple[str, ...]
    parent_netns: str
    expected_euid: int
    expected_egid: int
    require_no_capabilities: bool
    execution_environment: tuple[tuple[str, str], ...]


EXECUTION_ENVIRONMENT_KEYS = ("PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME", "VIRTUAL_ENV")
LOOPBACK_INIT_SCRIPT = '"$1" link set dev lo up && shift && exec "$@"'
SUDO_FALLBACK_TOOL_CANDIDATES = (
    ("sudo", ("/usr/bin/sudo",)),
    ("unshare", ("/usr/bin/unshare",)),
    ("shell", ("/bin/sh", "/usr/bin/sh")),
    ("ip", ("/usr/sbin/ip", "/usr/bin/ip")),
    ("setpriv", ("/usr/bin/setpriv",)),
)
_SYSTEM_TOOL_DIRECTORIES = frozenset(("/bin", "/sbin", "/usr/bin", "/usr/sbin"))


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise NetworkIsolationError("network isolation deadline expired")


def _validate_trusted_system_metadata(
    role: str,
    path: Path,
    metadata: os.stat_result,
    *,
    require_regular: bool,
) -> None:
    expected_kind = stat.S_ISREG if require_regular else stat.S_ISDIR
    kind = "regular file" if require_regular else "directory"
    if not expected_kind(metadata.st_mode):
        raise NetworkIsolationError(f"sudo fallback {role} is not a {kind}: {path}")
    if metadata.st_uid != 0:
        raise NetworkIsolationError(f"sudo fallback {role} is not root-owned: {path}")
    if metadata.st_mode & 0o022:
        raise NetworkIsolationError(
            f"sudo fallback {role} is group/world writable: {path}"
        )
    if require_regular and not metadata.st_mode & 0o111:
        raise NetworkIsolationError(f"sudo fallback {role} is not executable: {path}")


def _validate_trusted_directory_chain(role: str, directory: Path) -> None:
    current = directory
    while True:
        try:
            metadata = os.stat(current, follow_symlinks=False)
        except OSError as exc:
            raise NetworkIsolationError(
                f"cannot inspect sudo fallback {role} directory {current}: {exc}"
            ) from exc
        _validate_trusted_system_metadata(
            role, current, metadata, require_regular=False
        )
        if current.parent == current:
            return
        current = current.parent


def _inspect_system_tool_candidate(
    role: str,
    candidate: str,
) -> tuple[str, tuple[int, int]] | None:
    path = Path(candidate)
    if not path.is_absolute() or str(path) != candidate:
        raise NetworkIsolationError(
            f"sudo fallback {role} candidate is not a canonical absolute path: {candidate!r}"
        )
    try:
        alias_metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise NetworkIsolationError(
            f"cannot inspect sudo fallback {role} candidate {path}: {exc}"
        ) from exc
    if alias_metadata.st_uid != 0:
        raise NetworkIsolationError(
            f"sudo fallback {role} candidate is not root-owned: {path}"
        )
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NetworkIsolationError(
            f"sudo fallback {role} candidate has an unresolved or ambiguous symlink: {path}"
        ) from exc
    _validate_trusted_directory_chain(role, resolved_parent)
    if str(resolved.parent) not in _SYSTEM_TOOL_DIRECTORIES:
        raise NetworkIsolationError(
            f"sudo fallback {role} resolved outside fixed system tool directories: {resolved}"
        )
    try:
        metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise NetworkIsolationError(
            f"cannot inspect resolved sudo fallback {role} tool {resolved}: {exc}"
        ) from exc
    _validate_trusted_system_metadata(role, resolved, metadata, require_regular=True)
    return str(resolved), (metadata.st_dev, metadata.st_ino)


def _select_trusted_system_tool(role: str, candidates: Sequence[str]) -> str:
    inspected: list[tuple[str, tuple[int, int]]] = []
    for candidate in candidates:
        result = _inspect_system_tool_candidate(role, candidate)
        if result is not None:
            inspected.append(result)
    if not inspected:
        raise NetworkIsolationError(
            f"sudo fallback {role} is unavailable at fixed system paths"
        )
    identities = {identity for _, identity in inspected}
    if len(identities) != 1:
        raise NetworkIsolationError(
            f"sudo fallback {role} candidates resolve to ambiguous tool identities"
        )
    return inspected[0][0]


def _sudo_fallback_tools() -> dict[str, str] | None:
    try:
        return {
            role: _select_trusted_system_tool(role, candidates)
            for role, candidates in SUDO_FALLBACK_TOOL_CANDIDATES
        }
    except NetworkIsolationError:
        return None


def current_netns() -> str:
    try:
        return os.readlink("/proc/self/ns/net")
    except OSError as exc:
        raise NetworkIsolationError(f"cannot inspect current network namespace: {exc}") from exc


def process_security_state() -> tuple[dict[str, int], int]:
    try:
        capabilities: dict[str, int] = {}
        no_new_privs: int | None = None
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator != ":":
                continue
            if name in {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"}:
                capabilities[name] = int(value.strip(), 16)
            elif name == "NoNewPrivs":
                no_new_privs = int(value.strip())
    except (OSError, ValueError) as exc:
        raise NetworkIsolationError(f"cannot inspect process security state: {exc}") from exc
    expected = {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"}
    if set(capabilities) != expected or no_new_privs is None:
        raise NetworkIsolationError("cannot find complete capability/no-new-privs state")
    return capabilities, no_new_privs


def _normalize_hex(field: str, *, width: int, label: str) -> str:
    if len(field) != width or any(character not in _HEX_DIGITS for character in field):
        raise NetworkIsolationError(f"malformed {label}: expected {width} hexadecimal characters")
    return field.lower()


def _normalize_decimal(field: str, *, label: str) -> str:
    if not field or any(character not in "0123456789" for character in field):
        raise NetworkIsolationError(f"malformed {label}: expected an unsigned decimal integer")
    value = int(field, 10)
    if value > 0xFFFFFFFF:
        raise NetworkIsolationError(f"malformed {label}: value exceeds a 32-bit unsigned integer")
    return str(value)


def _normalize_interface(field: str, *, label: str) -> str:
    if not field or any(character.isspace() or character == "\x00" for character in field):
        raise NetworkIsolationError(f"malformed {label}: invalid interface name")
    return field


def _normalize_ipv4_routes(lines: Sequence[str]) -> RouteSnapshot:
    """Normalize a header-bearing IPv4 route table; the header may have no rows."""
    if not lines:
        raise NetworkIsolationError("malformed IPv4 route table: missing header")
    if tuple(lines[0].split()) != _IPV4_ROUTE_HEADER:
        raise NetworkIsolationError("malformed IPv4 route header")
    routes: list[Route] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 11:
            raise NetworkIsolationError(f"malformed IPv4 route line {line_number}: expected 11 columns")
        interface = _normalize_interface(fields[0], label=f"IPv4 route line {line_number} interface")
        destination = _normalize_hex(fields[1], width=8, label=f"IPv4 route line {line_number} destination")
        gateway = _normalize_hex(fields[2], width=8, label=f"IPv4 route line {line_number} gateway")
        flags = _normalize_hex(fields[3], width=4, label=f"IPv4 route line {line_number} flags")
        _normalize_decimal(fields[4], label=f"IPv4 route line {line_number} RefCnt")
        _normalize_decimal(fields[5], label=f"IPv4 route line {line_number} Use")
        metric = _normalize_decimal(fields[6], label=f"IPv4 route line {line_number} Metric")
        mask = _normalize_hex(fields[7], width=8, label=f"IPv4 route line {line_number} mask")
        mtu = _normalize_decimal(fields[8], label=f"IPv4 route line {line_number} MTU")
        window = _normalize_decimal(fields[9], label=f"IPv4 route line {line_number} Window")
        irtt = _normalize_decimal(fields[10], label=f"IPv4 route line {line_number} IRTT")
        routes.append((interface, destination, gateway, flags, metric, mask, mtu, window, irtt))
    return tuple(routes)


def _normalize_ipv6_prefix(field: str, *, label: str) -> str:
    normalized = _normalize_hex(field, width=2, label=label)
    if int(normalized, 16) > 128:
        raise NetworkIsolationError(f"malformed {label}: prefix length exceeds 128")
    return normalized


def _normalize_ipv6_routes(lines: Sequence[str]) -> RouteSnapshot:
    routes: list[Route] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 10:
            raise NetworkIsolationError(f"malformed IPv6 route line {line_number}: expected 10 columns")
        destination = _normalize_hex(fields[0], width=32, label=f"IPv6 route line {line_number} destination")
        destination_prefix = _normalize_ipv6_prefix(fields[1], label=f"IPv6 route line {line_number} destination prefix")
        source = _normalize_hex(fields[2], width=32, label=f"IPv6 route line {line_number} source")
        source_prefix = _normalize_ipv6_prefix(fields[3], label=f"IPv6 route line {line_number} source prefix")
        gateway = _normalize_hex(fields[4], width=32, label=f"IPv6 route line {line_number} gateway")
        metric = _normalize_hex(fields[5], width=8, label=f"IPv6 route line {line_number} metric")
        _normalize_hex(fields[6], width=8, label=f"IPv6 route line {line_number} ref")
        _normalize_hex(fields[7], width=8, label=f"IPv6 route line {line_number} use")
        flags = _normalize_hex(fields[8], width=8, label=f"IPv6 route line {line_number} flags")
        interface = _normalize_interface(fields[9], label=f"IPv6 route line {line_number} interface")
        routes.append((destination, destination_prefix, source, source_prefix, gateway, metric, flags, interface))
    return tuple(routes)


def _route_snapshot() -> tuple[RouteSnapshot, RouteSnapshot]:
    try:
        ipv4_lines = Path("/proc/net/route").read_text(encoding="ascii").splitlines()
        ipv6_lines = Path("/proc/net/ipv6_route").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise NetworkIsolationError(f"cannot inspect network routes: {exc}") from exc
    return _normalize_ipv4_routes(ipv4_lines), _normalize_ipv6_routes(ipv6_lines)


def _interface_names() -> tuple[str, ...]:
    try:
        return tuple(sorted(
            line.split(":", 1)[0].strip()
            for line in Path("/proc/net/dev").read_text(encoding="ascii").splitlines()[2:]
            if ":" in line
        ))
    except OSError as exc:
        raise NetworkIsolationError(f"cannot inspect network interfaces: {exc}") from exc


def parent_connectivity_snapshot() -> tuple[str, tuple[str, ...], RouteSnapshot, RouteSnapshot]:
    ipv4, ipv6 = _route_snapshot()
    return current_netns(), _interface_names(), ipv4, ipv6


def _assert_no_default_route() -> None:
    ipv4, ipv6 = _route_snapshot()
    if any(route[1] == "00000000" for route in ipv4):
        raise NetworkIsolationError("isolated namespace still has an IPv4 default route")
    if any(route[0] == "0" * 32 and route[1] == "00" and route[-1] != "lo" for route in ipv6):
        raise NetworkIsolationError("isolated namespace still has an IPv6 default route")


def _assert_loopback_only() -> None:
    interfaces = set(_interface_names())
    if interfaces != {"lo"}:
        raise NetworkIsolationError(f"isolated namespace exposes non-loopback interfaces: {sorted(interfaces)}")


def _assert_external_connect_fails(*, deadline: float | None = None) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            _check_deadline(deadline)
            remaining = 0.25 if deadline is None else min(0.25, deadline - time.monotonic())
            if remaining <= 0:
                raise NetworkIsolationError("network isolation deadline expired")
            stream.settimeout(remaining)
            result = stream.connect_ex(("198.51.100.1", 9))
    except OSError:
        _check_deadline(deadline)
        return
    _check_deadline(deadline)
    if result == 0:
        raise NetworkIsolationError("isolated namespace established an external network connection")


def assert_isolated(
    *,
    parent_netns: str,
    expected_euid: int,
    expected_egid: int,
    require_no_capabilities: bool,
    address_space_limit_bytes: int | None,
    outer_deadline: float | None = None,
) -> None:
    _check_deadline(outer_deadline)
    if current_netns() == parent_netns:
        raise NetworkIsolationError("test command remained in the parent network namespace")
    if os.geteuid() != expected_euid:
        raise NetworkIsolationError(f"test command UID mismatch: expected {expected_euid}, got {os.geteuid()}")
    if os.getegid() != expected_egid:
        raise NetworkIsolationError(f"test command GID mismatch: expected {expected_egid}, got {os.getegid()}")
    if require_no_capabilities:
        capabilities, no_new_privs = process_security_state()
        retained = sorted(name for name, value in capabilities.items() if value != 0)
        if retained:
            raise NetworkIsolationError("privilege-dropping fallback retained capabilities: " + ",".join(retained))
        if no_new_privs != 1:
            raise NetworkIsolationError("privilege-dropping fallback did not set no_new_privs")
        if os.getgroups():
            raise NetworkIsolationError("privilege-dropping fallback retained supplementary groups")
    if address_space_limit_bytes is not None:
        expected = (address_space_limit_bytes, address_space_limit_bytes)
        observed = resource.getrlimit(resource.RLIMIT_AS)
        if observed != expected:
            raise NetworkIsolationError(f"address-space limit was not inherited by the isolated command: expected={expected} observed={observed}")
    _assert_loopback_only()
    _assert_no_default_route()
    _assert_external_connect_fails(deadline=outer_deadline)
    _check_deadline(outer_deadline)


def _child_arguments(
    plan: IsolationPlan,
    command: Sequence[str] | None,
    *,
    address_space_limit_bytes: int | None = None,
    outer_deadline: float | None = None,
) -> list[str]:
    result = [
        sys.executable, str(SCRIPT), "--child", "--parent-netns", plan.parent_netns,
        "--expected-euid", str(plan.expected_euid), "--expected-egid", str(plan.expected_egid),
        "--strategy", plan.strategy,
    ]
    if plan.require_no_capabilities:
        result.append("--require-no-capabilities")
    for name, value in plan.execution_environment:
        result.extend(("--execution-env", f"{name}={value}"))
    if address_space_limit_bytes is not None:
        result.extend(("--address-space-limit-bytes", str(address_space_limit_bytes)))
    if outer_deadline is not None:
        result.extend(("--outer-deadline", repr(outer_deadline)))
    result.append("--probe" if command is None else "--")
    if command is not None:
        result.extend(command)
    return result


def _candidate_plans(parent_netns: str) -> list[IsolationPlan]:
    unshare = shutil.which("unshare")
    if not unshare:
        raise NetworkIsolationError("unshare is unavailable")
    uid = os.getuid()
    gid = os.getgid()
    environment = tuple((name, os.environ[name]) for name in EXECUTION_ENVIRONMENT_KEYS if name in os.environ)
    plans = [IsolationPlan(
        strategy="user-network-namespace",
        prefix=(unshare, "--user", "--map-root-user", "--net", "--fork"),
        parent_netns=parent_netns, expected_euid=0, expected_egid=0,
        require_no_capabilities=False, execution_environment=environment,
    )]
    sudo_tools = _sudo_fallback_tools()
    if sudo_tools is not None:
        plans.append(IsolationPlan(
            strategy="sudo-network-namespace-drop-privileges",
            prefix=(sudo_tools["sudo"], "-n", sudo_tools["unshare"], "--net", "--fork",
                    sudo_tools["shell"], "-c", LOOPBACK_INIT_SCRIPT, "sllm-loopback-init", sudo_tools["ip"],
                    sudo_tools["setpriv"],
                    f"--reuid={uid}", f"--regid={gid}", "--clear-groups",
                    "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all",
                    "--no-new-privs"),
            parent_netns=parent_netns, expected_euid=uid, expected_egid=gid,
            require_no_capabilities=True, execution_environment=environment,
        ))
    return plans


def wrap_command(
    plan: IsolationPlan,
    command: Sequence[str],
    *,
    address_space_limit_bytes: int | None,
    outer_deadline: float | None = None,
) -> list[str]:
    _check_deadline(outer_deadline)
    if not command:
        raise NetworkIsolationError("cannot isolate an empty command")
    return [*plan.prefix, *_child_arguments(plan, command, address_space_limit_bytes=address_space_limit_bytes, outer_deadline=outer_deadline)]


def _probe(
    plan: IsolationPlan,
    *,
    address_space_limit_bytes: int | None = None,
    outer_deadline: float | None = None,
) -> tuple[bool, str]:
    try:
        _check_deadline(outer_deadline)
        timeout = 10.0
        if outer_deadline is not None:
            timeout = outer_deadline - time.monotonic()
            if timeout <= 0:
                raise NetworkIsolationError("network isolation deadline expired")
        result = subprocess.run(
            [*plan.prefix, *_child_arguments(plan, None, address_space_limit_bytes=address_space_limit_bytes, outer_deadline=outer_deadline)],
            text=True, capture_output=True, check=False, timeout=timeout,
        )
        _check_deadline(outer_deadline)
    except (OSError, subprocess.TimeoutExpired, NetworkIsolationError) as exc:
        return False, str(exc)
    detail = (result.stderr or result.stdout).strip()
    return result.returncode == 0, detail


def prepare_isolation(*, outer_deadline: float | None = None) -> IsolationPlan:
    parent_netns = current_netns()
    _check_deadline(outer_deadline)
    if os.environ.get("SLLM_NETWORK_GUARD_ACTIVE") == "1":
        raise NetworkIsolationError("network guard cannot establish a nested required boundary")
    plans = _candidate_plans(parent_netns)
    _check_deadline(outer_deadline)
    failures: list[str] = []
    for plan in plans:
        passed, detail = _probe(plan, outer_deadline=outer_deadline)
        if current_netns() != parent_netns:
            raise NetworkIsolationError("network guard changed the parent network namespace")
        _check_deadline(outer_deadline)
        if passed:
            return plan
        failures.append(f"{plan.strategy}: {detail or 'probe failed'}")
    raise NetworkIsolationError("cannot establish network isolation: " + "; ".join(failures))


def verify_parent_restored(plan: IsolationPlan, *, outer_deadline: float | None = None) -> None:
    if current_netns() != plan.parent_netns:
        raise NetworkIsolationError("test execution changed the parent network namespace")
    _check_deadline(outer_deadline)


def child_main(args: argparse.Namespace) -> int:
    try:
        for assignment in args.execution_env:
            name, separator, value = assignment.partition("=")
            if separator != "=" or name not in EXECUTION_ENVIRONMENT_KEYS or "\x00" in value:
                raise NetworkIsolationError(f"invalid execution environment assignment: {assignment!r}")
            os.environ[name] = value
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if args.address_space_limit_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (args.address_space_limit_bytes, args.address_space_limit_bytes))
        assert_isolated(
            parent_netns=args.parent_netns,
            expected_euid=args.expected_euid,
            expected_egid=args.expected_egid,
            require_no_capabilities=args.require_no_capabilities,
            address_space_limit_bytes=args.address_space_limit_bytes,
            outer_deadline=args.outer_deadline,
        )
        _check_deadline(args.outer_deadline)
    except (NetworkIsolationError, OSError, ValueError) as exc:
        print(f"network guard: {exc}", file=sys.stderr)
        return 2
    if args.probe:
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("network guard: missing command", file=sys.stderr)
        return 2
    try:
        env = os.environ.copy()
        env["SLLM_NETWORK_GUARD_ACTIVE"] = "1"
        env["SLLM_CI_NETWORK_DISABLED"] = "1"
        env["SLLM_NETWORK_GUARD_STRATEGY"] = args.strategy or "unknown"
        env["SLLM_EMIT_TEST_COUNTS"] = "1"
        _check_deadline(args.outer_deadline)
        os.execvpe(command[0], command, env)
    except NetworkIsolationError as exc:
        print(f"network guard: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"network guard: cannot execute isolated command: {exc}", file=sys.stderr)
        return 127


def self_test() -> int:
    try:
        plan = prepare_isolation()
        inherited, detail = _probe(plan, address_space_limit_bytes=1024 * 1024 * 1024)
        if not inherited:
            raise NetworkIsolationError("address-space inheritance probe failed: " + (detail or "probe failed"))
        verify_parent_restored(plan)
    except NetworkIsolationError as exc:
        print(f"network guard self-test: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"network guard self-test: PASS strategy={plan.strategy} parent_restored=true address_space_inherited=true privilege_drop_verified=true")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--child", action="store_true")
    result.add_argument("--probe", action="store_true")
    result.add_argument("--parent-netns")
    result.add_argument("--expected-euid", type=int)
    result.add_argument("--expected-egid", type=int)
    result.add_argument("--require-no-capabilities", action="store_true")
    result.add_argument("--address-space-limit-bytes", type=int)
    result.add_argument("--outer-deadline", type=float)
    result.add_argument("--execution-env", action="append", default=[])
    result.add_argument("--strategy")
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        if args.child or args.command:
            parser().error("--self-test cannot be combined with child execution")
        return self_test()
    if not args.child or args.parent_netns is None or args.expected_euid is None or args.expected_egid is None:
        parser().error("only --self-test or a complete --child invocation is allowed")
    if args.address_space_limit_bytes is not None and args.address_space_limit_bytes <= 0:
        parser().error("--address-space-limit-bytes must be positive")
    if args.outer_deadline is not None and not math.isfinite(args.outer_deadline):
        parser().error("--outer-deadline must be finite")
    return child_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
