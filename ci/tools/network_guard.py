#!/usr/bin/env python3
"""Fail-closed, per-command network isolation for required host CI.

The guard executes each test command in a fresh Linux network namespace.  The
normal path is an unprivileged user+network namespace.  Some hardened hosts
disable unprivileged user namespaces, so a narrowly scoped ``sudo -n`` fallback
creates only the network namespace and immediately drops back to the invoking
UID/GID before project code starts.  The fallback rejects retained effective
capabilities.  In both cases the parent namespace is never modified, so
connectivity is restored automatically when the child exits.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class NetworkIsolationError(RuntimeError):
    """The required no-network boundary could not be established."""


SCRIPT = Path(__file__).resolve()

Route = tuple[str, ...]
RouteSnapshot = tuple[Route, ...]

_IPV4_ROUTE_HEADER = (
    "Iface",
    "Destination",
    "Gateway",
    "Flags",
    "RefCnt",
    "Use",
    "Metric",
    "Mask",
    "MTU",
    "Window",
    "IRTT",
)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class IsolationPlan:
    strategy: str
    prefix: tuple[str, ...]
    parent_netns: str
    parent_connectivity: tuple[str, tuple[str, ...], RouteSnapshot, RouteSnapshot]
    expected_euid: int
    expected_egid: int
    require_no_capabilities: bool
    execution_environment: tuple[tuple[str, str], ...]


EXECUTION_ENVIRONMENT_KEYS = (
    "PATH",
    "HOME",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "VIRTUAL_ENV",
)


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
        raise NetworkIsolationError(
            f"malformed {label}: expected {width} hexadecimal characters"
        )
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
    """Normalize IPv4 proc routes while excluding only RefCnt and Use."""
    # A fresh network namespace exposes an empty /proc/net/route rather than
    # the header, which is a valid no-route snapshot.  Any non-empty file
    # still requires the exact proc header and strict row validation.
    if not lines:
        return ()
    if tuple(lines[0].split()) != _IPV4_ROUTE_HEADER:
        raise NetworkIsolationError("malformed IPv4 route header")
    routes: list[Route] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 11:
            raise NetworkIsolationError(
                f"malformed IPv4 route line {line_number}: expected 11 columns"
            )
        interface = _normalize_interface(
            fields[0], label=f"IPv4 route line {line_number} interface"
        )
        destination = _normalize_hex(
            fields[1], width=8, label=f"IPv4 route line {line_number} destination"
        )
        gateway = _normalize_hex(
            fields[2], width=8, label=f"IPv4 route line {line_number} gateway"
        )
        flags = _normalize_hex(
            fields[3], width=4, label=f"IPv4 route line {line_number} flags"
        )
        # RefCnt and Use are validated but intentionally omitted from the
        # normalized route because Linux updates them while the topology is
        # unchanged.
        _normalize_decimal(fields[4], label=f"IPv4 route line {line_number} RefCnt")
        _normalize_decimal(fields[5], label=f"IPv4 route line {line_number} Use")
        metric = _normalize_decimal(fields[6], label=f"IPv4 route line {line_number} Metric")
        mask = _normalize_hex(
            fields[7], width=8, label=f"IPv4 route line {line_number} mask"
        )
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
    """Normalize IPv6 proc routes while excluding only ref and use counters."""
    routes: list[Route] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 10:
            raise NetworkIsolationError(
                f"malformed IPv6 route line {line_number}: expected 10 columns"
            )
        destination = _normalize_hex(
            fields[0], width=32, label=f"IPv6 route line {line_number} destination"
        )
        destination_prefix = _normalize_ipv6_prefix(
            fields[1], label=f"IPv6 route line {line_number} destination prefix"
        )
        source = _normalize_hex(
            fields[2], width=32, label=f"IPv6 route line {line_number} source"
        )
        source_prefix = _normalize_ipv6_prefix(
            fields[3], label=f"IPv6 route line {line_number} source prefix"
        )
        gateway = _normalize_hex(
            fields[4], width=32, label=f"IPv6 route line {line_number} gateway"
        )
        metric = _normalize_hex(
            fields[5], width=8, label=f"IPv6 route line {line_number} metric"
        )
        _normalize_hex(fields[6], width=8, label=f"IPv6 route line {line_number} ref")
        _normalize_hex(fields[7], width=8, label=f"IPv6 route line {line_number} use")
        flags = _normalize_hex(
            fields[8], width=8, label=f"IPv6 route line {line_number} flags"
        )
        interface = _normalize_interface(
            fields[9], label=f"IPv6 route line {line_number} interface"
        )
        routes.append(
            (
                destination,
                destination_prefix,
                source,
                source_prefix,
                gateway,
                metric,
                flags,
                interface,
            )
        )
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
        names = tuple(
            sorted(
                line.split(":", 1)[0].strip()
                for line in Path("/proc/net/dev").read_text(encoding="ascii").splitlines()[2:]
                if ":" in line
            )
        )
    except OSError as exc:
        raise NetworkIsolationError(f"cannot inspect network interfaces: {exc}") from exc
    return names


def parent_connectivity_snapshot() -> tuple[str, tuple[str, ...], RouteSnapshot, RouteSnapshot]:
    """Capture route/interface topology without relying on external connectivity."""
    ipv4, ipv6 = _route_snapshot()
    return current_netns(), _interface_names(), ipv4, ipv6


def _assert_no_default_route() -> None:
    ipv4, ipv6 = _route_snapshot()
    for route in ipv4:
        if route[1] == "00000000":
            raise NetworkIsolationError("isolated namespace still has an IPv4 default route")
    for route in ipv6:
        # A fresh namespace can retain Linux's unreachable loopback default
        # entries.  They are harmless; any default through another device is
        # an externally usable route and must fail closed.
        if (
            route[0] == "0" * 32
            and route[1] == "00"
            and route[-1] != "lo"
        ):
            raise NetworkIsolationError("isolated namespace still has an IPv6 default route")


def _assert_loopback_only() -> None:
    interfaces = set(_interface_names())
    if interfaces != {"lo"}:
        raise NetworkIsolationError(
            f"isolated namespace exposes non-loopback interfaces: {sorted(interfaces)}"
        )


def _assert_external_connect_fails() -> None:
    """Use a numeric documentation address so DNS is never consulted."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.settimeout(0.25)
            result = stream.connect_ex(("198.51.100.1", 9))
    except OSError:
        return
    if result == 0:
        raise NetworkIsolationError("isolated namespace established an external network connection")


def assert_isolated(
    *,
    parent_netns: str,
    expected_euid: int,
    expected_egid: int,
    require_no_capabilities: bool,
    address_space_limit_bytes: int | None,
) -> None:
    if current_netns() == parent_netns:
        raise NetworkIsolationError("test command remained in the parent network namespace")
    if os.geteuid() != expected_euid:
        raise NetworkIsolationError(
            f"test command UID mismatch: expected {expected_euid}, got {os.geteuid()}"
        )
    if os.getegid() != expected_egid:
        raise NetworkIsolationError(
            f"test command GID mismatch: expected {expected_egid}, got {os.getegid()}"
        )
    if require_no_capabilities:
        capabilities, no_new_privs = process_security_state()
        retained = sorted(name for name, value in capabilities.items() if value != 0)
        if retained:
            raise NetworkIsolationError(
                "privilege-dropping fallback retained capabilities: "
                + ",".join(retained)
            )
        if no_new_privs != 1:
            raise NetworkIsolationError(
                "privilege-dropping fallback did not set no_new_privs"
            )
        if os.getgroups():
            raise NetworkIsolationError(
                "privilege-dropping fallback retained supplementary groups"
            )
    if address_space_limit_bytes is not None:
        observed = resource.getrlimit(resource.RLIMIT_AS)
        expected = (address_space_limit_bytes, address_space_limit_bytes)
        if observed != expected:
            raise NetworkIsolationError(
                "address-space limit was not inherited by the isolated command: "
                f"expected={expected} observed={observed}"
            )
    _assert_loopback_only()
    _assert_no_default_route()
    _assert_external_connect_fails()


def _child_arguments(
    plan: IsolationPlan,
    command: Sequence[str] | None,
    *,
    address_space_limit_bytes: int | None = None,
) -> list[str]:
    result = [
        sys.executable,
        str(SCRIPT),
        "--child",
        "--parent-netns",
        plan.parent_netns,
        "--expected-euid",
        str(plan.expected_euid),
        "--expected-egid",
        str(plan.expected_egid),
        "--strategy",
        plan.strategy,
    ]
    if plan.require_no_capabilities:
        result.append("--require-no-capabilities")
    for name, value in plan.execution_environment:
        result.extend(("--execution-env", f"{name}={value}"))
    if address_space_limit_bytes is not None:
        result.extend(
            ("--address-space-limit-bytes", str(address_space_limit_bytes))
        )
    if command is None:
        result.append("--probe")
    else:
        result.extend(("--", *command))
    return result


def _candidate_plans() -> list[IsolationPlan]:
    unshare = shutil.which("unshare")
    if not unshare:
        raise NetworkIsolationError("unshare is unavailable")
    parent, interfaces, ipv4, ipv6 = parent_connectivity_snapshot()
    uid = os.getuid()
    gid = os.getgid()
    execution_environment = tuple(
        (name, os.environ[name])
        for name in EXECUTION_ENVIRONMENT_KEYS
        if name in os.environ
    )
    plans = [
        IsolationPlan(
            strategy="user-network-namespace",
            prefix=(unshare, "--user", "--map-root-user", "--net", "--fork"),
            parent_netns=parent,
            parent_connectivity=(parent, interfaces, ipv4, ipv6),
            expected_euid=0,
            expected_egid=0,
            require_no_capabilities=False,
            execution_environment=execution_environment,
        )
    ]
    sudo = shutil.which("sudo")
    setpriv = shutil.which("setpriv")
    if sudo and setpriv:
        plans.append(
            IsolationPlan(
                strategy="sudo-network-namespace-drop-privileges",
                prefix=(
                    sudo,
                    "-n",
                    unshare,
                    "--net",
                    "--fork",
                    setpriv,
                    f"--reuid={uid}",
                    f"--regid={gid}",
                    "--clear-groups",
                    "--inh-caps=-all",
                    "--ambient-caps=-all",
                    "--bounding-set=-all",
                    "--no-new-privs",
                ),
                parent_netns=parent,
                parent_connectivity=(parent, interfaces, ipv4, ipv6),
                expected_euid=uid,
                expected_egid=gid,
                require_no_capabilities=True,
                execution_environment=execution_environment,
            )
        )
    return plans


def wrap_command(
    plan: IsolationPlan,
    command: Sequence[str],
    *,
    address_space_limit_bytes: int | None,
) -> list[str]:
    if not command:
        raise NetworkIsolationError("cannot isolate an empty command")
    return [
        *plan.prefix,
        *_child_arguments(
            plan,
            command,
            address_space_limit_bytes=address_space_limit_bytes,
        ),
    ]


def _probe(
    plan: IsolationPlan, *, address_space_limit_bytes: int | None = None
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                *plan.prefix,
                *_child_arguments(
                    plan,
                    None,
                    address_space_limit_bytes=address_space_limit_bytes,
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (result.stderr or result.stdout).strip()
    return result.returncode == 0, detail


def prepare_isolation() -> IsolationPlan:
    """Select a tested isolation method and prove parent connectivity is untouched."""
    if os.environ.get("ULLM_NETWORK_GUARD_ACTIVE") == "1":
        raise NetworkIsolationError("network guard cannot establish a nested required boundary")
    parent_before = parent_connectivity_snapshot()
    failures: list[str] = []
    for plan in _candidate_plans():
        passed, detail = _probe(plan)
        parent_after = parent_connectivity_snapshot()
        if parent_after != parent_before:
            raise NetworkIsolationError("network guard changed parent network connectivity")
        if passed:
            return plan
        failures.append(f"{plan.strategy}: {detail or 'probe failed'}")
    raise NetworkIsolationError("cannot establish network isolation: " + "; ".join(failures))


def verify_parent_restored(plan: IsolationPlan) -> None:
    if current_netns() != plan.parent_netns:
        raise NetworkIsolationError("test execution did not restore the parent network namespace")
    if parent_connectivity_snapshot() != plan.parent_connectivity:
        raise NetworkIsolationError("test execution did not restore parent connectivity")


def child_main(args: argparse.Namespace) -> int:
    try:
        for assignment in args.execution_env:
            name, separator, value = assignment.partition("=")
            if (
                separator != "="
                or name not in EXECUTION_ENVIRONMENT_KEYS
                or "\x00" in value
            ):
                raise NetworkIsolationError(
                    f"invalid execution environment assignment: {assignment!r}"
                )
            os.environ[name] = value
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if args.address_space_limit_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (
                    args.address_space_limit_bytes,
                    args.address_space_limit_bytes,
                ),
            )
        assert_isolated(
            parent_netns=args.parent_netns,
            expected_euid=args.expected_euid,
            expected_egid=args.expected_egid,
            require_no_capabilities=args.require_no_capabilities,
            address_space_limit_bytes=args.address_space_limit_bytes,
        )
    except (NetworkIsolationError, OSError, ValueError) as exc:
        print(f"network guard: {exc}", file=sys.stderr)
        return 2
    if args.probe:
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("network guard: missing command", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env["ULLM_NETWORK_GUARD_ACTIVE"] = "1"
    env["ULLM_CI_NETWORK_DISABLED"] = "1"
    env["ULLM_NETWORK_GUARD_STRATEGY"] = args.strategy or "unknown"
    env["ULLM_EMIT_TEST_COUNTS"] = "1"
    try:
        os.execvpe(command[0], command, env)
    except OSError as exc:
        print(f"network guard: cannot execute isolated command: {exc}", file=sys.stderr)
        return 127


def self_test() -> int:
    try:
        plan = prepare_isolation()
        inherited, detail = _probe(
            plan, address_space_limit_bytes=1024 * 1024 * 1024
        )
        if not inherited:
            raise NetworkIsolationError(
                "address-space inheritance probe failed: "
                + (detail or "probe failed")
            )
        verify_parent_restored(plan)
    except NetworkIsolationError as exc:
        print(f"network guard self-test: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "network guard self-test: PASS "
        f"strategy={plan.strategy} parent_restored=true "
        "address_space_inherited=true privilege_drop_verified=true"
    )
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
    if (
        not args.child
        or args.parent_netns is None
        or args.expected_euid is None
        or args.expected_egid is None
    ):
        parser().error("only --self-test or a complete --child invocation is allowed")
    if (
        args.address_space_limit_bytes is not None
        and args.address_space_limit_bytes <= 0
    ):
        parser().error("--address-space-limit-bytes must be positive")
    return child_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
