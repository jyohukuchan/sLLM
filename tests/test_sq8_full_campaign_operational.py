from __future__ import annotations

import dataclasses
import gc
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "sq8_full_campaign_operational.py"


def load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "sq8_full_campaign_operational_test", TOOL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()

CONTAINER_ID = "1" * 64
IMAGE_ID = TOOL.OPENWEBUI_IMAGE_ID
NETWORK_ID = TOOL.OPENWEBUI_NETWORK_ID
GATEWAY_URL = "http://172.20.0.1:8000/readyz"
OPENWEBUI_URL = "http://127.0.0.1:3000/health"


def compact(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def proc_stat(pid: int, ppid: int, starttime: int) -> str:
    remaining = [str(field) for field in range(4, 53)]
    remaining[0] = str(ppid)
    remaining[18] = str(starttime)
    return f"{pid} (ullm-sq8-worker) S " + " ".join(remaining) + "\n"


def make_fake_worker_proc(root: Path, pid: int) -> None:
    process = root / str(pid)
    (process / "fd").mkdir(parents=True)
    (process / "task" / str(pid)).mkdir(parents=True)
    (process / "stat").write_text(proc_stat(pid, 101, 55), encoding="ascii")
    (process / "status").write_text(
        "Name:\tullm-sq8-worker\nVmRSS:\t  12345 kB\nThreads:\t7\n",
        encoding="ascii",
    )
    (process / "task" / str(pid) / "children").write_text("\n", encoding="ascii")
    (process / "fd" / "0").write_bytes(b"")
    executable = root / "ullm-sq8-worker"
    executable.write_bytes(b"binary")
    os.symlink(executable, process / "exe")


def systemd_raw(**changes: str) -> bytes:
    fields = {
        "ActiveState": "active",
        "SubState": "running",
        "Result": "success",
        "MainPID": "101",
        "NRestarts": "2",
        "Restart": "on-failure",
        "RestartUSec": "10s",
        "StartLimitIntervalUSec": "15min",
        "StartLimitBurst": "3",
        "ActiveEnterTimestampMonotonic": "100000000",
    }
    fields.update(changes)
    return "".join(f"{key}={value}\n" for key, value in fields.items()).encode()


def container_document() -> list[dict[str, Any]]:
    return [
        {
            "Id": CONTAINER_ID,
            "Name": "/open-webui",
            "Image": IMAGE_ID,
            "RestartCount": 0,
            "State": {
                "Status": "running",
                "Running": True,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "Pid": 404,
                "StartedAt": "2026-07-11T12:00:00.000000000Z",
                "Health": {"Status": "healthy"},
            },
            "NetworkSettings": {
                "Networks": {"open-webui-network": {"NetworkID": NETWORK_ID}}
            },
        }
    ]


def expectation() -> Any:
    return TOOL.OperationalExpectation(
        service_unit="ullm-openai.service",
        gateway_pid=101,
        worker_pid=202,
        container_name="open-webui",
        container_id=CONTAINER_ID,
        image_id=IMAGE_ID,
        network_name="open-webui-network",
        network_id=NETWORK_ID,
        gateway_ready_url=GATEWAY_URL,
        openwebui_health_url=OPENWEBUI_URL,
        observer_socket=Path("/run/ullm/lifecycle-observer.sock"),
        observer_parent_uid=1000,
        observer_parent_gid=1000,
    )


def parent_identity(*, inode: int = 7, mode: int | None = None) -> Any:
    return TOOL.FileIdentity(
        1,
        inode,
        stat.S_IFDIR | (0o750 if mode is None else mode),
        2,
        1000,
        1000,
    )


def observer_snapshot(
    *,
    parent: Any | None = None,
    named: Any | None = None,
    child: Any | None = None,
) -> Any:
    actual = parent_identity() if parent is None else parent
    return TOOL.ObserverPathSnapshot(
        actual,
        actual if named is None else named,
        child,
    )


def gpu_snapshot(**changes: Any) -> Any:
    values = {
        "worker_pid": 202,
        "worker_starttime_ticks": 55,
        "amd_vram_bytes": 20_000_000_000,
        "kfd_vram_bytes": 20_000_000_000,
        "positive_kfd_pids": (202,),
        "amd_list_sha256": "a" * 64,
        "kfd_attempt_count": 1,
    }
    values.update(changes)
    return TOOL.GpuIsolationSnapshot(**values)


class FakeCommands:
    def __init__(
        self,
        systemd: list[bytes] | None = None,
        container: list[bytes] | None = None,
        extras: dict[tuple[str, ...], list[bytes]] | None = None,
    ) -> None:
        self.responses = {
            TOOL._systemd_command("ullm-openai.service"): list(
                systemd or [systemd_raw(), systemd_raw()]
            ),
            TOOL._docker_command("open-webui"): list(
                container
                or [compact(container_document()), compact(container_document())]
            ),
        }
        if extras:
            self.responses.update(extras)
        self.calls: list[tuple[tuple[str, ...], str, float, int]] = []

    def run(
        self,
        arguments: tuple[str, ...],
        *,
        label: str,
        timeout_seconds: float,
        maximum_stdout_bytes: int,
    ) -> bytes:
        self.calls.append((arguments, label, timeout_seconds, maximum_stdout_bytes))
        values = self.responses.get(arguments)
        if not values:
            raise AssertionError(f"unexpected or exhausted command: {arguments}")
        return values.pop(0)


class FakeHttp:
    def __init__(
        self,
        health: list[Any] | None = None,
    ) -> None:
        self.responses = {
            OPENWEBUI_URL: list(
                health
                or [
                    TOOL.HttpResponse(OPENWEBUI_URL, 200, TOOL.OPENWEBUI_HEALTH_BODY),
                    TOOL.HttpResponse(OPENWEBUI_URL, 200, TOOL.OPENWEBUI_HEALTH_BODY),
                ]
            ),
        }
        self.calls: list[tuple[str, float, int]] = []

    def get(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_body_bytes: int,
    ) -> Any:
        self.calls.append((url, timeout_seconds, maximum_body_bytes))
        values = self.responses.get(url)
        if not values:
            raise AssertionError(f"unexpected or exhausted HTTP GET: {url}")
        return values.pop(0)


class FakeGatewayHttp:
    def __init__(self, ready: list[Any] | None = None) -> None:
        self.responses = list(
            ready
            or [
                TOOL.HttpResponse(GATEWAY_URL, 200, TOOL.GATEWAY_READY_BODY),
                TOOL.HttpResponse(GATEWAY_URL, 200, TOOL.GATEWAY_READY_BODY),
            ]
        )
        self.calls: list[tuple[int, float, int]] = []

    def get(
        self,
        container_pid: int,
        *,
        timeout_seconds: float,
        maximum_body_bytes: int,
    ) -> Any:
        self.calls.append((container_pid, timeout_seconds, maximum_body_bytes))
        if not self.responses:
            raise AssertionError("gateway HTTP fixture exhausted")
        return self.responses.pop(0)


class FakeObserverHandle:
    def __init__(
        self,
        snapshots: list[Any] | None = None,
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.snapshots = list(snapshots or [observer_snapshot(), observer_snapshot()])
        self.closed = 0
        self.close_error = close_error

    def snapshot(self) -> Any:
        if not self.snapshots:
            raise AssertionError("observer snapshot fixture exhausted")
        return self.snapshots.pop(0)

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


class FakeObserverReader:
    def __init__(self, handle: FakeObserverHandle | None = None) -> None:
        self.handle = handle or FakeObserverHandle()
        self.paths: list[Path] = []

    def open_path(self, path: Path) -> FakeObserverHandle:
        self.paths.append(path)
        return self.handle


class FakeGpu:
    def __init__(self, result: Any | None = None) -> None:
        self.result = gpu_snapshot() if result is None else result
        self.pids: list[int] = []

    def capture(self, worker_pid: int) -> Any:
        self.pids.append(worker_pid)
        return self.result


class FakeClock:
    def __init__(self, values: list[int] | None = None) -> None:
        self.values = list(values or [1_100_000_000_000, 1_101_000_000_000])

    def __call__(self) -> int:
        if not self.values:
            raise AssertionError("clock fixture exhausted")
        return self.values.pop(0)


def dependencies(
    *,
    commands: Any | None = None,
    http: Any | None = None,
    gateway_http: Any | None = None,
    observer: Any | None = None,
    gpu: Any | None = None,
    clock: Callable[[], int] | None = None,
) -> Any:
    return TOOL.OperationalDependencies(
        commands or FakeCommands(),
        http or FakeHttp(),
        gateway_http or FakeGatewayHttp(),
        observer or FakeObserverReader(),
        gpu or FakeGpu(),
        clock or FakeClock(),
    )


class OperationalPreflightTests(unittest.TestCase):
    def test_production_container_discovery_allowlist_is_one_inspect(self) -> None:
        commands = TOOL.production_container_discovery_commands()
        self.assertEqual(
            commands,
            frozenset({TOOL._docker_command(TOOL.OPENWEBUI_CONTAINER_NAME)}),
        )
        flattened = " ".join(next(iter(commands)))
        self.assertNotIn(" run ", f" {flattened} ")
        self.assertNotIn(" exec ", f" {flattened} ")

    def test_production_container_discovery_feeds_full_preflight_identity(self) -> None:
        commands = FakeCommands(
            container=[
                compact(container_document()),
                compact(container_document()),
                compact(container_document()),
            ]
        )
        discovered = TOOL.discover_production_openwebui_container(commands)
        self.assertEqual(discovered.container_id, CONTAINER_ID)
        self.assertEqual(discovered.image_id, TOOL.OPENWEBUI_IMAGE_ID)
        self.assertEqual(
            discovered.network_ids,
            ((TOOL.OPENWEBUI_NETWORK_NAME, TOOL.OPENWEBUI_NETWORK_ID),),
        )
        self.assertEqual(discovered.pid, 404)
        self.assertEqual(discovered.restart_count, 0)
        self.assertEqual(discovered.started_at, "2026-07-11T12:00:00.000000000Z")

        bound = dataclasses.replace(expectation(), container_id=discovered.container_id)
        result = TOOL.run_operational_preflight(
            bound,
            dependencies(commands=commands),
        )
        self.assertEqual(result.container, discovered)
        docker_calls = [
            call[0] for call in commands.calls if call[0][0] == TOOL.DOCKER_BIN
        ]
        self.assertEqual(
            docker_calls,
            [TOOL._docker_command(TOOL.OPENWEBUI_CONTAINER_NAME)] * 3,
        )
        flattened = " ".join(item for call in docker_calls for item in call)
        self.assertNotIn(" run ", f" {flattened} ")
        self.assertNotIn(" exec ", f" {flattened} ")

    def test_production_container_discovery_rejects_identity_or_state_drift(
        self,
    ) -> None:
        cases: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
            (
                "container-id",
                lambda item: item.__setitem__("Id", "not-an-id"),
                "container or image",
            ),
            (
                "image",
                lambda item: item.__setitem__("Image", "sha256:" + "f" * 64),
                "container or image",
            ),
            (
                "network",
                lambda item: item["NetworkSettings"]["Networks"][
                    "open-webui-network"
                ].__setitem__("NetworkID", "f" * 64),
                "network content",
            ),
            (
                "restarting",
                lambda item: item["State"].__setitem__("Restarting", True),
                "running and healthy",
            ),
            (
                "oom",
                lambda item: item["State"].__setitem__("OOMKilled", True),
                "running and healthy",
            ),
        )
        for label, mutate, message in cases:
            document = container_document()
            mutate(document[0])
            commands = FakeCommands(container=[compact(document)])
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                TOOL.discover_production_openwebui_container(commands)
            self.assertEqual(
                [call[0] for call in commands.calls],
                [TOOL._docker_command("open-webui")],
            )

    def test_success_is_bounded_read_only_and_double_checked(self) -> None:
        commands = FakeCommands()
        http = FakeHttp()
        gateway_http = FakeGatewayHttp()
        observer = FakeObserverReader()
        gpu = FakeGpu()
        result = TOOL.run_operational_preflight(
            expectation(),
            dependencies(
                commands=commands,
                http=http,
                gateway_http=gateway_http,
                observer=observer,
                gpu=gpu,
            ),
        )

        self.assertEqual(result.systemd.main_pid, 101)
        self.assertGreaterEqual(
            result.systemd.active_age_ns, TOOL.MINIMUM_ACTIVE_AGE_NS
        )
        self.assertEqual(result.container.container_id, CONTAINER_ID)
        self.assertEqual(result.gpu.positive_kfd_pids, (202,))
        self.assertEqual(gpu.pids, [202, 202])
        self.assertEqual(
            [call[0][:3] for call in commands.calls],
            [
                (TOOL.SYSTEMCTL_BIN, "show", "ullm-openai.service"),
                (TOOL.DOCKER_BIN, "container", "inspect"),
                (TOOL.DOCKER_BIN, "container", "inspect"),
                (TOOL.SYSTEMCTL_BIN, "show", "ullm-openai.service"),
            ],
        )
        self.assertEqual(
            [call[0] for call in http.calls],
            [OPENWEBUI_URL, OPENWEBUI_URL],
        )
        self.assertTrue(
            all(call[1] == TOOL.HTTP_TIMEOUT_SECONDS for call in http.calls)
        )
        self.assertEqual([call[0] for call in gateway_http.calls], [404, 404])
        self.assertEqual(observer.paths, [Path("/run/ullm/lifecycle-observer.sock")])
        self.assertEqual(observer.handle.closed, 1)
        flattened = " ".join(item for call in commands.calls for item in call[0])
        for forbidden in ("restart", "start", "stop", "kill", "run", "exec"):
            self.assertNotIn(f" {forbidden} ", f" {flattened} ")

    def test_gateway_and_openwebui_require_exact_200_bodies(self) -> None:
        cases = (
            (
                FakeGatewayHttp(
                    ready=[TOOL.HttpResponse(GATEWAY_URL, 503, TOOL.GATEWAY_READY_BODY)]
                ),
                "gateway readiness",
            ),
            (
                FakeGatewayHttp(
                    ready=[TOOL.HttpResponse(GATEWAY_URL, 200, b'{"status": "ready"}')]
                ),
                "gateway readiness",
            ),
            (
                FakeHttp(
                    health=[
                        TOOL.HttpResponse(
                            OPENWEBUI_URL, 200, TOOL.OPENWEBUI_HEALTH_BODY + b"\n"
                        )
                    ]
                ),
                "OpenWebUI health",
            ),
        )
        for reader, message in cases:
            with self.subTest(message=message):
                observer = FakeObserverReader()
                with self.assertRaisesRegex(TOOL.OperationalError, message):
                    TOOL.run_operational_preflight(
                        expectation(),
                        dependencies(
                            http=reader if isinstance(reader, FakeHttp) else None,
                            gateway_http=(
                                reader if isinstance(reader, FakeGatewayHttp) else None
                            ),
                            observer=observer,
                        ),
                    )
                self.assertEqual(observer.handle.closed, 1)

    def test_final_http_recheck_is_authoritative(self) -> None:
        health = [
            TOOL.HttpResponse(OPENWEBUI_URL, 200, TOOL.OPENWEBUI_HEALTH_BODY),
            TOOL.HttpResponse(OPENWEBUI_URL, 503, TOOL.OPENWEBUI_HEALTH_BODY),
        ]
        with self.assertRaisesRegex(TOOL.OperationalError, "final OpenWebUI health"):
            TOOL.run_operational_preflight(
                expectation(), dependencies(http=FakeHttp(health=health))
            )

    def test_container_rejects_unhealthy_or_mutating_state(self) -> None:
        mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
            ("running", lambda item: item["State"].__setitem__("Running", False)),
            ("restarting", lambda item: item["State"].__setitem__("Restarting", True)),
            ("OOM", lambda item: item["State"].__setitem__("OOMKilled", True)),
            ("dead", lambda item: item["State"].__setitem__("Dead", True)),
            (
                "unhealthy",
                lambda item: item["State"]["Health"].__setitem__("Status", "unhealthy"),
            ),
        )
        for label, mutate in mutations:
            document = container_document()
            mutate(document[0])
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    TOOL.OperationalError, "not running and healthy"
                ),
            ):
                TOOL.run_operational_preflight(
                    expectation(),
                    dependencies(commands=FakeCommands(container=[compact(document)])),
                )

    def test_container_requires_exact_content_and_network_identities(self) -> None:
        cases: tuple[tuple[str, Callable[[dict[str, Any]], None], str], ...] = (
            (
                "container",
                lambda item: item.__setitem__("Id", "f" * 64),
                "container or image",
            ),
            (
                "image",
                lambda item: item.__setitem__("Image", "sha256:" + "f" * 64),
                "container or image",
            ),
            (
                "network-id",
                lambda item: item["NetworkSettings"]["Networks"][
                    "open-webui-network"
                ].__setitem__("NetworkID", "f" * 64),
                "network content",
            ),
            (
                "extra-network",
                lambda item: item["NetworkSettings"]["Networks"].__setitem__(
                    "bridge", {"NetworkID": "f" * 64}
                ),
                "attachment set",
            ),
        )
        for label, mutate, message in cases:
            document = container_document()
            mutate(document[0])
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                TOOL.run_operational_preflight(
                    expectation(),
                    dependencies(commands=FakeCommands(container=[compact(document)])),
                )

    def test_container_toctou_change_and_duplicate_json_are_rejected(self) -> None:
        changed = container_document()
        changed[0]["State"]["Pid"] = 405
        with self.assertRaisesRegex(TOOL.OperationalError, "changed during"):
            TOOL.run_operational_preflight(
                expectation(),
                dependencies(
                    commands=FakeCommands(
                        container=[compact(container_document()), compact(changed)]
                    )
                ),
            )
        duplicate = compact(container_document()).replace(
            b'"RestartCount":0', b'"RestartCount":0,"RestartCount":0'
        )
        with self.assertRaisesRegex(TOOL.OperationalError, "duplicate JSON"):
            TOOL.run_operational_preflight(
                expectation(),
                dependencies(commands=FakeCommands(container=[duplicate])),
            )

    def test_systemd_requires_fixed_state_restart_and_start_limit_policy(self) -> None:
        cases = (
            ("state", {"ActiveState": "failed"}, "active, running"),
            ("result", {"Result": "exit-code"}, "active, running"),
            ("pid", {"MainPID": "999"}, "MainPID differs"),
            ("restart", {"Restart": "always"}, "policy differs"),
            ("delay", {"RestartUSec": "9s"}, "policy differs"),
            ("interval", {"StartLimitIntervalUSec": "14min"}, "policy differs"),
            ("burst", {"StartLimitBurst": "4"}, "policy differs"),
        )
        for label, changes, message in cases:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                TOOL.run_operational_preflight(
                    expectation(),
                    dependencies(
                        commands=FakeCommands(systemd=[systemd_raw(**changes)])
                    ),
                )

    def test_systemd_active_age_is_at_least_900_seconds(self) -> None:
        with self.assertRaisesRegex(TOOL.OperationalError, "900 seconds"):
            TOOL.run_operational_preflight(
                expectation(),
                dependencies(clock=FakeClock([999_999_999_999])),
            )
        with self.assertRaisesRegex(TOOL.OperationalError, "in the future"):
            TOOL.run_operational_preflight(
                expectation(),
                dependencies(clock=FakeClock([99_999_999_999])),
            )

    def test_systemd_toctou_restart_or_active_epoch_change_is_rejected(self) -> None:
        for changed in (
            systemd_raw(NRestarts="3"),
            systemd_raw(ActiveEnterTimestampMonotonic="100000001"),
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(
                    TOOL.OperationalError, "systemd identity changed"
                ),
            ):
                TOOL.run_operational_preflight(
                    expectation(),
                    dependencies(
                        commands=FakeCommands(systemd=[systemd_raw(), changed])
                    ),
                )

    def test_observer_path_must_stay_missing_under_same_pinned_parent(self) -> None:
        child = TOOL.FileIdentity(1, 8, stat.S_IFSOCK | 0o600, 1, 1000, 1000)
        changed_parent = parent_identity(inode=8)
        cases = (
            (
                "present initially",
                [observer_snapshot(child=child)],
                "required missing path",
            ),
            (
                "appeared",
                [observer_snapshot(), observer_snapshot(child=child)],
                "required missing path",
            ),
            (
                "parent replaced",
                [observer_snapshot(), observer_snapshot(parent=changed_parent)],
                "parent identity changed",
            ),
            (
                "named parent differs",
                [observer_snapshot(named=changed_parent)],
                "parent pathname",
            ),
            (
                "parent mode",
                [observer_snapshot(parent=parent_identity(mode=0o755))],
                "parent identity",
            ),
        )
        for label, snapshots, message in cases:
            handle = FakeObserverHandle(snapshots)
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                TOOL.run_operational_preflight(
                    expectation(),
                    dependencies(observer=FakeObserverReader(handle)),
                )
            self.assertEqual(handle.closed, 1)

    def test_observer_cleanup_never_masks_primary_failure(self) -> None:
        handle = FakeObserverHandle(
            close_error=TOOL.OperationalError("observer cleanup sentinel")
        )
        gateway_http = FakeGatewayHttp(
            ready=[TOOL.HttpResponse(GATEWAY_URL, 503, TOOL.GATEWAY_READY_BODY)]
        )
        with self.assertRaisesRegex(
            TOOL.OperationalError, "gateway readiness"
        ) as caught:
            TOOL.run_operational_preflight(
                expectation(),
                dependencies(
                    gateway_http=gateway_http,
                    observer=FakeObserverReader(handle),
                ),
            )
        self.assertTrue(
            any(
                "observer cleanup also failed" in note
                for note in caught.exception.__notes__
            )
        )
        self.assertEqual(handle.closed, 1)

    def test_observer_cleanup_failure_after_success_is_reported(self) -> None:
        handle = FakeObserverHandle(
            close_error=TOOL.OperationalError("observer cleanup sentinel")
        )
        with self.assertRaisesRegex(TOOL.OperationalError, "cleanup sentinel"):
            TOOL.run_operational_preflight(
                expectation(), dependencies(observer=FakeObserverReader(handle))
            )
        self.assertEqual(handle.closed, 1)

    def test_gpu_result_rejects_other_or_inconsistent_positive_vram(self) -> None:
        cases = (
            gpu_snapshot(positive_kfd_pids=(202, 303)),
            gpu_snapshot(kfd_vram_bytes=19_000_000_000),
            gpu_snapshot(worker_pid=303, positive_kfd_pids=(303,)),
            gpu_snapshot(amd_vram_bytes=0, kfd_vram_bytes=0),
            gpu_snapshot(amd_list_sha256="bad"),
        )
        for result in cases:
            with (
                self.subTest(result=result),
                self.assertRaisesRegex(
                    TOOL.OperationalError, "single-worker isolation"
                ),
            ):
                TOOL.run_operational_preflight(
                    expectation(), dependencies(gpu=FakeGpu(result))
                )

    def test_gpu_worker_or_vram_toctou_change_is_rejected(self) -> None:
        class SequencedGpu:
            def __init__(self, results: list[Any]) -> None:
                self.results = results

            def capture(self, _worker_pid: int) -> Any:
                return self.results.pop(0)

        for changed in (
            gpu_snapshot(worker_starttime_ticks=56),
            gpu_snapshot(amd_vram_bytes=20_000_000_001, kfd_vram_bytes=20_000_000_001),
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(TOOL.OperationalError, "isolation changed"),
            ):
                TOOL.run_operational_preflight(
                    expectation(),
                    dependencies(gpu=SequencedGpu([gpu_snapshot(), changed])),
                )

    def test_expectation_rejects_nonexact_endpoints_and_identifiers(self) -> None:
        cases = (
            dataclasses.replace(expectation(), gateway_ready_url="http://host/health"),
            dataclasses.replace(
                expectation(), openwebui_health_url="http://host/health?full=1"
            ),
            dataclasses.replace(expectation(), image_id="2" * 64),
            dataclasses.replace(expectation(), observer_parent_mode=0o755),
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(TOOL.OperationalError):
                TOOL.run_operational_preflight(value, dependencies())


class FakePopenProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        return_code: int = 0,
        running: bool = False,
        hold_pipes_open: bool = False,
        cleanup_wait_error: bool = False,
    ) -> None:
        self.pid = 424_242
        self.return_code = return_code
        self.running = running
        self.cleanup_wait_error = cleanup_wait_error
        self.wait_calls = 0
        self._writers: list[int] = []
        self.stdout = self._pipe(stdout, hold_pipes_open)
        self.stderr = self._pipe(stderr, hold_pipes_open)

    def _pipe(self, raw: bytes, hold_open: bool) -> Any:
        reader, writer = os.pipe()
        if raw:
            os.write(writer, raw)
        if hold_open:
            self._writers.append(writer)
        else:
            os.close(writer)
        return os.fdopen(reader, "rb", buffering=0)

    def poll(self) -> int | None:
        return None if self.running else self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        if self.cleanup_wait_error:
            raise OSError("cleanup wait failed")
        while self._writers:
            os.close(self._writers.pop())
        self.running = False
        return self.return_code


class FakePopenFactory:
    def __init__(self, process: FakePopenProcess) -> None:
        self.process = process
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, arguments: list[str], **kwargs: Any) -> FakePopenProcess:
        self.calls.append((arguments, kwargs))
        return self.process


class BoundedDependencyTests(unittest.TestCase):
    def command_reader(self) -> Any:
        command = TOOL._systemd_command("ullm-openai.service")
        return TOOL.BoundedReadOnlyCommandReader(frozenset({command}))

    def run_fake_process(
        self,
        process: FakePopenProcess,
        *,
        timeout_seconds: float = 0.1,
        maximum_stdout_bytes: int = 4096,
    ) -> tuple[bytes, FakePopenFactory, Any]:
        factory = FakePopenFactory(process)
        killpg = mock.Mock()
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(TOOL.os, "killpg", killpg),
        ):
            raw = self.command_reader().run(
                TOOL._systemd_command("ullm-openai.service"),
                label="fake read-only command",
                timeout_seconds=timeout_seconds,
                maximum_stdout_bytes=maximum_stdout_bytes,
            )
        return raw, factory, killpg

    def test_command_reader_streams_with_fixed_environment_and_closes_pipes(
        self,
    ) -> None:
        process = FakePopenProcess(stdout=b"bounded-output")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            raw, factory, killpg = self.run_fake_process(process)
            gc.collect()
        self.assertEqual(raw, b"bounded-output")
        self.assertEqual(len(factory.calls), 1)
        arguments, kwargs = factory.calls[0]
        self.assertEqual(arguments[0], TOOL.SYSTEMCTL_BIN)
        self.assertEqual(kwargs["env"], dict(TOOL.COMMAND_ENVIRONMENT))
        self.assertEqual(
            set(kwargs["env"]), {key for key, _ in TOOL.COMMAND_ENVIRONMENT}
        )
        self.assertIs(kwargs["stdout"], TOOL.subprocess.PIPE)
        self.assertIs(kwargs["stderr"], TOOL.subprocess.PIPE)
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        killpg.assert_not_called()

    def test_command_reader_kills_on_timeout_and_closes_held_pipes(self) -> None:
        process = FakePopenProcess(running=True, hold_pipes_open=True)
        factory = FakePopenFactory(process)
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(TOOL.os, "killpg") as killpg,
            self.assertRaisesRegex(TOOL.OperationalError, "timed out"),
        ):
            self.command_reader().run(
                TOOL._systemd_command("ullm-openai.service"),
                label="fake timeout",
                timeout_seconds=0.001,
                maximum_stdout_bytes=32,
            )
        killpg.assert_called_once_with(process.pid, TOOL.signal.SIGKILL)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_command_reader_kills_during_stdout_and_stderr_overflow(self) -> None:
        cases = (
            (b"x" * 33, b"", "stdout", 32, None),
            (b"", b"x" * 17, "stderr", 32, 16),
        )
        for stdout, stderr, message, stdout_maximum, stderr_maximum in cases:
            process = FakePopenProcess(stdout=stdout, stderr=stderr, running=True)
            factory = FakePopenFactory(process)
            stderr_patch = (
                mock.patch.object(TOOL, "MAX_COMMAND_STDERR_BYTES", stderr_maximum)
                if stderr_maximum is not None
                else mock.patch.object(
                    TOOL,
                    "MAX_COMMAND_STDERR_BYTES",
                    TOOL.MAX_COMMAND_STDERR_BYTES,
                )
            )
            with (
                self.subTest(channel=message),
                mock.patch.object(TOOL.subprocess, "Popen", factory),
                mock.patch.object(TOOL.os, "killpg") as killpg,
                stderr_patch,
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                self.command_reader().run(
                    TOOL._systemd_command("ullm-openai.service"),
                    label="fake overflow",
                    timeout_seconds=0.1,
                    maximum_stdout_bytes=stdout_maximum,
                )
            killpg.assert_called_once_with(process.pid, TOOL.signal.SIGKILL)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)

    def test_success_requires_empty_stderr(self) -> None:
        process = FakePopenProcess(stdout=b"ok", stderr=b"warning")
        factory = FakePopenFactory(process)
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(TOOL.os, "killpg") as killpg,
            self.assertRaisesRegex(TOOL.OperationalError, "emitted stderr"),
        ):
            self.command_reader().run(
                TOOL._systemd_command("ullm-openai.service"),
                label="fake stderr",
                timeout_seconds=0.1,
                maximum_stdout_bytes=32,
            )
        killpg.assert_not_called()
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_cleanup_failure_does_not_mask_primary_bound_error(self) -> None:
        process = FakePopenProcess(
            stdout=b"x" * 33,
            running=True,
            cleanup_wait_error=True,
        )
        factory = FakePopenFactory(process)
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(TOOL.os, "killpg"),
            self.assertRaisesRegex(TOOL.OperationalError, "stdout exceeds") as caught,
        ):
            self.command_reader().run(
                TOOL._systemd_command("ullm-openai.service"),
                label="fake cleanup",
                timeout_seconds=0.1,
                maximum_stdout_bytes=32,
            )
        self.assertTrue(
            any("cleanup also failed" in note for note in caught.exception.__notes__)
        )

    def test_command_reader_rejects_mutation_even_if_caller_allowlists_it(
        self,
    ) -> None:
        reader = TOOL.BoundedReadOnlyCommandReader(
            frozenset(
                {
                    (TOOL.SYSTEMCTL_BIN, "show", "ullm-openai.service"),
                    (TOOL.SYSTEMCTL_BIN, "restart", "ullm-openai.service"),
                    (TOOL.DOCKER_BIN, "restart", "open-webui"),
                    ("kill", "101"),
                }
            )
        )
        for arguments in (
            (TOOL.SYSTEMCTL_BIN, "restart", "ullm-openai.service"),
            (TOOL.DOCKER_BIN, "restart", "open-webui"),
            ("kill", "101"),
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(TOOL.OperationalError, "allowlist"),
            ):
                reader.run(
                    arguments,
                    label="forbidden",
                    timeout_seconds=1,
                    maximum_stdout_bytes=1,
                )

    def test_production_command_set_contains_only_four_read_only_vectors(self) -> None:
        commands = TOOL.production_read_only_commands(expectation())
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(TOOL._is_read_only_command(item) for item in commands))
        self.assertIn(TOOL._systemd_command("ullm-openai.service"), commands)
        self.assertIn(TOOL._docker_command("open-webui"), commands)

    def test_http_reader_rejects_nonallowlisted_url_before_network_io(self) -> None:
        reader = TOOL.BoundedHttpReader(frozenset({GATEWAY_URL}))
        with self.assertRaisesRegex(TOOL.OperationalError, "allowlist"):
            reader.get(
                "http://127.0.0.1:3000/health",
                timeout_seconds=1,
                maximum_body_bytes=100,
            )

    def run_fake_gateway_process(
        self, process: FakePopenProcess
    ) -> tuple[Any, FakePopenFactory, Any]:
        factory = FakePopenFactory(process)
        killpg = mock.Mock()
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(TOOL.os, "killpg", killpg),
        ):
            response = TOOL.ProductionGatewayNamespaceReader().get(
                404,
                timeout_seconds=TOOL.HTTP_TIMEOUT_SECONDS,
                maximum_body_bytes=TOOL.MAX_HTTP_BODY_BYTES,
            )
        return response, factory, killpg

    def test_gateway_namespace_reader_uses_only_the_fixed_get_vector(self) -> None:
        process = FakePopenProcess(stdout=TOOL.GATEWAY_NAMESPACE_OUTPUT)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            response, factory, killpg = self.run_fake_gateway_process(process)
            gc.collect()

        self.assertEqual(
            response,
            TOOL.HttpResponse(GATEWAY_URL, 200, TOOL.GATEWAY_READY_BODY),
        )
        self.assertEqual(
            factory.calls[0][0],
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/nsenter",
                "--target",
                "404",
                "--net",
                "--setgid",
                "1000",
                "--setuid",
                "1000",
                "/usr/bin/python3",
                "-I",
                "-c",
                TOOL.GATEWAY_NAMESPACE_SOURCE,
            ],
        )
        source = factory.calls[0][0][-1]
        self.assertIn("HTTPConnection('172.20.0.1',8000,timeout=5.0)", source)
        self.assertIn("c.request('GET','/readyz'", source)
        flattened = " ".join(factory.calls[0][0])
        for forbidden in ("docker run", "docker exec", " restart ", " kill "):
            self.assertNotIn(forbidden, flattened)
        self.assertEqual(dataclasses.fields(TOOL.ProductionGatewayNamespaceReader), ())
        self.assertIs(factory.calls[0][1]["stdin"], TOOL.subprocess.DEVNULL)
        self.assertEqual(factory.calls[0][1]["env"], dict(TOOL.COMMAND_ENVIRONMENT))
        self.assertFalse(factory.calls[0][1]["start_new_session"])
        self.assertEqual(factory.calls[0][1]["process_group"], 0)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        killpg.assert_not_called()

    def test_gateway_namespace_reader_rejects_pid_and_limit_drift(self) -> None:
        reader = TOOL.ProductionGatewayNamespaceReader()
        cases = (
            (False, TOOL.HTTP_TIMEOUT_SECONDS, TOOL.MAX_HTTP_BODY_BYTES, "PID"),
            (0, TOOL.HTTP_TIMEOUT_SECONDS, TOOL.MAX_HTTP_BODY_BYTES, "PID"),
            (404, 9.0, TOOL.MAX_HTTP_BODY_BYTES, "timeout"),
            (404, TOOL.HTTP_TIMEOUT_SECONDS, 4095, "body bound"),
        )
        for pid, timeout, maximum, message in cases:
            with (
                self.subTest(pid=pid, timeout=timeout, maximum=maximum),
                mock.patch.object(TOOL.subprocess, "Popen") as popen,
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                reader.get(
                    pid,
                    timeout_seconds=timeout,
                    maximum_body_bytes=maximum,
                )
            popen.assert_not_called()

    def test_gateway_namespace_reader_rejects_output_and_stderr_drift(self) -> None:
        cases = (
            (
                FakePopenProcess(stdout=b'201\n{"status":"ready"}'),
                "output differs",
            ),
            (
                FakePopenProcess(
                    stdout=TOOL.GATEWAY_NAMESPACE_OUTPUT, stderr=b"warning"
                ),
                "emitted stderr",
            ),
        )
        for process, message in cases:
            factory = FakePopenFactory(process)
            with (
                self.subTest(message=message),
                warnings.catch_warnings(),
                mock.patch.object(TOOL.subprocess, "Popen", factory),
                mock.patch.object(TOOL.os, "killpg") as killpg,
                self.assertRaisesRegex(TOOL.OperationalError, message),
            ):
                warnings.simplefilter("error", ResourceWarning)
                TOOL.ProductionGatewayNamespaceReader().get(
                    404,
                    timeout_seconds=TOOL.HTTP_TIMEOUT_SECONDS,
                    maximum_body_bytes=TOOL.MAX_HTTP_BODY_BYTES,
                )
            gc.collect()
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
            killpg.assert_not_called()

    def test_gateway_namespace_reader_preserves_interrupt_and_cleans_process(
        self,
    ) -> None:
        process = FakePopenProcess(running=True, hold_pipes_open=True)
        factory = FakePopenFactory(process)
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(
                TOOL.selectors.DefaultSelector,
                "select",
                side_effect=KeyboardInterrupt,
            ),
            mock.patch.object(TOOL.os, "killpg") as killpg,
            self.assertRaises(KeyboardInterrupt),
        ):
            TOOL.ProductionGatewayNamespaceReader().get(
                404,
                timeout_seconds=TOOL.HTTP_TIMEOUT_SECONDS,
                maximum_body_bytes=TOOL.MAX_HTTP_BODY_BYTES,
            )
        killpg.assert_called_once_with(process.pid, TOOL.signal.SIGKILL)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_gateway_namespace_reader_rejects_sudo_or_nsenter_failure(self) -> None:
        process = FakePopenProcess(
            stderr=b"sudo or nsenter failed",
            return_code=1,
        )
        factory = FakePopenFactory(process)
        with (
            mock.patch.object(TOOL.subprocess, "Popen", factory),
            mock.patch.object(TOOL.os, "killpg") as killpg,
            self.assertRaisesRegex(TOOL.OperationalError, "exited 1"),
        ):
            TOOL.ProductionGatewayNamespaceReader().get(
                404,
                timeout_seconds=TOOL.HTTP_TIMEOUT_SECONDS,
                maximum_body_bytes=TOOL.MAX_HTTP_BODY_BYTES,
            )
        killpg.assert_not_called()
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)


def collector_capture(
    context: Any,
    worker_pid: int,
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], int, int]:
    process_raw = context.command_runner(
        [context.amd_smi, "process", "--gpu", "2", "--general", "--json"],
        "amd-smi process",
    )
    if process_raw != b"process-json":
        raise AssertionError("process fixture differs")
    identity = {"pid": worker_pid, "st_dev": 9, "st_ino": 10}
    captured = {
        "worker": {
            "pid": worker_pid,
            "starttime_ticks_before": 55,
            "starttime_ticks_after": 55,
            "exe_target": "/opt/ullm-sq8-worker",
            "children": [],
            "vmrss_bytes": 1234,
        },
        "gpu": {
            "index": TOOL.GPU_INDEX,
            "bdf": TOOL.GPU_BDF,
            "uuid": TOOL.GPU_UUID,
            "kfd_gpu_id": TOOL.KFD_GPU_ID,
            "worker_pid": worker_pid,
            "mem_usage_value": 20_000,
            "mem_usage_unit": "B",
            "kfd_snapshot": {
                "acquisition_started_monotonic_ns": 1,
                "acquisition_completed_monotonic_ns": 2,
                "deadline_monotonic_ns": 1_000_000_001,
                "attempt_count": 1,
                "attempts": [{"outcome": "stable"}],
                "before_identities": [identity],
                "after_identities": [dict(identity)],
                "processes": [
                    {**identity, "vram_bytes": 20_000},
                ],
            },
        },
    }
    if mutate is not None:
        mutate(captured)
    return captured, 1234, 20_000


class FakeCollector:
    GPU_INDEX = TOOL.GPU_INDEX
    GPU_BDF = TOOL.GPU_BDF
    GPU_UUID = TOOL.GPU_UUID
    KFD_GPU_ID = TOOL.KFD_GPU_ID

    class ProbeContext:
        def __init__(
            self,
            amd_smi: str,
            proc_root: Path,
            kfd_proc_root: Path,
            command_runner: Callable[[list[str], str], bytes],
        ) -> None:
            self.amd_smi = amd_smi
            self.proc_root = proc_root
            self.kfd_proc_root = kfd_proc_root
            self.command_runner = command_runner

    @staticmethod
    def parse_amd_smi_list(raw: bytes) -> None:
        if raw != b"list-json":
            raise ValueError("list differs")

    @staticmethod
    def capture_resource_sample(
        context: Any, worker_pid: int
    ) -> tuple[dict[str, Any], int, int]:
        return collector_capture(context, worker_pid)


class WorkerAcceptanceGpuReaderTests(unittest.TestCase):
    def command_fixture(self) -> FakeCommands:
        return FakeCommands(
            extras={
                ("/opt/rocm/bin/amd-smi", "list", "--json"): [b"list-json"],
                (
                    "/opt/rocm/bin/amd-smi",
                    "process",
                    "--gpu",
                    "2",
                    "--general",
                    "--json",
                ): [b"process-json"],
            }
        )

    def test_adapter_reuses_collector_capture_with_bounded_commands(self) -> None:
        commands = self.command_fixture()
        reader = TOOL.WorkerAcceptanceGpuReader.from_collector_module(
            FakeCollector, commands
        )
        result = reader.capture(202)
        self.assertEqual(result.worker_pid, 202)
        self.assertEqual(result.amd_vram_bytes, 20_000)
        self.assertEqual(result.kfd_vram_bytes, 20_000)
        self.assertEqual(result.positive_kfd_pids, (202,))
        self.assertEqual([call[0][1] for call in commands.calls], ["list", "process"])
        self.assertTrue(
            all(
                call[2] == TOOL.COMMAND_TIMEOUT_SECONDS
                and call[3] == TOOL.MAX_COMMAND_STDOUT_BYTES
                for call in commands.calls
            )
        )

    def test_fixed_loader_runs_real_collector_against_fake_proc_and_kfd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc_root = root / "proc"
            kfd_root = root / "kfd"
            proc_root.mkdir()
            kfd_root.mkdir()
            make_fake_worker_proc(proc_root, 202)
            kfd_process = kfd_root / "202"
            kfd_process.mkdir()
            (kfd_process / f"vram_{TOOL.KFD_GPU_ID}").write_text(
                "20000\n", encoding="ascii"
            )
            commands = FakeCommands(
                extras={
                    (TOOL.AMD_SMI_BIN, "list", "--json"): [
                        compact(
                            [
                                {
                                    "gpu": TOOL.GPU_INDEX,
                                    "bdf": TOOL.GPU_BDF,
                                    "uuid": TOOL.GPU_UUID,
                                    "kfd_id": TOOL.KFD_GPU_ID,
                                }
                            ]
                        )
                    ],
                    (
                        TOOL.AMD_SMI_BIN,
                        "process",
                        "--gpu",
                        "2",
                        "--general",
                        "--json",
                    ): [
                        compact(
                            [
                                {
                                    "gpu": TOOL.GPU_INDEX,
                                    "process_list": [
                                        {
                                            "process_info": {
                                                "pid": 202,
                                                "mem_usage": {
                                                    "value": 20_000,
                                                    "unit": "B",
                                                },
                                            }
                                        }
                                    ],
                                }
                            ]
                        )
                    ],
                }
            )
            reader = TOOL.load_worker_acceptance_gpu_reader(
                commands, proc_root=proc_root, kfd_proc_root=kfd_root
            )
            result = reader.capture(202)
            self.assertEqual(result.worker_pid, 202)
            self.assertEqual(result.worker_starttime_ticks, 55)
            self.assertEqual(result.amd_vram_bytes, 20_000)
            self.assertEqual(result.kfd_vram_bytes, 20_000)
            self.assertEqual(result.positive_kfd_pids, (202,))

    def test_adapter_rejects_inconsistent_amd_and_kfd_snapshots(self) -> None:
        mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
            (
                "other owner",
                lambda value: value["gpu"]["kfd_snapshot"]["processes"].append(
                    {"pid": 303, "vram_bytes": 1}
                ),
            ),
            (
                "different amount",
                lambda value: value["gpu"]["kfd_snapshot"]["processes"][0].__setitem__(
                    "vram_bytes", 19_999
                ),
            ),
            (
                "KFD identity",
                lambda value: value["gpu"]["kfd_snapshot"]["after_identities"][
                    0
                ].__setitem__("st_ino", 11),
            ),
            (
                "proc identity",
                lambda value: value["worker"].__setitem__("starttime_ticks_after", 56),
            ),
        )
        for label, mutate in mutations:

            class MutatingCollector(FakeCollector):
                @staticmethod
                def capture_resource_sample(
                    context: Any, worker_pid: int
                ) -> tuple[dict[str, Any], int, int]:
                    return collector_capture(context, worker_pid, mutate=mutate)

            with self.subTest(label=label), self.assertRaises(TOOL.OperationalError):
                TOOL.WorkerAcceptanceGpuReader.from_collector_module(
                    MutatingCollector, self.command_fixture()
                ).capture(202)

    def test_adapter_rejects_collector_constant_drift(self) -> None:
        class WrongCollector(FakeCollector):
            GPU_INDEX = 1

        with self.assertRaisesRegex(TOOL.OperationalError, "constants differ"):
            TOOL.WorkerAcceptanceGpuReader.from_collector_module(
                WrongCollector, self.command_fixture()
            )

    def test_adapter_wraps_collector_parse_failure(self) -> None:
        class RejectingCollector(FakeCollector):
            @staticmethod
            def parse_amd_smi_list(_raw: bytes) -> None:
                raise ValueError("wrong physical GPU")

        reader = TOOL.WorkerAcceptanceGpuReader.from_collector_module(
            RejectingCollector, self.command_fixture()
        )
        with self.assertRaisesRegex(TOOL.OperationalError, "R9700 isolation"):
            reader.capture(202)


if __name__ == "__main__":
    unittest.main()
