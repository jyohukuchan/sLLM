from __future__ import annotations

import importlib.util
import os
import signal
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SPEC = importlib.util.spec_from_file_location(
    "test_sq8_full_campaign_backend_module", TOOLS / "sq8_full_campaign_backend.py"
)
assert SPEC is not None and SPEC.loader is not None
BACKEND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKEND
SPEC.loader.exec_module(BACKEND)


class FakeProcess:
    def __init__(
        self,
        *,
        hold_open: bool = False,
        fileno_error: BaseException | None = None,
        returncode: int = 0,
        wait_interrupts: int = 0,
    ) -> None:
        self.pid = 1234
        self.returncode: int | None = returncode
        self.waited = False
        self.wait_interrupts = wait_interrupts
        self._writes: list[int] = []
        self.stdout = self._stream(hold_open, fileno_error)
        self.stderr = self._stream(hold_open, None)

    def _stream(self, hold_open: bool, error: BaseException | None) -> Any:
        read_fd, write_fd = os.pipe()
        if hold_open:
            self._writes.append(write_fd)
        else:
            os.close(write_fd)
        return FakeStream(read_fd, error)

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_interrupts:
            self.wait_interrupts -= 1
            raise InterruptedError
        self.waited = True
        for descriptor in self._writes:
            os.close(descriptor)
        self._writes.clear()
        assert self.returncode is not None
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


class FakeStream:
    def __init__(self, descriptor: int, error: BaseException | None) -> None:
        self.descriptor = descriptor
        self.error = error

    def fileno(self) -> int:
        if self.error is not None:
            raise self.error
        return self.descriptor

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run_gate(
        self, gate: Any, output_dir: Path, secrets: Any, deployment: Any
    ) -> Any:
        self.commands.append(
            BACKEND.build_gate_argv(gate, output_dir, secrets, deployment)
        )
        return BACKEND.GateCommandResult(b"", b"")


class Owner:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def close(self) -> None:
        self.events.append(self.name)


class SecretOwner:
    def __init__(self) -> None:
        self.api_key_path = Path("/run/user/1000/campaign/gateway-api-key")
        self.openwebui_token_path = Path("/run/user/1000/campaign/openwebui-token")
        self.revalidations = 0

    def revalidate(self) -> None:
        self.revalidations += 1


class Bridge(Owner):
    def now_ns(self) -> int:
        return 1

    def scan_evidence(self, raw: bytes, label: str) -> None:
        pass

    def make_session_writer(self, path: Path) -> object:
        return object()

    def make_resource_writer(self, path: Path) -> object:
        return object()

    def make_journal_capture(self, path: Path, boot_id: str, epoch: Any) -> object:
        return object()

    def preflight(self, work_dir: Path) -> str:
        return "preflight"

    def make_resource_adapter(self, **arguments: Any) -> str:
        return "resource"

    def final(self, work_dir: Path) -> str:
        return "final"


class Bindings:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def build(self, gate: str, work_dir: Path) -> str:
        self.events.append(f"bind:{gate}")
        return str(gate)


class Collector:
    def collect_normal(self, *, expected_identity: Any = None) -> str:
        return "normal"

    def collect_restart(
        self, normal_identity: Any, *, expected_identity: Any = None
    ) -> str:
        return "restart"


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.secret_owner = SecretOwner()
        self.secrets = BACKEND.SecretMasterPaths.from_owner(self.secret_owner)
        self.deployment = BACKEND.GateDeployment(
            BACKEND.PRODUCTION_IMAGE_ID,
            BACKEND.PRODUCTION_NETWORK_ID,
            "browser@sha256:" + "3" * 64,
            "probe@sha256:" + "4" * 64,
            "http://192.168.0.66:3000/",
            "http://172.20.0.1:8000/readyz",
            "open-webui-network",
            "ullm-openai.service",
            Path("/run/user/1000/restart-epoch.json"),
        )

    def test_all_six_argv_are_allowlisted_and_use_only_master_paths(self) -> None:
        original = "/etc/ullm/original-api-key"
        for gate in BACKEND.GATE_ORDER:
            command = BACKEND.build_gate_argv(
                gate, Path(f"/tmp/{gate}"), self.secrets, self.deployment
            )
            self.assertTrue(BACKEND._is_allowed_command(command))
            joined = "\0".join(command)
            self.assertNotIn(original, joined)
            if gate in {"api_contract", "direct_cancel", "latency"}:
                self.assertIn(str(self.secrets.api_key), command)
            else:
                self.assertIn(str(self.secrets.openwebui_token), command)

    def test_runner_kills_and_reaps_group_on_timeout_and_interrupt(self) -> None:
        for process, error_type in (
            (FakeProcess(hold_open=True), BACKEND.ProductionBackendError),
            (FakeProcess(fileno_error=KeyboardInterrupt()), KeyboardInterrupt),
        ):
            kills: list[tuple[int, int]] = []
            runner = BACKEND.BoundedGateRunner(
                timeout_seconds=1.0,
                process_factory=lambda *args, **kwargs: process,
                kill_group=lambda pid, sig: kills.append((pid, sig)),
            )
            with self.assertRaises(error_type):
                runner.run_gate(
                    "api_contract", Path("/tmp/api"), self.secrets, self.deployment
                )
            self.assertEqual(kills, [(1234, signal.SIGKILL)])
            self.assertTrue(process.waited)

    def test_runner_has_no_public_arbitrary_argv_entrypoint(self) -> None:
        runner = BACKEND.BoundedGateRunner(timeout_seconds=1.0)
        self.assertFalse(hasattr(runner, "run"))
        with self.assertRaises(BACKEND.ProductionBackendError):
            runner.run_gate("unknown", Path("/tmp/api"), self.secrets, self.deployment)

    def test_public_runner_rejects_unowned_secrets_deployment_and_relative_output(
        self,
    ) -> None:
        runner = BACKEND.BoundedGateRunner(timeout_seconds=1.0)
        for output, secrets, deployment in (
            (Path("relative"), self.secrets, self.deployment),
            (Path("/tmp/api"), object(), self.deployment),
            (Path("/tmp/api"), self.secrets, object()),
        ):
            with self.assertRaises(BACKEND.ProductionBackendError):
                runner.run_gate("api_contract", output, secrets, deployment)

    def test_nonzero_after_normal_wait_does_not_kill_reaped_group(self) -> None:
        process = FakeProcess(returncode=7)
        kills: list[tuple[int, int]] = []
        runner = BACKEND.BoundedGateRunner(
            timeout_seconds=1.0,
            process_factory=lambda *args, **kwargs: process,
            kill_group=lambda pid, sig: kills.append((pid, sig)),
        )
        with self.assertRaises(BACKEND.ProductionBackendError):
            runner.run_gate(
                "api_contract", Path("/tmp/api"), self.secrets, self.deployment
            )
            self.assertTrue(process.waited)
            self.assertEqual(kills, [])

    def test_successful_sigkill_retries_interrupted_unbounded_reap(self) -> None:
        process = FakeProcess(fileno_error=KeyboardInterrupt(), wait_interrupts=1)
        runner = BACKEND.BoundedGateRunner(
            timeout_seconds=1.0,
            process_factory=lambda *args, **kwargs: process,
            kill_group=lambda pid, sig: None,
        )
        with self.assertRaises(KeyboardInterrupt):
            runner.run_gate(
                "api_contract", Path("/tmp/api"), self.secrets, self.deployment
            )
        self.assertTrue(process.waited)

    def test_allowlist_rejects_extra_duplicate_and_reordered_options(self) -> None:
        command = BACKEND.build_gate_argv(
            "api_contract", Path("/tmp/api"), self.secrets, self.deployment
        )
        variants = (
            command + ("--timeout", "1"),
            command + ("--api-key-file", str(self.secrets.api_key)),
            command[:5] + command[7:9] + command[5:7] + command[9:],
        )
        for variant in variants:
            self.assertFalse(BACKEND._is_allowed_command(variant))

    def test_invalid_gate_is_normalized(self) -> None:
        with self.assertRaises(BACKEND.ProductionBackendError):
            BACKEND.build_gate_argv(
                "unknown", Path("/tmp/unknown"), self.secrets, self.deployment
            )

    def test_secret_paths_require_owner_and_deployment_is_fixed(self) -> None:
        with self.assertRaises(TypeError):
            BACKEND.SecretMasterPaths(Path("/tmp/key"), Path("/tmp/token"))
        with self.assertRaises(BACKEND.ProductionBackendError):
            BACKEND.GateDeployment(
                "sha256:" + "0" * 64,
                BACKEND.PRODUCTION_NETWORK_ID,
                "browser@sha256:" + "3" * 64,
                "probe@sha256:" + "4" * 64,
                "http://192.168.0.66:3000/",
                "http://172.20.0.1:8000/readyz",
                "open-webui-network",
                "ullm-openai.service",
                Path("/run/user/1000/epoch.json"),
            )

    def test_backend_enforces_gate_order_and_only_one_restart_gate(self) -> None:
        events: list[str] = []
        runner = RecordingRunner()

        def ingest(path: Path, binding: Any) -> str:
            events.append(f"ingest:{binding}")
            return str(binding)

        ingestors = BACKEND.GateIngestors(*(ingest for _ in range(6)))
        backend = BACKEND.ProductionCampaignBackend(
            bridge=Bridge("bridge", events),
            runner=runner,
            secrets=self.secrets,
            deployment=self.deployment,
            bindings=Bindings(events),
            ingestors=ingestors,
        )
        methods = (
            backend.api_contract,
            backend.combined,
            backend.direct_cancel,
            backend.stop,
            backend.failure,
            backend.latency,
        )
        for gate, method in zip(BACKEND.GATE_ORDER, methods, strict=True):
            self.assertEqual(method(Path(f"/tmp/{gate}")), gate)
        self.assertEqual(backend.final(Path("/tmp/final")), "final")
        failure_commands = [
            command
            for command in runner.commands
            if "run-openwebui-failure-gate.py" in command[2]
        ]
        self.assertEqual(len(failure_commands), 1)
        with self.assertRaises(BACKEND.ProductionBackendError):
            backend.failure(Path("/tmp/failure-again"))
        self.assertEqual(
            events,
            [
                item
                for gate in BACKEND.GATE_ORDER
                for item in (f"bind:{gate}", f"ingest:{gate}")
            ],
        )
        self.assertEqual(self.secret_owner.revalidations, 1 + len(BACKEND.GATE_ORDER))

    def test_failed_runner_or_ingestor_poisons_backend(self) -> None:
        events: list[str] = []

        class FailingRunner(RecordingRunner):
            def run_gate(
                self, gate: Any, output_dir: Path, secrets: Any, deployment: Any
            ) -> Any:
                super().run_gate(gate, output_dir, secrets, deployment)
                raise RuntimeError("runner failed")

        def reject(path: Path, binding: Any) -> Any:
            raise RuntimeError("ingestor failed")

        def accept(path: Path, binding: Any) -> Any:
            return binding

        for runner, ingestor in (
            (FailingRunner(), accept),
            (RecordingRunner(), reject),
        ):
            backend = BACKEND.ProductionCampaignBackend(
                bridge=Bridge("bridge", events),
                runner=runner,
                secrets=self.secrets,
                deployment=self.deployment,
                bindings=Bindings(events),
                ingestors=BACKEND.GateIngestors(*(ingestor for _ in range(6))),
            )
            with self.assertRaises(RuntimeError):
                backend.api_contract(Path("/tmp/api"))
            self.assertTrue(backend.poisoned)
            with self.assertRaises(BACKEND.ProductionBackendError):
                backend.api_contract(Path("/tmp/api-retry"))

    def test_backend_and_resource_owners_close_in_reverse_order(self) -> None:
        events: list[str] = []
        backend = BACKEND.ProductionCampaignBackend(
            bridge=Bridge("bridge", events),
            runner=RecordingRunner(),
            secrets=self.secrets,
            deployment=self.deployment,
            bindings=Bindings(events),
            ingestors=BACKEND.GateIngestors(
                *(lambda path, binding: None for _ in range(6))
            ),
            owners=(Owner("first", events), Owner("second", events)),
        )
        backend.close()
        self.assertEqual(events, ["bridge", "second", "first"])
        events.clear()
        adapter = BACKEND.ResourceAdapter(
            Collector(), (Owner("one", events), Owner("two", events))
        )
        self.assertEqual(adapter.collect_normal(), "normal")
        self.assertEqual(adapter.collect_restart(object()), "restart")
        adapter.close()
        self.assertEqual(events, ["two", "one"])


if __name__ == "__main__":
    unittest.main()
