"""Shared, local-only fixtures for the host test suite."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"
_DESELECTED = 0
_OUTCOMES: dict[str, str] = {}


def _record_outcome(nodeid: str, outcome: str) -> None:
    priority = {"passed": 0, "skipped": 1, "failed": 2}
    previous = _OUTCOMES.get(nodeid)
    if previous is None or priority[outcome] > priority[previous]:
        _OUTCOMES[nodeid] = outcome


def pytest_deselected(items) -> None:
    """Count tests removed by the registered marker expression."""

    global _DESELECTED
    _DESELECTED += len(items)


def pytest_runtest_logreport(report) -> None:
    """Record one terminal outcome for every selected pytest item."""

    if report.when == "setup" and report.failed:
        _record_outcome(report.nodeid, "failed")
    elif report.when == "setup" and report.skipped:
        _record_outcome(report.nodeid, "skipped")
    elif report.when == "call":
        if getattr(report, "wasxfail", False):
            _record_outcome(
                report.nodeid, "skipped" if report.skipped else "failed"
            )
        else:
            _record_outcome(report.nodeid, report.outcome)
    elif report.when == "teardown" and report.failed:
        _record_outcome(report.nodeid, "failed")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Emit a single machine-readable count record for the outer CI runner."""

    if os.environ.get("SLLM_EMIT_TEST_COUNTS") != "1":
        return
    selected_ids = [item.nodeid for item in session.items]
    outcomes = {"passed": 0, "failed": 0, "skipped": 0}
    for nodeid in selected_ids:
        # An interrupted item without a terminal report is not a pass.
        outcomes[_OUTCOMES.get(nodeid, "failed")] += 1
    counts = {
        "collected": len(selected_ids) + _DESELECTED,
        "selected": len(selected_ids),
        **outcomes,
        "deselected": _DESELECTED,
    }
    print(
        "\nSLLM_PYTEST_COUNTS="
        + json.dumps(counts, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """Fail immediately if a required host test attempts network access."""

    def blocked_socket(*_args, **_kwargs):
        raise RuntimeError("network access is disabled in required host tests")

    monkeypatch.setattr(socket, "socket", blocked_socket)


@pytest.fixture
def load_json_fixture():
    """Load one checked-in fixture without consulting caches or the network."""

    def load(name: str) -> Any:
        path = FIXTURE_DIR / name
        if path.parent != FIXTURE_DIR or path.suffix != ".json":
            raise ValueError(f"fixture must be a direct JSON file: {name!r}")
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    return load
