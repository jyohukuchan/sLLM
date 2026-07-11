from __future__ import annotations

from pathlib import Path
from typing import Any

from ullm_openai_gateway import __main__ as entrypoint
from ullm_openai_gateway.settings import GatewaySettings


def test_entrypoint_disables_access_and_proxy_logging(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = GatewaySettings(
        worker_binary=tmp_path / "worker",
        artifact_dir=tmp_path / "artifact",
        package_dir=tmp_path / "package",
        tokenizer_dir=tmp_path / "tokenizer",
        api_key_file=tmp_path / "key",
        gpu_lock_file=tmp_path / "lock",
    )
    application = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(entrypoint.GatewaySettings, "from_env", lambda: settings)
    monkeypatch.setattr(entrypoint, "create_app", lambda _: application)

    def capture(app: object, **options: Any) -> None:
        captured["app"] = app
        captured.update(options)

    monkeypatch.setattr(entrypoint.uvicorn, "run", capture)
    entrypoint.main()
    assert captured == {
        "app": application,
        "host": "127.0.0.1",
        "port": 8000,
        "workers": 1,
        "reload": False,
        "proxy_headers": False,
        "server_header": False,
        "access_log": False,
    }
