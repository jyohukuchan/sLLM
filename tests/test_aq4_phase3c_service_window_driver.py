from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_service_window_driver_is_explicitly_single_use_and_preserves_lock_contract() -> None:
    source = (ROOT / "tools" / "run-aq4-phase3c-service-window.sh").read_text(encoding="utf-8")

    assert "--confirm-single-window" in source
    assert "systemctl restart" not in source
    assert "exec 9< \"$lock\"" in source
    assert "flock -n 9" in source
    assert "run-aq4-phase3c-r9700-guard.py" in source
    assert "guard-before" in source
    assert "guard-after" in source
    assert "test -x /opt/rocm/bin/amd-smi" not in source
    assert "! -x /opt/rocm/bin/amd-smi" in source
