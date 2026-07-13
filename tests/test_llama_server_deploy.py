"""Static deployment contract checks for the llama.cpp Qwen3.5 server."""

from __future__ import annotations

import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "deploy/systemd/llama-qwen35-udq4.service"
ENV = ROOT / "deploy/systemd/llama-qwen35-udq4.env.example"
NFT = ROOT / "deploy/nftables/ullm-openai.nft"
FIREWALL = ROOT / "deploy/systemd/ullm-openai-firewall.service"


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _value_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator, f"{path}: malformed environment line: {line!r}"
        values[key] = value
    return values


def _directive_values(name: str) -> list[str]:
    prefix = f"{name}="
    return [line[len(prefix) :] for line in _lines(UNIT) if line.startswith(prefix)]


def test_llama_environment_profile_is_complete_and_secret_free() -> None:
    values = _value_map(ENV)
    assert values == {
        "HIP_VISIBLE_DEVICES": "1",
        "LLAMA_SERVER_BINARY": "/home/homelab1/llama.cpp-src/build-rdna4/bin/llama-server",
        "LLAMA_MODEL": (
            "/home/homelab1/datapool/ai_models/gguf/unsloth/"
            "Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
        ),
        "LLAMA_DEVICE": "ROCm0",
        "LLAMA_HOST": "172.20.0.1",
        "LLAMA_PORT": "8001",
        "LLAMA_ALIAS": "llama-qwen3.5-9b-ud-q4",
        "LLAMA_CTX_SIZE": "4096",
        "LLAMA_API_KEY_FILE": "/etc/ullm/llama-qwen35-udq4-api-key",
    }
    # The example may point at the key file, but must not contain a key value.
    assert all("=" not in value for value in values.values())


def test_llama_unit_dependencies_identity_and_mount_are_fixed() -> None:
    text = UNIT.read_text(encoding="utf-8")
    assert "User=homelab1\n" in text
    assert "Group=homelab1\n" in text
    assert "SupplementaryGroups=video render\n" in text
    assert "Requires=docker.service ullm-openai-firewall.service\n" in text
    assert "After=network-online.target docker.service ullm-openai-firewall.service\n" in text
    assert "RequiresMountsFor=/home/homelab1/datapool\n" in text
    assert "EnvironmentFile=/etc/ullm/llama-qwen35-udq4.env\n" in text
    assert "Environment=HOME=/var/cache/llama-qwen35-udq4\n" in text
    assert "Environment=XDG_CACHE_HOME=/var/cache/llama-qwen35-udq4\n" in text
    assert "PrivateDevices=" not in text
    assert "/run/ullm/r9700.lock" not in text


def test_llama_unit_preflight_restart_and_hardening_contract() -> None:
    text = UNIT.read_text(encoding="utf-8")
    assert "ExecStartPre=/usr/bin/test -x ${LLAMA_SERVER_BINARY}\n" in text
    assert "ExecStartPre=/usr/bin/test -r ${LLAMA_MODEL}\n" in text
    assert "ExecStartPre=/usr/bin/test -s ${LLAMA_API_KEY_FILE}\n" in text
    assert "Restart=on-failure\n" in text
    assert "RestartSec=10s\n" in text
    assert "TimeoutStartSec=650s\n" in text
    assert "TimeoutStopSec=60s\n" in text
    assert "KillMode=control-group\n" in text
    assert "OOMPolicy=stop\n" in text
    assert "UMask=0077\n" in text
    assert "LimitNOFILE=65536\n" in text
    for directive in (
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "LockPersonality=yes",
        "SystemCallArchitectures=native",
    ):
        assert f"{directive}\n" in text


def test_llama_execstart_is_shell_free_and_has_exact_runtime_options() -> None:
    starts = _directive_values("ExecStart")
    assert len(starts) == 1
    command = starts[0]
    assert not any(shell_token in command for shell_token in ("/bin/sh", "/bin/bash", " -c ", "&&", "||", "|"))
    argv = shlex.split(command)
    assert argv[0] == "/home/homelab1/llama.cpp-src/build-rdna4/bin/llama-server"
    expected_pairs = {
        "--model": "${LLAMA_MODEL}",
        "--device": "${LLAMA_DEVICE}",
        "--host": "${LLAMA_HOST}",
        "--port": "${LLAMA_PORT}",
        "--alias": "${LLAMA_ALIAS}",
        "--ctx-size": "${LLAMA_CTX_SIZE}",
        "--api-key-file": "${LLAMA_API_KEY_FILE}",
        "--gpu-layers": "999",
        "--fit": "off",
        "--parallel": "1",
        "--cache-ram": "0",
        "--threads": "32",
        "--threads-batch": "64",
    }
    for option, value in expected_pairs.items():
        assert argv[argv.index(option) + 1] == value
    for flag in ("--no-mmproj", "--no-ui", "--no-warmup"):
        assert argv.count(flag) == 1
    assert not any("mmproj" in arg and arg != "--no-mmproj" for arg in argv)


def test_nftables_protect_both_openai_ports_to_openwebui_bridge() -> None:
    lines = [line.strip() for line in _lines(NFT) if "tcp dport" in line]
    assert len(lines) == 2
    assert all("tcp dport { 8000, 8001 }" in line for line in lines)
    assert lines[0].endswith('iifname != "br-79bb7cfca31c" counter drop')
    assert lines[1].endswith(
        'iifname "br-79bb7cfca31c" ip saddr != 172.20.0.0/16 counter drop'
    )
    assert all("ip daddr 172.20.0.1" in line for line in lines)


def test_firewall_orders_both_gateway_units_after_nft_install() -> None:
    text = FIREWALL.read_text(encoding="utf-8")
    assert "Before=ullm-openai.service llama-qwen35-udq4.service\n" in text
    assert "ExecStart=/usr/local/libexec/ullm-openai-firewall install\n" in text
    assert "ExecStop=/usr/local/libexec/ullm-openai-firewall remove\n" in text
