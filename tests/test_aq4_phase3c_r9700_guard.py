from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_guard():
    path = ROOT / "tools" / "run-aq4-phase3c-r9700-guard.py"
    spec = importlib.util.spec_from_file_location("aq4_phase3c_r9700_guard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUARD = load_guard()


def hip_payload(bdf: str = "0000:47:00.0") -> dict:
    return {
        "schema_version": "ullm.r9700_hip_device_guard.v1",
        "status": "valid",
        "required": {
            "hip_visible_devices": "1",
            "ullm_hip_visible_devices": "1",
            "visible_hip_device_count": 1,
            "architecture": "gfx1201",
        },
        "actual": {
            "hip_visible_devices": "1",
            "ullm_hip_visible_devices": "1",
            "visible_hip_device_count": 1,
            "filtered_hip_ordinal": 0,
            "architecture": "gfx1201",
            "name": "AMD Radeon Graphics",
            "pci_bdf": bdf,
        },
    }


def amd_smi_payload(bdf: str = "0000:47:00.0") -> dict:
    return {
        "gpu": {
            "target": {
                "asic": {
                    "target_graphics_version": "gfx1201:feature",
                    "device_id": "0x7551",
                    "market_name": "AMD Radeon AI PRO R9700",
                },
                "bus": {"bdf": bdf},
            }
        }
    }


def test_runuser_contract_keeps_visibility_and_uses_absolute_amd_smi_path() -> None:
    command = GUARD.build_runuser_command(
        "/usr/sbin/runuser",
        [str(GUARD.AMD_SMI_PATH), "static", "--gpu", "0000:47:00.0", "--asic", "--bus", "--json"],
    )

    assert command[:8] == [
        "/usr/sbin/runuser",
        "-u",
        "homelab1",
        "--",
        "/usr/bin/env",
        "HOME=/home/homelab1",
        "HIP_VISIBLE_DEVICES=1",
        "ULLM_HIP_VISIBLE_DEVICES=1",
    ]
    assert command[8] == "/opt/rocm/bin/amd-smi"


def test_identity_cross_check_accepts_only_the_r9700_mapping() -> None:
    actual, bdf = GUARD.validate_hip_payload(hip_payload())
    amd_smi = GUARD.validate_amd_smi_payload(amd_smi_payload(), bdf)

    assert actual["architecture"] == "gfx1201"
    assert amd_smi == {
        "pci_bdf": "0000:47:00.0",
        "architecture": "gfx1201",
        "pci_device_id": "0x7551",
        "market_name": "AMD Radeon AI PRO R9700",
    }


def test_identity_cross_check_refuses_an_unexpected_bdf_before_amd_smi() -> None:
    with pytest.raises(GUARD.GuardError, match="fixed R9700 BDF"):
        GUARD.validate_hip_payload(hip_payload("0000:41:00.0"))
