#!/usr/bin/env python3
"""Check the repository-level license and provenance control files."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def imported_blob_sha256(commit: str, relative: str) -> str | None:
    """Hash the immutable imported bytes, not later maintenance revisions."""

    completed = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or completed.stderr:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def main() -> int:
    errors: list[str] = []
    license_path = ROOT / "LICENSE"
    provenance = ROOT / "docs/provenance/README.md"
    source_lock = ROOT / "docs/references/source-lock.md"
    notices = ROOT / "THIRD_PARTY_NOTICES.md"
    if not license_path.exists() or "MIT License" not in license_path.read_text(encoding="utf-8"):
        errors.append("LICENSE is missing or is not the declared MIT license")
    if not provenance.exists():
        errors.append("docs/provenance/README.md is missing")
    else:
        text = provenance.read_text(encoding="utf-8")
        for marker in ("THIRD_PARTY_NOTICES.md", "full commit SHA", "imported_sha256", "reuse:"):
            if marker not in text:
                errors.append(f"provenance policy is missing required marker: {marker}")
    if not source_lock.exists():
        errors.append("docs/references/source-lock.md is missing")
    else:
        text = source_lock.read_text(encoding="utf-8")
        if len(re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", text)) < 7:
            errors.append("source-lock does not contain complete source revisions")
    if notices.exists():
        text = notices.read_text(encoding="utf-8")
        sampling = ROOT / "crates/sllm-core/src/sampling.rs"
        sampling_tests = ROOT / "crates/sllm-core/tests/sampling_contract.rs"
        retained_license = ROOT / "docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt"
        expected_source_hash = "0965ba54bc21bad846f050143b4f8034129b03c6180d950790500a104ecb8013"
        import_commit = "b3fbfdccda87628b94d1440df1bf25707cd93c35"
        for marker in (
            "llama-cpp-profile-v1-sampling-001",
            "llama-cpp-profile-v1-sampling-tests-001",
            "f5919bf458ef190468b5c329bb293f8a54a1e69c",
            "a9cb6bee5fd78728e5c94d5d1d008c3022abf330",
            "2aecff90e7bb4b8c09e32ae3dab24d41ca2138f0",
            expected_source_hash,
            "431b4892ddd431c5933c1188ff446d58362a686e24535baf1b5b7d9b0f580079",
            import_commit,
        ):
            if marker not in text:
                errors.append(f"THIRD_PARTY_NOTICES.md is missing required import marker: {marker}")
        if sampling.exists():
            source_text = sampling.read_text(encoding="utf-8")
            if "THIRD_PARTY_NOTICES.md#llama-cpp-profile-v1-sampling-001" not in source_text:
                errors.append("sampling source is missing its provenance header")
            observed = imported_blob_sha256(import_commit, "crates/sllm-core/src/sampling.rs")
            if observed != expected_source_hash:
                errors.append("A3 sampling import-commit bytes differ from imported_sha256")
        else:
            errors.append("A3 sampling import source is missing")
        if sampling_tests.exists():
            test_text = sampling_tests.read_text(encoding="utf-8")
            if "THIRD_PARTY_NOTICES.md#llama-cpp-profile-v1-sampling-tests-001" not in test_text:
                errors.append("sampling contract tests are missing their provenance header")
            observed = imported_blob_sha256(
                import_commit, "crates/sllm-core/tests/sampling_contract.rs"
            )
            if observed != "431b4892ddd431c5933c1188ff446d58362a686e24535baf1b5b7d9b0f580079":
                errors.append("A3 sampling test import-commit bytes differ from imported_sha256")
        else:
            errors.append("A3 sampling contract tests are missing")
        if not retained_license.exists() or hashlib.sha256(retained_license.read_bytes()).hexdigest() != "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d":
            errors.append("retained llama.cpp MIT license is missing or changed")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("license/provenance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
