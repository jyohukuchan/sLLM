from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import run_phase36_session_c as runner  # noqa: E402


def _identity_files(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    result = []
    for name, content in (("cli", b"cli"), ("bf16-model", b"bf16"), ("bf16-lock", b"lock"), ("fp8-model", b"fp8"), ("fp8-lock", b"fp8-lock"), ("source", b"source")):
        path = root / name
        path.write_bytes(content)
        result.append(path)
    return tuple(result)  # type: ignore[return-value]


def _identity(root: Path) -> dict[str, object]:
    cli, model, lock, fp8_model, fp8_lock, source = _identity_files(root)
    fact = lambda path: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
    return {"binary": fact(cli), "cli_binary": fact(cli), "server_binary": fact(cli), "model": fact(model), "lock": fact(lock), "bf16_model": fact(model), "bf16_lock": fact(lock), "fp8_model": fact(fp8_model), "fp8_lock": fact(fp8_lock), "source": fact(source)}


def _common(identity: dict[str, object]) -> dict[str, object]:
    return {"state": "PASS", "pass": True, "target": "gfx942", "selected_backend": "hip", "gpu_execution": True, "fallback_used": False, "cpu_fallback_used": False, "fallback_allowed": False, "partial_offload": False, "identity": identity, "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True}, "amd_smi_metric": {"state": "unavailable", "reason": "provider Namespace.partition"}}


def _populate(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    cli, model, lock, fp8_model, fp8_lock, source = _identity_files(root)
    identity = _identity(root)
    raw = root / "raw"
    raw.mkdir()
    rows = []
    for target_dtype, target_kv, widths in (("bf16", "fp16", runner.MTP_WIDTHS_BF16_FP16), ("fp8", "fp8", runner.MTP_WIDTHS_FP8)):
        for width in widths:
            rows.append({"width": width, "target_dtype": target_dtype, "target_kv": target_kv, "mtp_weight_dtype": "bf16", "mtp_kv_dtype": "fp16", "visible_token_ids": [10, 11], "target_only_token_ids": [10, 11], "accepted_prefix": [], "rejected_prefix": [], "state_publication_match": True, "rewind_replay_match": True, "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True}})
    mtp = _common(identity)
    mtp["rows"] = rows
    (raw / "mtp-final-v1.json").write_text(json.dumps(mtp), encoding="utf-8")
    assets = raw / "assets"
    assets.mkdir()
    hashes = {}
    for fmt, suffix in (("png", ".png"), ("jpeg", ".jpg"), ("webp", ".webp")):
        path = assets / f"reference{suffix}"
        path.write_bytes(f"{fmt}-asset".encode())
        hashes[fmt] = hashlib.sha256(path.read_bytes()).hexdigest()
    (raw / "vision-assets.sha256").write_text("\n".join(f"{digest} /private/reference.{('jpg' if fmt == 'jpeg' else fmt)}" for fmt, digest in hashes.items()) + "\n", encoding="utf-8")
    vision = _common(identity)
    vision["rows"] = [{"format": fmt, "asset_sha256": hashes[fmt], "image_pad_tokens": 64, "output_sha256": "a" * 64, "numerical_match": True, "identical_output": True, "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True}} for fmt in ("png", "jpeg", "webp")]
    (raw / "vision-cli-final-v1.json").write_text(json.dumps(vision), encoding="utf-8")
    lazy = _common(identity)
    lazy.update({"server_started": True, "lazy_residency": True, "initial_model_without_vision": True, "vision_resident_after_image": True, "vision_released_after_request": True, "graceful_shutdown": True, "memory": {"before_image_bytes": 1, "during_image_bytes": 2, "after_shutdown_bytes": 0, "gtt_spill_bytes": 0}})
    (raw / "vision-lazy-residency-v1.json").write_text(json.dumps(lazy), encoding="utf-8")
    openai = _common(identity)
    openai["checks"] = {check: True for check in runner.OPENAI_CHECKS}
    (raw / "openai-a6-final-v1.json").write_text(json.dumps(openai), encoding="utf-8")
    return cli, model, lock, fp8_model, fp8_lock, source


class Phase36SessionCRunnerTests(unittest.TestCase):
    def test_complete_session_c_aggregates_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cli, model, lock, fp8_model, fp8_lock, source = _populate(root)
            summary = runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source)
        self.assertEqual(summary["state"], "PASS")
        self.assertEqual(summary["mtp"]["selected_rows"], 8)
        self.assertEqual(summary["vision_cli"]["selected_formats"], 3)
        self.assertTrue(all(summary["openai_a6"]["checks"].values()))

    def test_derived_lock_output_binds_model_and_digest_lock_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cli, model, _lock, fp8_model, _fp8_lock, source = _identity_files(root)
            bf16_lock = root / "bf16.lock.json"
            fp8_lock = root / "fp8.lock.json"

            def write_lock(path: Path, model_path: Path, digest: str | None = None) -> None:
                output_sha = digest or hashlib.sha256(model_path.read_bytes()).hexdigest()
                path.write_text(json.dumps({"schema_version": "derived-gguf-lock-v1", "output": {"sha256": "sha256:" + output_sha}}), encoding="utf-8")

            write_lock(bf16_lock, model)
            write_lock(fp8_lock, fp8_model)
            identity = runner._identity_from_args(cli, model, bf16_lock, source, fp8_model=fp8_model, fp8_lock=fp8_lock)
            self.assertEqual(identity["bf16_lock"]["output_sha256"], identity["bf16_model"]["sha256"])
            self.assertEqual(identity["fp8_lock"]["output_sha256"], identity["fp8_model"]["sha256"])

            write_lock(bf16_lock, model, digest="0" * 64)
            with self.assertRaisesRegex(runner.SessionCError, "BF16 model lock: derived lock output SHA-256 does not match model digest"):
                runner._identity_from_args(cli, model, bf16_lock, source, fp8_model=fp8_model, fp8_lock=fp8_lock)

            digest_lock = "sha256:" + "a" * 64
            fp8_digest_lock = "sha256:" + "b" * 64
            digest_identity = runner._identity_from_args(cli, model, Path(digest_lock), source, fp8_model=fp8_model, fp8_lock=fp8_digest_lock)
            self.assertEqual(digest_identity["bf16_lock"]["sha256"], "a" * 64)
            self.assertEqual(digest_identity["fp8_lock"]["sha256"], "b" * 64)

    def test_wrong_target_and_missing_fp8_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cli, model, lock, fp8_model, fp8_lock, source = _populate(root)
            with self.assertRaisesRegex(runner.SessionCError, "exact target"):
                runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source, target="gfx1201")
            with self.assertRaisesRegex(runner.SessionCError, "FP8 model and lock"):
                runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, source_identity=source)

    def test_mtp_width_and_fallback_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cli, model, lock, fp8_model, fp8_lock, source = _populate(root)
            path = root / "raw" / "mtp-final-v1.json"
            report = json.loads(path.read_text())
            report["rows"][1]["width"] = 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(runner.SessionCError, "width order"):
                runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source)
            report["rows"][1]["width"] = 2
            report["rows"][1]["fallback_used"] = True
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(runner.SessionCError, "fallback_used"):
                runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source)

    def test_vision_pad_asset_and_metric_states_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cli, model, lock, fp8_model, fp8_lock, source = _populate(root)
            path = root / "raw" / "vision-cli-final-v1.json"
            report = json.loads(path.read_text())
            report["rows"][0]["image_pad_tokens"] = 63
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(runner.SessionCError, "image-pad"):
                runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source)
            report["rows"][0]["image_pad_tokens"] = 64
            report["amd_smi_metric"] = {"state": "unavailable"}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(runner.SessionCError, "unavailable amd-smi"):
                runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source)


if __name__ == "__main__":
    unittest.main()
