from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "tools" / "export-sq8-chat-template-fixtures.py"
VALIDATOR_PATH = REPO_ROOT / "tools" / "validate-sq8-chat-template-fixtures.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORTER = load_module("sq8_chat_template_exporter", EXPORTER_PATH)
VALIDATOR = load_module("sq8_chat_template_validator", VALIDATOR_PATH)


class FakeTokenizer:
    chat_template = "fake-qwen3-chat-template-v1"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Any:
        self.calls.append(
            {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        if add_generation_prompt is not True or enable_thinking is not False:
            raise AssertionError("template options are not frozen")
        rendered = (
            "<fake-chat>"
            + json.dumps(
                messages,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "<fake-assistant>"
        )
        if not tokenize:
            return rendered

        exact_repeat = None
        if (
            len(messages) == 1
            and messages[0].get("role") == "user"
            and isinstance(messages[0].get("content"), str)
        ):
            content = messages[0]["content"]
            candidate = len(content) // len(EXPORTER.REPEATED_UNIT)
            if content == EXPORTER.REPEATED_UNIT * candidate:
                exact_repeat = candidate
        if exact_repeat is not None:
            length = EXPORTER.BASE_PROMPT_TOKENS + exact_repeat
        else:
            length = EXPORTER.BASE_PROMPT_TOKENS + max(
                1, len(rendered.encode("utf-8")) // 16
            )
        seed = int(hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:8], 16)
        return {"input_ids": [(seed + index) % 100_000 for index in range(length)]}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_fake_model(model_dir: Path) -> None:
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "qwen3", "vocab_size": 100_000})
    for index, name in enumerate(EXPORTER.TOKENIZER_REQUIRED_FILES):
        (model_dir / name).write_bytes(f"fake-tokenizer-file-{index}\n".encode("ascii"))
    metadata = model_dir / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for name in ("config.json", *EXPORTER.TOKENIZER_REQUIRED_FILES):
        (metadata / f"{name}.metadata").write_text(
            EXPORTER.EXPECTED_REVISION + "\n", encoding="ascii"
        )


class Sq8ChatTemplateFixtureToolTests(unittest.TestCase):
    def export_fake(self, root: Path) -> tuple[Path, Path, FakeTokenizer]:
        model = root / "model"
        output = root / "fixtures"
        write_fake_model(model)
        tokenizer = FakeTokenizer()
        EXPORTER.export_fixture_set(
            output,
            model,
            tokenizer=tokenizer,
            transformers_version="fake-transformers-1",
        )
        return model, output, tokenizer

    def test_fake_export_has_exact_lengths_and_validator_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model, output, exporter_tokenizer = self.export_fake(Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))

            self.assertEqual(manifest["schema_version"], EXPORTER.SCHEMA_VERSION)
            self.assertEqual(
                manifest["exact_length_contract"]["target_prompt_tokens"],
                list(EXPORTER.EXACT_PROMPT_LENGTHS),
            )
            exact_records = {
                record["prompt_tokens"]
                for record in manifest["fixture_files"]
                if record["kind"] == "exact_length"
            }
            self.assertEqual(exact_records, set(EXPORTER.EXACT_PROMPT_LENGTHS))
            self.assertEqual(len(manifest["fixture_files"]), 10)
            self.assertEqual(
                manifest["tokenizer"]["loader"],
                {"local_files_only": True, "trust_remote_code": False},
            )

            validator_tokenizer = FakeTokenizer()
            result = VALIDATOR.validate_fixture_set(
                output,
                model,
                tokenizer=validator_tokenizer,
                transformers_version="fake-transformers-1",
            )
            self.assertEqual(result["fixtures"], 10)
            self.assertEqual(
                result["exact_prompt_lengths"], list(EXPORTER.EXACT_PROMPT_LENGTHS)
            )
            for call in [*exporter_tokenizer.calls, *validator_tokenizer.calls]:
                self.assertTrue(call["add_generation_prompt"])
                self.assertFalse(call["enable_thinking"])

    def test_existing_output_is_no_clobber_before_model_or_tokenizer_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fixtures"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("unchanged", encoding="ascii")
            tokenizer = FakeTokenizer()

            with self.assertRaisesRegex(EXPORTER.ExportError, "refusing to overwrite"):
                EXPORTER.export_fixture_set(
                    output,
                    root / "missing-model",
                    tokenizer=tokenizer,
                    transformers_version="fake-transformers-1",
                )

            self.assertEqual(marker.read_text(encoding="ascii"), "unchanged")
            self.assertEqual(tokenizer.calls, [])

    def test_atomic_publish_does_not_replace_raced_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "source-marker").write_text("source", encoding="ascii")
            (destination / "destination-marker").write_text(
                "destination", encoding="ascii"
            )

            with self.assertRaises(FileExistsError):
                EXPORTER.rename_noreplace(source, destination)

            self.assertTrue((source / "source-marker").is_file())
            self.assertTrue((destination / "destination-marker").is_file())

    def test_validator_rejects_self_consistent_token_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model, output, _ = self.export_fake(Path(temporary))
            relative = "fixtures/exact-p0032.json"
            fixture_path = output / relative
            fixture = json.loads(fixture_path.read_text(encoding="ascii"))
            fixture["expected"]["token_ids"][0] += 1
            encoded = b"".join(
                struct.pack("<I", token_id)
                for token_id in fixture["expected"]["token_ids"]
            )
            fixture["expected"]["token_ids_u32le_sha256"] = hashlib.sha256(
                encoded
            ).hexdigest()
            write_json(fixture_path, fixture)

            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            for record in manifest["fixture_files"]:
                if record["file"] == relative:
                    record["bytes"] = fixture_path.stat().st_size
                    record["sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
                    break
            else:
                self.fail("tampered fixture is absent from the manifest")
            write_json(manifest_path, manifest)

            with self.assertRaisesRegex(VALIDATOR.ValidationError, "token IDs mismatch"):
                VALIDATOR.validate_fixture_set(
                    output,
                    model,
                    tokenizer=FakeTokenizer(),
                    transformers_version="fake-transformers-1",
                )

    @unittest.skipUnless(
        EXPORTER.DEFAULT_MODEL_DIR.is_dir(),
        "local Qwen3-14B-FP8 tokenizer is absent",
    )
    def test_real_local_tokenizer_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "real-fixtures"
            exported = subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER_PATH),
                    "--model-dir",
                    str(EXPORTER.DEFAULT_MODEL_DIR),
                    "--output-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=180,
            )
            self.assertEqual(exported.returncode, 0, exported.stderr)
            validated = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_PATH),
                    str(output),
                    "--model-dir",
                    str(EXPORTER.DEFAULT_MODEL_DIR),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=180,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertIn("passed=true independent_recompute=true", validated.stdout)
            self.assertIn("exact_prompt_lengths=32,128,512,2048,3584", validated.stdout)


if __name__ == "__main__":
    unittest.main()
