#!/usr/bin/env python3
"""Merge per-tensor prototype .ullm.d directories into one prototype directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def copy_file(src: Path, dst: Path, overwrite: bool) -> int:
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def relative_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too short to contain a safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return 8 + int(header_len), header


def copy_safetensors_payload(
    src_file: Path,
    tensor_name: str,
    dst: Path,
    overwrite: bool,
    buffer_bytes: int,
) -> tuple[int, str]:
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data_start, header = read_safetensors_header(src_file)
    tensor = header.get(tensor_name)
    if tensor is None:
        raise KeyError(f"{tensor_name} not found in {src_file}")
    start, end = tensor["data_offsets"]
    if end < start:
        raise ValueError(f"{tensor_name} has invalid data_offsets {tensor['data_offsets']}")
    remaining = int(end) - int(start)
    offset = data_start + int(start)
    digest = hashlib.sha256()
    copied = 0
    with src_file.open("rb") as src, dst.open("wb") as out:
        src.seek(offset)
        while remaining > 0:
            chunk = src.read(min(buffer_bytes, remaining))
            if not chunk:
                raise EOFError(f"unexpected EOF while copying {tensor_name} from {src_file}")
            out.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
            remaining -= len(chunk)
    return copied, digest.hexdigest()


def merge_passthrough_tensors(args: argparse.Namespace, copied_files: list[dict]) -> list[dict]:
    if not args.include_passthrough:
        return []
    if args.plan_json is None:
        raise ValueError("--include-passthrough requires --plan-json")
    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    passthrough: list[dict] = []
    for index, tensor in enumerate(plan.get("tensors", [])):
        if tensor.get("action") != "passthrough":
            continue
        name = tensor["name"]
        src_file = Path(tensor["source_file"])
        tensor_stem = f"{index:03d}-{sanitize(name)}"
        dst_rel = Path("passthrough") / f"{tensor_stem}.raw"
        bytes_copied, sha256 = copy_safetensors_payload(
            src_file,
            name,
            args.output_dir / dst_rel,
            args.overwrite,
            args.copy_buffer_bytes,
        )
        expected_bytes = int(tensor["n_bytes"])
        if bytes_copied != expected_bytes:
            raise RuntimeError(
                f"copied {bytes_copied} bytes for {name}, expected {expected_bytes}"
            )
        copied_files.append({"path": relative_path(dst_rel), "bytes": bytes_copied})
        passthrough.append(
            {
                "name": name,
                "source_file": str(src_file),
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "family": tensor["family"],
                "elements": tensor["n_elements"],
                "payload_bytes": bytes_copied,
                "payload_file": relative_path(dst_rel),
                "payload_encoding": "raw_safetensors_payload",
                "payload_sha256": sha256,
            }
        )
    return passthrough


def merge(args: argparse.Namespace) -> dict:
    summary = json.loads(args.policy_summary.read_text(encoding="utf-8"))
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    tensors_dir = args.output_dir / "tensors"
    codebooks_dir = args.output_dir / "codebooks"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    codebooks_dir.mkdir(parents=True, exist_ok=True)

    merged_tensors: list[dict] = []
    merged_codebooks: dict[tuple[str, str], dict] = {}
    copied_files: list[dict] = []
    source_model_dir = None

    for result_index, result in enumerate(summary.get("results", [])):
        if result.get("returncode") != 0:
            raise RuntimeError(f"cannot merge failed result {result_index}: {result}")
        src_dir = Path(result["output_dir"])
        manifest_path = src_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_model_dir = source_model_dir or manifest.get("source_model_dir")
        for tensor in manifest.get("tensors", []):
            tensor_stem = f"{result_index:03d}-{sanitize(tensor['name'])}"
            src_index = src_dir / tensor["index_file"]
            src_scale = src_dir / tensor["scale_file"]
            dst_index_rel = Path("tensors") / f"{tensor_stem}.idx4"
            dst_scale_rel = Path("tensors") / f"{tensor_stem}.scale_u8"
            index_bytes = copy_file(src_index, args.output_dir / dst_index_rel, args.overwrite)
            scale_bytes = copy_file(src_scale, args.output_dir / dst_scale_rel, args.overwrite)
            copied_files.append({"path": relative_path(dst_index_rel), "bytes": index_bytes})
            copied_files.append({"path": relative_path(dst_scale_rel), "bytes": scale_bytes})

            codebook_key = (tensor["family"], tensor["candidate_id"])
            if codebook_key not in merged_codebooks:
                src_codebook = src_dir / tensor["codebook_file"]
                codebook_rel = Path("codebooks") / f"{sanitize(codebook_key[0] + '__' + codebook_key[1])}.f32"
                codebook_bytes = copy_file(
                    src_codebook,
                    args.output_dir / codebook_rel,
                    args.overwrite,
                )
                merged_codebooks[codebook_key] = {
                    "family": tensor["family"],
                    "candidate_id": tensor["candidate_id"],
                    "file": relative_path(codebook_rel),
                    "encoding": "f32_le",
                    "entries": 16,
                }
                copied_files.append({"path": relative_path(codebook_rel), "bytes": codebook_bytes})

            merged = dict(tensor)
            merged["index_file"] = relative_path(dst_index_rel)
            merged["scale_file"] = relative_path(dst_scale_rel)
            merged["codebook_file"] = merged_codebooks[codebook_key]["file"]
            merged_tensors.append(merged)

    passthrough_tensors = merge_passthrough_tensors(args, copied_files)
    manifest = {
        "schema_version": "ullm-prototype-manifest-v0.1",
        "source_model_dir": source_model_dir,
        "tensors": merged_tensors,
        "codebooks": list(merged_codebooks.values()),
    }
    if passthrough_tensors:
        manifest["passthrough_tensors"] = passthrough_tensors
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied_files.append({"path": "manifest.json", "bytes": manifest_path.stat().st_size})

    total_bytes = sum(item["bytes"] for item in copied_files)
    merge_summary = {
        "schema_version": "ullm-prototype-merge-summary-v0.1",
        "policy_summary": str(args.policy_summary),
        "output_dir": str(args.output_dir),
        "tensor_count": len(merged_tensors),
        "passthrough_tensor_count": len(passthrough_tensors),
        "codebook_count": len(merged_codebooks),
        "total_file_bytes": total_bytes,
        "files": copied_files,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(merge_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return merge_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-summary", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--include-passthrough", action="store_true")
    parser.add_argument("--copy-buffer-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.copy_buffer_bytes < 1:
        raise SystemExit("--copy-buffer-bytes must be >= 1")
    merge(args)


if __name__ == "__main__":
    main()
