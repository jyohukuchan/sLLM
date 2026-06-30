#!/usr/bin/env python3
"""Merge per-tensor prototype .ullm.d directories into one prototype directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def copy_file(src: Path, dst: Path, overwrite: bool) -> int:
    if dst.exists() and not overwrite:
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


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
            copied_files.append({"path": str(dst_index_rel), "bytes": index_bytes})
            copied_files.append({"path": str(dst_scale_rel), "bytes": scale_bytes})

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
                    "file": str(codebook_rel).replace("\\", "/"),
                    "encoding": "f32_le",
                    "entries": 16,
                }
                copied_files.append({"path": str(codebook_rel), "bytes": codebook_bytes})

            merged = dict(tensor)
            merged["index_file"] = str(dst_index_rel).replace("\\", "/")
            merged["scale_file"] = str(dst_scale_rel).replace("\\", "/")
            merged["codebook_file"] = merged_codebooks[codebook_key]["file"]
            merged_tensors.append(merged)

    manifest = {
        "schema_version": "ullm-prototype-manifest-v0.1",
        "source_model_dir": source_model_dir,
        "tensors": merged_tensors,
        "codebooks": list(merged_codebooks.values()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied_files.append({"path": "manifest.json", "bytes": manifest_path.stat().st_size})

    total_bytes = sum(item["bytes"] for item in copied_files)
    merge_summary = {
        "schema_version": "ullm-prototype-merge-summary-v0.1",
        "policy_summary": str(args.policy_summary),
        "output_dir": str(args.output_dir),
        "tensor_count": len(merged_tensors),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge(args)


if __name__ == "__main__":
    main()
