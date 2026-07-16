#!/usr/bin/env python3
"""Compare transient AQ4 CPU and production-GPU layer-0 stage streams.

Both inputs use the framed ``f32le`` protocol emitted by the CPU hybrid probe
and the opt-in GPU kernel-stage sidecar.  The CPU probe emits every replay
timestep; this tool joins only the final timestep of each of the three
hash-bound contexts, which is the value retained by the GPU M=1 trace.
Full tensors are consumed per comparison and never written to the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator


SCHEMA = "ullm.aq4_layer0_cpu_gpu_stage_compare.v1"
MAX_HEADER_BYTES = 64 * 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024
STAGES = (
    "qkv_dequant_row_scale",
    "z_dequant_row_scale",
    "recurrent_gate",
    "recurrent_beta",
    "recurrent_state_after",
    "recurrent_output",
    "attention_residual",
    "post_norm",
    "mlp_activation",
    "layer_output",
)
CONTEXTS = (
    (
        "fixture-prompt-0",
        0,
        "42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c",
        3,
    ),
    (
        "fixture-prompt-0",
        1,
        "6af1601b9bf35d095b24c5bac3a95a01bf77d047b576441d0a5f9510eec66249",
        4,
    ),
    (
        "fixture-prompt-1",
        0,
        "3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6",
        2,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_keys() -> set[tuple[str, int, str, int, int, str]]:
    return {
        (case_id, step, context_hash, context_length, context_length - 1, stage)
        for case_id, step, context_hash, context_length in CONTEXTS
        for stage in STAGES
    }


def frame_key(header: dict[str, Any]) -> tuple[str, int, str, int, int, str]:
    return (
        header["case_id"],
        header["step"],
        header["context_token_ids_sha256"],
        header["context_length"],
        header["timestep"],
        header["stage"],
    )


def read_frames(source: BinaryIO) -> Iterator[tuple[dict[str, Any], bytes]]:
    while True:
        line = source.readline(MAX_HEADER_BYTES + 1)
        if not line:
            raise ValueError("stage stream ended without terminal frame")
        if len(line) > MAX_HEADER_BYTES or not line.endswith(b"\n"):
            raise ValueError("stage stream frame header is oversized or unterminated")
        header = json.loads(line)
        if header == {"kind": "end"}:
            if source.read(1):
                raise ValueError("stage stream has trailing data after terminal frame")
            return
        expected = {
            "kind",
            "case_id",
            "step",
            "context_token_ids_sha256",
            "context_length",
            "timestep",
            "stage",
            "dtype",
            "shape",
            "bytes",
        }
        if not isinstance(header, dict) or set(header) != expected:
            raise ValueError("stage stream header fields differ")
        elements = header["shape"][0] if isinstance(header["shape"], list) and len(header["shape"]) == 1 else None
        byte_count = header["bytes"]
        if (
            header["kind"] != "stage"
            or not isinstance(header["case_id"], str)
            or not isinstance(header["step"], int)
            or not isinstance(header["context_token_ids_sha256"], str)
            or len(header["context_token_ids_sha256"]) != 64
            or not isinstance(header["context_length"], int)
            or not isinstance(header["timestep"], int)
            or not isinstance(header["stage"], str)
            or header["dtype"] != "f32le"
            or not isinstance(elements, int)
            or elements <= 0
            or not isinstance(byte_count, int)
            or byte_count != elements * 4
            or byte_count > MAX_FRAME_BYTES
        ):
            raise ValueError("stage stream frame contract differs")
        payload = source.read(byte_count)
        if len(payload) != byte_count:
            raise ValueError("stage stream frame payload is truncated")
        yield header, payload


def selected_cpu_frames(path: Path, required: set[tuple[str, int, str, int, int, str]]) -> dict[tuple[str, int, str, int, int, str], bytes]:
    frames: dict[tuple[str, int, str, int, int, str], bytes] = {}
    with path.open("rb") as source:
        for header, payload in read_frames(source):
            key = frame_key(header)
            if key not in required:
                continue
            if key in frames:
                raise ValueError(f"CPU stage stream repeats frame {key}")
            frames[key] = payload
    if set(frames) != required:
        raise ValueError("CPU stage stream is missing one or more final-context frames")
    return frames


def fixed_coordinates(elements: int) -> list[int]:
    candidates = (0, 1, 31, 127, 1024, 2048, 4095, elements - 1)
    return list(dict.fromkeys(index for index in candidates if 0 <= index < elements))


@dataclass
class Metrics:
    records: int = 0
    elements_per_record: int | None = None
    diff_sq: float = 0.0
    cpu_sq: float = 0.0
    gpu_sq: float = 0.0
    dot: float = 0.0
    max_abs: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def update(self, header: dict[str, Any], gpu: bytes, cpu: bytes) -> None:
        if len(gpu) != len(cpu) or len(gpu) % 4:
            raise ValueError(f"stage payload geometry differs: {header['stage']}")
        elements = len(gpu) // 4
        if self.elements_per_record is None:
            self.elements_per_record = elements
        elif self.elements_per_record != elements:
            raise ValueError(f"stage geometry changed: {header['stage']}")
        coordinates = fixed_coordinates(elements)
        sampled: dict[int, tuple[float, float, float]] = {}
        record_diff_sq = 0.0
        record_cpu_sq = 0.0
        record_gpu_sq = 0.0
        record_dot = 0.0
        record_max_abs = 0.0
        for index, ((gpu_value,), (cpu_value,)) in enumerate(
            zip(struct.iter_unpack("<f", gpu), struct.iter_unpack("<f", cpu))
        ):
            if not math.isfinite(gpu_value) or not math.isfinite(cpu_value):
                raise ValueError(f"non-finite value in stage {header['stage']}")
            difference = gpu_value - cpu_value
            record_diff_sq += difference * difference
            record_cpu_sq += cpu_value * cpu_value
            record_gpu_sq += gpu_value * gpu_value
            record_dot += gpu_value * cpu_value
            record_max_abs = max(record_max_abs, abs(difference))
            if index in coordinates:
                sampled[index] = (gpu_value, cpu_value, abs(difference))
        self.records += 1
        self.diff_sq += record_diff_sq
        self.cpu_sq += record_cpu_sq
        self.gpu_sq += record_gpu_sq
        self.dot += record_dot
        self.max_abs = max(self.max_abs, record_max_abs)
        self.samples.append(
            {
                "case_id": header["case_id"],
                "step": header["step"],
                "context_token_ids_sha256": header["context_token_ids_sha256"],
                "context_length": header["context_length"],
                "timestep": header["timestep"],
                "elements": elements,
                "coordinates": coordinates,
                "gpu_values": [sampled[index][0] for index in coordinates],
                "cpu_values": [sampled[index][1] for index in coordinates],
                "abs_diff_values": [sampled[index][2] for index in coordinates],
                "max_abs": record_max_abs,
                "relative_l2": math.sqrt(record_diff_sq) / math.sqrt(record_cpu_sq)
                if record_cpu_sq
                else (0.0 if record_diff_sq == 0.0 else math.inf),
                "cosine": record_dot / math.sqrt(record_gpu_sq * record_cpu_sq)
                if record_gpu_sq and record_cpu_sq
                else None,
            }
        )

    def report(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "elements_per_record": self.elements_per_record,
            "max_abs": self.max_abs,
            "relative_l2": math.sqrt(self.diff_sq) / math.sqrt(self.cpu_sq)
            if self.cpu_sq
            else (0.0 if self.diff_sq == 0.0 else math.inf),
            "cosine": self.dot / math.sqrt(self.gpu_sq * self.cpu_sq)
            if self.gpu_sq and self.cpu_sq
            else None,
            "samples": self.samples,
        }


def compare(cpu_stream: Path, gpu_stream: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    required = expected_keys()
    cpu_frames = selected_cpu_frames(cpu_stream, required)
    metrics = {stage: Metrics() for stage in STAGES}
    seen: set[tuple[str, int, str, int, int, str]] = set()
    with gpu_stream.open("rb") as source:
        for header, payload in read_frames(source):
            key = frame_key(header)
            if key not in required:
                raise ValueError(f"GPU stage stream has an unexpected frame {key}")
            if key in seen:
                raise ValueError(f"GPU stage stream repeats frame {key}")
            seen.add(key)
            metrics[header["stage"]].update(header, payload, cpu_frames.pop(key))
    if seen != required or cpu_frames:
        raise ValueError("GPU stage stream is missing one or more required frames")
    result = {
        "schema_version": SCHEMA,
        "status": "valid",
        "classification": "unclassified",
        "promotion": False,
        "comparison_contract": "GPU f32le frames are compared immediately with matching AQ4 CPU f32le frames at the final token of each hash-bound context; full tensors are not persisted in this report.",
        "cpu_stream": {"path": str(cpu_stream), "sha256": sha256_file(cpu_stream)},
        "gpu_stream": {"path": str(gpu_stream), "sha256": sha256_file(gpu_stream)},
        "contexts": [
            {
                "case_id": case_id,
                "step": step,
                "context_token_ids_sha256": context_hash,
                "context_length": context_length,
                "timestep": context_length - 1,
            }
            for case_id, step, context_hash, context_length in CONTEXTS
        ],
        "stages": {stage: metrics[stage].report() for stage in STAGES},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_name("SHA256SUMS").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="ascii"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-stream", type=Path, required=True)
    parser.add_argument("--gpu-stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.cpu_stream, args.gpu_stream, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, struct.error) as error:
        print(f"AQ4 CPU/GPU stage comparison failed: {error}")
        return 1
    print(json.dumps({"status": result["status"], "stages": len(result["stages"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
