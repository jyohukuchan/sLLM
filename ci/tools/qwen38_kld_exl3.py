#!/usr/bin/env python3
"""Full-vocabulary, teacher-forced EXL3 logits; no sampler or top-k truncation."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--k-bits", type=int, default=16)
    ap.add_argument("--v-bits", type=int, default=16)
    args = ap.parse_args()
    assert args.chunk > 0 and args.context % 256 == 0
    import numpy as np
    import torch
    from exllamav3 import Config, Model, Cache, CacheLayer_quant

    assert torch.version.hip and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1201"
    source = json.loads(args.manifest.read_text())
    assert source["schema_version"] == "qwen38-kld-manifest-v1"
    assert source["cases"]
    assert max(len(c["token_ids"]) for c in source["cases"]) <= args.context
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert not (args.output_dir / "manifest.json").exists(), "Use a fresh output directory"
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    kwargs = {}
    if (args.k_bits, args.v_bits) != (16, 16):
        assert 2 <= args.k_bits <= 8 and 2 <= args.v_bits <= 8
        kwargs = dict(layer_type=CacheLayer_quant, k_bits=args.k_bits, v_bits=args.v_bits)
    cache = Cache(model, max_num_tokens=args.context, max_batch_size=1, **kwargs)
    report = {"engine": "rocm_exl3", "model": args.model, "gpu": props.gcnArchName,
              "torch": torch.__version__, "kv": [args.k_bits, args.v_bits],
              "chunk_size": args.chunk, "context": args.context,
              "input_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "position_contract": "row t predicts next token after consuming input[t]",
              "cases": [], "complete": False}
    try:
        model.load(device="cuda:0", max_chunk_size=args.chunk,
                   max_output_size=args.chunk, progressbar=True)
        for case in source["cases"]:
            case_id = case["id"]
            assert case_id and all(x.isalnum() or x in "_-" for x in case_id)
            tokens = case["token_ids"]
            wanted = sorted(case.get("positions", range(len(tokens))))
            assert wanted and len(set(wanted)) == len(wanted)
            assert wanted[0] >= 0 and wanted[-1] < len(tokens)
            file = args.output_dir / f"{case_id}.f32"
            assert not file.exists(), file
            params = {"attn_mode": "flash_attn", "cache": cache,
                      "past_len": 0, "batch_shape": (1, args.context)}
            rows = 0
            vocab = None
            started = time.monotonic()
            try:
                with file.open("wb") as out, torch.inference_mode():
                    for start in range(0, len(tokens), args.chunk):
                        end = min(len(tokens), start + args.chunk)
                        ids = torch.tensor([tokens[start:end]], dtype=torch.long)
                        params["past_len"] = start
                        logits = model.forward(ids, params)
                        assert logits.ndim == 3 and logits.shape[:2] == (1, end-start), logits.shape
                        assert torch.isfinite(logits).all().item(), (case_id, start)
                        vocab = int(logits.shape[-1])
                        offsets = [p-start for p in wanted if start <= p < end]
                        if offsets:
                            data = logits[0, offsets].float().cpu().numpy().astype("<f4", copy=False)
                            assert np.isfinite(data).all()
                            out.write(data.tobytes(order="C"))
                            rows += len(offsets)
                        del logits
                torch.cuda.synchronize()
            finally:
                for state in params.get("recurrent_states", []):
                    cache.release_state(state)
                params.clear()
            assert rows == len(wanted) and vocab
            assert file.stat().st_size == rows * vocab * 4
            item = {"id": case_id, "logits_file": file.name,
                    "shape": [rows, vocab], "positions": wanted,
                    "input_token_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                    "nonfinite_count": 0, "elapsed_seconds": time.monotonic()-started,
                    "sha256": hashlib.file_digest(file.open("rb"), "sha256").hexdigest()}
            report["cases"].append(item)
            print(json.dumps({k:v for k,v in item.items() if k != "positions"}), flush=True)
            (args.output_dir / "progress.json").write_text(json.dumps(report, indent=2)+"\n")
        report["complete"] = True
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    finally:
        model.unload()
        torch.cuda.synchronize()
    print("All full-logit rows written; normal teardown follows.", flush=True)


if __name__ == "__main__":
    main()
