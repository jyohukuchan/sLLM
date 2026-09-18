#!/usr/bin/env python3
"""Exact full-valid-vocabulary KL(reference || candidate), FP64 accumulation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows-per-block", type=int, default=8)
    args = ap.parse_args()
    import numpy as np

    assert args.rows_per_block > 0
    refs = json.loads((args.reference / "manifest.json").read_text())
    cans = json.loads((args.candidate / "manifest.json").read_text())
    inputs = {x["id"]: x for x in json.loads(args.inputs.read_text())["cases"]}
    vocab = np.asarray(json.loads(args.vocab.read_text())["valid_token_ids"], dtype=np.int64)
    assert len(vocab) > 1 and len(np.unique(vocab)) == len(vocab)
    assert np.all(vocab[1:] > vocab[:-1]), "Vocabulary IDs must be sorted"
    rmap = {x["id"]: x for x in refs["cases"]}
    cmap = {x["id"]: x for x in cans["cases"]}
    assert set(rmap) == set(cmap) == set(inputs), "Compare an explicit identical case set"

    def matrix(root, item):
        shape = item.get("shape", [item.get("rows"), item.get("vocab_size")])
        assert len(shape) == 2 and shape[0] > 0 and shape[1] > int(vocab.max())
        file = root / item["logits_file"]
        assert file.stat().st_size == shape[0] * shape[1] * 4, file
        assert item.get("nonfinite_count", 0) == 0
        return np.memmap(file, dtype="<f4", mode="r", shape=tuple(shape))

    def stats(values):
        x = np.asarray(values, dtype=np.float64)
        assert x.size and np.isfinite(x).all() and x.min() >= -1e-10
        return {"count": int(x.size), "mean": float(x.mean()),
                "median": float(np.median(x)), "p95": float(np.quantile(x, .95)),
                "p99": float(np.quantile(x, .99)), "max": float(x.max())}

    def log_probs(x):
        assert np.isfinite(x).all(), "Nonfinite raw logits"
        x -= x.max(axis=1, keepdims=True)
        x -= np.log(np.exp(x).sum(axis=1, keepdims=True))
        return x

    output = {"definition": "KL(softmax(reference)||softmax(candidate)), temperature1, nats",
              "dtype": "FP64 normalization/accumulation from raw FP32 logits",
              "valid_vocab_size": int(len(vocab)), "reference": str(args.reference),
              "candidate": str(args.candidate), "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "vocab_sha256": hashlib.sha256(args.vocab.read_bytes()).hexdigest(), "cases": []}
    all_kld, main_kld, all_top, main_top, case_means = [], [], [], [], []
    for case_id, source in inputs.items():
        ri, qi = rmap[case_id], cmap[case_id]
        expected_hash = hashlib.sha256(json.dumps(source["token_ids"], separators=(",", ":")).encode()).hexdigest()
        expected_binary_hash = hashlib.sha256(struct.pack("<" + "i"*len(source["token_ids"]), *source["token_ids"])).hexdigest()
        for item in (ri, qi):
            h = item.get("input_token_ids_sha256", item.get("token_ids_sha256", ""))
            assert h.removeprefix("sha256:") in (expected_hash, expected_binary_hash), case_id
        assert ri["positions"] == qi["positions"] == sorted(source.get("positions", range(len(source["token_ids"])))), case_id
        ref, cand = matrix(args.reference, ri), matrix(args.candidate, qi)
        assert ref.shape[0] == cand.shape[0] == len(ri["positions"])
        klds, tops, reference_nll, candidate_nll = [], [], [], []
        for begin in range(0, len(ref), args.rows_per_block):
            end = min(len(ref), begin + args.rows_per_block)
            p = log_probs(np.asarray(ref[begin:end, vocab], dtype=np.float64).copy())
            q = log_probs(np.asarray(cand[begin:end, vocab], dtype=np.float64).copy())
            kl = np.sum(np.exp(p) * (p-q), axis=1)
            assert np.isfinite(kl).all() and kl.min() >= -1e-10
            klds.extend(np.maximum(kl, 0).tolist())
            tops.extend((p.argmax(axis=1) == q.argmax(axis=1)).tolist())
            for j, pos in enumerate(ri["positions"][begin:end]):
                if pos+1 < len(source["token_ids"]):
                    col = np.searchsorted(vocab, source["token_ids"][pos+1])
                    assert col < len(vocab) and vocab[col] == source["token_ids"][pos+1]
                    reference_nll.append(float(-p[j,col])); candidate_nll.append(float(-q[j,col]))
        item = {"id": case_id, "kld": stats(klds), "top1_agreement": float(np.mean(tops)),
                "positions": ri["positions"], "position_kld": klds,
                "reference_mean_next_token_nll": float(np.mean(reference_nll)) if reference_nll else None,
                "candidate_mean_next_token_nll": float(np.mean(candidate_nll)) if candidate_nll else None}
        output["cases"].append(item)
        all_kld.extend(klds); all_top.extend(tops)
        if not case_id.endswith("_repeat"):
            main_kld.extend(klds)
            main_top.extend(tops)
            case_means.append(item["kld"]["mean"])
        print(json.dumps({"id": case_id, **item["kld"], "top1_agreement": item["top1_agreement"]}), flush=True)
    output["all_rows"] = stats(all_kld)
    output["primary_excluding_repeat_control"] = stats(main_kld)
    output["all_rows_top1_agreement"] = float(np.mean(all_top))
    output["primary_top1_agreement"] = float(np.mean(main_top))
    output["primary_macro_case_mean_kld"] = float(np.mean(case_means))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
