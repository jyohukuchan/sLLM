#!/usr/bin/env python3
"""Measure llama-server with the token prompts used by the EXL3 comparison.

Run inside the isolated comparison container. Native timings are retained;
they are not silently redefined to match another engine's timing boundaries.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


def request(url, body):
    req = urllib.request.Request(
        url + "/completion", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18089")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())
    records = []
    for case in prompts["cases"]:
        body = {
            "prompt": case["token_ids"], "n_predict": case["output_tokens"],
            "cache_prompt": False, "ignore_eos": True, "stream": False,
            "temperature": 0.8, "min_p": 0.08, "top_k": 0, "top_p": 1.0,
            "repeat_penalty": 1.0, "repeat_last_n": 0,
            "samplers": ["min_p", "temperature"], "seed": 1234,
            "return_tokens": True,
        }
        response, wall = request(args.url, body)
        timings = response["timings"]
        assert timings["cache_n"] == 0, timings
        assert timings["prompt_n"] == case["input_tokens"], timings
        assert response["tokens_predicted"] == case["output_tokens"], response
        assert timings["predicted_n"] == case["output_tokens"], timings
        assert len(response["tokens"]) == case["output_tokens"], response
        assert not response.get("truncated"), response
        assert response["stop_type"] == "limit", response
        key = "prompt_per_second" if case["phase"] == "prefill" else "predicted_per_second"
        speed = timings[key]
        assert math.isfinite(speed) and speed > 0, timings
        numerator = timings["prompt_n"] if case["phase"] == "prefill" else timings["predicted_n"] - 1
        elapsed_ms = timings["prompt_ms"] if case["phase"] == "prefill" else timings["predicted_ms"]
        assert math.isclose(speed, 1000 * numerator / elapsed_ms, rel_tol=1e-12), timings
        filename = f"{case['phase']}-{case['input_tokens']}-{case['repeat']}.json"
        raw = {"request": body, "response": response, "http_wall_seconds": wall}
        path = args.output_dir / filename
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        record = {k: v for k, v in case.items() if k != "token_ids"}
        record.update(timings=timings, tokens_per_second=speed,
                      http_wall_seconds=wall, response_file=filename,
                      response_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      protocol="PASS")
        records.append(record)
        print(json.dumps(record), flush=True)
    rows = []
    for phase, length in (("prefill", 128), ("prefill", 512), ("decode", 128)):
        samples = [r["tokens_per_second"] for r in records
                   if r["phase"] == phase and r["input_tokens"] == length and not r["warmup"]]
        assert len(samples) == 2, samples
        median = statistics.median(samples)
        rows.append({"phase": phase, "input_tokens": length,
                     "output_tokens": 1 if phase == "prefill" else 64,
                     "tokens_per_second": median, "samples": samples,
                     "spread_percent": (max(samples) - min(samples)) / median * 100})
    prompt = ("<|im_start|>user\n2 + 3 の計算結果を、数字だけで答えてください。"
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    smoke, wall = request(args.url, {
        "prompt": prompt, "n_predict": 32, "cache_prompt": False,
        "temperature": 0, "seed": 1234, "return_tokens": True})
    (args.output_dir / "smoke.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + "\n")
    assert smoke["content"].strip() == "5", smoke
    assert smoke["stop_type"] == "eos", smoke
    summary = {"schema_version": "llama-q4k-exl3-comparison-v1",
               "prompt_manifest_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
               "warmups_per_row": 1, "measured_per_row": 2,
               "rows": rows, "records": records,
               "smoke": {"content": smoke["content"], "stop_type": smoke["stop_type"],
                         "timings": smoke["timings"], "http_wall_seconds": wall}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
