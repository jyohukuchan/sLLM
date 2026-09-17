#!/usr/bin/env python3
"""External rocm_exl3 research probe; run only in its isolated ROCm environment.

This is not an sLLM correctness test or a throughput benchmark. Runtime imports
are intentionally deferred so ordinary host tooling does not require PyTorch.
"""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    args = parser.parse_args()

    import numpy as np
    import torch
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler import DefaultSampler

    assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1201", props
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    started = time.monotonic()
    model.load(device="cuda:0", max_chunk_size=512, progressbar=False)
    load_seconds = time.monotonic() - started
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    prompt = ("<|im_start|>user\n" + args.prompt
              + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = tokenizer.encode(prompt, add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=args.tokens,
              sampler=DefaultSampler(), seed=1234, return_logits=True,
              stop_conditions=config.eos_token_id_list)
    generator.enqueue(job)
    pieces = []
    logit_rows = 0
    first_logits = None
    final_metrics = {}
    started = time.monotonic()
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result["stage"] != "streaming":
                continue
            pieces.append(result.get("text", ""))
            logits = result.get("logits")
            if logits is not None and logits.numel():
                logits = logits.reshape(-1, logits.shape[-1])[:, :tokenizer.actual_vocab_size].float()
                assert torch.isfinite(logits).all().item(), "Nonfinite vocabulary logits"
                logit_rows += logits.shape[0]
                if first_logits is None:
                    first_logits = logits[0].cpu().numpy()
            if result.get("eos"):
                for key in ("prompt_tokens", "new_tokens", "time_prefill", "time_generate", "cached_tokens"):
                    final_metrics[key] = result.get(key)
    torch.cuda.synchronize()
    assert logit_rows > 0 and first_logits is not None, "No numerical observations"
    assert pieces and "".join(pieces).strip(), "Empty generation"
    report = {
        "model": args.model, "gpu": props.gcnArchName, "torch": torch.__version__,
        "torch_hip": torch.version.hip, "prompt": prompt, "seed": 1234,
        "sampler": "DefaultSampler(min_p=0.08, temperature=0.8)",
        "requested_tokens": args.tokens, "stop_token_ids": config.eos_token_id_list,
        "load_seconds": load_seconds,
        "generation_wall_seconds": time.monotonic() - started,
        "observed_finite_logit_rows": logit_rows, "text": "".join(pieces),
        "first_logit_top1": int(first_logits.argmax()), "metrics": final_metrics,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "status": "generation-and-finite-logits-observed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".first-logits.npy"), first_logits)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    model.unload()
    torch.cuda.synchronize()
    print("Explicit model.unload completed; normal interpreter teardown follows.", flush=True)


if __name__ == "__main__":
    main()
