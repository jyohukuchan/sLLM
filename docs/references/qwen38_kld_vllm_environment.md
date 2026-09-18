# Qwen3.8 KLD vLLM environment

2026-09-18時点のローカル検査記録。Docker検査は`--network none`、GPUデバイス未指定、ホストhome未mountの短命コンテナで行った。実機推論のPASSを示す記録ではない。

## イメージとruntime

| image | local image ID | vLLM | PyTorch | Transformers | Triton | ROCm platform |
| --- | --- | --- | --- | --- | --- | --- |
| `vllm/vllm-openai-rocm:latest` | `950cac514567` | `0.21.0+rocm722` | `2.10.0+git8514f05` | `5.8.1` | `3.6.0` | packaged ROCm 7.2.2 |
| `vllm-rdna4-mxfp4:latest` | `a692e6265e6b` | `0.21.0+rocm722` | `2.10.0+git8514f05` | `5.8.1` | `3.6.0` | packaged ROCm 7.2.2 |

The `vllm/vllm-openai-rocm:latest` image currently resolves to local digest
`sha256:98a77b20df03adeb1cfc0ced009b4df6dd52b0a994ab99a32421f30876a9ae0c`.
The local `vllm-rdna4-mxfp4:latest` image has no registry digest. Both images
advertise the same `PYTORCH_ROCM_ARCH` list, including `gfx1201`.

The CPU-only inspection reported `torch.cuda.is_available() == False`, zero
devices, and an unspecified vLLM platform. `rocminfo` could load ROCr but could
not open `/dev/kfd`, as expected without a GPU device passed to the container.

## Qwen and FP8 support

The installed model registry contains
`Qwen3_5ForConditionalGeneration` in `vllm/model_executor/models/registry.py`,
implemented by `vllm/model_executor/models/qwen3_5.py`. The model implementation
also registers the `qwen3_5_text` configuration used by the local Qwen3.8-27B
FP8 artifact.

Both images expose `LLM(..., tensor_parallel_size=..., dtype=..., ...)`,
`SamplingParams(prompt_logprobs=..., logprobs=...)`, and `ModelConfig` fields
`max_logprobs` and `logprobs_mode`. The supported logprob modes include
`raw_logits`, `raw_logprobs`, `processed_logits`, and `processed_logprobs`.
The ROCm platform lists `fp8`, `fp8_per_block`, `mxfp8`, and related methods;
its platform mapping lists Radeon RX 9070 XT and Radeon R9700 as `gfx1201`, and
`supports_fp8()` returns true on `gfx12x`. RDNA3/RDNA4 attention selection uses
the FlashAttention Triton backend.

The images set `AITER_ROCM_ARCH` to `gfx942;gfx950`, so the installed AITER
FP8 path is not built for `gfx1201`. For the fine-grained block-scaled
artifact, vLLM's ROCm kernel registry consequently offers the Triton
block-scaled FP8 path as the available RDNA4 candidate. This is a source-level
selection finding; without `/dev/kfd` the container inspection cannot prove a
successful R9700 kernel launch.

The FP8 artifact inspected for this study is
`/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-FP8`. Its config is
`Qwen3_5ForConditionalGeneration`, text config `qwen3_5_text`, vocabulary size
248,320, and fine-grained FP8 `e4m3` with weight block size `[128, 128]`.

## Raw-logit capture caveat

In vLLM `0.21.0+rocm722`, the V1 implementation in
`vllm/v1/worker/gpu_model_runner.py` computes prompt logprobs through
`self.sampler.compute_logprobs(logits)` unconditionally. Thus setting
`logprobs_mode="raw_logits"` alone does not make prompt-logprob output raw.
The normal output path also converts every full-vocabulary row into Python
objects, which is impractical for long Qwen sequences.

`ci/tools/qwen38_kld_vllm.py` therefore forces
`VLLM_USE_V2_MODEL_RUNNER=0`, installs its importable `RawLogitWorker` through
the vLLM `worker_cls` hook, and replaces the V1 prompt-logprob callback. The
callback invokes `model.compute_logits()` at the same hidden-state boundary as
the built-in prompt path, copies each complete FP32 row to CPU, and writes it
to a preallocated little-endian `.f32` file using positional writes. A small
completion marker prevents a missing final chunk from being mistaken for a
zero-filled file. The ordinary output processor receives a zero-width prompt
logprob request and never builds the full Python dictionaries.

The adapter passes `language_model_only=True`, which vLLM documents as
equivalent to zero image/video limits. Qwen3.5's `_mark_tower_model` then
constructs the vision module as missing layers and does not load its weights;
the corpus remains text-only while the language-model wrapper and tokenizer are
kept intact.

`--kv-cache-dtype` is passed to vLLM and accepts the image's cache dtype set,
including `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, and the
TurboQuant/NVFP4 names exposed by this build. The ROCm 0.21 Triton
reshape-and-cache path asserts for an explicit native `float16`/`bfloat16`
string, so the adapter maps those two requests to the backend's `auto` spelling
while preserving the requested value. vLLM's `auto` means the loaded model
dtype; the adapter resolves and records that effective dtype from the model
configuration. The output manifest includes `kv_cache_dtype_requested`,
`kv_cache_dtype_engine_arg`, and `kv_cache_dtype_resolved`. It does not label
`auto` as FP16 by assumption.

For an FP8 KV diagnostic, `fp8_e4m3` is the clear ROCm choice on gfx12x:
`current_platform.fp8_dtype()` returns `torch.float8_e4m3fn` there. Although
the image accepts the `fp8_e5m2` spelling, the inspected Triton reshape path
uses that platform dtype for every quantized KV spelling; an E5M2 run therefore
needs runtime cache dtype evidence before it can be called a true E5M2 result.

The source trace makes the E5 caveat precise. `STR_DTYPE_TO_TORCH_DTYPE` maps
`fp8`, `fp8_e4m3`, and `fp8_e5m2` all to `torch.uint8`; `Attention` then builds
the same `FullAttentionSpec(dtype=torch.uint8,
kv_quant_mode=FP8_PER_TENSOR)` for all three spellings. On gfx12x, the ROCm
Triton cache writer views that byte storage as `current_platform.fp8_dtype()`
(`float8_e4m3fn`) for every quantized spelling. The read path selects E4M3 for
`fp8`/`fp8_e4m3`, but reinterprets the same bytes as `float8_e5m2` for
`fp8_e5m2`. Thus E5M2 is an API-level reader reinterpretation over E4M3-written
bytes in this image, not an independently encoded E5M2 cache variant. The
adapter's worker report records the physical storage dtype, writer encoding,
reader encoding, and an `fp8_e5m2_platform_alias_warning` flag; E5M2 must not be
counted as a separate valid KLD variant when that flag is true.

The adapter now exposes `--fix-fp8-e5m2-writer` as an explicit experiment.
When enabled, the worker wraps only the existing ROCm
`triton_reshape_and_cache_flash` call. During an `fp8_e5m2` cache update it
temporarily supplies `torch.float8_e5m2` through that function's platform
binding, then restores the binding immediately; model weight kernels and the
global ROCm platform FP8 dtype remain unchanged. The worker report marks both
the opt-in and whether the wrapper was installed. This is a scoped image
compatibility experiment and requires the tiny writer/read oracle below before
any full E5M2 capture is accepted.

After the R9700 is available, run this tiny oracle inside the same container
before loading the full model. It uses scale 1.0 and finite values whose raw
bytes are independently checked against PyTorch casts for both cache formats:

```sh
python3 - <<'PY'
import os, sys, torch
sys.path.insert(0, "/work")
os.environ["SLLM_VLLM_FIX_FP8_E5M2_WRITER"] = "1"
import qwen38_kld_vllm as adapter
adapter._install_fp8_e5m2_writer_patch()
from vllm.v1.attention.backends import rocm_attn
triton_reshape_and_cache_flash = rocm_attn.triton_reshape_and_cache_flash

key = torch.tensor([[[-2., -1., -.5, 0., .5, 1., 2., 3.,
                      -3., -1.5, -.25, .25, 1.5, 2.5, 4., 5.]]],
                   device="cuda", dtype=torch.float32)
value = -key
slot = torch.tensor([0], device="cuda", dtype=torch.int64)
raw = torch.empty((2, 1, 16, 1, 16), device="cuda", dtype=torch.uint8)
for scale_value in (1.0, 2.0):
    scale = torch.tensor(scale_value, device="cuda", dtype=torch.float32)
    for name, dtype in (("e4", torch.float8_e4m3fn),
                        ("e5", torch.float8_e5m2)):
        raw.zero_()
        triton_reshape_and_cache_flash(
            key, value, raw[0].view(1, 1, 1, 16, 16),
            raw[1].view(1, 1, 16, 16), slot,
            "fp8_e4m3" if name == "e4" else "fp8_e5m2", scale, scale)
        torch.cuda.synchronize()
        # Convert on CPU so the expected bytes are independent of the GPU
        # writer's cast implementation.
        expected = (key[0, 0].cpu() / scale_value).to(dtype).view(torch.uint8)
        actual = raw[0, 0, 0, 0, :].cpu()
        assert torch.equal(actual, expected), (name, scale_value, actual.tolist(), expected.tolist())
        expected_v = (value[0, 0].cpu() / scale_value).to(dtype).view(torch.uint8)
        actual_v = raw[1].view(1, 1, 16, 16)[0, 0, :, 0].cpu()
        assert torch.equal(actual_v, expected_v), (name, scale_value, actual_v.tolist(), expected_v.tolist())
        decoded = actual.to(device="cuda").view(dtype).float().cpu()
        assert torch.equal(decoded, (key[0, 0].cpu() / scale_value).to(dtype).float()), name
        print(name, scale_value, actual.tolist())
print("PASS: E4 and E5 writer/read byte oracle")
PY
```

Use `--max-num-batched-tokens` to bound the GPU staging chunk (the adapter
defaults to 32). The adapter defaults `--fla-autotune single`, which keeps the
first-run Triton FLA compile to the first valid configuration in each GDN
module; `--fla-autotune auto` benchmarks the complete candidate set for a
performance study. Both modes use the same kernel operations and raw-logit
capture boundary. Prefix caching is disabled unless
`--enable-prefix-caching` is explicitly supplied; the latter is a separate KV
diagnostic and should not be mixed into the baseline KLD row.

The R9700 smoke and full capture used `--cpu-offload-gb 8
--gpu-memory-utilization 0.80`; vLLM logged 8.09 GiB of CPU-offloaded
parameters and 20.08 GiB of model weights on GPU. This is a GPU compute run
with weight offload, rather than a fully GPU-resident model run. The successful
baseline also used `--fla-autotune single`; the output report records each
selected first Triton configuration and its original candidate count.

The vLLM R9700 attempt ledger is retained in the run log directory. Attempts
1 and 2 exited with engine initialization failure (`HIP out of memory`, a
180 MiB allocation with only 170 MiB free; the allocator setting change did
not alter the model footprint). Attempts 3 and 4 did not reach a clean shell
exit: their engine processes were stopped with SIGKILL after the FLA cache
showed one candidate compile still incomplete and `py-spy` repeatedly showed
the same `make_amdgcn -> autotuner.do_bench -> chunk_scaled_dot_kkt_fwd`
stack; their logs end with the resource-tracker warning. Attempt 5 exited
with `HIP out of memory` while allocating a 236 MiB KV block (28 MiB free).
Attempt 6 exited with the ROCm assertion `unsupported kv_cache_dtype (str), got
bfloat16`. Attempt 7 and the full run exited 0 after native BF16 was passed as
`auto` and FLA search was bounded to one candidate. Logs are under
`/home/homelab1/datapool/qwen38-kld-20260918/logs/`.

Example inside a compatible vLLM ROCm runtime:

```sh
python3 ci/tools/qwen38_kld_vllm.py \
  --model-root /models/Qwen3.8-27B-FP8 \
  --manifest /work/cases-v1.json \
  --output-dir /work/vllm-fp8 \
  --tensor-parallel-size 2 \
  --max-model-len 2050
```

The adapter writes `manifest.json` and one dense `<case-id>.f32` file per
case. Each row is the logits after consuming `token_ids[:position + 1]`, so it
predicts the next token at that position. KLD consumers should crop the raw
vocabulary to the official token IDs according to the shared measurement
contract before normalization.

## Phase2 measurement ledger

The following R9700 runs used the same FP8 checkpoint, TP1, BF16 model dtype,
8 GiB CPU weight offload, `gpu_memory_utilization=0.80`, eager execution,
32-token prefill chunks, and single FLA configurations. Each completed run
ended with exit code 0 and reported finite full-vocabulary rows.

| output | corpus | KV runtime evidence |
| --- | --- | --- |
| `results/vllm-fp8-e4m3` | `cases-v1.json`, 9 cases / 2,761 rows | physical `uint8`, writer/read `float8_e4m3fn` |
| `results/vllm-bf16-long` | `cases-long-v1.json`, 514 rows | native BF16 cache |
| `results/vllm-fp8-e4m3-long` | `cases-long-v1.json`, 514 rows | physical `uint8`, writer/read `float8_e4m3fn` |

The short E4M3 run compared with llama BF16 has primary mean KLD
`0.01564249699`, p95 `0.05097701885`, max `2.25977363002`, and top-1
agreement `0.95516717`. Against the same-run vLLM BF16 output, its mean KLD
is `0.00535145071`, p95 `0.01710457151`, max `2.57929763594`, and top-1
agreement `0.97492401`.

For the two long 4,097-token prompts, vLLM BF16 versus llama BF16 has mean
KLD `0.00239955582`, p95 `0.01403844780`, max `0.14986482293`, and top-1
agreement `0.99610895`. E4M3 versus llama BF16 has mean `0.00203316292`, p95
`0.01217048707`, max `0.07465210442`, and top-1 agreement `0.99221790`. E4M3
versus vLLM BF16 has mean `0.00208864350`, p95 `0.01147796604`, max
`0.12114881706`, and top-1 agreement `0.99027237`.

The corresponding KLD JSON files are
`kld-vllm-fp8-e4m3-vs-llama-bf16.json`,
`kld-vllm-fp8-e4m3-vs-vllm-bf16.json`,
`kld-vllm-bf16-long-vs-llama-bf16-long.json`,
`kld-vllm-fp8-e4m3-long-vs-llama-bf16-long.json`, and
`kld-vllm-fp8-e4m3-long-vs-vllm-bf16-long.json` under the run's `results/`
directory.

The patched E5M2 full-model attempt is retained as an unsupported experiment:
it exited 1 during model construction with vLLM's explicit
`fp8_e5m2 kv-cache is not supported with fp8 checkpoints` guard, before any
logit rows were produced. The independent CPU-reference GPU oracle
`/home/homelab1/datapool/qwen38-kld-20260918/qwen38_kv_oracle_independent.py`
completed with exit 0 and tested all finite E4M3 codes (254) and E5M2 codes
(248) at scales 1.0 and 2.0. It found zero K/V byte mismatches and confirmed
that the global platform dtype remained E4M3; its durable evidence is
`/home/homelab1/datapool/qwen38-kld-20260918/vllm-kv-writer-independent-oracle.json`.
This validates the scoped writer wrapper's byte-level GPU path, but is not
full-model E5M2 evidence. The opt-in constructor bypass is retained for future
investigation only; the local checkpoint declares weight `fmt=e4m3` and has no
independent E5M2 KV scale calibration. No E5M2 KLD result is included.
