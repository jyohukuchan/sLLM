# Phase 46 tools

This document is the operator-facing closeout for the Phase 46 host tools.  The
normative scope and ownership remain in the [Phase 46 plan](../plans/archive/2026/08/21-31/phase46-conversion-quantization-benchmark-quality-tools.md).
This implementation is deliberately offline and bounded: it verifies inputs,
produces digest-bound summaries, and never turns an artifact or evaluator error
into CPU inference or a quality pass.

## Entry points

Build the binaries from the workspace with `cargo run -p <package> --bin
<binary> -- ...`.  Every command has `--help`; unknown options and missing
required values are errors.

| Binary | Purpose | Invocation |
| --- | --- | --- |
| `sllm-artifact` (`sllm-tools`) | Capability dispatch, GGUF split/merge, LoRA conversion, layout repack, bounded quantization and imatrix | `capabilities`, `split`, `merge`, `lora`, `repack`, `quantize`, `imatrix` (examples below) |
| `sllm-convert-gguf` (`sllm-cli`) | Reviewed model-lock/cache to GGUF and derived lock | `--kind qwen35-bf16`, `qwen35-fp8`, `gemma4-nvfp4`, or `qwen35moe-mxfp4` |
| `sllm-bench` (`sllm-tools`) | Identity-bound aggregation of already-collected benchmark samples | `aggregate --input INPUT.json --output-bundle DIR --tool-commit SHA40` |
| `sllm-eval` (`sllm-tools`) | Bounded perplexity, logit KLD/top-1, task, or long-context evaluation | `--input INPUT.json --manifest RUN.json [--output RESULT.json]` |
| `sllm-qwen35-quality-baseline` (`sllm-hip`) | Exact HIP, BF16-only Qwen3.5-4B baseline | `LOCK CACHE DATASET_JSON DEVICE_INDEX GFX_TARGET REPEATS OUTPUT_JSON` |

`sllm-artifact` writes machine-readable JSON to stdout.  Successful commands
exit 0; errors are written to stderr with an `sllm-artifact:` prefix and exit
2.  The benchmark, evaluator, and artifact binaries do not catch a failed
operation and label it `PASS`.

### Artifact commands

```text
cargo run -p sllm-tools --bin sllm-artifact -- capabilities --architecture qwen35
cargo run -p sllm-tools --bin sllm-artifact -- split \
  --input MODEL.gguf --output-dir PARTS --max-part-bytes 1073741824
cargo run -p sllm-tools --bin sllm-artifact -- merge \
  --manifest PARTS/manifest.json --output-dir MERGED
cargo run -p sllm-tools --bin sllm-artifact -- lora \
  --input SOURCE.json --output-dir ADAPTER
cargo run -p sllm-tools --bin sllm-artifact -- repack \
  --encoding mxfp4 --values values.bin --scales scales.bin \
  --rows ROWS --columns COLUMNS --output-dir REPACKED
cargo run -p sllm-tools --bin sllm-artifact -- quantize \
  --recipe RECIPE --input-json values.json --rows ROWS --columns COLUMNS \
  --output-dir QUANTIZED
cargo run -p sllm-tools --bin sllm-artifact -- imatrix \
  --input-json values.json --rows ROWS --columns COLUMNS --seed SEED \
  --output-dir IMATRIX
```

`split` publishes `part-*.gguf`, `manifest.json`, and `run-manifest.json`.
`merge` publishes `model.gguf`, `merge-report.json`, and
`run-manifest.json`.  The LoRA bundle contains `adapter.lock.json`,
`adapter.payload`, `manifest.json`, and `run-manifest.json`.  Repack,
quantize, and imatrix similarly publish their payload/summary plus a common
run manifest.

`sllm-convert-gguf` first verifies the reviewed model lock and its cache (and,
for FP8, the sidecar manifest and artifact).  `--dry-run` performs planning
and identity checks without creating output and does not require
`--converter-commit`.  A real conversion requires a 40-character lower-case
commit through `--converter-commit` and should use the directory transaction:

| `--kind` | Required identity inputs |
| --- | --- |
| `qwen35-bf16` (default) | `--lock` for the reviewed Qwen3.5 lock and `--cache` for its verified cache |
| `qwen35-fp8` | `--lock`, `--cache`, `--manifest`, and `--artifact`; the sidecar is checked against the lock |
| `gemma4-nvfp4` | reviewed Gemma 4 `--lock` and the verified artifact in `--cache` |
| `qwen35moe-mxfp4` | the reviewed Qwen3.5-MoE artifact in `--cache` (the fixed model fingerprint is checked) |

`--output-bundle` cannot be combined with `--output` or `--derived-lock`.
A non-dry run requires `--output-bundle`; the legacy two-file form is rejected
because GGUF and derived lock cannot be published as one atomic transaction.

```text
cargo run -p sllm-cli --bin sllm-convert-gguf -- \
  --kind qwen35-bf16 --lock LOCK.json --cache VERIFIED_CACHE \
  --output-bundle CONVERSION_DIR --converter-commit SHA40
```

The bundle contains `model.gguf`, `model.derived-lock.json`, and
`run-manifest.json`.  There is no partially published two-file compatibility
path.

## Identity and schemas

The common envelope is
[`sllm-phase46-tool-run-v1`](../../ci/schema/phase46-tool-run-v1.schema.json),
with `struct_size: 13` and sorted-JSON canonicalization.  A manifest binds:

- operation, state (`PASS`, `FAIL`, or `INSUFFICIENT-EVIDENCE`) and a non-zero
  `selected_count`;
- repository, exact nonzero 40-hex commit, package/version, executable
  SHA-256, arguments, OS/architecture, and compile-time
  `rustc --version --verbose` identity;
- recipe ID/version and a SHA-256 of its canonical configuration;
- every source, output, and raw-evidence file by logical name, byte size, and
  lower-case SHA-256; and
- typed identity and metric maps.  Additive data belongs in `extensions`;
  adding a required field requires a new schema version.

The common file identity uses a bare 64-hex SHA-256.  Artifact/core manifests
use their own versioned fields and may use the explicit `sha256:<hex>` form;
consumers must not silently normalize one into the other.

Artifact-specific checks are intentionally stronger than a filename check:

- **GGUF split/merge:** split opens a `VerifiedGguf`, rejects zero tensors,
  keeps the header and each tensor wholly within one contiguous part, and
  records source size/digest, metadata digest, tensor-catalog digest, complete
  tensor ranges, part order/ranges, names, sizes, and digests.  Merge rejects
  non-canonical manifests, missing/duplicate/out-of-order/foreign/tampered
  parts, then verifies both byte identity and the GGUF semantic catalog before
  publication.
- **LoRA:** `sllm-lora-source-v1` requires a base-model fingerprint and weight
  plan digest, explicit target shape/rank/dtype, provenance, and A/B
  orientation.  A is normalized to input-by-rank and B to rank-by-output,
  converted to finite little-endian BF16, and bound into a
  `sllm-adapter-lock-v1` and conversion manifest.
- **Repack:** MXFP4 and NVFP4 preserve separate logical and physical digests,
  dimensions, scale-plane byte count, recipe version, and standard-block tail
  policy.  Alignment, truncation, and non-finite scale bytes fail closed.
- **Quantize/imatrix:** only reviewed FP8 channel/F32-scale, NVFP4 block16,
  and MXFP4 block32 recipes are accepted.  Inputs are bounded finite F32
  matrices.  Quantized output records values/scales/tensor-scale digests;
  imatrix uses deterministic row-major F64 sum-of-squares with the caller's
  seed and input sample-order digest.

## Atomic publication and partial artifacts

Directory-producing artifact operations stage every member under a fresh
hidden directory, fsync each member, write and validate identity manifests,
then publish with one directory rename.  Any error, cancellation, duplicate
destination, stale staging name, malformed input, or digest mismatch removes
the staging directory and never exposes a completed-name artifact.  The
common `AtomicBundleV1` follows the same rule and also syncs the parent
directory.  The GGUF converter's `--output-bundle` uses this full transaction;
the evaluator and HIP baseline use an atomic temporary-file rename for their
single JSON result.  Debug dumps are opt-in and use a reserved partial file;
dropping or failing the writer removes it.

`run-manifest.json` is created after the other members and therefore does not
include itself in its output list (self-inclusion would be a digest cycle).
The older split/merge and artifact operation publishers sync files before the
final rename but currently do not issue an explicit parent-directory fsync;
this is a durability caveat, not a reason to treat a partial directory as a
valid result.

## Benchmark and quality contracts

`sllm-bench aggregate` consumes the versioned input schema and rejects empty
warmups/measured sets, duplicate iterations, missing identities, non-positive
timings, and failed samples without a reason.  It reports wall and GPU timing
separately (min/p10/median/p90/max/MAD), retains rejected samples and reasons,
and records model-load, E2E, TTFT, TPOT, prefill, decode, request/parallelism/
context/sampling, provider, GPU identity, KV encoding, fallback, and cleanup.
HBM/GTT, model-resident, KV logical/physical, and workspace measurements are
`measured`, `unsupported`, or `missing`; zero is not a substitute for an
unavailable sampler.  A fallback, cleanup failure, rejected measurement, or
zero accepted sample cannot become a performance `PASS`.  The aggregate is
published as [`phase46-benchmark-result-v1`](../../ci/schema/phase46-benchmark-result-v1.schema.json)
and retains a digest-bound copy of its bounded input as raw evidence.

`sllm-eval` requires a validated common run manifest (`--manifest`) and accepts
exactly one section in the [`phase46-quality-result-v1`](../../ci/schema/phase46-quality-result-v1.schema.json)
input envelope:

- perplexity from a finite loss sum/token count or finite loss list;
- baseline/candidate logit comparison with KLD, top-1 agreement, max and
  quantile absolute differences, and first divergence position;
- task exact-match and/or multiple-choice accuracy with task version, renderer,
  and few-shot metadata; or
- long-context coverage with early/middle/tail positions, K/V planes, layers,
  heads, and block-tail samples.

Empty, duplicate, non-finite, over-limit, or mismatched samples fail closed.
No evaluator fetches a model, prompt, tokenizer, leaderboard, or network
resource.  `--output` uses an atomic single-file publication; without it the
canonical result is printed to stdout.  Evaluator errors and unknown options
exit 2.

The bounded debug writer is disabled by default.  Enabling it requires an
identity-bound tool manifest and enforces hard limits (16 MiB, 128 tensors,
4096 tokens, 64 top-k, 256 layers, 4096 positions).  Its closed metadata
allow-list rejects prompt/response text, credentials, API keys, payloads,
pointers, and device addresses.  Tensor dtype, shape, layout, endianness,
quantization, and scale plane are explicit; packed KV cannot be mislabeled as
FP16.

## Exact-HIP Qwen3.5 baseline

The baseline binary is not a generic evaluator and has no CPU fallback.  It
accepts only the reviewed Qwen3.5-4B lock, verified cache, the checked-in
[`phase46-kv-quality-baseline-v1` fixture](../../ci/fixtures/phase46-kv-quality-baseline-v1.json),
an exact `gfx*` target, device index, 3–16 repeats, and a new output path.
The fixture is project-authored CC0 token-ID data (seed 1729), covers lengths
1, 15, 16, 17, 255, 256, 257, 511, 512, and 513, and explicitly covers K/V,
early/middle/tail, selected layers/heads, and block tails.

Each case compares a baseline request with the first measured request for both
the final prefill row and a one-token decode continuation, then repeats the
same prefill/decode transition.  The report records perplexity/top-1, KLD, maximum logit delta,
first divergence, exact-HIP dispatch audit, executable and dataset digests,
model lock identity, and allocation cleanup.  Any non-HIP dispatch, fallback,
zero/non-finite metric, failed cleanup, wrong lock/cache/fixture, or output
publication error fails.  Runtime/argument errors exit 1; report
serialization/publication errors exit 2.  A reproducible invocation is:

```text
cargo run -p sllm-hip --bin sllm-qwen35-quality-baseline -- \
  LOCK.json VERIFIED_CACHE ci/fixtures/phase46-kv-quality-baseline-v1.json \
  DEVICE_INDEX GFX_TARGET REPEATS BASELINE.json
```

The decode continuation consumes the committed KV state at every listed
boundary, including block tails and the allocated-capacity edge. This baseline
is evidence for a later KV policy; it does not implement or
adopt a new KV encoding, target selector, or default.

The Phase 46 closeout froze one exact `gfx1030` FP16 baseline.  Its external
report digest and executable digest are bound in `kv-cache-default-v1`; the
`gfx1201` and `gfx942` baseline entries remain `required-before-candidate` and
cannot inherit the `gfx1030` result.  The boundary fixture's task score is
synthetic next-token correctness, so it is a determinism/coverage signal and
not a standalone semantic benchmark.

## Evidence and repository hygiene

Tracked content is limited to source, schemas, small deterministic fixtures,
policy/summary JSON, and reproduction instructions.  Model weights and
tokenizer caches, converted GGUF/adapter payloads, raw prompts/responses,
unbounded logits, profiler traces, memory dumps, and full benchmark sample
streams remain outside Git (normally under the local artifact area).  A report
must carry their digest and identity rather than embedding the bytes.  See the
[repository hygiene policy](repository-hygiene.md) before selecting an output
directory.

## Verification commands

The focused host contract checks are:

```text
cargo test -p sllm-tools
cargo test -p sllm-tools --test artifact_contract
python3 -m unittest ci.tests.test_phase46_ci_contract
```

For schema/markdown changes, also run the repository's JSON-manifest and
Markdown-link validators described in [host testing](testing.md).  GPU quality
evidence requires the HIP target and verified model/cache above; CPU tests and
an empty, timed-out, crashed, or fallback run are not GPU proof.
