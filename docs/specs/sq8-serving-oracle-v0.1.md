# SQ8 Serving Oracle v0.1

Status: P8-A input contract and trust-boundary specification; real vLLM
numerical oracle capture is pending

## 1. Scope

This specification fixes the independent reference inputs, identities, numerical
comparison rules, and validation boundary for the SQ8 serving path. The checked-in
fixture set is `tests/fixtures/sq8-serving-v0.1/`.

The checked-in v0.1 fixture set contains deterministic raw-token inputs and
explicit placeholders for numerical vLLM outputs. It MUST NOT be presented as a
completed oracle, and it is not promotion eligible. Synthetic hidden states,
logits, generated token IDs, hashes, or success fields are forbidden. A later
capture from the fixed real vLLM environment must replace the placeholders under
a separately anchored real-oracle schema before P8-B can pass its vLLM gates.

This specification does not modify the P7 result schemas or their validators.

## 2. Frozen identities

### 2.1 Source model and package

| Property | Fixed value |
| --- | --- |
| model | `Qwen/Qwen3-14B-FP8` |
| revision | `9a283b4a5efbc09ce247e0ae5b02b744739e525a` |
| artifact content SHA-256 | `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147` |
| package manifest SHA-256 | `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb` |
| context length | 4096 |
| hidden size | 5120 |
| vocabulary/logit size | 151936 |
| EOS token IDs | 151645, 151643 |

The exact checkpoint filenames, byte counts, and SHA-256 values are part of
`manifest.json`. Both the exporter and the independent validator contain the
same frozen list. A revision name without the per-file hashes is insufficient.

### 2.2 Tokenizer

| Property | Fixed value |
| --- | --- |
| class | `Qwen2Tokenizer` |
| revision | source-model revision above |
| `tokenizer.json` SHA-256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` SHA-256 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `vocab.json` SHA-256 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` SHA-256 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `generation_config.json` SHA-256 | `231c22c0b89ffbbb785d0e68b2f3f922244f263487af79f6542fc82dbee37dbf` |
| chat-template UTF-8 bytes | 4168 |
| chat-template SHA-256 | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |

The exact-length chat-message cases are root-owned work. The checked-in
`chat-template.pending.json` contains no fabricated text or token IDs and fixes
only the required lengths 32, 128, 512, 2048, and 3584, with
`add_generation_prompt=true` and `enable_thinking=false`.

### 2.3 vLLM capture environment

| Property | Fixed value |
| --- | --- |
| runner | `LLM.generate` |
| vLLM package | `0.23.1rc1.dev618+g8cf7c4d8a.rocm723` |
| source revision exposed by package | `8cf7c4d8a` |
| Python | `3.12.3` |
| PyTorch | `2.11.0+gitd0c8b1f` |
| PyTorch git revision | `d0c8b1f364ecacff4dd8bc06a645d0fb9324cd37` |
| HIP | `7.2.53211` |
| Transformers | `5.12.1` |
| model dtype | `bfloat16` |
| device | visible index 0, `AMD Radeon Graphics`, `gfx1201` |
| device memory | 34208743424 bytes |

The environment uses tensor parallel 1, pipeline parallel 1, `max_num_seqs=1`,
eager execution, no prefix caching, no asynchronous scheduling, and
`ROCR_VISIBLE_DEVICES=1`. A real export MUST bind every one of these values and a
full exporter source commit. The short revision embedded in the package version
does not substitute for the exporter source commit.

## 3. Deterministic raw prompts

Raw prompt lengths are exactly 1, 8, 32, 128, 512, and 4095. For length `N`:

- token IDs are the unsigned integers `[1, 2, ..., N]`;
- the payload is contiguous little-endian `u32`, with exactly `4*N` bytes;
- positions are `[0, 1, ..., N-1]`;
- attention is causal; and
- the validator recomputes the bytes, token values, and SHA-256 instead of
  trusting manifest metadata.

Generation cases are fixed as follows:

| Case | Maximum new tokens | EOS handling | Scope |
| --- | ---: | --- | --- |
| `greedy-g1` | 1 | stop on EOS | product |
| `greedy-g8` | 8 | stop on EOS | product |
| `greedy-g64` | 64 | stop on EOS | product |
| `greedy-g512-ignore-eos-boundary` | 512 | ignore EOS | test only |

Sampling is greedy with temperature zero. A case is attached to a prompt only
when `prompt_tokens + max_new_tokens <= 4096`; therefore the 4095-token prompt has
only `greedy-g1`. Normal cases record the actual token count and `eos` or `length`
finish reason. The boundary case must emit exactly 512 tokens.

## 4. Required real oracle payload

Schema `ullm.sq8.serving_oracle.v1` is reserved for a completed real capture. For
each raw prompt, a completed record MUST contain:

- the complete source-model, tokenizer, vLLM, device, dtype, position, and causal
  attention identities fixed above;
- the raw prompt file, count, byte count, and recomputed payload SHA-256;
- a full 40-hex exporter source commit and an explicit clean/dirty source state;
- final-prompt hidden state as little-endian F32, shape `[5120]`, exactly 20480
  bytes, with file SHA-256;
- final-prompt logits as little-endian F32, shape `[151936]`, exactly 607744
  bytes, with file SHA-256;
- one little-endian U32 generated-token payload for every feasible generation
  case, including requested count, actual count, finish reason, byte count, and
  SHA-256; and
- a metadata SHA-256 anchor and a payload-manifest SHA-256 anchor outside the
  producer-controlled success fields.

Every floating-point value must be finite. Tensor shape, dtype, endianness, byte
count, and hash are checked before comparison. Greedy generated token sequences
must match exactly through their recorded terminal boundary. A `passed` value
written by an exporter is ignored and SHOULD be rejected as an unknown field.

The current `ullm.sq8.serving_oracle_placeholder.v1` files deliberately set all
output filenames, byte counts, payload hashes, token counts, and metadata anchors
to `null`. Their status is `pending_real_vllm_export` and their trust block says
that a real export is required. They are scheduling records, not evidence.

## 5. Comparison contract

For finite F32 vectors `actual` and `reference`:

```text
relative_l2 = ||actual - reference||_2 / max(||reference||_2, 1e-30)
cosine_similarity = dot(actual, reference) / (||actual||_2 * ||reference||_2)
top_10_overlap = |set(actual_top_10_token_ids) intersect
                   set(reference_top_10_token_ids)|
```

A zero or non-finite cosine denominator fails validation. Logit ranking is by
descending value, with ascending token ID as the deterministic tie-breaker.

| Gate | Non-finite | Max relative L2 | Min cosine | Top-1 | Min top-10 overlap |
| --- | ---: | ---: | ---: | --- | ---: |
| SQ8 M=1 against vLLM source model | 0 | 0.20 | 0.98 | exact | 3 |
| SQ8 M=8 chunk against verified M=1 | 0 | 0.10 | 0.995 | exact | 5 |

The first row is the source-model correctness gate. The second row is the uLLM
path-equivalence gate used before enabling fixed single-request M=8 prompt
chunks. Metrics are recomputed from payloads by a consumer-side validator. An
exporter summary or a precomputed `passed` bit is never authoritative.

## 6. Fixture layout and trust boundary

The deterministic bootstrap exporter writes a new directory and refuses to
overwrite any existing path, including a dangling symlink. It stages the complete
tree beside the destination and publishes with Linux `renameat2(RENAME_NOREPLACE)`.

```text
tests/fixtures/sq8-serving-v0.1/
  manifest.json
  SHA256SUMS
  chat-template.pending.json
  raw/prompt-NNNN.u32le
  oracles/raw-pNNNN.pending.json
  openwebui/capture.json
  openwebui/stream-request.json
  openwebui/nonstream-request.json
```

`manifest.json` hashes every artifact except itself and `SHA256SUMS`.
`SHA256SUMS` hashes every artifact plus `manifest.json`. The independent validator
contains the trusted `manifest.json` SHA-256 in source and, in promotion mode,
checks that anchor before parsing producer-controlled metadata. It then checks
strict JSON types and exact keys, rejects duplicate keys and non-finite JSON
numbers, rejects symlinks/path traversal/extra files, recomputes all hashes and
raw token values, and verifies every placeholder is still empty.

`--contract-only` skips only the compiled manifest anchor. It retains all other
checks and never makes a fixture promotion eligible. The validator does not
import the exporter.

The checked-in fixture-set status is
`input_contract_ready_oracles_pending`, and `promotion_eligible` is `false`.
The trusted bootstrap manifest SHA-256 is
`eea3e6b48583b429b0f36bd82756db0d9967474c8d2af1d7143de274e18bc313`.

## 7. OpenWebUI interoperability capture

The sanitized request fixtures were captured through the actual
`/api/chat/completions` proxy of this fixed image:

| Property | Fixed value |
| --- | --- |
| OpenWebUI | `v0.9.4` |
| source revision | `f51d2b026f1b0e7283b15f093412be8b67d24770` |
| image digest | `sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff` |

The fixture records one stream and one non-stream forwarded request body. The
capture observed that OpenWebUI removed `metadata` and forwarded
`max_completion_tokens` as `max_tokens`. Authorization, cookies, and secrets are
absent. The capture proves these sanitized request shapes and transformations;
it contains no response payload and is not a numerical SQ8 oracle.

## 8. OOM and capture procedure

The bootstrap exporter imports neither vLLM nor the model and writes prompts one
token at a time. The validator hashes and validates payloads in bounded chunks.

The later real exporter must load one model instance, run only one prompt/case at
a time, write each CPU tensor payload immediately, hash it in bounded chunks,
release per-case tensors, and synchronize before advancing. It must not retain a
matrix of full logits, submit concurrent model requests, or batch the six fixture
prompts. A failed or interrupted export is staged outside the trusted fixture
path and is never published as partial evidence.

## 9. Promotion conditions

P8-B may consume the vLLM reference only after all of the following are true:

1. Every placeholder has been replaced by a real capture under the completed
   schema.
2. The independent validator has a reviewed external metadata/manifest anchor.
3. It recomputes tensor hashes, finiteness, shapes, metrics, top-token gates, and
   greedy sequences without exporter verdicts.
4. Model, tokenizer, vLLM, device, dtype, exporter commit, positions, and causal
   attention identities match this specification exactly.
5. The chat-template exact-length fixture has separately passed its root-owned
   validator.

Until then, the deterministic fixture directory is useful for implementation and
validator tests, but it cannot satisfy the P8-A real-oracle acceptance item.
