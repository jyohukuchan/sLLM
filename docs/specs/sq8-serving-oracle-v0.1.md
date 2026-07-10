# SQ8 Serving Oracle v0.1

Status: P8-A completed real-vLLM capture and trust-boundary specification

## 1. Scope

This specification fixes the independent reference inputs, identities, numerical
comparison rules, and validation boundary for the SQ8 serving path. The checked-in
fixture set is `tests/fixtures/sq8-serving-v0.1/`.

The completed v0.1 fixture set contains deterministic raw-token inputs and a real
vLLM capture under the separately anchored completed-oracle schema. The original
pending input tree remains byte-exact under `provenance/` but is not an active
oracle. Synthetic hidden states, logits, generated token IDs, hashes, or success
fields are forbidden.

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

The checked-in `chat-template/` fixture contains real tokenizer output for exact
lengths 32, 128, 512, 2048, and 3584 plus English, Japanese, system/user,
two-turn, and code-block cases. Its manifest SHA-256 is
`6324b74e2604b86d46bf2dfdc259c1ca68d8cc9a47e90bfb765919f4aa9d54e0`.
`tools/validate-sq8-chat-template-fixtures.py` independently reloads the frozen
local tokenizer with `local_files_only=true`, `trust_remote_code=false`,
`add_generation_prompt=true`, and `enable_thinking=false`, then recomputes the
rendered text, token IDs, lengths, and hashes for every case.

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

### 2.4 Completed capture identity

The promotion validator accepted the 21-run real capture. Its immutable
read-only product copy is:

```text
/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/oracles/vllm-source-v0.1
```

| Property | Fixed value |
| --- | --- |
| metadata SHA-256 | `1710ebf504c3cf84616f265f57575d48b91804635a0c0151875eadc91fbc122b` |
| payload-manifest SHA-256 | `5972a024c91509b432e68ee39a3dd1cf7a0f0ba2ba48fe7ef5c0bfb02957405c` |
| source bootstrap manifest SHA-256 | `c5b502fe54a5f1563eaf48b8308d7f1d479d11afcbf4cb4a7567bb31b65b61af` |
| prompt count | 6 |
| generation run count | 21 |

The metadata and payload-manifest hashes are fixed in the independent serving
fixture validator, outside the producer-controlled oracle tree.

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
only `greedy-g1`. Normal cases record the actual token count and `stop` or `length`
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

The preserved `ullm.sq8.serving_oracle_placeholder.v1` files deliberately set all
output filenames, byte counts, payload hashes, token counts, and metadata anchors
to `null`. Their status is `pending_real_vllm_export` and their trust block says
that a real export is required. They are historical source scheduling records
under `provenance/bootstrap-input-v0.1/`, not active evidence.

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

The deterministic exporter refuses to overwrite any existing path, including a
dangling symlink. It stages the complete tree beside the destination and
publishes with Linux `renameat2(RENAME_NOREPLACE)`. Symlinked output parents and
source/destination overlap are rejected. Its default mode continues to emit the
byte-exact pending bootstrap tree. Completed mode requires both the trusted
bootstrap tree and trusted real-oracle tree as explicit inputs. The staged tree
is independently validated before publication, so a changing source cannot
publish a mixed snapshot.

```text
tests/fixtures/sq8-serving-v0.1/
  manifest.json
  SHA256SUMS
  chat-template/manifest.json
  chat-template/fixtures/*.json
  raw/prompt-NNNN.u32le
  openwebui/capture.json
  openwebui/stream-request.json
  openwebui/nonstream-request.json
  provenance/bootstrap-input-v0.1/
    manifest.json
    SHA256SUMS
    chat-template/...
    raw/...
    oracles/raw-pNNNN.pending.json
    openwebui/...
  oracles/vllm-source-v0.1/
    metadata.json
    payload-manifest.json
    SHA256SUMS
    captured-exporter.py
    input-fixture-manifest.json
    inputs/raw-pNNNN.u32le
    prompts/raw-pNNNN/final-hidden.f32le
    prompts/raw-pNNNN/prefill-logits.f32le
    prompts/raw-pNNNN/greedy-*.u32le
```

`manifest.json` hashes every artifact except itself and `SHA256SUMS`.
The root `SHA256SUMS` hashes every artifact plus `manifest.json`, including both
nested `SHA256SUMS` files. Each nested sums file excludes only itself. This avoids
a circular hash while retaining both nested integrity chains.

The independent validator contains separate trusted hashes for the pending
bootstrap manifest, completed root manifest, real-oracle metadata, and real
payload manifest. In promotion mode it checks those anchors before parsing
producer-controlled metadata. It then checks strict JSON types and exact keys,
rejects duplicate keys, non-finite JSON numbers, symlinks, path traversal, and
extra files, and recomputes all hashes, raw tokens, tensor finiteness, top tokens,
and greedy sequences. It rejects producer `passed` fields.

`--contract-only` skips only the completed root-manifest anchor. It still performs
trusted nested-bootstrap validation and fixed-anchor real-oracle numerical
validation, but never makes the root fixture promotion eligible. The validator
does not import the exporter.

The completed fixture-set status is
`input_contract_ready_real_oracles_complete`, and its manifest declares
`promotion_eligible=true`. Effective promotion additionally requires the
completed root manifest to match the reviewed hash fixed in validator source.
There are no active `.pending.json` files; pending records exist only below the
provenance subtree.

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

The real exporter loaded one model instance, ran only one prompt/case at a time,
wrote each CPU tensor payload immediately, hashed it in bounded chunks, released
per-case tensors, and synchronized before advancing. It did not retain a matrix
of full logits, submit concurrent model requests, or batch the six fixture
prompts. The five feasible 512-token boundary runs used `ignore_eos=true` and
completed at exactly 512 tokens. A failed or interrupted export is staged outside
the trusted fixture path and is never published as partial evidence.

## 9. Promotion conditions

P8-B may consume the checked-in vLLM reference only while all of the following
remain true:

1. Every active placeholder has been replaced by the 21-run real capture under
   the completed schema; historical placeholders occur only under provenance.
2. The independent validator has a reviewed external metadata/manifest anchor.
3. It recomputes tensor hashes, finiteness, shapes, metrics, top-token gates, and
   greedy sequences without exporter verdicts.
4. Model, tokenizer, vLLM, device, dtype, exporter commit, positions, and causal
   attention identities match this specification exactly.
5. The chat-template exact-length fixture has separately passed its root-owned
   validator.

An unanchored completed root may pass `--contract-only` for review, but it cannot
satisfy the P8-A real-oracle acceptance item or be consumed by P8-B.
