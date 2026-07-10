# OpenWebUI Single-Worker Product Plan v0.1

Status: in progress; P8-B2 is complete with selected M=128 correctness, deep-boundary, and formal performance gates passed; P8-C is next

Date: 2026-07-10

Baseline commit: `2bf3d16`

Schedule basis: acceptance-driven. There is no fixed completion deadline. The minimum usable product is completed before request batching or optional micro-optimization begins. The fixed long-prompt latency gates are part of product functionality, not optional tuning.

Supersedes: `docs/plans/sq8-recovery-plan-v0.2.md` as the active plan after its P0-P7 completion. That document remains the implementation and evidence history for the SQ8 runtime.

Concurrency definition: v0.1 has one active GPU request and no waiting request. A fixed-size prefill chunk containing tokens from that one request is allowed; it is not request batching.

## 前回の要点

- SQ8 P0-P7は完了し、Qwen3-14B-FP8をR9700上で8 token実生成できる。
- canonical SQ8 generation runtimeは、prompt `[1,2,3,4,5,6,7,8]`、context 16、最大8 generated token、greedyに固定された監査経路である。
- model、embedding、40-layer stack、paged KV、final norm、lm_headは常駐し、完了後のresetと再実行まで検証済みである。
- HTTP server、OpenAI request/response、SSE、常駐tokenizer、全chat履歴のtemplate、実行中cancelは未実装である。
- 旧Python wrapperにはTransformers tokenizer、Qwen3 chat template、decodeがあるが、旧CLIをrequestごとに起動する診断経路であり、製品serverにはそのまま使えない。

## 今回の変更点

- 次の目標を、batchなしのB=1 OpenWebUI対応製品に変更する。
- multi-request batching、continuous batching、prefix cache、request queueを初期製品の必須条件から外す。
- 固定M=8 prefillを任意長へ直接拡張せず、M=1 token stepでpromptを順番にKVへ入れる低リスク経路を先に作る。
- M=1経路を独立oracleで正しいと確認した後、既存M=8実行を一request内の固定prefill chunkとして再利用し、4096 contextのhard TTFT gateを満たす。
- 現監査runtimeを壊さず、逐次tokenを返すlean serving sessionを別APIとして追加する。
- Rust常駐workerがGPU/model stateを所有し、Python FastAPI gatewayがtokenizer、chat template、OpenAI互換HTTP、SSEを担当する。
- 自動履歴切捨て、request `stop`文字列、waiting queue、request batchingはv0.1後の拡張とする。

## 次の行動

最初の実装単位はP8-AとP8-Bである。HTTP依存を追加する前に、可変raw token prompt、4096 context、逐次`next_token`をR9700で成立させ、新M=1経路をP7とvLLMの独立oracleへ照合する。cancelとreset完了protocolはP8-Cで固定する。

## 1. Objective

WRX80上のuLLMをOpenWebUIから通常のtext chat modelとして利用できる状態にする。

「通常利用可能」は次を満たすことを指す。

1. service起動時にQwen3-14B-FP8 SQ8 modelをR9700へ一度だけloadする。
2. OpenWebUIのmodel selectorに`ullm-qwen3-14b-sq8`が表示される。
3. system、user、assistant履歴をQwen3 chat templateでtokenizeできる。
4. 生成tokenがSSEで逐次表示される。
5. Stop操作またはclient disconnectで生成を中断し、次requestを処理できる。
6. 日本語、英語、code block、複数turnを処理できる。
7. 1 active requestだけを実行し、同時requestは即時`429`にする。
8. 失敗、context超過、同時requestをOpenAI互換errorとして返し、GPU stateを壊さない。
9. 連続利用後もKV allocationとVRAMが規定状態へ戻る。

OpenWebUIの初期接続はOpenAI Chat Completions protocolを使う。公式仕様上、`POST /v1/chat/completions`が必須で、`GET /v1/models`はmodel自動検出に推奨される。

Reference:

- https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/

## 2. Fixed Product Target

### 2.1 Model and device

- model: `Qwen/Qwen3-14B-FP8`
- model revision: `9a283b4a5efbc09ce247e0ae5b02b744739e525a`
- canonical artifact SHA-256: `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`
- thin package manifest SHA-256: `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`
- GPU: Radeon AI PRO R9700 / `gfx1201`
- execution profile: `rdna4_w8a8_block_ck`
- tensor parallel: 1
- batch size: 1
- concurrent GPU execution: 1

Model、artifact、package、GPU、profile、required HIP guardのいずれかが一致しない場合、workerはReadyにならない。

### 2.2 Request limits

- context length: 4096 tokens
- default maximum completion: 256 tokens
- hard maximum completion: 512 tokens
- minimum prompt length: 1 token
- active GPU requests: 1
- waiting queue capacity: 0
- request while active: immediate `429` with `Retry-After: 1`
- request body limit: 2 MiB
- worker command line limit: 4 MiB
- `n`: 1 only
- `max_tokens` / `max_completion_tokens`: integer 1 through 512

`prompt_tokens + max_completion_tokens <= 4096`をGPU mutation前に検査する。

### 2.3 Chat behavior

- text-only Chat Completions
- roles: `system`, `user`, `assistant`
- message content: stringまたはtext content part
- chat template: local Qwen3 tokenizer config
- `add_generation_prompt=true`
- `enable_thinking=false` by default
- conversation state: stateless at server API boundary
- request間KV reuse: none
- history source: OpenWebUIが毎requestで送る`messages`
- message order: optional first `system`、その後は`user`と`assistant`が交互、最後は`user`
- context overflow policy: v0.1 does not truncate; return `400 context_length_exceeded`

OpenWebUI側が送る全履歴を毎回templateへ適用する。自動履歴切捨ては、利用実績を確認した後にcomplete turn単位の別仕様として追加する。

### 2.4 Sampling

Supported:

- greedy when `temperature=0`
- temperature sampling
- `top_p`
- `top_k` as a fixed uLLM model default/config option, not an OpenAI request field
- `seed`
- `max_tokens`
- `max_completion_tokens`
- stop token IDs from model generation config

Initial Qwen3 defaults:

- temperature: `0.6`
- top_p: `0.95`
- top_k: `20`
- EOS IDs: `151645`, `151643`

Validation and default rules:

- `temperature`: 0 through 2 inclusive;
- `top_p`: greater than 0 and at most 1;
- `seed`: signed 64-bit integer;
- when `seed` is omitted, the gateway selects and records an OS-random seed for the request;
- specifying both maximum-token fields is rejected, even when their values match;
- `stream` defaults to `false`;
- `stop: null` is accepted, while non-empty `stop` is rejected in v0.1;
- unknown non-null request fields are rejected unless the API subset spec explicitly marks them as accepted metadata.

Accepted only at neutral/default value:

- `frequency_penalty=0`
- `presence_penalty=0`

Rejected with `400 unsupported_parameter`:

- `n != 1`
- non-empty `logit_bias`
- `tools` / `tool_choice`
- non-empty request `stop`
- image, audio, or file content parts
- JSON schema/structured output guarantees
- nonzero frequency/presence penalty

Unsupported fields are not silently treated as implemented.

## 3. Non-Goals for v0.1

- request batch size greater than 1
- continuous batching
- parallel request execution
- waiting request queue
- tensor parallel
- prefix cache or request-to-request KV reuse
- speculative decode
- adaptive or autotuned prefill chunking beyond the fixed single-request v0.1 route
- full 40960-token context
- GPU sampling kernel
- fused sampling/lm_head kernel
- vLLM performance parity
- tools/function calling
- multimodal input
- embeddings and RAG endpoint
- OpenAI Responses API
- audio/image endpoints
- multi-tenant authorization
- TLS termination
- model hot reload
- multiple loaded models
- automatic chat-history truncation
- request stop-string matching

These items cannot delay the first OpenWebUI release unless they expose a correctness or resource-safety defect in the fixed target.

## 4. Current Baseline and Gaps

Reusable:

- source-correct canonical SQ8 artifact and thin package;
- resident embedding, 40-layer CK stack, paged KV, final norm, lm_head;
- M=1 decode and token feedback;
- EOS and max-token completion reason;
- scheduler allocation/release;
- completed-request reset;
- fail-closed Poisoned state;
- independent P6/P7 result validators;
- Python `AutoTokenizer`, `apply_chat_template`, encode/decode reference code.

Missing:

- arbitrary prompt token IDs in canonical runtime;
- more than one 16-token KV block;
- 4096-token context allocation contract;
- prompt M=1 token-step path from position 0;
- one-request M=8 prompt chunks that attend both prior KV and the current causal chunk;
- per-token step/callback interface;
- cancel transition and safe reset before normal completion;
- sampling other than greedy;
- lean result that does not retain every hidden/logit tensor;
- persistent worker command protocol;
- OpenAI HTTP types and endpoints;
- incremental UTF-8 detokenization;
- atomic one-request HTTP admission guard;
- readiness, health, auth, structured logs, service unit;
- real OpenWebUI end-to-end evidence.

## 5. Architecture

### 5.1 Rust serving runtime

New library module:

- `crates/ullm-engine/src/sq8_serving_runtime.rs`

Required existing-module changes:

- `crates/ullm-engine/src/sq8_stack_runtime.rs`: admit M=1 initial prefill and fixed-size cached-prefix prefill chunks;
- `crates/ullm-engine/src/sq8_layer_runtime.rs`: add the cached-prefix paged-attention mode required by later M=8 chunks;
- `crates/ullm-engine/src/decoder.rs`: pass resident K/V cache into the existing cached-prefix attention boundary without host staging;
- `crates/ullm-engine/src/sq8_model_head_runtime.rs`: expose the lean current-position logits boundary used by serving;
- `crates/ullm-engine/src/sq8_generation_runtime.rs`: extract a resident model core shared by audited and serving paths without changing the P7 result schema;
- `crates/ullm-engine/src/qwen3_loader.rs`: move the example-private verified loader for all 40 layer norms and the final norm into the library;
- `crates/ullm-engine/src/lib.rs`: export the serving boundary;
- `crates/ullm-engine/Cargo.toml`: register the worker binary and require the existing `rocm-ck-gfx1201` feature.

Responsibilities:

- own one resident model core shared with `Qwen3Sq8GenerationRuntime`;
- validate variable raw-token requests;
- allocate/configure 4096-token paged KV at startup;
- execute prompt tokens sequentially with M=1;
- after the M=1 oracle gate, execute full prompt chunks with the existing M=8 stack and use M=1 for the remainder;
- produce one token step at a time;
- sample locally without transferring full logits across a process boundary;
- handle EOS, max-token, cancel, finish, and reset;
- expose typed status and metrics;
- preserve existing audited P7 runtime unchanged.

Proposed public boundary:

```rust
pub struct Sq8ServingRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<usize>,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<usize>,
    pub sampling: Sq8SamplingParams,
}

#[derive(Clone)]
pub struct Sq8CancellationToken { /* Arc<AtomicBool> */ }

pub enum Sq8ServingStep {
    Token { token_id: usize, generated_index: usize },
    Finished { reason: Sq8ServingFinishReason },
}

impl Qwen3Sq8ServingSession {
    pub fn start(&mut self, request: Sq8ServingRequest, cancel: Sq8CancellationToken,
        stream: &mut RuntimeStream)
        -> Result<(), Sq8ServingError>;
    pub fn next_token(&mut self, stream: &mut RuntimeStream)
        -> Result<Sq8ServingStep, Sq8ServingError>;
    pub fn abort_and_reset(&mut self, stream: &mut RuntimeStream)
        -> Result<(), Sq8ServingError>;
    pub fn reset(&mut self, stream: &mut RuntimeStream)
        -> Result<(), Sq8ServingError>;
}
```

The exact names may change, but start, step, terminal cleanup, and reset must remain separate operations. `RuntimeContext`, model buffers, and `RuntimeStream` are constructed and remain on the inference thread because the runtime owners are not `Send`. The cloned cancellation token is the only cross-thread control object.

### 5.2 Rust resident worker

New binary:

- `crates/ullm-engine/src/bin/ullm-sq8-worker.rs`

Responsibilities:

- load model once;
- write one `ready` event after identity validation;
- read versioned JSON Lines commands from stdin on a dedicated reader thread;
- construct and execute all GPU/model state on one inference thread;
- let the stdin reader set the matching active request's atomic cancellation token directly, without waiting behind generation work;
- emit internal `progress` events during prefill so the gateway can detect a hung GPU call before the first token;
- emit one token event per generated token;
- emit a terminal `released` event only after scheduler release and KV reset make the worker reusable;
- reserve stdout exclusively for protocol events;
- send structured operational logs to stderr;
- exit nonzero after an unrecoverable HIP/runtime poison.

The worker accepts only one active `generate`; it has no waiting queue. A second `generate` is a protocol violation. The worker does not implement public HTTP or tokenizer behavior.

### 5.3 Python OpenAI gateway

New package:

- `services/openai-gateway/pyproject.toml`
- `services/openai-gateway/src/ullm_openai_gateway/app.py`
- `services/openai-gateway/src/ullm_openai_gateway/tokenizer.py`
- `services/openai-gateway/src/ullm_openai_gateway/worker.py`
- `services/openai-gateway/src/ullm_openai_gateway/schemas.py`

Dependencies are pinned in a dedicated virtual environment. The vLLM development environment is not the product runtime environment.

Responsibilities:

- load local tokenizer with `local_files_only=true`;
- use `trust_remote_code=false` unless a frozen audit proves it is required;
- validate OpenAI request subset;
- apply the full Qwen3 chat template;
- enforce the fixed context rejection policy;
- admit at most one request through an atomic active-slot guard;
- communicate token IDs with the Rust worker;
- incrementally decode generated token IDs;
- provide stream and non-stream Chat Completions;
- detect disconnect and issue cancel;
- map worker failures to OpenAI-shaped errors;
- enforce the configured Bearer API key.

Python does not load model weights or allocate GPU tensors.

The gateway runs as exactly one Uvicorn process with `--workers 1` and reload disabled. It holds a per-GPU singleton lock before spawning the worker. Unexpected worker EOF or a fatal worker error terminates the gateway nonzero; systemd restarts the complete control group. Partially streamed requests are never retried.

The gateway also owns hard watchdogs: 600 seconds to worker Ready, 180 seconds for a complete request, 30 seconds without worker protocol progress, and 5 seconds from cancellation request to terminal release. A watchdog failure marks readiness false, attempts a nonblocking SSE error flush for at most 250 ms, closes the active transport, sends worker termination with a 2-second grace before kill, and exits the gateway nonzero within 5 seconds. Client backpressure cannot delay this sequence.

### 5.4 OpenWebUI

Fixed release topology:

- systemd system service runs as `homelab1`;
- OpenWebUI attaches to the existing Docker network `open-webui-network` (`172.20.0.0/16`, host gateway `172.20.0.1` as observed on 2026-07-10);
- gateway listens only on `172.20.0.1:8000`, not on the LAN interface;
- OpenWebUI Base URL is `http://172.20.0.1:8000/v1`;
- model ID is `ullm-qwen3-14b-sq8`;
- a Bearer API key is mandatory.

P8-F revalidates the network ID, subnet, gateway, and route before enabling the service. A topology change requires updating the environment and firewall evidence; it is not handled by widening the bind to `0.0.0.0`.

During the single-worker phase, disable OpenWebUI title, follow-up, and tag generation, or route those tasks to another model. These features can create background completion requests that collide with the one active slot.

## 6. State and Protocol Contracts

### 6.1 Serving session state

```text
Loading -> Ready
Ready -> Prefilling -> Decoding -> Finishing -> Resetting -> Ready
Ready -> Prefilling/Decoding -> Cancelling -> Resetting -> Ready
Loading/Prefilling/Decoding/Resetting -> Failed
```

Rules:

- request validation errors occur before `Ready -> Prefilling`;
- only one request may own KV state;
- cancel is checked between prompt/decode token steps;
- a token is emitted only after the corresponding step is complete and finite;
- no token is emitted after the inference thread observes cancellation;
- normal completion and cancel both release scheduler state and reset KV lengths;
- the terminal `released` event is emitted only after release/reset is complete;
- the gateway cannot clear its active slot or send another `generate` before `released`;
- a poisoned HIP/runtime state transitions to `Failed`, not `Ready`;
- `Failed` requires worker process restart;
- `/readyz` describes model lifecycle health, not an idle request slot: it returns `200` during healthy Prefilling/Decoding and `503` during Loading, Restarting, Failed, or Shutdown.

### 6.2 Worker protocol v0.1

Commands:

```json
{"schema_version":"ullm.worker.v1","type":"generate","request_id":"req-1","prompt_token_ids":[1,2,3],"max_new_tokens":256,"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20,"seed":0},"eos_token_ids":[151645,151643]}
{"schema_version":"ullm.worker.v1","type":"cancel","request_id":"req-1","reason":"client_disconnect"}
{"schema_version":"ullm.worker.v1","type":"shutdown"}
```

Events:

```json
{"schema_version":"ullm.worker.v1","type":"ready","model":"ullm-qwen3-14b-sq8","context_length":4096,"max_new_tokens":512}
{"schema_version":"ullm.worker.v1","type":"started","request_id":"req-1","prompt_tokens":42}
{"schema_version":"ullm.worker.v1","type":"progress","request_id":"req-1","phase":"prefill","processed_prompt_tokens":8}
{"schema_version":"ullm.worker.v1","type":"token","request_id":"req-1","index":0,"token_id":123}
{"schema_version":"ullm.worker.v1","type":"released","request_id":"req-1","outcome":"stop","prompt_tokens":42,"completion_tokens":17,"reset_complete":true}
{"schema_version":"ullm.worker.v1","type":"released","request_id":"req-2","outcome":"cancelled","cancel_reason":"client_disconnect","prompt_tokens":42,"completion_tokens":3,"reset_complete":true}
{"schema_version":"ullm.worker.v1","type":"error","request_id":"req-1","code":"runtime_failed","recoverable":false,"message":"..."}
```

Protocol requirements:

- one JSON object per line;
- duplicate keys, NaN, Infinity, unknown schema version, oversized lines, invalid UTF-8 are rejected;
- every output line is flushed immediately;
- stdout contains no human log lines;
- prefill emits `progress` after each eight processed prompt tokens and at the prefill/decode transition; decode token events count as progress;
- request IDs are unique while active;
- `released.outcome` is one of `stop`, `length`, or `cancelled`;
- `released` is the only successful request-terminal event and guarantees that a new request may start;
- a fatal `error` does not claim reset completion and is followed by nonzero worker exit;
- unexpected worker EOF makes gateway unready and fails the active request;
- gateway does not automatically retry a partially streamed request.

No-progress, request, startup, and cancel-release deadline failures are fatal even when the worker has not emitted `error` or EOF.

### 6.3 OpenAI API subset v0.1

Endpoints:

| endpoint | method | behavior |
| --- | --- | --- |
| `/healthz` | GET | process alive; does not require model Ready |
| `/readyz` | GET | model loaded and worker healthy, including while busy; otherwise 503 |
| `/v1/models` | GET | returns one fixed model; never triggers model load |
| `/v1/chat/completions` | POST | stream and non-stream text chat |

`GET /v1/models` minimum response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "ullm-qwen3-14b-sq8",
      "object": "model",
      "owned_by": "ullm"
    }
  ]
}
```

Streaming sequence:

1. assistant role delta with stable `id`, `created`, `model`, `object`, and choice `index=0`;
2. zero or more content deltas with the same envelope identity;
3. exactly one empty-delta chunk with `finish_reason`;
4. when `stream_options.include_usage=true`, one usage chunk with empty `choices`;
5. `data: [DONE]\n\n` exactly once;
6. HTTP stream close.

`stream` defaults to false. SSE responses use `Content-Type: text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`. Both public `/v1` endpoints require the same configured Bearer key. `/healthz` and `/readyz` are unauthenticated and expose no model paths or secrets.

Finish reason mapping:

- model EOS: `stop`
- maximum completion reached: `length`
- client cancellation: stream closes after worker release; no false `stop` completion is recorded
- worker failure: error event/connection close, request marked failed

Failure after headers are sent has a fixed wire contract: emit one `data: {"error":{...}}\n\n` event, close the stream, and emit neither a final choice nor `[DONE]`. Client disconnect emits nothing further, but the gateway continues draining worker stdout until terminal release or process exit. Failures before headers use the normal OpenAI-shaped JSON error response.

Error mapping:

| condition | HTTP | code |
| --- | ---: | --- |
| invalid JSON/schema/messages | 400 | `invalid_request_error` |
| context limit exceeded | 400 | `context_length_exceeded` |
| unsupported non-default parameter | 400 | `unsupported_parameter` |
| invalid/missing configured API key | 401 | `invalid_api_key` |
| unknown model | 404 | `model_not_found` |
| another request is active | 429 | `request_busy` |
| worker loading/failed | 503 | `model_not_ready` |
| unexpected internal failure | 500 | `internal_error` |

### 6.4 Incremental decoding and backpressure

- gateway accumulates generated token IDs and decodes with `skip_special_tokens=true` and cleanup disabled;
- only stable decoded suffixes are emitted;
- partial UTF-8, byte fallback, Japanese, emoji, combining marks, code fences are covered by golden tests;
- stream and non-stream responses must concatenate to identical final text for the same seed.

The worker stdout and stderr pumps never wait on the client-facing stream queue. The client queue holds at most 32 token events. The first nonblocking enqueue failure immediately requests cancellation; the pump discards later token events while continuing to drain both pipes, waits for terminal release, and then closes the HTTP stream without `[DONE]`.

## 7. Phase Plan

### P8-A: Contract Freeze and Fixtures

Tasks:

- add `docs/specs/sq8-serving-session-v0.1.md`;
- add `docs/specs/sq8-worker-protocol-v0.1.md`;
- add `docs/specs/openai-chat-subset-v0.1.md`;
- add `docs/specs/sq8-serving-oracle-v0.1.md`;
- freeze product model/artifact/package/tokenizer hashes;
- stream-copy the exact artifact and thin package from `/tmp` into immutable versioned directories under `/home/homelab1/datapool/ullm/product/`, then verify their frozen hashes before runtime work starts;
- freeze limits, parameter subset, state machine, error codes, and finish reasons;
- export deterministic raw-token and chat-template fixtures;
- export vLLM source-model final hidden/logit oracles for prompt lengths 1/8/32/128/512/4095;
- add an oracle validator that independently checks metadata, payload hashes, tensor shapes/dtypes, finite values, and frozen comparison calculations;
- capture and redact one actual request from the installed OpenWebUI version, then freeze it as an interoperability fixture;
- define exact non-stream response and SSE chunk schemas, null handling, unknown-field policy, auth scope, and post-header error behavior;
- define the `released` event as the sole reusable-worker acknowledgement;
- freeze latency percentile interpolation and the RSS/VRAM robust-slope calculation used by the release validator;
- freeze the local RSS/VRAM/FD/thread/child measurement commands and their parsing schema after probing the installed systemd, KFD, and AMD monitoring tools;
- export chat-message fixtures whose post-template prompt lengths are exactly 32/128/512/2048/3584 tokens;
- add a validator that rejects contract drift.

Fixtures:

- raw prompt lengths: 1, 8, 32, 128, 512, 4095;
- generated lengths: 1, 8, 64, plus one test-only 512-token `ignore_eos` boundary case;
- English user message;
- Japanese user message;
- system plus user;
- two-turn conversation;
- code block;
- EOS-forced fixture;
- context-overflow fixture;
- invalid token ID and malformed request fixtures;
- actual OpenWebUI stream and non-stream request fixtures, with secrets and message text replaced by fixed test text.

Deliverables:

- four specs;
- persistent read-only product artifact and package directories with a checksum manifest;
- fixed fixture directory under `tests/fixtures/sq8-serving-v0.1/`;
- fixture exporter, oracle validator, and contract validator tests.

Acceptance:

- model/tokenizer hashes are deterministic;
- persistent artifact/package copies match the frozen content and manifest hashes and are read-only to the service account;
- Transformers chat template output text and token IDs are frozen;
- each oracle binds model revision, vLLM commit/package version, dtype, backend, token IDs, positions, attention semantics, tensor hashes, and exporter source commit;
- the independent oracle validator passes without trusting an exporter-provided `passed` field;
- all supported and unsupported API fields are enumerated;
- the request schema covers `stream`, both maximum-token field names, `stream_options.include_usage`, `user`, string content, and `type=text` content parts;
- the M=1 serving oracle and the M=8 single-request chunk oracle have frozen comparison tolerances;
- P7 evidence and validators remain unchanged.

Stop condition:

- do not change runtime state until context, sampling, EOS, and API subset are unambiguous.

### P8-B: Variable-Length Lean Serving Session

Tasks:

- separate audited generation result collection from lean serving execution;
- parameterize prompt token IDs and max generated tokens;
- allocate a 4096-token paged KV layout;
- use block size 16 and 256 logical blocks per layer unless measurement proves a smaller safe layout is required;
- implement M=1 prompt ingestion beginning at position 0;
- write each prompt token K/V before attending through its prefix;
- skip lm_head for all prompt tokens except the final prompt token;
- continue generated-token feedback through the existing M=1 decode path;
- add synchronous `start`, `next_token`, `finish/reset`, and `abort_and_reset` session operations;
- discard per-step full hidden/logit evidence after sampling;
- retain finite/logit health checks and required execution identity;
- keep the P7 audited path callable without changed output schema.

Memory note:

- F32 KV at context 4096 is approximately 1.25 GiB for 40 layers, 8 KV heads, key/value width 128;
- allocate with checked arithmetic and record exact bytes;
- allocation failure occurs during worker startup, not after accepting an HTTP request.

Tests:

- prompt 1/8/32/128 with generation 1/8/64;
- cache positions from 0 through prompt and decode boundaries;
- crossing block boundaries at 15/16/17 and 255/256/257 tokens;
- total context exactly 4096;
- total context 4097 rejected before mutation;
- invalid token ID rejected before mutation;
- EOS on first output and during decode;
- maximum token completion;
- reset followed by a different prompt;
- existing P7 fixed oracle regression;
- new M=1 serving path reproduces the P7 `[1..8]` greedy token sequence;
- final prompt hidden/logits for `[1..8]` match the P7 M=8 path within the P8-A frozen tolerances and have exact top-1;
- prompt 1/8/32/128 greedy outputs match frozen vLLM source-model oracle fixtures under the specified token/logit gates.

Acceptance:

- all layer cache lengths equal `prompt_tokens + generated_tokens - 1` at completion;
- token feedback and positions are contiguous;
- no fallback or host weight staging;
- P7 audited result remains identical, and the new serving path independently reproduces its fixed 8-token sequence;
- M=1 prompt 8 final hidden/logits pass direct comparison with P7 M=8;
- prompt 1/8/32/128 pass the independent vLLM oracle gates;
- prompt 128 / generation 64 completes on R9700;
- one context-boundary run reaches 4096 without OOM;
- completion/reset returns allocator and cache lengths to baseline.

Stop condition:

- do not add the worker while the variable-prompt oracle, normal reset, or context-boundary behavior is unstable.

### P8-B2: Fixed Single-Request Prefill Chunks

This phase improves one request at a time. It does not combine requests and does not add a batch scheduler.

Tasks:

- reuse the verified M=8 stack for complete eight-token prompt chunks;
- make each chunk attend the prior paged KV plus its own causal positions;
- use the verified M=1 path for the tail and for decode;
- compute lm_head only for the final prompt position;
- compare chunked and all-M=1 execution before treating timing as valid;
- keep chunk size fixed at 8 for v0.1 unless a correctness or hard performance gate requires another already-verified shape.

Correctness tests:

- prompt lengths 1/7/8/9/15/16/17/32/128/512/4095;
- exact cache position and block-table transitions across every chunk;
- final hidden/logits and greedy tokens agree with all-M=1 under frozen gates;
- prompt 8/32/128/512/4095 agree with frozen vLLM final hidden/logit oracle gates;
- a test-only `ignore_eos` run performs prompt 3584 plus 512 actual generated tokens, reaches total context 4096, and verifies final KV length 4095 and every deep decode position;
- the existing P7 M=8 audit remains unchanged.

Hard runtime performance gate:

- R9700 only, resident model, load excluded, warmups 2, measured repeats 5;
- raw-token prompt lengths 32/128/512/2048/3584 with `max_new_tokens=512`; measure the first generated token, then call synchronous `abort_and_reset` so the 3584 case reserves the exact 4096 boundary;
- record p50 and p95 from request start to the first generated token;
- p50/p95 limits are 2.5/3 seconds at 32, 4/5 seconds at 128, 10/12 seconds at 512, 30/35 seconds at 2048, and 50/60 seconds at 3584;
- prompt 32 with 64 generated tokens records decode p50 at least 15 token/s and p95 inter-token latency at most 100 ms;
- allocator, KV lengths, and VRAM are checked after every sample.

For raw runtime TTFT, time starts immediately before synchronous `session.start` after the previous sample has fully reset and all model state is resident. It ends when the first sampled token ID is available on the host. Fixture construction and model load are excluded; prompt execution, final head, readback, and sampling are included. Every sample performs terminal cleanup before the next sample. p50/p95 use the P7 linear percentile interpolation frozen in P8-A. No other KFD process may use the R9700; GPU temperature, power, and clocks are recorded before and after each workload length.

Acceptance:

- all correctness comparisons pass;
- the 512- and 4095-token independent vLLM oracle comparisons pass;
- the 3584-prompt/512-generation deep boundary run passes without EOS shortening;
- every p50 and p95 limit passes;
- prompt 3584 plus maximum completion reservation 512 proves the advertised 4096-token boundary;
- no request batching, waiting queue, or request-to-request KV reuse is introduced.

If a latency gate fails, v0.1 is not releasable. Further work is limited to measured single-request prefill changes; request batching cannot be introduced to satisfy this phase. Reducing the advertised context requires an explicit new plan version that refreezes every dependent fixture, configuration value, OpenWebUI setting, and release-validator anchor.

Stop condition:

- do not add the worker until the advertised context has both correctness and hard latency evidence.

#### P8-B2 formal gate result and bounded recovery plan (2026-07-10)

##### 前回の要点

The fixed M=8 path passed the full correctness matrix and the exact 3584 prompt plus
512 generated-token boundary. The first clean formal performance run used one
resident load, two warmups, five measured samples, and a complete reset after every
sample.

##### 今回の変更点

- Evidence is frozen under
  `benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/performance-clean-08bdcec/`.
- Prompt 32/128/512/2048 TTFT p50 was
  `0.144360 / 0.602628 / 3.035701 / 23.481711` seconds and passed.
- Prompt 3584 TTFT was p50 `61.023836` seconds and p95 `61.025951`
  seconds. It failed the fixed `50 / 60` second limits; the limits and advertised
  4096-token context remain unchanged.
- Prompt 32 / generation 64 decode passed at p50 `27.779928` token/s and
  p95 inter-token latency `0.036897` seconds.
- The measured prompt-length curve is closely described by
  `T(N) ~= 0.0195 + 0.004036 N + 3.623e-6 N^2` seconds. This is a diagnostic
  inference, not a release contract: the linear term is consistent with repeating
  the M=8 stack, while the quadratic term is consistent with cached-prefix
  attention work.
- CK projection and layer-shape evidence already covers M=16, M=32, and M=128.
  A larger fixed chunk still contains tokens from exactly one request and therefore
  does not change the no-batching product contract.

##### 次の行動

1. Generalize the serving-only cached-prefix chunk plumbing to measured fixed
   widths M=32 and M=128 while retaining M=8 as the frozen oracle. Do not expose
   either candidate to the product configuration yet.
2. Before long timing, require planner/cache-transition tests at 31/32/33,
   127/128/129, and 4095 tokens, plus prompt 32/128/512 final hidden/logit
   comparison against the existing M=8 and source-oracle gates.
3. On the isolated R9700, run a bounded candidate screen at prompt 3584 with one
   resident load, two warmups, and three measured first-token samples per width.
   Select the faster correct width only if it passes both existing 50/60-second
   limits; do not tune thresholds from these samples.
4. For the selected width, repeat the prompt 4095 independent oracle and the
   3584+512 deep-boundary run, then rerun the unchanged formal 2-warmup/5-measured
   TTFT and decode matrix.
5. If neither measured width passes, collect one kernel-summary profile of a
   3584-token request and make only the change justified by the dominant component.
   Request batching, a waiting queue, relaxed latency limits, or a reduced context
   are not fallback options inside plan v0.1.

P8-C remains blocked until the selected single-request prefill path passes the
unchanged formal gate. This bounds the optimization work to one candidate screen,
one selected-path validation cycle, and, only if necessary, one profiler-directed
change before the plan is explicitly reconsidered.

#### P8-B2 M=128 selection update (2026-07-10)

##### 前回の要点

The first M=8 formal run failed only prompt 3584 TTFT at p50/p95
`61.023836 / 61.025951` seconds. The bounded recovery plan allowed measured wider
single-request chunks without changing batching, context, or latency limits.

##### 今回の変更点

- The serving-only chunk path now accepts measured M=32 and M=128 while the P7
  audited M=8 path and old M=8 evidence schemas remain unchanged.
- M=128 prompt 128 and 512 produced final hidden tensors and logits that were
  bitwise identical to the existing M=8 captures. Their resident request times were
  `0.325525 / 1.015741` seconds.
- A prompt 3584 M=128 diagnostic used exactly 28 chunks, emitted the same first
  token `1`, reached cache length 3584 and position 3583, reset completely, and
  took `31.310532` seconds. This is `48.7%` faster than the M=8 formal p50 and
  `18.69` seconds below the fixed 50-second p50 limit.
- The diagnostic ran while validator files were being edited, so
  `runner_worktree_clean=false`; it selects the implementation but is not release
  evidence.
- M=128 therefore closes the candidate screen early. Measuring M=32, adding a
  sequence KV-write API, changing F32 KV, or redesigning attention would add work
  without being required by the observed gate margin. Those changes remain
  conditional on a later clean formal failure.

##### 次の行動

1. Produce one clean-binary M=128 and all-M=1 oracle pair for prompt
   32/128/512/4095, validate all unit/cache transitions, and compare final
   hidden/logits against all-M=1 and the frozen vLLM source fixtures.
2. Run the exact prompt 3584 plus 512 generated-token deep boundary through M=128
   and validate every decode position and the final 4095-token KV state.
3. Run the unchanged formal resident matrix with two warmups and five measured
   samples at every prompt length, plus prompt 32 / generation 64 decode.
4. Advance to P8-C only if the clean M=128 correctness, boundary, and every formal
   performance threshold pass. Otherwise profile the failing component once before
   approving any further implementation change.

#### P8-B2 M=128 clean correctness result (2026-07-10)

##### 前回の要点

The bounded candidate screen selected M=128 after a 3584-token diagnostic took
`31.310532` seconds, but that dirty-worktree diagnostic was not release evidence.
The selected path still required one clean M=128/all-M1 oracle pair before the
deep-boundary and formal performance runs.

##### 今回の変更点

- Clean evidence is frozen under
  `benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/m128-p32-p4095-clean-72008b9/`.
- Both producers used clean commit
  `72008b91d3e2ada892208803b1891a5af466c5f2` and binary SHA-256
  `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`.
- Prompt 32/128/512/4095 M=128 request times were
  `1.131062 / 0.176792 / 1.005426 / 56.753855` seconds. The matching all-M1
  times were `1.160059 / 3.979503 / 18.786001 / 369.124277` seconds.
- The 4095-token path used 31 M=128 calls and a 127-token M=1 tail, emitted token
  `291`, reached KV length 4095 at position 4094/block 255, and reset all runtime,
  scheduler, allocator, and 40-layer cache state.
- The independent validator passed every required prompt against both all-M1 and
  the frozen vLLM source. M=128 versus all-M1 had worst relative L2
  `0.055494862`, minimum cosine `0.998492050`, exact top-1 on all prompts, and
  top-10 overlap 10. Only prompt 32 was bitwise equal; the other prompts passed
  the defined numeric gates.

##### 次の行動

1. Run and independently validate the exact prompt 3584 plus 512 generated-token
   boundary through the clean M=128 path.
2. Run the unchanged formal resident TTFT/decode matrix with two warmups and five
   measured samples.
3. Complete P8-B2 and advance to P8-C only if the boundary evidence and every
   formal threshold pass.

#### P8-B2 M=128 clean deep-boundary result (2026-07-10)

##### 前回の要点

The selected M=128 path passed its clean prompt 32/128/512/4095 correctness
oracle. The remaining correctness condition was the exact 3584-token prompt plus
512 generated-token boundary before formal performance could close P8-B2.

##### 今回の変更点

- Clean evidence is frozen under
  `benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/deep-boundary-p3584-g512-m128-clean-3bb1ef2/`.
- The run used clean commit `3bb1ef206e05aafc47bde82f105eea0bd8278443`
  and binary SHA-256
  `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`.
- It executed 28 M=128 prefill calls and 511 M=1 decode calls, recorded all 512
  generated steps, and reached final KV length 4095 at position 4094/block 255.
- Resident request time was `107.083953` seconds and reset time was
  `0.003267` seconds. Reset returned Ready, active0/waiting0, zero allocated
  blocks, and all 40 cache lengths zero.
- The independent validator accepted every prefill, decode, terminal, reset, and
  external build-identity condition. The raw result SHA-256 is
  `885bbd1a84fdd18c81829bc87f0e558d46f1267180263c5adf865a55cb07235e`.

##### 次の行動

1. Run the unchanged M=128 formal resident matrix with two warmups and five
   measured TTFT samples at every required prompt, plus prompt 32 / generation 64
   decode.
2. Independently validate timing, terminal/reset, VRAM, isolation, and clean build
   identity without accepting producer self-reported pass state.
3. Complete P8-B2 and begin P8-C only if every fixed threshold passes. On failure,
   collect the one bounded profiler run already allowed by this plan.

#### P8-B2 M=128 formal performance result (2026-07-10)

##### 前回の要点

The selected M=128 path passed clean prompt correctness and the exact 3584+512
deep boundary. P8-B2 had one remaining condition: repeat the unchanged formal
resident TTFT/decode matrix with complete reset, isolation, and VRAM evidence.

##### 今回の変更点

- Clean evidence is frozen under
  `benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/performance-m128-clean-c271e01/`.
- The run used clean commit `c271e010f18e6683dc53834188c45287434a34ef`
  and binary SHA-256
  `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`.
- Prompt 32/128/512/2048/3584 TTFT p50/p95 was
  `0.958687/0.960489`, `0.150361/0.150400`, `0.995855/1.216792`,
  `10.817689/10.825768`, and `31.286809/31.291056` seconds. Every unchanged
  threshold passed.
- Prompt 32 remains below M=128 and therefore uses 32 M=1 calls. Its TTFT is
  slower than the old M=8 path but remains below the fixed `2.5/3.0` second gate;
  hybrid-tail tuning is not required for the minimum product and stays deferred.
- Prompt 32 / generation 64 decode passed at p50 `27.757735` token/s and p95
  inter-token latency `0.036882` seconds.
- All 42 requests reset completely, and all 44 AMD SMI/KFD VRAM captures agreed
  with no unrelated process. The independent validator reported no gate errors.
- P8-B2 is complete. M=128 is now the selected single-request prefill path for the
  v0.1 resident worker; this does not add request batching or a waiting queue.

##### 次の行動

1. Begin P8-C with deterministic CPU sampling and its frozen RNG/ordering
   contract, while preserving greedy behavior as a supported special case.
2. Add cross-thread cancellation at the worker boundary and prove that no token is
   emitted after cancellation acknowledgment.
3. Implement the single resident worker protocol with one active request, reject
   concurrent work as busy, and keep request batching/waiting0 unchanged.

### P8-C: Sampling, Cancellation, and Resident Worker

Tasks:

- implement deterministic CPU sampling over the current-step logits;
- use a pinned RNG implementation and explicit seed;
- implement temperature, top-k, top-p ordering and normalization;
- reject nonfinite logits and invalid probability mass;
- add the cross-thread atomic cancellation token to the existing synchronous session operations;
- add cancellation checks between token steps;
- make cancel release scheduler/KV ownership without marking normal runtime failures as recoverable;
- implement JSONL worker reader thread and inference thread;
- implement ready/started/progress/token/released/error events;
- implement graceful shutdown and fatal exit;
- add structured stderr logs with request ID, phase, token counts, latency, and error code;
- never log message content by default.

Tests:

- temperature 0 equals greedy top1;
- same seed and parameters produce identical token IDs;
- different seeds exercise more than one valid sample on a synthetic distribution;
- top-k and top-p boundary tests;
- NaN/Inf and all-masked distribution rejection;
- cancel during prompt and during decode;
- cancel for an inactive/mismatched request ID is rejected without touching GPU state;
- cancel immediately after a token without emitting a later token;
- second request succeeds after every recoverable cancel;
- malformed/oversized JSONL input does not reach GPU mutation;
- worker EOF/fatal poison behavior;
- progress-event cadence during long prefill;
- 100 sequential raw-token requests with cancel mixed in.

Acceptance:

- token events are ordered and flushed;
- cancellation observation is bounded by one token-step boundary and terminal release p95 is at most 2 seconds;
- a subsequent request is not admitted before `released(reset_complete=true)`;
- allocator/cache return to baseline after completion and cancel;
- unrecoverable runtime errors exit the worker and do not silently continue;
- 100-request run has no monotonic host or VRAM growth.

Cancel latency uses 2 warmups and 10 measured repeats. Time starts when the reader validates the matching cancel command and sets the atomic flag; it ends after the inference thread completes reset and the flushed `released(reset_complete=true)` event is observed by the protocol reader. p95 uses the frozen linear interpolation.

Stop condition:

- do not expose streaming HTTP until worker cancellation and event ordering are deterministic.

### P8-D: Tokenizer and Non-Streaming OpenAI Gateway

Tasks:

- create the dedicated Python package and locked dependencies;
- load local tokenizer without network access;
- support complete system/user/assistant history;
- support string and text-only content parts;
- apply Qwen3 chat template with generation prompt;
- reject invalid role order and context overflow before worker mutation;
- implement stable final detokenization;
- acquire the GPU singleton lock and spawn exactly one Rust worker;
- terminate the gateway nonzero on unexpected worker EOF or fatal worker error;
- implement startup, total-request, no-progress, and cancel-release hard watchdogs;
- implement bounded worker termination and nonzero gateway exit for every watchdog breach;
- implement one atomic active request slot with no waiting queue;
- implement `/healthz`, `/readyz`, `/v1/models`;
- implement non-stream `/v1/chat/completions`;
- implement OpenAI-shaped errors and usage counts;
- implement Bearer auth for `/v1/models` and `/v1/chat/completions`;
- fail startup when binding beyond loopback without an API key.

Tests:

- Python token IDs equal frozen Transformers fixtures;
- English, Japanese, system, two-turn, and code prompts;
- malformed role order returns 400;
- prompt plus completion reservation over 4096 returns 400 without worker mutation;
- model mismatch returns 404;
- a second request while one is active returns 429 with `Retry-After: 1`;
- worker not ready returns 503;
- invalid API key returns 401;
- unsupported fields return 400;
- injected startup, prefill, decode, and reset hangs make readiness false and terminate the gateway within the frozen deadline;
- non-stream generated token IDs/text match direct worker output.

Acceptance:

- `curl` can list the model and obtain one complete Japanese answer;
- worker remains loaded between requests;
- no request path launches a new model-loading subprocess;
- there is no waiting request storage;
- `/readyz` remains 200 during a healthy active request;
- fatal worker exit makes the gateway exit rather than remain falsely healthy.
- every hard-deadline breach kills the worker, exits the gateway in bounded time, and leaves recovery to systemd.

Stop condition:

- do not add OpenWebUI-specific workarounds if the base OpenAI contract is wrong.

### P8-E: SSE Streaming and Disconnect Safety

Tasks:

- implement `text/event-stream` ChatCompletion chunks;
- emit role, content deltas, final finish reason, and `[DONE]` in order;
- implement bounded worker-to-client channel;
- apply incremental UTF-8 decoder;
- detect client disconnect and send worker cancel;
- keep stdout/stderr pumps draining independently of client backpressure;
- cancel immediately on the first failed nonblocking enqueue into the 32-event client queue;
- implement optional final usage chunk when requested;
- implement the frozen post-header error event and no-`[DONE]` failure path;
- implement the fixed fatal ordering: unready, bounded best-effort error, transport close, worker termination, gateway nonzero exit;
- ensure health/models endpoints remain responsive during generation.

Tests:

- SSE syntax and headers;
- role before content;
- finish reason exactly once;
- `[DONE]` exactly once and last;
- stream concatenation equals non-stream response;
- Japanese, emoji, combining character, code fence;
- disconnect before first token, mid-generation, and after final token;
- slow-reader bounded-memory test;
- worker failure before and after the first SSE chunk;
- worker `SIGSTOP` during prefill/decode/reset triggers the correct watchdog and bounded gateway exit;
- second concurrent request receives 429 while health/readiness remain responsive;
- next request succeeds after disconnect.

Acceptance:

- `curl -N` shows token deltas as generated;
- a client disconnect releases the worker and the following request succeeds;
- no invalid replacement characters, duplicated text, or missing suffixes occur in the UTF-8 suite.

Stop condition:

- a disconnect that leaves the worker busy or KV allocated blocks the release.

### P8-F: OpenWebUI Integration, Deployment, and Release Gate

Tasks:

- add `deploy/systemd/ullm-openai.service` and environment example;
- install it as a system service with `User=homelab1`, `RuntimeDirectory=ullm`, `RequiresMountsFor=/home/homelab1/datapool`, and ordering after Docker/network availability;
- verify that the service uses the P8-A persistent artifact/package copies and rejects any hash mismatch before model load;
- launch Uvicorn with one worker, reload disabled, a GPU singleton lock, and `KillMode=control-group`;
- make fatal worker exit terminate the gateway so systemd restarts the whole service with bounded backoff;
- handle SIGTERM with bounded cancellation and shutdown;
- record startup/load/readiness timings;
- configure bind address and Bearer key;
- bind only to the `open-webui-network` host gateway and restrict port 8000 to that bridge with host firewall rules;
- connect the current OpenWebUI instance through the OpenAI connection UI;
- revalidate the fixed `open-webui-network` route and record its network ID, subnet, gateway, bridge interface, and firewall rules;
- set model context to 4096 in OpenWebUI;
- disable title/follow-up/tag background generation for initial single-worker operation;
- record OpenWebUI version/image digest and connection settings without secrets;
- run the release smoke and soak matrix;
- add `tools/validate-sq8-openwebui-release.py` to derive pass/fail from the machine-readable release matrix;
- write operator README with start, stop, health, logs, upgrade, and recovery commands.

Required OpenWebUI smoke:

1. model appears in selector;
2. Japanese one-turn answer streams;
3. English one-turn answer streams;
4. system prompt affects output;
5. two-turn conversation uses prior assistant/user history;
6. code block renders without corruption;
7. Stop button cancels generation;
8. next request succeeds after Stop;
9. max token completion reports `length`;
10. EOS completion reports `stop`;
11. context overflow returns a visible 400 and the next valid chat succeeds;
12. a concurrent second request receives a visible 429 and does not poison the service;
13. fixed-seed sampling is repeatable and default sampling produces a valid response;
14. a post-header injected worker failure is shown as failed, never as a completed answer.

Soak:

- 20 sequential OpenWebUI chats;
- at least 5 cancellations at different phases;
- 100 sequential HTTP chats after warmup, including at least 20 sampled responses;
- at least 2 context-overflow rejections;
- at least 1 malformed API request between valid requests;
- FD, thread, child-process, RSS, VRAM, allocator, and KV observations before and after;
- kill the worker once and prove automatic full-service restart followed by a successful chat;
- after restart and re-warmup, run 20 additional sequential HTTP chats;
- service restart followed by a successful chat.

Measurement rules:

- the HTTP latency client runs from the fixed OpenWebUI Docker network and uses the exact post-template-length fixtures from P8-A;
- HTTP TTFT starts after the client writes the last request-body byte and ends when it receives the first non-empty SSE content delta, so auth, schema validation, tokenization, admission, IPC, prefill, sampling, detokenization, and local network transport are included;
- each HTTP length has 2 warmups and 10 measured repeats, with terminal release/reset between samples and the frozen linear percentile calculation;
- resource soak has a normal-operation segment and a later fatal-restart segment; their baselines are never mixed;
- after 10 warmup chats, the normal segment baseline is the median of five one-second samples taken after a five-second idle settle;
- primary host memory is the systemd service cgroup `MemoryCurrent`; gateway and worker RSS are recorded as diagnostics;
- primary VRAM is isolated-R9700 process VRAM from the P8-A frozen command; no unrelated KFD process may use the R9700;
- after each terminal release, wait five seconds and record the median of five one-second resource samples;
- compute Theil-Sen slope over the 100 normal-operation post-release medians;
- after the intentional fatal restart, perform 10 new warmup chats and establish a separate baseline before judging the post-restart segment.

Acceptance:

- all smoke items pass;
- no worker reload occurs between normal requests;
- allocator and KV state exactly match the post-warmup baseline after every terminal release;
- FD, thread, and child-process counts return to the post-warmup baseline;
- in each separately baselined soak segment, final RSS and VRAM are each no more than 64 MiB above that segment's post-warmup baseline, and their frozen robust-slope calculation is at most 256 KiB per request;
- readiness is correct during load, Ready, fatal failure, and shutdown;
- the P8-B2 latency matrix is repeated through HTTP with 10 measured runs, and first non-empty SSE content passes the same p50/p95 limits;
- logs contain request IDs and timings but not prompt/response content by default;
- a machine-readable release matrix and independent validator derive the final result;
- the validator recomputes oracle differences, token/logit gates, percentiles, request counts, resource deltas, Theil-Sen slopes, and all evidence hashes from raw samples rather than trusting any producer `passed` field;
- the validator binds git commit/dirty state, model/artifact/package/tokenizer identities, vLLM oracle identity, gateway/worker source hashes, OpenWebUI image digest, and every input evidence SHA-256;
- correctness, resource safety, auth, cancellation recovery, and hard latency gates cannot be waived;
- evidence bundle and checksums are committed;
- batch processing remains absent from the critical path.

## 8. Configuration Contract

Initial environment keys:

```text
HIP_VISIBLE_DEVICES=1
ULLM_MODEL_ID=ullm-qwen3-14b-sq8
ULLM_ARTIFACT_DIR=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
ULLM_PACKAGE_DIR=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
ULLM_TOKENIZER_DIR=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8
ULLM_CONTEXT_LENGTH=4096
ULLM_DEFAULT_MAX_TOKENS=256
ULLM_MAX_TOKENS=512
ULLM_ACTIVE_REQUESTS=1
ULLM_WAITING_REQUESTS=0
ULLM_ENABLE_THINKING=false
ULLM_LISTEN=172.20.0.1:8000
ULLM_API_KEY_FILE=/etc/ullm/openai-api-key
ULLM_GPU_LOCK_FILE=/run/ullm/r9700.lock
ULLM_SSE_QUEUE_EVENTS=32
ULLM_WORKER_STARTUP_DEADLINE_SECONDS=600
ULLM_REQUEST_DEADLINE_SECONDS=180
ULLM_PROGRESS_DEADLINE_SECONDS=30
ULLM_CANCEL_RELEASE_DEADLINE_SECONDS=5
ULLM_FATAL_FLUSH_DEADLINE_MILLISECONDS=250
ULLM_LOG_LEVEL=info
```

The existing ten `ULLM_REQUIRE_HIP_*` SQ8 guards remain mandatory in the worker service environment.

Configuration validation occurs before model load. Secrets are not accepted as command-line arguments and are not written to evidence.

## 9. Verification Matrix

| area | required verification |
| --- | --- |
| P7 regression | existing Rust tests, P6/P7 validators, fixed generation evidence |
| raw runtime | M=1 and M=8-chunk oracles, lengths, block boundaries, EOS, max length, reset, invalid IDs |
| prefill latency | 32/128/512/2048/3584 p50/p95 hard gates, decode p50/p95 |
| sampling | greedy equivalence, seeded stability, top-k/top-p boundaries, nonfinite rejection |
| cancellation | prompt, decode, disconnect, terminal release, next-request recovery |
| tokenizer | exact template text and IDs against frozen Transformers version |
| detokenizer | Japanese, emoji, byte fallback, combining characters, code fences |
| protocol | duplicate keys, schema version, oversized line, event order, worker EOF |
| HTTP | models, chat stream/non-stream, auth, status/error schemas, usage |
| admission | active 1, waiting 0, concurrent second 429, health endpoints remain responsive |
| context | exact 4096 reservation, 4097 reject, no silent truncation |
| resource | KV blocks, FD/thread/child counts, VRAM/RSS baseline/peak/post, 100 raw and HTTP requests, 20 UI chats |
| operations | startup, readiness, SIGTERM, fatal worker restart, service restart |

Tests that mutate GPU/session state run serially. Python/HTTP schema tests may run in parallel when they do not spawn a model worker.

## 10. Evidence Layout

Use:

```text
benchmarks/results/YYYY-MM-DD/sq8-openwebui-v0.1/
  environment.json
  model-identity.json
  raw-session-results.json
  sampling-results.json
  cancel-results.json
  prefill-latency-results.json
  api-contract-results.json
  openwebui-smoke.json
  soak-results.json
  release-matrix.json
  service.log
  summary.md
  SHA256SUMS
```

Record:

- git commit and dirty state;
- worker/gateway source hashes;
- model/artifact/package/tokenizer hashes;
- Rust/Python/package versions;
- OpenWebUI version or image digest;
- GPU identity and visibility variables;
- exact limits and sampling defaults;
- commands and environment without secrets;
- prompt/completion token counts;
- TTFT, per-token latency, request latency;
- finish reason and cancel phase;
- allocator/cache state before and after;
- VRAM/RSS baseline, peak, and post-run;
- validator output and evidence checksums.

## 11. Risk and Stop Policy

| risk | mitigation / stop rule |
| --- | --- |
| sequential M=1 prefill is slow | use M=1 as the correctness oracle, then require fixed single-request M=8 chunks to pass the full hard latency matrix |
| serving changes break P7 evidence | keep audited runtime/API separate and run P7 regression on every phase |
| full logits readback limits decode | accept for v0.1 CPU sampling; GPU sampling is a post-product optimization |
| Python/Rust protocol desynchronizes | versioned JSONL, stdout protocol only, line limits, duplicate-key rejection |
| UTF-8 streaming corrupts text | stable incremental decoder and multilingual golden suite before OpenWebUI release |
| disconnect leaves worker occupied | cancel at token boundary; release/reset is a hard release gate |
| fatal HIP error is treated as recoverable | transition to Failed, fail readiness, exit worker for supervisor restart |
| OpenWebUI background tasks collide with active chat | disable initial background task generation or route it to another model |
| context allocation causes OOM | checked startup allocation, fixed 4096 cap, no per-request growth |
| oversized chat history exceeds context | return explicit 400; add whole-turn truncation only under a later frozen contract |
| dependency drift changes tokenization | dedicated locked environment plus tokenizer fixture hashes |
| LAN endpoint is exposed without auth | default to loopback; require Bearer key and host firewall restriction before non-loopback bind |

Do not spend time on speculative micro-optimization after all P8-F acceptance gates pass. Create a separate optimization plan with measured bottlenecks.

## 12. Recommended Commit Boundaries

1. `Define SQ8 serving and worker contracts`
2. `Add variable-length SQ8 serving session`
3. `Add fixed single-request SQ8 prefill chunks`
4. `Add deterministic SQ8 sampling and cancellation`
5. `Add resident SQ8 worker protocol`
6. `Add Qwen3 tokenizer gateway`
7. `Add OpenAI model and non-stream chat endpoints`
8. `Add SSE streaming and disconnect cancellation`
9. `Add OpenWebUI service deployment`
10. `Record OpenWebUI product evidence`

Each commit must pass the tests for its phase and preserve the prior phase gates.

## 13. Product Exit Criteria

The v0.1 product goal is complete only when all conditions below are true.

1. P7 correctness evidence remains valid.
2. Variable M=1 prompts pass P7 and independent vLLM oracle gates on R9700.
3. Fixed single-request prefill chunks pass correctness and every 4096-context latency gate.
4. Sampling and cancellation are deterministic and recoverable.
5. Model loads once and remains resident across normal requests.
6. `/v1/models` and stream/non-stream `/v1/chat/completions` pass contract tests.
7. OpenWebUI displays live Japanese and English responses.
8. Multi-turn chat, EOS, length finish, Stop, post-Stop recovery, context error, and concurrent-request 429 pass.
9. The 100-HTTP/20-OpenWebUI soak plus cancel mix passes all resource limits.
10. Fatal worker exit automatically restarts the complete service and the next chat succeeds.
11. systemd startup, readiness, shutdown, and restart are documented and verified.
12. The independent release validator passes with no waiver of hard gates.
13. Evidence bundle and SHA-256 checksums are committed.
14. Known limitations are shown in operator documentation.

After this gate, the next decision is based on observed usage:

- latency headroom is small: optimize measured single-request prefill bottlenecks;
- decode limited by logits readback: add GPU sampling/top-k path;
- users need automatic history management: specify whole-turn truncation;
- clients need custom stop strings: add bounded stop matching;
- isolated concurrent requests need waiting: add a one-entry queue before considering batching;
- sustained queue pressure from real users: plan request batching;
- context pressure: evaluate larger KV cache or cache dtype change;
- tools are required: create a separate function-calling/API extension plan.

None of these are prerequisites for the first OpenWebUI product release.
