# reasoning・thinking・OpenWebUI互換拡張履歴

## 2026-08-14: 受入条件と実装

Phase 6のstrict DTOは`max_completion_tokens`だけを受理し、production Qwen backendは
`ThinkingModeV1::Disabled`を固定していた。その一方、frontend rendererはすでに
typed enabled/disabled modeとhistorical `reasoning_content`の正規化を持っていた。そのため、
rendererを再実装せずrequest controlとresponse framingだけを拡張した。

- `sllm.thinking`/`sllm.separate_reasoning`をclosed typed extensionとして追加した。
- separationはUTF-8 chunkを壊さず`</think>`をchunk境界越しで認識するstate machineとした。
- non-streamとSSEは同じsplitterを使い、opt-in時だけ`reasoning_content`を返す。
- `ServerConfigV1::openwebui_compatible`とCLI `--compatibility-profile openwebui`を追加し、
  strict profileの`max_tokens`拒否は維持した。
- assistant messageのstring `reasoning_content`をtyped frontendへ渡し、multi-turn履歴の
  round-tripを可能にした。system/userの同fieldは拒否する。

## 2026-08-14: R9700 production smoke

canonical R9700をUUID `GPU-a8e9ddefa2d60f55`で単独可視化し、exact `gfx1201`、ROCm
7.14.0、Qwen3.5-4B lock
`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`で
`sllm-server --compatibility-profile openwebui`を起動した。ready eventは`openwebui`、`gfx1201`、
exact model fingerprintを返した。

- non-stream: legacy `max_tokens=37`とthinking/separationを受理し、prompt 19、completion 37、
  total 56 tokenを返した。`</think>`前のlength終了のため、生成文全体が
  `reasoning_content`、最終`content` emptyとなった。
- SSE: `max_tokens=17`でrole chunk、連続する`delta.reasoning_content`、final usage
  15+17=32 token、`finish_reason=length`、exact `[DONE]`を確認した。
- shutdown auditは2 requestとも`selected_backend=hip`、`all_dispatches_hip=true`、
  `fallback_used=false`だった。request/workspace cleanup、retryable cleanup、durable quarantine、
  final memoryは全て0で、shutdown後の全3 GPU processは0だった。

これはdirty local candidateのdraft smokeであり、immutable release evidenceや性能claimではない。

## 2026-08-14: verification

- sllm-server unit 13/13、HTTP contract 10/10: PASS。
- workspace Rust test、sllm-server fmt/clippy `-D warnings`: PASS。
- OpenAI profile fixture/validator、Markdown link、JSON/schema/matrix、`git diff --check`: PASS。
- H0 full row 504/504: PASS（local-development、immutable=false）。
- R9700 production non-stream/SSE smoke 2/2、shutdown cleanup: PASS。

[対応する計画](../../../../plans/archive/2026/08/11-20/api-reasoning-openwebui-extension.md)
