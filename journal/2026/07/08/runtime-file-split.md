# Runtime file split

## 前回の要点

- `runtime/src/ullm_runtime.cpp` は23k行を超え、prefill/attention最適化を続けるには単純に長すぎる状態だった。
- `crates/ullm-runtime-sys/build.rs` は `ullm_runtime.cpp` 1ファイルだけをC++ build対象にしていた。

## 今回の変更点

- まずtranslation unitは維持したままinclude分割した。
- `HipRtcRuntime` 内のHIPRTC source-string builder群を `runtime/src/ullm_runtime_hiprtc_sources.inc` へ移動した。
- 末尾の公開C ABI wrapper群を `runtime/src/ullm_runtime_api.inc` へ移動した。
- `ullm_runtime_api.inc` はmanifestにし、実体を次の役割別ファイルへ分けた。
  - `ullm_runtime_api_core.inc`
  - `ullm_runtime_api_aq4.inc`
  - `ullm_runtime_api_linear_attn_prepare.inc`
  - `ullm_runtime_api_primitives.inc`
  - `ullm_runtime_api_attention.inc`
  - `ullm_runtime_api_linear_attn.inc`
  - `ullm_runtime_api_smoke.inc`
- `crates/ullm-runtime-sys/build.rs` に各include fileの `cargo:rerun-if-changed` を追加した。

Line count after split:

| file | lines |
| --- | ---: |
| `runtime/src/ullm_runtime.cpp` | 11699 |
| `runtime/src/ullm_runtime_hiprtc_sources.inc` | 4725 |
| `runtime/src/ullm_runtime_api_aq4.inc` | 2588 |
| `runtime/src/ullm_runtime_api_primitives.inc` | 2004 |
| `runtime/src/ullm_runtime_api_attention.inc` | 1113 |
| `runtime/src/ullm_runtime_api_linear_attn_prepare.inc` | 354 |
| `runtime/src/ullm_runtime_api_linear_attn.inc` | 321 |
| `runtime/src/ullm_runtime_api_core.inc` | 315 |
| `runtime/src/ullm_runtime_api_smoke.inc` | 15 |
| `runtime/src/ullm_runtime_api.inc` | 10 |

確認:

- `cargo fmt --all --check`
- `git diff --check`
- `cargo check -p ullm-runtime-sys`
- `cargo test -p ullm-runtime-sys -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`

## 次の行動

- 次に分けるなら、runtime internal側のhost fallback、HIP kernel cache/launcher、staging pathを役割別に切る。
- 完全な複数`.cpp`化は、匿名namespace helperの境界を先に整理してから行う。
