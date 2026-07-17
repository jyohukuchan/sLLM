# uLLM Current Project Overview, Progress, And Environment

Date: 2026-07-10

## 前回の要点

- 2026-07-09 時点では、uLLM は Rust 制御層、C++20/HIP runtime、Python 評価ツールからなる Qwen3/Qwen3.5 向け研究用推論エンジンだった。
- AQ4 prototype は R9700/RDNA4 と V620/RDNA2 で単一 request 推論まで到達していた。
- SQ は FP8 候補評価として整理されており、batch throughput、resident runtime、serving 比較は次の課題だった。

## 今回の変更点

- 現在のソース、Git 履歴、計画、benchmark result、OS、GPU、ROCm、toolchain を再監査した。
- SQ の正式 public format ID は `SQ8_0` となり、`AQ4_0` と合わせて format ID と legacy alias の正規化が実装済みである。
- Qwen3-14B-FP8 の 40 layer mixed-request-state path は、b2/b4/b8 で direct SQ8_0 batch projection `6720/6720`、host staging `0/0` まで進んだ。
- HEAD `8b08d98` の offline-serving candidate CLI を現在の debug binary に再構築し、R9700 で短い 40 layer 実モデル実行を行った。
- 実環境の RAM は記載上の 16GB x 8 ではなく、DMI 上は CHANNEL D が空の 16GB x 7 であることを確認した。

## 次の行動

1. `origin/main` より先行する 92 commits をレビューし、remote へ push または別 remote へバックアップする。
2. offline-serving candidate report の空白付き値、input source、load-excluded latency の変換を修正し、実モデル統合 test を追加する。
3. `README.md`、SQ8 plan 冒頭、overview、memo を現在の実装状態へ同期する。
4. offline synchronous batch と server-style serving の境界を固定し、vLLM parity gate へ昇格させる条件を満たす。
5. CHANNEL D の DIMM 未認識を確認し、CPU memory bandwidth を使う benchmark の前提を揃える。

## Repository And Git

- Workspace root: `/home/homelab1/coding-local/ultimateLLM`
- Git repository: `uLLM-project/`
- Remote: `git@github.com:jyohukuchan/uLLM-project.git`
- Branch: `main`
- HEAD: `8b08d98 Add SQ8 offline serving throughput CLI`
- Fetch 後の状態: `origin/main=3138206`, ahead 92, behind 0
- Worktree: clean
- 最新 92 commits は 2026-07-09 から 2026-07-10 の SQ8_0 loader、dispatch、real batch、host staging 除去、vLLM comparison gate、offline-serving candidate 追加が中心である。
- `journal/` は `.gitignore` 対象であり、検証履歴は commit と一緒には保存されない。

## Project Summary

uLLM は、低ビット LLM 向けの次の四つを一体で開発するプロジェクトである。

1. `.ullm.d` directory package と manifest
2. 精度重視の `AQ4_0` と速度重視の FP8 E4M3 `SQ8_0`
3. Rust control plane と C++20/HIP runtime
4. correctness guard と throughput comparison の benchmark 基盤

現在は production server ではなく、Qwen3/Qwen3.5 decoder を対象にした高度な研究・性能検証用 CLI engine である。HTTP server、native tokenizer、online arrival、tensor parallel、multi-node、汎用 Model IR は未完成または将来構想である。

## Current Architecture

- Cargo workspace:
  - `crates/ullm-engine`: package loader、scheduler、decoder、Qwen3/Qwen3.5 runtime、AQ/SQ、backend dispatch、CLI
  - `crates/ullm-quant`: safetensors planning、AQ conversion、prototype/full `.ullm.d` package generation
  - `crates/ullm-runtime-sys`: C ABI FFI と RAII wrapper
- Rust owns CLI, request lifecycle, scheduler, metadata, telemetry, and benchmark orchestration.
- C++20 owns runtime buffers/streams, CPU fallback, HIP/HIPRTC kernels, and low-level operators.
- GPU libraries are loaded dynamically. HIP kernels are embedded as source and compiled at runtime through HIPRTC.
- Python is used for package tools, comparison runners, result normalization, and quality guards.
- Runtime includes paged KV cache, request-ready batches, Qwen3 self-attention, Qwen3.5 linear attention, RMSNorm, RoPE, causal/cached-prefix/paged attention, AQ4_0/SQ8_0 projection primitives, and CPU/HIP paths.

## Progress Assessment

### Completed Or Demonstrated

- `AQ4_0` / `SQ8_0` naming and legacy alias normalization.
- SQ8_0 artifact builder, checksum/scale metadata, resident tensor loader, and direct single/batch/pair/triple matvec APIs.
- Qwen3/Qwen3.5 package namespace aliases and thin-package plus SQ8 sidecar loading.
- Scheduler ready-batch execution and paged KV cache primitives.
- R9700-specific active direct matvec descriptor selection.
- Qwen3-14B-FP8 40 layer real-batch model-loop diagnostics on R9700.
- Latest no-host-staging result rows:

| requests | prefill tok/s | decode tok/s | total tok/s | SQ8 batch | host staging | VRAM consumed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 13.955 | 16.475 | 14.705 | 6720/6720 | 0/0 | 14.34 GiB |
| 4 | 15.529 | 16.704 | 15.902 | 6720/6720 | 0/0 | 14.51 GiB |
| 8 | 16.135 | 16.659 | 16.306 | 6720/6720 | 0/0 | 15.02 GiB |

These rows are classified as `cli_model_loop_diagnostic`, not serving parity rows. The matching vLLM b2/b4/b8 rows use `serving_throughput_benchmark`, so the raw throughput values are not final equivalent-harness comparisons. uLLM throughput also shows little batch scaling, which remains a performance issue even before final parity work.

### Partial Or Not Done

- SQ8_0 payload remains a sidecar artifact rather than an integrated `.ullm.d` format.
- Higher-level fused QKV, MLP, and linear-attention descriptors are catalog entries only; active fused C++ kernels and family switching are not implemented.
- Stable SQ8_0-named regression coverage still coexists with many legacy `sq-fp8-*` CLI names.
- Tokenizer, HTTP API, asynchronous/online arrival, and final server-style serving semantics are not included.
- Generic Model IR, MoE, MTP, diffusion, multimodal, CUDA/NVIDIA, MI300X production validation, TPU/JAX, Ascend, and NPU backends remain future work.

## Latest Offline-Serving Candidate Check

The existing `target/debug/ullm-engine` was stale and did not contain HEAD's new command. After `cargo build -p ullm-engine`, the following R9700 run completed successfully:

```text
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 \
target/debug/ullm-engine sq-fp8-token-ids-offline-serving-throughput \
  /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d \
  /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact \
  2 1048576 manifest-all len:2x2 1 0 1024 128 1000000 0
```

Observed:

- 40 layers, two requests, prompt 2 each, generated 1 each.
- `prefill_real_batch=true`, `decode_real_batch=true`.
- `sq_execution_mode=direct_fp8_dequant_matvec`.
- `sq_fp8_batch_matvec_count=840/840`.
- host staging read/write `0/0`.
- prefill `4.290 tok/s`, decode `16.422 tok/s`, end-to-end `5.691 tok/s`.
- Candidate blockers correctly include `tokenizer_not_included` and `http_server_not_included`.

Confirmed report issues:

1. `parse_offline_serving_candidate_value_map` uses `split_whitespace()`. The quoted runtime value `name="AMD Radeon Graphics"` becomes `device_name="AMD"`.
2. The parser whitelist omits `input_source` and `first_layer_input_source`. The report falls back to `mixed_request_state` and `unknown` even though the underlying model-loop path reports the actual sources.
3. The candidate declares `load_excluded_from_total=true`, but `batch_wall_ms` and request latency are derived from `outer_wall_ms`, which includes the approximately 10.7 second load. Throughput uses the approximately 1.05 second inference total. Latency and throughput therefore use different load semantics.
4. The current unit test uses `MockGPU` without spaces and does not cover these input-source or latency semantics.

## Development Environment

### Host

- Host: `homelab1-WRX80-Creator`
- OS: Ubuntu 24.04.4 LTS
- Kernel: 6.17.0-35-generic
- CPU: AMD Ryzen Threadripper PRO 3995WX, 64 cores / 128 threads
- NUMA: one node
- ISA: AVX2, no AVX-512
- RAM: 112 GB physically reported as 7 x 16 GB; OS `MemTotal` approximately 109 GiB
- DIMM status: P0 CHANNEL D reports `No Module Installed`
- Swap: 8 GiB, approximately 560 MiB used during audit
- Root: Samsung 990 PRO 4 TB, ext4, approximately 2.4 TB free
- `~/datapool`: two 7 TB NVMe devices as local ZFS pool, approximately 8.5 TB free

### GPU And ROCm

- ROCm: 7.2.1
- HIP: 7.2.53211
- amdgpu driver reported by ROCm SMI: 6.16.13
- uLLM runtime devices:
  - `0`: CPU fallback
  - `1`: Radeon Pro V620, gfx1030, approximately 32 GB
  - `2`: Radeon AI PRO R9700 class device, gfx1201, approximately 34.2 GB decimal, reported as `AMD Radeon Graphics`
  - `3`: Radeon Pro V620, gfx1030, approximately 32 GB
- All three GPUs were visible through ROCm and uLLM. CPU plus all GPU allocation/copy/add smokes passed.
- Runtime device `arch` text is empty, while compute versions 10.3/12.0 are available. Current dispatch derives RDNA4 from compute major and canonicalizes the local generic R9700 name conservatively.

### Toolchain

- Rust/Cargo: 1.96.0, Edition 2024 workspace
- rustfmt: 1.9.0
- clippy: 0.1.96
- System clang/clang++: 18.1.3
- g++: 13.3.0
- ROCm clang: 22.0.0git
- CMake: 3.28.3
- Ninja: 1.11.1
- mold: 2.30.0
- Python: 3.12.3
- Link config: clang plus mold through `.cargo/config.toml`
- `cc` crate C++ builds use the default C++ compiler because `CXX` is not set; currently this is g++ 13.3.
- `build/envs/` contains atom, SGLang, vLLM, and vLLM ROCm nightly environments. The known comparison environment is `build/envs/vllm-rocm-nightly`.
- `tools/fast-build-env.sh` is not sourced in the current shell, so recommended sccache and parallel-build variables are not active.

### Local Storage Use

- `build/`: approximately 36 GB
- `target/`: approximately 1.5 GB
- `reference-src/`: approximately 2.6 GB
- tracked/ignored benchmark results: approximately 674 MB under `benchmarks/results/`

### Network Note

- The WRX80 `datapool` is local ZFS and NFS server service is active; no client NFS mount exists on this host.
- The active i40e Ethernet port negotiated 1 Gb/s during the audit and the other was down. An Omni-Path RDMA link was active. The stated 25 Gb connection to T7610 was not confirmed from the current WRX80 Ethernet state.

## Verification Performed

- `git fetch --prune origin`: passed.
- `cargo fmt --all --check`: passed.
- `cargo check --workspace`: passed.
- `cargo build -p ullm-engine`: passed.
- `cargo test --workspace -- --test-threads=1`: 345 Rust tests passed.
- `python3 -m unittest discover -s tests -p 'test_*.py'`: 98 tests passed.
- Targeted offline-serving report unit test: passed.
- CPU and all three GPU allocation/copy/add smokes: passed.
- R9700 40 layer SQ8_0 offline-serving candidate smoke: passed with the report issues noted above.

The C++ build emits three `-Wsubobject-linkage` warnings because public runtime context/buffer/stream structs contain an anonymous-namespace `BackendKind`. These warnings do not currently fail the build.

## Documentation And Process Risks

- `README.md` is only three lines and does not describe build, test, current support, or prototype limitations.
- The SQ8 plan's early `Current Implementation State` says dispatch and 40 layer rows are unfinished, while later sections document their completion. The later source/results are the current evidence.
- The 2026-07-09 overview and root memo do not include the latest 92 commits.
- There is no repository CI configuration. Validation is manual and journal-based.
- The ignored `journal/` directory means important verification evidence is not preserved by Git commits unless separately backed up.

## Integrated Assessment

The project has moved beyond isolated operator experiments. It now has a working Qwen3/Qwen3.5 package/runtime path, request scheduling, paged KV primitives, AQ4_0 execution, and 40 layer resident SQ8_0 real-batch model-loop execution on R9700. The current boundary is a credible research engine and benchmark harness.

It is not yet a production or parity-grade serving engine. The immediate blocker is no longer SQ8_0 connectivity or host staging. It is the correctness of the new serving result contract, the absence of tokenizer/HTTP/online arrival semantics, limited batch scaling, sidecar packaging, and missing fused kernel families.
