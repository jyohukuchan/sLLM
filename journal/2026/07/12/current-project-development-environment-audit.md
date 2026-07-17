# ultimateLLM current project and development environment audit

Date: 2026-07-12 12:35 JST

## 前回の要点

- 2026-07-10朝のoverviewでは、uLLMはQwen3/Qwen3.5向けの研究用CLI engineで、tokenizer、HTTP server、OpenWebUI製品経路は未完成だった。
- 同時期に旧SQ8 sidecarがsource checkpointの2D `weight_scale_inv`を適用していない問題が見つかり、2026-07-09/10の該当uLLM比較行は接続診断へ隔離された。
- その後、source-correctな`v0.2` canonical artifact、CK ABScale FP8経路、40-layer generation、resident worker、OpenAI gateway、OpenWebUI deploymentまで実装された。

## 今回の変更点

- 現行repository、実行コード、Git履歴、製品plan/spec、release evidence、hardware/toolchain、storage/network、systemd/Dockerの実稼働状態を横断確認した。
- `uLLM-project/main`は`origin/main`と一致し、現HEADは`85753121f688dc38790a4d89c0f59cab391cfeea`である。
- SQ8/OpenWebUI v0.1は独立validatorを含む製品releaseまで完了しており、現在もrelease時と同じworker、service unit、environment file、OpenWebUI image、artifact/packageで稼働している。
- release bundleの`SHA256SUMS` 19件、Rust formatting、Git diff check、Docker bridge経由gateway readinessを再確認し、すべて成功した。
- コード変更は行っていない。workspace rootの進捗ファイルとこのjournalだけを更新した。

## 次の行動

1. 完了したv0.1を維持する運用計画と、次の性能・機能計画を分けて新規に決める。request batchingは既存planどおり、自動的には開始しない。
2. `README.md`を現在の製品状態へ更新し、staleな`sq8-recovery-plan-v0.2.md`のstatus、release tag不在、`.rocprofv3/`未ignoreを整理する。
3. 次の性能作業をするなら、M128未満promptのM1 tail反復、151,936 logitsの毎step host readback/CPU sampling、prototype package schemaを、batchingより先に評価する。
4. AGENTS.mdと実測が異なるRAM容量、NFS/RDMA/25Gb状態を別途再確認する。WRX80側で現在確認できた通常LANは1Gbpsで、NFS/RDMAの通信中状態は確認できなかった。

## Scope and method

- Workspace root: `/home/homelab1/coding-local/ultimateLLM`
- Git repository: `/home/homelab1/coding-local/ultimateLLM/uLLM-project`
- Local files、Git object database、systemd、Docker、ROCm、ZFS、release evidenceを読み取り中心で調査した。
- コード構成、開発環境、Git/evidenceを3 subagentへ分担し、親agentでコマンド結果とsourceを再照合した。
- Internet fetch、Git fetch、remote write、service restart、GPU benchmark、推論request、full test suiteは実行していない。
- `sudo -n dmidecode`はpassword-requiredで実行できなかったため、DIMM slotは今回再取得していない。OSが認識するmemoryを実測値とした。

## Integrated conclusion

現在のuLLMには、二つの異なる成熟度が同居している。

1. **固定製品vertical slice**
   - Qwen3-14B-FP8 / SQ8_0
   - Radeon AI PRO R9700 / `gfx1201`
   - `rdna4_w8a8_block_ck`
   - 1 active request、waiting queueなし
   - resident Rust worker、OpenAI Chat Completions gateway、OpenWebUI
   - source/model/binary/deploymentをhashで固定したrelease evidenceあり
   - 現在も稼働中

2. **汎用推論engine構想と研究基盤**
   - AQ4_0 quantizer、Qwen3/Qwen3.5 partial runtime、scheduler、paged KV、CPU/HIP operators、多数のdiagnostic CLIとbenchmark toolsがある。
   - concept文書にあるgeneric Model IR、MoE、multiple backend、CUDA/NVIDIA、MI300X production、TPU/JAX、NPU、multi-model、continuous batchingなどは未完成である。

したがって「uLLM v0.1製品は完成している」は正しい。一方、「ultimate LLM inference engine全体が完成している」わけではない。

## Repository and Git

- Branch: `main`
- HEAD: `85753121f688dc38790a4d89c0f59cab391cfeea`
- Local tracking state: `HEAD == origin/main == origin/HEAD`
- Remote: `git@github.com:jyohukuchan/uLLM-project.git`
- Local branch: `main`のみ
- Git tag: 0
- Tracked/staged changes: なし
- Untracked: `.rocprofv3/`のcounter data 3件、合計16KiB
- 直近の重要commit:
  - `f647a8a`: final campaign実行元
  - `64370fe`: release evidence収録とplan complete化
  - `8575312`: v0.1 limitationsをoperator documentationへ追加
- `f647a8a..HEAD`はevidence、plan、documentationのみで、worker/gateway実行コードはcampaign時から変わっていない。
- Tracked filesは5,674件で、そのうち5,213件が`benchmarks/`配下である。
- Repository CI configurationはない。検証はlocal test harness、Git commit、tracked evidenceが中心である。
- Root `README.md`は3行だけで、現行build/test/deployment/product scopeを説明していない。

## Current architecture

### Product request path

```text
OpenWebUI :3000
  -> Docker bridge 172.20.0.1:8000
  -> FastAPI OpenAI gateway
  -> local Qwen3 tokenizer / chat template
  -> strict JSONL over stdin/stdout
  -> resident ullm-sq8-worker
  -> Qwen3 SQ8 serving session in Rust
  -> ullm-runtime-sys C ABI / RAII wrapper
  -> C++20 runtime + HIP/Composable Kernel on gfx1201
  -> token event
  -> gateway incremental detokenization / SSE
  -> OpenWebUI
```

### Main components

- `crates/ullm-engine`
  - package loader、scheduler、decoder、AQ/SQ format、Qwen3 runtime、SQ8 serving、worker protocol/backend
  - default `ullm-engine` diagnostic CLIとfeature-gated `ullm-sq8-worker`
- `crates/ullm-runtime-sys`
  - Rust FFI、runtime buffer/context/stream wrapper
  - C++20 CPU fallback、HIP/HIPRTC kernels、gfx1201 CK integration
- `crates/ullm-quant`
  - safetensors planning、AQ4 conversion、streamed `.ullm.d` package generation
- `services/openai-gateway`
  - Python 3.12、FastAPI、Uvicorn、Transformers tokenizer
  - strict request normalization、Bearer auth、single-request gate、worker supervision、SSE/cancel
- `deploy/`
  - systemd、nftables、derived OpenWebUI image/compose/browser gates
- `tools/`、`tests/`、`benchmarks/`
  - artifact builder、oracle、promotion/release validator、performance/quality tooling、raw evidence

### Language boundary

- Rust: control plane、metadata、loader、scheduler、request/session lifecycle、worker protocol
- C++20/HIP: device runtime、buffers、operators、HIPRTC、CK FP8 execution
- Python: OpenAI gateway、tokenizer、evaluation、artifact/release orchestration

## Product v0.1 status

- Model: `Qwen/Qwen3-14B-FP8`
- Served ID: `ullm-qwen3-14b-sq8`
- Model revision: `9a283b4a5efbc09ce247e0ae5b02b744739e525a`
- Artifact schema: `sq-fp8-artifact-v0.2`
- Artifact content SHA-256: `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`
- Thin package manifest SHA-256: `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`
- Worker SHA-256: `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950`
- Context: 4,096 tokens
- Max completion: 512 tokens
- Active GPU request: 1
- Waiting queue: 0
- Product root: 約16GiB
- Source tokenizer/model root: 約16GiB
- Runtime KFD VRAM: 18,276,048,896 bytes on the gfx1201 GPU

Current limitations:

- request batching、continuous batching、waiting queueなし
- single model、text-only Chat Completions
- history auto-truncationなし。overflowはgatewayが400を返す
- tools/function calling、structured output guarantee、multimodal、embeddings、Responses APIなし
- request stop stringなし
- TLS termination、multi-tenant authなし
- OpenWebUI v0.9.4はupstream 429をlegacy routeでvisible 400として表示することがある

## Old invalid results and current correctness

- 旧v0.1 sidecarはsource F8 payloadに対応する128x128 `weight_scale_inv`を適用せず、数学的に元checkpointと異なっていた。
- 該当する2026-07-09/10のuLLM same-model result rowsは`source_fp8_weight_scale_inv_not_applied`でquarantine済みで、default summariesとcomparison gateから除外される。
- 現製品は別のsource-correct `sq-fp8-artifact-v0.2`である。F8 payloadと2D block scaleを保持し、promotion validatorも`verified=true`である。
- 旧quarantine warningは現在のv0.1 product artifactを無効にしない。ただし旧rowを新しい性能・品質比較へ再利用してはいけない。

## Release evidence

Primary bundle:

`uLLM-project/benchmarks/results/2026-07-12/sq8-openwebui-product-20260712-v0.1/`

Independent validation:

- `release_status=complete`
- `full_campaign_validated=true`
- source commit `f647a8a`
- fixed source files 70
- 21 successful OpenWebUI requests
- 5 cancellation phases
- 100 normal resource requests
- planned worker failure and full systemd recovery
- 20 post-restart resource requests
- 72 latency requests
- 610 resource samples
- restart count 7 -> 8

Measured results:

- decode p50: 27.534 token/s
- inter-content p95: 37.208 ms
- TTFT p50:
  - prompt 32: 0.965 s
  - prompt 128: 0.157 s
  - prompt 512: 1.017 s
  - prompt 2048: 10.834 s
  - prompt 3584: 31.315 s
- normal 100-request final cgroup memory delta: +1,519,616 bytes
- normal process VRAM delta: 0
- restart 20-request final cgroup memory delta: -1,044,480 bytes
- restart process VRAM delta: 0
- slowest required cancel-to-release case: 約1.301 s、gate 5 s以内

Evidence reliability is high because the independent validator reconstructs outcomes from raw session、systemd journal、resource samples and binds Git source、worker binary、model、artifact、package、tokenizer、OpenWebUI image by SHA-256. Campaign Git status was dirty only because the same untracked rocprof files were present; tracked source was pinned and individually hashed.

## Current live deployment

- `ullm-openai.service`: enabled、active/running
- Gateway PID: 736263
- Worker PID: 736390
- Current `NRestarts`: 8
- Service start: 2026-07-12 09:23 JST
- Service cgroup memory: 約390MiB at audit time
- Worker binary hash、systemd unit hash、environment-file hashはrelease evidenceと一致
- Gateway: `172.20.0.1:8000`のみlisten
- Docker bridgeから`/readyz`: `{"status":"ready"}`
- OpenWebUI container: running/healthy、restart 0
- OpenWebUI image: `ullm/open-webui:0.9.4-ullm.1`
- Derived image IDはrelease evidenceと一致
- Host port 3000 `/health`: HTTP 200
- Firewall unitはdisabled表示だが、enabled gateway unitの`Requires=`でactive/exitedになる構成

## Development environment

### Host

- OS: Ubuntu 24.04.4 LTS
- Kernel: 6.17.0-35-generic
- systemd: 255
- CPU: AMD Ryzen Threadripper PRO 3995WX、64 cores / 128 threads
- NUMA: 1 node
- ISA: AVX2、AVX-512なし
- Memory: 114GiB online、約109.8GiB usable、audit時available約76GiB
- Swap: 8GiB、約0.6GiB使用

AGENTS.mdの`16GB x 8`/128GBとは一致しない。2026-07-10のDMI auditでは16GB x 7と記録されているが、今回DIMM slotをroot権限で再確認していない。

### GPU and ROCm

- Radeon Pro V620 / `gfx1030` x2、各約30GiB VRAM
- Radeon AI PRO R9700 class / `gfx1201` x1、約31.9GiB VRAM
- ROCm: 7.2.1
- HIP: 7.2.53211
- amdgpu: 6.16.13
- AMD clang: 22.0.0git
- ROCm-SMI、HSA/HIP、uLLM runtimeのdevice ordinalは一致しないため、production identityは`gfx1201`とBDFで扱う。

### Toolchain

- Rust/Cargo: 1.96.0 stable、Edition 2024
- GCC/G++: 13.3.0
- System clang: 18.1.3
- CMake: 3.28.3
- Ninja: 1.11.1
- mold: 2.30.0
- ccache: 4.9.1
- sccache: 0.7.7
- Python: 3.12.3
- uv: 0.11.25
- `.cargo/config.toml`: clang linker + mold
- Gateway environment: Python 3.12、FastAPI 0.116.1、Uvicorn 0.35.0、Transformers 5.12.1

Current shell does not have `RUSTC_WRAPPER`、`CMAKE_BUILD_PARALLEL_LEVEL`、`ULLM_HIP_BUILD_JOBS`、`ROCM_PATH`、`GPU_ARCH` set. `tools/fast-build-env.sh` is not sourced. ccacheは0.1/50GB、356/1514 hit、sccacheは利用実績0である。

Reference environments under `build/envs/`:

- `vllm-rocm-nightly`
- `vllm-rocm`
- `sglang-rocm`
- `atom-rocm`

Reference sources under ignored `reference-src/`:

- llama.cpp
- vLLM
- SGLang
- ATOM/AITER
- TensorRT-LLM

## Storage and network

- Root filesystem: 3.6TiB ext4、約2.4TiB free
- `~/datapool`: local ZFS、14.0TiB、5.47TiB used、8.50TiB free、ONLINE
- ZFSは表示上2台のstripeで、mirrorではない
- Repository working tree全体: 約50GiB
  - `build/`: 約36GiB
  - `target/`: 約11GiB
  - `reference-src/`: 約2.6GiB
  - `benchmarks/`: 約755MiB
- WRX80のdatapoolはlocal ZFSであり、このhost上のNFS client mountではない。
- NFS serverとexport設定は存在し、T7610 (`192.168.0.129`)へのpingは成功した。
- 通常LAN linkの実測は1Gbps。NFS/RDMA kernel module/transportのactive useと25Gb pathは今回確認できなかった。

## Code and maintainability assessment

Strengths:

- source-correct artifactと旧invalid resultのquarantineが明示されている。
- Worker/gatewayはfail-closed validation、singleton lock、watchdog、bounded buffering、cancel/resetを持つ。
- Releaseはraw evidenceから独立再構築され、source/binary/model/deployment identityまでhashで固定される。
- Runtime operators、scheduler、paged KV、AQ/SQ、CPU/HIP pathsに広いlocal test coverageがある。
- 普通のsource fileは約10,000 lines未満へ分割されている。

Technical debt / boundary:

- Production routeはR9700/gfx1201とQwen3-14Bにhard-fixedされている。
- `FixedM128Chunks`は128-token chunkだけをM128で実行し、promptの128未満部分とremainderをM1反復する。32-token TTFTが128-token TTFTより遅い計測はこの構造と整合する。
- Serving model headは毎stepで151,936個のF32 logitsをhostへreadbackし、CPU samplerを使う。
- Product package manifestは`ullm-prototype-manifest-v0.1`で、正式な統合container schemaではない。
- Default diagnostic CLIは分割後も合計52,616 linesの`main_parts`を持ち、production workerとは別に大きな保守面積がある。
- Generic Model IR、backend plugin model、multiple hardware、MoE/MTP等はconceptと実装に大きな差がある。
- No repository CI、no release Git tag、3-line README、ignored journalにより、再現性の強さがlocal evidence harnessへ偏っている。
- `sq8-recovery-plan-v0.2.md`は新planにsupersedeされたのに`Status: active`と古いnext actionを保持している。
- `systemctl is-system-running`は`starting`のままで、NetworkManager wait-online失敗と起動job残留がある。uLLM/OpenWebUI自体は正常である。

## Verification performed in this audit

Passed:

- `git status --short --branch`
- `HEAD == origin/main`
- current worker SHA-256 matches release anchor
- installed service/environment hashes match release evidence
- product artifact/package/promotion manifest hashes match release evidence
- release bundle `sha256sum -c SHA256SUMS`: 19/19 OK
- `cargo fmt --all --check`
- `git diff --check`
- `systemctl` active/read-only inspection
- Docker/OpenWebUI health inspection
- Docker bridge -> gateway `/readyz`
- KFD current VRAM readback
- OS/toolchain/GPU/ZFS/network inspection

Not run:

- Full Rust/Python tests
- Independent validator re-execution
- GPU smoke/benchmark
- Inference request
- Service restart/failure injection
- Remote T7610 login or remote mount inspection

The full release campaign already contains stronger functional evidence; this audit intentionally avoided disturbing the live service.
