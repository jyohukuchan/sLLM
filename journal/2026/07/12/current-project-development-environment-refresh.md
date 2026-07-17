# ultimateLLM 現状・開発環境の再把握

日時: 2026-07-12 19:11 JST

## 前回の要点

- 12:35時点の監査では、Qwen3-14B SQ8_0製品経路がrelease済みで、OpenWebUI、OpenAI gateway、常駐workerまで稼働していた。
- 旧SQ8 sidecarのsource scale欠落結果は隔離済みで、現SQ8製品はsource-correctな別artifactと独立release evidenceを使う。
- 当時は汎用推論API、served-model manifest、Qwen3.5 AQ4常駐sessionが未完成だった。

## 今回の変更点

- `uLLM-project/main`は`d0cc2df`で、`origin/main`より39 commit先行している。追跡済み差分はなく、未追跡は`.rocprofv3/`だけである。
- 12:35以降に、量子化非依存の`InferenceSession`/worker driver、厳密な`ullm.served_model.v1`、atomic activation/rollback、Qwen3.5-9B AQ4常駐model/session、AQ4 resident worker、OpenWebUI実機検証まで実装された。
- 現active manifestは`ullm-qwen3.5-9b-aq4`、`AQ4_0`、`qwen35_aq4_rdna4_v1`を指す。promotion receiptは実在し、active manifest記載のSHA-256と一致する。
- AQ4 resident promotion evidenceはlegacy経路と2ケースでtoken完全一致、子processなし、clean shutdownを記録する。OpenWebUI evidenceは1011-token prompt、EOS、length、cancel後復帰、browser、timings表示を`verified=true`で記録する。
- ただし19:03にgateway serviceとOpenWebUIが外部から停止され、19:11時点では`ullm-openai.service`がinactive、OpenWebUI containerがexited/unhealthy、3000/8000番portは非listenである。直前までのrequestは正常完了しており、systemd上はTERMによる正常停止で、クラッシュではない。今回の把握作業では再起動していない。
- コード変更やservice変更は行わず、このjournalとworkspace rootの進捗表示だけを更新した。

## 現在の構成

```text
OpenWebUI
  -> Python OpenAI gateway
  -> ullm.worker.v1 JSONL
  -> quantization-independent worker driver
  -> InferenceSession
  -> Qwen3 SQ8 または Qwen3.5 AQ4 resident model runtime
  -> ullm-runtime-sys
  -> C++20 / HIP / HIPRTC / CK
  -> R9700 (gfx1201)
```

- `crates/ullm-engine`: loader、scheduler、decoder、AQ4/SQ8、manifest、session、worker。
- `crates/ullm-runtime-sys`: Rust FFI、CPU/HIP runtime、gfx1201 CK feature。
- `crates/ullm-quant`: streaming/chunked AQ4 package conversion。
- `services/openai-gateway`: FastAPI、tokenizer、single-request admission、SSE、cancel、worker supervision。
- `deploy/`: systemd、nftables、served-model profiles、OpenWebUI image/compose/browser gates。
- `tools/`、`tests/`、`benchmarks/`: artifact、oracle、promotion/release validator、実測証拠。

## 成熟度

1. Qwen3-14B SQ8_0 / R9700の製品vertical sliceはfull release evidenceまで完成している。
2. Qwen3.5-9B AQ4_0は常駐session、manifest配備、promotion receipt、OpenWebUI smokeまで到達した。直近の主開発対象はこちらである。
3. engine全体はまだ汎用完成ではない。単一active request、待ち行列・batchなし、context 4096、text Chat Completionsのみである。tools、structured output保証、multimodal、embeddings、Responses API、request stop string、自動履歴切詰め、TLS、multi-tenant authは未対応である。
4. AQ4 legacy compatibility child経路と移行aliasが残る。計画P8の完成監査・削除条件は未完了である。
5. repository CI、release tag、十分なroot READMEがない。local evidence harnessは強いが、入口文書とremote共有状態は弱い。

## 開発環境

- Ubuntu 24.04.4 LTS、kernel 6.17.0-35-generic。
- Threadripper PRO 3995WX、64 core / 128 thread、1 NUMA node。
- OS認識memoryは109GiB、swap 8GiB。AGENTS.mdの16GB x 8とは一致しない。
- GPUはV620/gfx1030 x2とR9700 class/gfx1201 x1。
- ROCm 7.2.1、HIP 7.2.53211、AMD clang 22.0.0git。
- Rust/Cargo 1.96.0、Edition 2024、Python 3.12.3、uv 0.11.25、CMake 3.28.3、Ninja 1.11.1。
- repositoryは約52GiBで、主に`build/`約36GiB、`target/`約13GiB、`reference-src/`約2.6GiBである。OOM回避のためGPU検証と大規模buildの並列実行は避ける。
- root READMEは3行だけで、実際の開発・配備入口は`deploy/README.md`、gateway README、`docs/plans/`、`docs/specs/`、journalに分散している。

## 次の行動

1. 現在の停止が意図したものかを確認し、必要ならAQ4 manifest serviceとOpenWebUIを再起動してreadinessを再確認する。
2. `inference-api-quantization-separation-plan-v0.1.md`のP8として、legacy AQ4 child経路・一時alias・量子化名依存が残る箇所を監査し、削除可否を決める。
3. `origin/main`より39 commit先行しているため、公開単位を整理してpush/tag方針を決める。
4. root `README.md`を現行のbuild、test、manifest activation、製品scopeへの入口として更新する。
5. 次の性能作業ではbatchingを自動開始せず、AQ4/SQ8それぞれのprefill、device logits/sampling、F32 fallback未実装箇所を測定して優先順位を決める。

