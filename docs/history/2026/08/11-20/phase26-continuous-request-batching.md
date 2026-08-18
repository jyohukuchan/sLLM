# Phase 26 continuous request batching history

## 2026-08-18: Phase 23結果と製品要件を受けた詳細計画

- ユーザーの明示指示により、Phase 23 shortlistの`P23-O3`をPhase 26へ割り当てた。
- `sLLM.md`は単一requestと複数requestのリクエストバッチ処理を製品目標とする。Phase 23のconcurrency=2はV620
  0.471/0.937 s、R9700 0.325/0.651 sでほぼ完全に直列化し、HTTP/SSE residualは約0.5〜0.6 msだった。
- current single FIFO worker、whole-generation blocking trait、backend state mutexを、waiting/running set、stepwise request state、
  immutable resident model、per-request execution owner、GPU `B>1` decodeへ分解する計画とした。
- primaryをQwen3.5-4B dense BF16 text、canonical `gfx1030`/`gfx1201`、OpenAI-compatible non-stream/SSE、
  concurrency `1,2,3,4,7,8`とした。MTP、multimodal、MoE/low-bit、Gemmaは安全なsingleton compatibility laneを許容する。
- requestごとのtoken/position、KV/GDN、sampler RNG、stop/usage、output、cancellation、error、cleanupを分離し、
  cross-request alias、unbounded memory、cancel propagationをcorrectness blockerとした。
- 提案primary performance基準はC1非悪化、C2 aggregate completion tokens/sが一targetで30%以上、他targetで20%以上改善である。
  `B>1` dispatch、fairness、tail latency、VRAM、backpressure isolationも採否へ含める。
- chunked prefill、GPU sampling、prefix cache、multi-GPU、TurboQuant、DeepSeek V4をPhase 26へ混ぜない。
- Phase 26はPhase 25の成功には依存せず、採用またはrollback後のstable source identityから開始する。Phase 25 providerは
  `B>1`で再計測し、`M=1`の選択を無条件に引き継がない。
- 本更新は計画のみである。source、public API、scheduler、GPU evidence、production defaultは変更していない。

## 2026-08-18: host contract、fresh C2 baseline、GPU candidate棄却

- Phase 25のnegative closeout後のcurrent sourceでserverをexact `gfx1030`/`gfx1201`向けに再buildし、13 input / 17 output、
  3 warmup + 10 measured、non-stream/SSEと同時2 requestを再取得した。全28 request/targetはHIP-only、fallbackなし、
  request/workspace cleanup 0だった。
- C2完了時刻はV620 0.457/0.908秒、R9700 0.327/0.646秒で、finish ratioは1.987/1.980だった。single HTTP中央値から
  算出したaggregate request/s差も+4.12%/-1.87%に留まり、current FIFO/whole-generation mutexの直列性を再確認した。
- bounded active admission、waiting/decode-ready、unique row map、compatibility class、round-robin、prefill挿入bound、backpressure、
  request-local finish/cancel/failを持つhost plannerを追加した。in-flight cancelはcompletionまでactive resourceを保持して結果を非公開にする。
  `C=1,2,3,4,7,8`を含む5 focused testとserver全32 testをPASSした。
  plannerはmodel/device resourceを所有せず、GPU backendへ未接続である。
- GPU接続監査では、現行`QwenExecutionCore`がrequest ownerごとに一つの`committed_length`、KV state map、linear/GDN state mapを持ち、
  `run_transition`のposition/state長もscalarであることを確認した。既存`decode_block(M>1)`はMTP用の同一request内連続tokenであり、
  独立request rowsではない。
- したがってhost row mapを現行`M>1`へ接続すると、異なるrequestが一つのcausal KV/GDN historyを共有してwrong token/stateを生む。
  正しい実装にはper-row position、独立KV/GDN binding、row-local transactional publicationをcore、HIP wrapper、public native ABI、
  kernel、production ownerへadditiveに通す必要がある。
- このstateful ABI拡張は固定したPhase 26見積りの1.5倍を超えるため、計画の停止・再計画条件を適用した。不完全なGPU pathや
  host-only同時workerをリクエストバッチ処理として採用せず、production scheduler/backend mutex/defaultは変更していない。
- 結論はPhase 26 candidate棄却である。host plannerだけをnonproduction infrastructureとして保持し、GPU `B>1`、throughput改善、
  continuous request batching成功は主張しない。次回はmulti-sequence KV/GDN ABIとtiny numerical oracleを独立見積りしてから接続する。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase26-continuous-request-batching.md)
[bounded summary](../../../../../ci/matrix/phase26-continuous-request-batching-summary-v1.json)
