# Phase 16 KV cache FP8/NVFP4履歴

## 2026-08-16: 実装とcanonical RDNA closeout

### encoding、state owner、ABI

- `KvCacheEncoding`へFP16、`kv-fp8-v1`、`kv-nvfp4-v1`を追加し、scalar `DType`とscale/packingを表す
  `Encoding`を分離した。FP8はtoken/head単位のE4M3FN valueと独立FP32 scale、NVFP4はlow-nibble-firstの
  E2M1 value、head dimension方向block-16のE4M3FN scale、token/head単位のFP32 outer scaleである。
- legacy `sllm_kv_state_create_info_t`/`sllm_kv_state_create`をexact FP16 ABIとして維持し、additiveな
  `sllm_kv_state_create_info_v2_t`/`sllm_kv_state_create_v2`でdtype、encoding、block、scale dtypeを指定する。
  C/C++/Rust layout probeへ新構造体と定数を追加し、binding parityをPASSした。
- opaque native stateがK/V value、block scale、outer scale planeを所有する。VMM `virtual-contiguous`と
  `contiguous-resident`の両providerでchecked byte計算、grow、query、releaseを実装した。appendは新規BF16 tokenだけを
  量子化し、K/Vと必要scale planeの全処理が完了するまでpublished length/generationを進めない。
- causal attention provider ID 3 `causal_attention.online_softmax_gqa.packed_kv.v3`を追加した。FP8/NVFP4 valueと
  scaleをkernel内で直接loadし、request全体のFP16/BF16 mirrorやCPU/別encoding fallbackを作らない。
- FP16専用のprivate evidence readback v1が低bit stateをFP16としてover-readし得る監査findingを修正した。低bitでは
  `SLLM_STATUS_UNSUPPORTED_ENCODING`を返し、数値証拠はpacked attentionの独立oracle比較から取得する。

### 独立oracle、境界、memory

- sLLM実装をimportしないNumPy oracleでE4M3FN/E2M1 code、RNE、saturation、zero/nonfinite、block tail、padding、
  head dimension `255/256/257`、block `15/16/17`、query `1/3/7/37`、token `255/256/257`と
  `1023/1024/1025`を検証した。quantization 12、attention 8、padding 6、nonfinite 2 caseをPASSした。FP8はNaNを
  E4M3FN NaN、Infを最大有限値へ写し、NVFP4はNaNをcanonical zero、Infを有限値由来のrow scaleで表現可能な上限へ飽和する。
- exact V620 `gfx1030`のpublic Rust execution pathでFP8/NVFP4各17 caseをPASSした。prefill M `1/3/17/37/255/256/257`、
  decode prefix `3/255/256/257`、KV `1023/1024/1025/8193`をscalar oracleと照合し、全caseでnumerical match、
  packed provider metadata、fallback false、cleanup 0だった。追加したquery NaN/value +Infも同じ独立oracleと一致した。
- logical KまたはV 1 planeはFP16比でFP8 `49.21875%`、NVFP4 `71.09375%`削減する。V620の2 MiB VMM granularityを
  含むKV=8193の実commitは、FP16 `18,874,368` byteに対しFP8 `12,582,912` byte（`33.33%`削減）、NVFP4
  `10,485,760` byte（`44.44%`削減）だった。短contextでは独立scale planeの最小pageにより理論削減が物理削減へ
  直結しないため、logical値だけをVRAM削減根拠にしない。

### full model、採用判断、service

- Qwen3.5-4B FP8 weight modelでFP8 KVを使ったteacher-forced 3 promptはtop-1 `3/3`、最大KLD
  `0.016834498446670534`、nonfiniteなしで`0.05` budget内だった。Qwen3.5-2B NVFP4 weight modelではNVFP4 KVが
  最大KLD約`0.3090`へ悪化したため棄却し、FP8 KVは約`0.2619540`で既存weight-only約`0.2637523`と同等だった。
- Qwen sidecarのweight encodingはKV recipeを指定していない。4B FP8 weightのV620 short-oddを3 warmup＋10 measuredで
  比較すると、FP8 KVとFP16 KVのprefill/decode/E2E差はnoise envelope内だった一方、greedy token列は一致しなかった。
  このためweight dtypeから低bit KVを推測する自動連動を棄却し、通常Qwen loaderはFP16 KVを維持する。内部の
  `*_with_kv_cache_encoding` builderだけが、検証済みmodel metadataを持つPhase 16F adapter/evidenceから選択できる。
  起動flag、確認、通常警告は追加していない。
- final Qwen BF16/FP16-KV service candidateをV620で実行し、OpenAI公式client 2.44.0のnon-stream/SSE、固定request、
  連続request、capacity `1023/1024/1025`、disconnect/cancel、直後のrecoveryをPASSした。全dispatch HIP、fallback false、
  shutdown時のcurrent/request/workspace byteとretryable/durable quarantineは0だった。
- host public runtime CTest 3/3、Rust workspace tests、exact `gfx1030`/`gfx1201`/`gfx942` compile/linkを最終sourceでPASSした。
  `gfx942`は利用可能なVMがないためcompile-onlyでありGPU PASSへ一般化しない。
- local R9700をphysical HIP index 2だけ可視化し、runnerのlogical device 0へexact `gfx1201` artifactを対応づけた。
  FP8/NVFP4各17 caseが独立oracle、packed provider metadata、fallback false、cleanup 0を含めてPASSした。KV=8193の
  committed byteはFP16 `18,874,368`に対してFP8 `12,582,912`（`33.33%`削減）、NVFP4 `10,485,760`
  （`44.44%`削減）でV620と一致した。終了後のR9700使用率とVRAMは0、温度はedge/hotspot/memory
  `34/34/32`℃だった。複数GPUを同時可視化したglobal physical indexでtarget別artifactを実行する契約は採用せず、
  stable device mappingで一台だけを可視化してlogical device 0を使う既存契約を維持する。

## 2026-08-16: 詳細計画作成

- 残タスクの依存関係を見直し、FP8 KVを先に、NVFP4 KVを次に実装する順序を維持した。
- Phase 16Fのprimary artifact `unsloth/gemma-4-12b-it-NVFP4`がmixed recipeでFP8 KVを要求するため、
  first-class FP4 full-model integrationよりPhase 16を先に完了する順序へ固定した。
- Phase 6のopaque KV、VMM virtual-contiguous、Phase 11のcontiguous-resident、Phase 13のtransactionを維持し、
  value/scale planeだけをversioned encodingとして追加する計画とした。
- append時の一度だけの量子化、attentionからの直接消費、全cache FP16/BF16 mirror禁止、K/V atomic publication、
  cancel/recovery、quality/memory/performanceの受入条件を固定した。
- canonical runtime matrixはexact `gfx1030`/`gfx1201`とし、利用可能な実機がない`gfx942`はcompile/host contractを
  超えてPASSとしない。本時点ではsource、ABI、kernel、model artifactを変更していない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase16-kv-cache-fp8-nvfp4.md)
