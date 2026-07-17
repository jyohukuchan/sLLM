# Qwen3.5-9B AQ4 長文時のprefill/decode低下診断

日時: 2026-07-12

## 前回の要点

- Qwen3.5-9B AQ4 resident workerは、OpenWebUIで短文時に約70～80 tok/s、長文時に約40 tok/sのdecodeが観測された。
- 過去の最適化結果から、prefillが数千tok/sになっている期待があった。

## 今回の変更点

- 実装、Git履歴、既存benchmark、実会話時のsystemd lifecycle logを読み取り専用で照合した。
- コード、service、GPU状態は変更していない。追加のGPU負荷試験も実施していない。

## 結論

観測値は現実装と一致する。異常や設定漏れではなく、次の二つが原因である。

1. AQ4 residentのprefillはbatched/tiled prefillへ接続されておらず、32層を1 tokenずつdecode-likeに実行している。
2. 8個のself-attention層は各tokenで現在のKV cache全体を走査するため、contextが長いほどdecodeが遅くなる。

128-token単位のprefill progressは通知のcoalescingであり、計算幅128を意味しない。resident sessionが返す`execution_width`は常に1である。

## 実会話ログとの照合

同じ時間帯の二要求は次のとおりだった。

- 短文: prompt 14、completion 10。first-tokenからreleaseまでの9区間で約69.7 tok/s。
- 長文: prompt 1339、completion 256。prompt開始からfirst-tokenまで約22.51秒、約59.5 prompt tok/s。first-tokenからreleaseまでの255区間で約42.64 tok/s。

これはユーザー観測の70～80 tok/s対約40 tok/sと一致する。別のAQ4 OpenWebUI evidenceでもprompt 1011のprefillは63.638 tok/sだった。

長文requestのprogress eventは128 tokenごとに見えるが、内部では1 tokenずつ処理している。OpenWebUIは会話履歴全体をgatewayへ渡し、gatewayは全messagesをchat templateへ再適用する。AQ4 sessionはrequest終了時にKV/conv/recurrent stateをresetするため、turn間のprefix cache再利用もない。

## prefillが遅い実装上の理由

`Qwen35Aq4InferenceSession::prepare_advance`はpromptから次の1 tokenだけを選び、`Qwen35Aq4ModelRuntime::dispatch_token`を1回呼ぶ。model runtimeはembeddingから32層までをsingle-token stepで直列に流す。

Qwen3.5-9Bの32層は次の構成である。

- linear attention: 24層。recurrent step自体はcontext長非依存だが、prompt token数だけ逐次反復する。
- self-attention: layer 3,7,11,15,19,23,27,31の8層。各prompt tokenでもKV write後に`written_len`全体をpaged decode attentionで走査する。

したがって、projection/MLPは概ねprompt長に比例し、8 self-attention層のattention workはtokenwise prefill全体で概ね二次的に増える。full-model prefill用の`[tokens, hidden]` buffer、batched RMSNorm/MLP/projection、chunked causal attentionは現product pathにない。

## decodeがcontext長で落ちる理由

self-attention HIP kernelはquery headごとに`source_timestep < cache_len`を反復する。`cache_len`はprompt tokensと既生成tokensの合計である。

- 24 linear-attention層のrecurrent step、KV write、LM headは大きな固定コストだが、context長には比例しない。
- 8 self-attention層のpaged attentionだけがcontext全体を毎token読む。
- prompt 14付近では固定コストが支配的だが、prompt 1339付近ではKV走査コストが追加され、約70 tok/sから約43 tok/sへ低下する。

長い生成中もcache_lenが1 tokenずつ増えるため、response後半ほどさらに遅くなる。

## 「数千tok/s」の由来

数千tok/sの値は現uLLM AQ4 full-model resident prefillではない。

1. `3275 tok/s` at prompt 256、`3527 tok/s` at prompt 1024は、llama.cppのQwen3.5-9B UD-Q4_K_XL、R9700、batch 2048 / ubatch 512の結果である。同じ比較時点のuLLM AQ4 prefill平均は54.923 tok/sだった。
2. uLLMの`19063.596 tok/s`はAQ4 package由来ではあるが、layer 3の`self_attn.k_proj.weight`一投影だけを2 tokens同時に処理したcomponent microbenchmarkである。
3. synthetic attention componentの`850713 tok/s`もfull modelではない。各記録はfull package throughput判断へ使わないと明記されている。

batch AQ4 matvec APIとoffline mixed-request batch runnerは存在するが、`Qwen35Aq4InferenceSession`/resident model/OpenWebUI経路からは呼ばれていない。

## 根拠

- `uLLM-project/crates/ullm-engine/src/qwen35_aq4_session.rs:332-414`
- `uLLM-project/crates/ullm-engine/src/qwen35_aq4_model_runtime.rs:386-448,494-553`
- `uLLM-project/crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs:2032-2125`
- `uLLM-project/runtime/src/ullm_runtime_hiprtc_sources.inc:4913-5010`
- `uLLM-project/crates/ullm-engine/src/sq8_worker_protocol.rs:2024-2053`
- `uLLM-project/services/openai-gateway/src/ullm_openai_gateway/tokenizer.py:156-176`
- `journal/2026/07/07/aq-runtime-continued-improvements.md:97-128`
- `uLLM-project/benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1.md`
- `uLLM-project/benchmarks/results/2026-07-12/qwen35-9b-aq4-resident-openwebui-v0.1/summary.json`

## 次の行動

1. AQ4 residentへ専用prefill APIを追加し、prompt token chunkを`[M, hidden]`で全層へ流す。既存のrequest batchとtoken batchを混同しない。
2. まずembedding、RMSNorm、AQ4 projection、MLPをM=16/32/64/128でbatch化する。
3. 24 linear-attention層はprojection/MLP batchと、状態を順序どおり更新するscanを分離する。
4. 8 self-attention層はdecode用paged kernelのtoken loopをやめ、chunked causal/cached-prefix attentionと一括KV writeを接続する。
5. prompt 128/512/1024/2048、decode開始context 16/512/1024/2048/3584でcomponent timingとend-to-endを測る。
6. turn間prefix cachingはbatched prefillと別機能として扱い、正しいsession/KV ownership設計後に検討する。

