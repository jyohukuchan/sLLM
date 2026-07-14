# P3 AQ4 prefill candidate shortlist v0.1

## 前回の要点

- P2の24-row fidelity capture hardeningは固定split、source provenance、active/package/device/guard identity、strict metrics validationまで進んだが、GPU captureは未実行である。
- 既存のP1 bottleneck auditはM=128 cold prefillのwall timeを記録する一方、operation別wall timeを0としており、候補順位はまだ仮説である。
- family-exclusive profiler（commit `28ec343`）は、rocprofv3のGPU interval unionとfamily attributionを行う診断器だが、profile artifactを性能証拠へ転用しない契約である。

## 今回の変更点

AQ4 native prefill hot pathを読み取り、実装範囲を分離しやすく、CPU oracleとfull-model gateへ落とし込みやすい候補を3つに絞った。候補はP2 R9700 profileで再順位付けするまで未確定とする。

| 候補 | hot pathと期待効果 | 主な変更範囲 | 正しさ・OOMリスク | 測定とfallback |
|---|---|---|---|---|
| A. sequence出力のdevice-to-device copy削減 | `dispatch_prefill_chunk_for_phase`が各layerでsequence workspaceの出力を次のping/pongへコピーする。layer outputを安全に次bufferへ直接書ければ、M×hiddenのlayer間copyと関連launchを削減できる | `crates/ullm-engine/src/qwen35_aq4_model_runtime.rs`、`crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs`。runtime ABIは変更しない第一案 | ping/pong alias、最終row保持、request failure後のresetが壊れやすい。workspace geometry・buffer bytesの事前検証を必須とし、条件外は既存copy pathへ戻す | profilerでprefill phaseのD2D bytes/launch、family union、p50/p95を比較。CPU sequence oracle、hidden/logit/greedy/top-k、chunk境界、cancel/resetを先に確認 |
| B. QKV projectionのgrouped/batched dispatch | self-attention sequence pathのQ/K/V projectionは個別`matvec_batch_for_phase`で3回発行される。重み・scale metadataを再利用するgrouped dispatchでlaunchと入力readを削減できる可能性がある | `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs`、AQ4 operation registry、必要なら新規runtime kernel/test。共有ABIはdescriptor凍結後のみ | Q/K/Vの異なる行幅、tail、scale residency、BM8 shape条件を誤ると数値差または未初期化出力になる。非対応shapeは既存3-call pathへfail-closed fallback | M=1/8/16/32/64/128とQ/K/V実shapeでimplementation/fallback、launch、workspace、finite、hidden/logit差を収集。grouped path失敗時は既存個別projectionへ戻す |
| C. paged KV writer/readerのmetadata・sync削減 | self-attention sequence pathでQK norm/RoPE後にpaged KV write、causal GQA readを実行する。metadata検証や不要syncをstream内で再利用すれば、validation overheadを減らせる可能性がある | `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs` とpaged chunk dispatch。registry/sessionの統合は後段 | writer/readerのplanned workspaceが各約1.61GBで、block table・cache position・KV stateの破損やOOMが致命的。writer後reader失敗時のrequest poison/resetが必須 | context 128/512/1011/1339/2048/3584、M grid、short decodeを含め、KV/cache/positionとpeak VRAMを比較。条件外はcanonical paged pathへ戻し、D2Hなしを成功扱いしない |

## 判断

最有力の第一案はA（sequence output copy削減）である。理由は、現行ループに明示的なlayer間copyが存在し、runtime ABIやkernel descriptorを変えずに候補化でき、失敗時の既存copy fallbackを維持できるためである。ただしP1のoperation wall timeが未計測なので、P2 profileでcopy bytes・GPU interval・全体prefill p50/p95が支配的でない場合は採用しない。Bは次点、Cは大きなworkspaceとstateリスクのため最後に回す。

## 最有力Aのpatch/test plan

1. `PackageLinearAttnSequenceWorkspace` と `PackageSelfAttnSequenceWorkspace` に、既存output bufferまたは呼出し側ping/pongを検証付きで選択する小さな内部経路を追加する。buffer aliasは同一layerの入力と重ならないこと、bytesが`M*hidden*sizeof(f32)`以上であることを拒否する。
2. `dispatch_prefill_chunk_for_phase`で直接出力経路を選び、最終layerのlast-row retainを直接出力bufferから行う。条件外、copy API不成立、途中operation失敗時は現行のworkspace output→ping/pong copyへ戻し、request stateをpoisonしてresetする。
3. CPU/source oracleでM=1/2/8/16/32/64/128、prompt 128/512/1011/1024/1339/2048/3584、chunk境界、cancel/resetを比較する。hidden/logit全要素、greedy、ordered top-k、KV/recurrent/conv stateに差がないことを確認する。
4. R9700 component/full-model gateで、同一active identity、実M、D2D bytes、launch/sync、workspace、peak VRAM、fallbackをraw evidenceへ保存する。2 warmup + 10 measuredの通常throughputとfamily-exclusive profileを分離する。
5. direct worker、production_server、OpenWebUIへ昇格する前に、P4/P5のtoken/state/reset/EOS/length、OOM、decode非回帰、identity hash gateを通す。差分が測定誤差を超えない場合は候補を破棄する。

## 次の行動

- 最新24-row captureの固定commitを独立validatorで再確認する。
- P2 R9700 profileで候補AのD2D copyとfamily intervalをまず測り、支配性が確認できた場合だけCPU patchを開始する。
- 支配性がなければB、Cの順に同じprofile契約で再順位付けし、候補を同時に二つ以上実装しない。
