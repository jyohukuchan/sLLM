# Phase 29実行履歴: GDN useful-workgroup並列化最適化

> 状態: 完了（N1 shared candidate採用）
> 実行日: 2026-08-18

## 結論

Phase 29固有のprimary metricをfull-modelではなく`GDN family device ns / committed decode step`のp50へ固定し、
Phase 28 production providerをbaselineとしてuseful-wave reductionを評価した。

Q/K L2 normとoutput RMSNormの128要素逐次和をwave32 reductionへ置き換えたfull candidateは、B0/B1/B2の全patternで
V620を2.15〜2.20%、R9700を8.10〜9.21%短縮した。全pattern非悪化かつ任意pattern 5%以上というGDN性能条件は満たした。

一方、演算順変更はmodel-free oracleと16-token生成では許容範囲内だったが、output 128では6 target/pattern中5 patternで
baselineとtoken列が分岐した。同一binary repeatは完全再現し、baseline/candidate間だけが分岐したため、process揺れではなく
reduction順変更による長期的な数値増幅と判定した。初回closeoutではtoken完全一致をhard conditionとしてcandidateを棄却した。

その後、2026-08-18のユーザー明示指示で数値変更規則を改訂した。今回の対象は全て非負の二乗値で、128項逐次和の依存深さ127を
固定wave treeの概ね8へ減らし、real-number semanticとBF16 RNE stageを維持したまま標準的な丸め誤差boundを縮小する。
差の原因が説明可能で同一provider repeatも再現するためN1へ分類し、token分岐を許容してshared candidateを再採用した。
target split、追加scratch、追加launch、ABI変更はない。

## 計測contract

- target: V620 exact `gfx1030` UUID `GPU-76a08c022586fed6`、R9700 exact `gfx1201` UUID `GPU-a8e9ddefa2d60f55`。
- model: Qwen3.5-4B dense BF16固定GGUF/derived lock。
- pattern: B0 prompt 17、B1 prompt 28、B2 prompt 255、各output 16。
- protocol: 14 request × 16 Argmaxから各request先頭を除く210 committed decode step。
- process: 各target/patternで`baseline-candidate-candidate-baseline-candidate-baseline`、各variant 3独立process。
- primary: committed step内の全GDN family kernel duration合計のprocess p50。full-model wall値はdiagnosticのみ。
- interference: Qwen service停止、UUIDで一GPUだけ可視化し、GPUを同時実行せず、run前後のforeign processなしを確認した。

## 正式GDN結果

| target | pattern | baseline p50 ns | candidate p50 ns | 改善 | baseline端点drift |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1030 | B0 | 1,359,040.0 | 1,329,098.5 | 2.20% | 0.44% |
| gfx1030 | B1 | 1,360,983.5 | 1,331,744.0 | 2.15% | 0.06% |
| gfx1030 | B2 | 1,363,004.0 | 1,333,446.5 | 2.17% | 0.06% |
| gfx1201 | B0 | 625,386.5 | 574,728.0 | 8.10% | 0.66% |
| gfx1201 | B1 | 619,072.0 | 567,607.0 | 8.31% | 0.10% |
| gfx1201 | B2 | 616,250.0 | 559,465.5 | 9.21% | 1.33% |

全baseline端点driftは2%以内だった。candidateはlaunch数、grid 4096 threads、workgroup 128、scratch 0を維持し、
各decode stepで24 GDN recurrent dispatchを計上した。

## Correctnessとbounded variant

- model-free G0はtoken count 1/3/17で両GPU PASS、fallbackなし、cleanup 0だった。
- output 16の全36 formal runはbaseline/candidateでtoken recordが一致した。
- output 128のfull candidateはgfx1030/B1だけ一致し、gfx1030 B0/B2は105/20 token目、gfx1201 B0/B1/B2は
  111/112/108 token目から分岐した。
- gfx1201/B0でbaseline/candidateを各1回repeatし、同一binary内は完全一致、cross variantは再び111 token目から分岐した。
- Q/K wave-onlyはgfx1201/B1を582,426.5 ns、5.92%短縮したが、gfx1201/B0 output 128が105 token目で分岐した。
- output wave-onlyはgfx1201/B1が608,885.5 ns、1.65%改善に留まり、5% thresholdを満たさなかった。
- 既探索の16/64/128/256単純workgroup分割はすべてPhase 28 providerより遅く、再採用しなかった。

## Full-model診断

profilerなしoutput 128のdecode tokens/s変化はV620がB0 -0.03%、B1 +0.24%、B2 +0.32%、R9700が
B0 +0.06%、B1 +0.40%、B2 +0.09%だった。GDN時間の短縮がfull-modelではほぼ相殺されることを確認したが、
Phase 29の性能採否には使用していない。token分岐は[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)へ記録し、
解析的に誤差を低減するN1変更のため数値gateを自動承認した。

## Evidenceとproduction状態

- bounded summary: `ci/matrix/phase29-gdn-device-summary-v1.json`
- schema: `ci/schema/phase29-gdn-device-summary-v1.schema.json`
- trace aggregator: `ci/tools/phase29_gdn_device.py`
- contract test: `ci/tests/test_phase29_gdn_device_summary.py`
- production GDN source SHA-256: `62b5d6caab9e06044e29c5f043046a65eace243faafa034aca1b3f2ce8eb3dc6`
- full candidate binary SHA-256: gfx1030 `040cb7e2103696cfb3c1b66343ae8791d734b0f743c2539becd95f86d1518b83`、
  gfx1201 `dd25b7cafa8712a94faf2eb983e9cc2cfd9fb9df18c5f66c40d5f02417c32b6b`。
- raw rocprof trace、model、binary、生成全文は追跡していない。

[Phase 29計画](../../../../plans/archive/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)
[Phase 29 bounded summary](../../../../../ci/matrix/phase29-gdn-device-summary-v1.json)
[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../plans/main-plan.md)
