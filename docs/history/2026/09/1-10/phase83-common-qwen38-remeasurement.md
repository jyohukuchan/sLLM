# Phase83.5共通化後のQwen3.8 27B再計測（2026-09-10）

現行 `c27e346d5f3332d6e8ce794b7913b87c42455c99` をPhase83.5最終版と比較した。今回の固定条件では速度低下・出力変化・実行経路の退行・資源解放の問題は観測しなかった。エンジンの修正は不要だった。

## 条件

Qwen3.8 27B NVFP4、MXFP8 E4 KV、8192 input／128 output、batch=1、seed=123、固定temperature=1.0／top_p=0.95／top_k=20。MTP幅2のon/offを各GPUで別processにより順次実行し、1 warmup＋3 measuredの中央値を比較した。chunk=2048、state capacity=8320。入力・model revision／hashは旧版と一致し、ビルドとGPU計測は重ねていない。

V620はGPU-76a08c022586fed6（gfx1030）、R9700はGPU-a8e9ddefa2d60f55（gfx1201）。現行の両target binaryは481 source hashがbuild前後・計測対象commitと一致することを確認した。旧版のsource対応はPhase83.5最終記録の `865a9e9e2201df2b8996007a12e64c739a009399`。

## 結果

単位tok/s。増加を新しい最適化効果と断定せず、今回の観測差として扱う。

| GPU | MTP | 旧prefill → 現行 | 旧decode → 現行 | decode差 |
| --- | --- | --- | --- | --- |
| V620 | off | 224.891 → 225.717 | 14.325 → 14.332 | +0.050% |
| V620 | on | 216.571 → 216.593 | 25.409 → 25.423 | +0.058% |
| R9700 | off | 550.149 → 551.210 | 19.083 → 19.183 | +0.524% |
| R9700 | on | 541.402 → 541.969 | 34.541 → 35.110 | +1.647% |

| GPU | MTP | 現行prefill MAD | 現行decode MAD | 現行TTFT秒 | 現行E2E秒 |
| --- | --- | --- | --- | --- | --- |
| V620 | off | 0.631 | 0.006 | 36.314 | 45.185 |
| V620 | on | 0.390 | 0.022 | 37.855 | 42.850 |
| R9700 | off | 1.236 | 0.019 | 14.881 | 21.547 |
| R9700 | on | 2.805 | 0.058 | 15.143 | 18.819 |

R9700のMTPあり初回warmup decodeは約24.683 tok/s（旧約24.55）、正式中央値は35.110 tok/s。warm値を初回性能として説明しない。prefillにはMTP prefix準備、decodeにはdraft／verify／棄却・replay／samplingを含める。出力128 tokenのうち最初の選択はprefill内で行うため、decode throughputの分子は127 transitionである。

## 正しさ・状態

- 4条件×4回の全16 runが実HIP、CPU fallbackなし、dispatch非zero、terminal logit非finite 0。各runのtoken ID配列・本文hash・audit全体・MTP統計は同条件の旧版と一致した。
- MTPありは全runでV620が提案106／採用74／棄却32（69.8%）、R9700が100／78／22（78.0%）。MTPなしとの出力一致を判定条件にはしていない。
- 各要求終了後のrequest-state／workspaceは0、全4processのresident drop後current allocation、retryable cleanup、durable quarantineも0。V620終了後VRAMは17,215,488 bytes、GPU busy 0。
- R9700既存サービスは計測中だけ停止し、元のunit／run script／server binaryのhash不変、active復帰、healthz／readyz HTTP 200を確認した。

## 範囲・証拠

今回の再計測は単一coding fixture・seedでのdirect benchmark回帰確認であり、BF16 full-model品質同等性や全promptを証明しない。CLI/API・SSE・cancel/recoveryは今回再実行せず、[共通化時の検証](phase83-common-speculation.md)を別証拠として参照する。速度目標の再設定やMTP量子化は行っていない。

[比較の数値・hash・各run・解放記録](phase83-common-qwen38-remeasurement.json)を追跡し、raw結果と実行manifestは `.local-artifacts/phase83-5/common-recheck-{v620,r9700}-mtp-{on,off}-r1` に保存した。旧値は[Phase83.5最終証拠](phase83-5-closeout-evidence.json)を使用した。

作業計画: [共通化計画](../../../../plans/archive/2026/09/1-10/phase83-common-speculation.md)。この再計測は完了済み共通化の追加確認である。
