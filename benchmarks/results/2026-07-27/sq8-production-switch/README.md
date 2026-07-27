# 依頼CF: SQ8_0 本番切替の品質再判定

## 結論

`SQ8_0` Qwen3-14B の `gqa_grouped_split` / `split_tile: 20` は**昇格しない**。
512 token 上限で direct と grouped を同じ 8 ケース（コード生成、日本語 multi-turn
を含む）で実行した結果、grouped だけに JavaScript の説明の事実誤認が残った。
従って、BQ の quality hold は短い token 上限だけによる交絡とは結論できない。

`javascript_debug_extended` の grouped 出力は、修正コード自体は
`values.filter(value => isFinite(value))` で期待出力も `2` と正しい一方、説明で
「`NaN`, `Infinity`, `0` はすべて falsy」および「そのため `Infinity` と `NaN` を
誤って含める」と記した。JavaScript では `Infinity` は truthy、`NaN` は falsy であり、
元の `filter(Boolean)` は `NaN` を含めない。この誤りは direct 側にはなく、同じ上限の
direct 出力は `Infinity` が truthy であることを正しく説明した。

自動 blocking finding は両側とも空だったが、軽量ポリシーに従い、数値閾値でこの
生成文の品質差を打ち消してはいない。raw text は
[`quality/direct/capture/cases/javascript_debug_extended.json`](quality/direct/capture/cases/javascript_debug_extended.json)
と
[`quality/gqa-grouped-tile20/capture/cases/javascript_debug_extended.json`](quality/gqa-grouped-tile20/capture/cases/javascript_debug_extended.json)
に保存した。

## 実行条件

- source base: `c5e7dc16c702e8bdada7da001ee8bc15f728b088`（開始時 main）。
  祖先確認済み: BH `3d914439`、BR `b46cc8ac`、BK `17a531a2`、CC `1c660223`、
  BX `8412e170`。
- worker build: sealed release worker SHA-256
  `550d6bfe36b403de86386af3ca4d469fc385a546d59bcddcfcb7db7415c5d8fc`。
  quality capture はこの single build を使用した。品質 fail のため `/opt/ullm` への
  release 配置は行っていない。
- device: R9700 (`gfx1201`, `HIP_VISIBLE_DEVICES=1`) のみ。KV は F32 default のまま。
  FP16 / S1E4M3 selector は設定していない。
- quality suite: 8 cases, code 2 件と日本語 multi-turn 1 件は `max_completion_tokens=512`。
  capture の両 manifest は validator を通過し、capture は各 8/8・自動 blocking なし。

## 実測（昇格候補と同じ F32 / tile-20 selector）

| workload | measured | comparison |
| --- | ---: | --- |
| decode, N=1024, 4 warmup + 16 steps x 5 | 27.394198 tok/s | llama.cpp 30.468075 の 89.91%; BH 27.378731 と同等、27.5 前後 |
| prefill N=128, adaptive M=128 | 426.744 tok/s | 一回だけの同期 sample。期待 887 前後を再現できず、5 warmup/median protocol ではないため採否根拠には使わない |
| prefill N=4095, adaptive M=2048 + 2047 real-token tail | 126.761 tok/s | CC 126.8 前後を再現; llama.cpp 1008.683 の 12.57% |

prefill の raw `synchronized_seconds` は N=128 で 0.299945524 s、N=4095 で
8.734469151 + 23.570326905 s。profiler range time は throughput として使用していない。
N=128 の実行 unit が実際に M=128 であることは raw result にあるが、この一発の冷えた
sample は CC の five-sample median と比較可能ではない。品質 fail により、追加の
本番停止を伴う再測定は行わなかった。

## 本番状態と rollback

昇格・artifact copy・active manifest 書換えは**未実施**。現在の active manifest は
`a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd` のままで、
`AQ4_0` worker
`/opt/ullm/aq4-overnight-consolidation-v0.1/releases/aq4-consolidated-840a1c7a-5a274733/ullm-aq4-worker`
が rollback 先（かつ現在の本番）である。

## 運用記録

CE/MoE worker は実行しておらず、lock は本番 gateway だけが保持していることを確認してから
開始した。隔離窓は 2 回（最初は suite schema preflight が GPU request 前に失敗、次が実測）
だった。第二窓で cleanup の start が隔離 gateway の lock 解放前に走り、AQ4 は
`WorkerBusy` で `NRestarts=3`、start-limit となった。lock 解放後に `reset-failed` と 1 回の
start で AQ4 を復旧した。最終状態は `ActiveState=active`, `NRestarts=0`、active manifest
SHA は上記のまま。これは SQ8 promotion ではない。
