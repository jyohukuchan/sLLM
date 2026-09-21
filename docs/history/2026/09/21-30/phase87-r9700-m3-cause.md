# R9700 FP8 M3並列化でMTPが遅くなる原因

2026-09-21〜22のユーザー指示に対応。対象はR9700でFP8 GDN QKV/ZのM3をforkしたときの
MTPあり35.6229→22.0052 token/sという退行。FP8 M1のみの候補は35.6223で同等だった。
**主因は複数queueでgraphを実行する際の同期・dispatchコスト**と確認した。
分岐graphを保ってqueue数だけ1に制限すると35.6604 token/sへ復帰した。
本番コード・採用範囲は変更せず、前の検証のfrozen binaryで調査した。

## 計算量ではなくgraph実行内の退行

既存 `a1/b1/b2` reportを比較すると、全候補で受理75/105（71.43%）、proposal block 53、
graph replay 52、discarded replay 0、native nodes 1325／kernel nodes 1308が同じ。
kernel選択とdispatch数、生成token列も一致している。

直列decode中央値3565.120 msに対しforkは5771.364 ms、**+2206.244 ms/request、+42.43 ms/replay**。
startup差は約4.8 msだけで、request setupやMTP受理率の低下では説明できない。

## トレースによる内訳

同じQwen3.8 NVFP4 revision `57926baca9a82b4d6906b43f2750d55315f5b10f`、
MXFP8 E4 KV、coding8192/128、固定sampling seed123、MTP幅2、1 warmup＋1 measured。
R9700 `GPU-a8e9ddefa2d60f55`、gfx1201、ROCm 7.14、scheduler=1。
rocprofv3のkernel/hip/marker traceを採取し、ROCTXのmeasured decode区間を比較した。
計測負荷を含むため、以下の絶対速度は通常性能の代用にしない。

| 区間 | 直列 | FP8 M3 fork | 増分 |
| --- | ---: | ---: | ---: |
| decode全体 | 3794.363 ms | 6107.112 ms | 2312.749 ms |
| GPU busy（区間の和集合） | 3434.889 ms | 4163.275 ms | 728.386 ms |
| カーネル間などの空白 | 359.474 ms | 1943.837 ms | 1584.363 ms |

増分の**68.51%がGPU空白**。kernel時間の単純合計は並列区間を重複計上するため用いない。
両方70313 trace dispatchで全run N0一致。52 graph replay（各1325 trace entries）を抽出した。
memcpy/memsetの内部処理もtrace entriesになるため、native kernel nodes 1308とは区別する。

48組×52回=2496 FP8ペアの中央値:

| 指標 | 直列 | fork |
| --- | ---: | ---: |
| QKV kernel | 130.120 µs | 211.241 µs |
| Z kernel | 65.201 µs | 194.121 µs |
| ペア開始〜両方完了 | 199.942 µs | 217.202 µs |
| 二つの実行の重なり | 0 µs | 187.541 µs |
| ペア完了→次kernel | 4.520 µs | 70.320 µs |
| 後続直列kernel間隔 | 4.440 µs | 24.640 µs |

forkでも実際に同時実行されている。個々のFP8 kernelは遅くなるが重なりがあるため、
ペア全体の悪化は約17 µsにとどまる。一方、合流と後続直列処理の間隔が増える。
小さなelementwise/quantize等も数µsから約20 µsになる。graph先頭の非fork部分にも影響があり、
「FP8二つがGPU資源を取り合う」だけではgraph全体の退行を説明できない。

直列のgraph traceはqueue 4に68900 entries。forkはqueue 4に66404、queue 1に2496。
Zだけが別queueへ移り、残りの計算の数・種類は変わっていない。

## ROCm側の経路と対照実験

導入版のrevision `2b22ab0195cc1461cd9abf3b969e9dd7c10af350` を参照した。
[graph executor](https://github.com/ROCm/rocm-systems/blob/2b22ab0195cc1461cd9abf3b969e9dd7c10af350/projects/clr/hipamd/src/hip_graph_internal.cpp) は
複数streamへの割当、segment間のcompletion signal、barrier packetとpatch list、
launch時のsignal取得・batch dispatch・leaf待ちを構築する（概ね398〜520、1051〜1128、1703〜1826行）。
classic経路にもstream間のevent wait listがある（2030〜2108行）。外部コードの閲覧のみで、移植・コピーなし。

プロファイラなし、同じ8192/128、1 warmup＋3 measuredでscheduler=0へ切り替えても、
直列35.1229／fork22.0324 token/s。全run N0一致。forkのmeasuredは19.2318／22.2235／22.0324で、
最初のsampleはprefillも長かったため、そのばらつきも記録した。classic設定でも退行は解消しない。

### 同じ分岐graphを1 queueへ制限

`DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1` を維持し、同じfork binaryに
`DEBUG_HIP_FORCE_GRAPH_QUEUES=1` だけを追加した。これはgraph stream数の上限で、
数学的な演算やcaptureの依存辺を変更しない。
通常8192/128、1 warmup＋3 measuredの結果は**35.660436 token/s**（MAD 0.018614）、
decode中央値3561.370 ms。measured 35.6418／35.6877／35.6604、全run N0、replay52回。
`single-queue-summary.json` と `fork-single-queue/execution.json` に保存した。

直列35.6229→複数queue fork22.0052→同じforkの1 queue制限35.6604と回復したため、
大幅退行の主因を複数queue graph実行に絞れる。FP8演算そのものの追加やMTP受理率ではなく、
複数queueの同期を避けることで回復する待ち時間である。
ただし、各barrierのcache fence・command processor・signal pollingの物理的な内訳までは測定していない。
1 queue設定は並列実行の利得もなくすので、FP8 M3並列化を高速化して採用できたという結果ではない。
本番は既存の直列を維持する。

## 計測上の制約

rocprofの標準queue interpositionではfork試行がwarmup開始後に同じsignal値1/2を待ち続けた。
SIGTERMで終了しなかったため、停止した計測childだけをSIGKILLし、runnerの復元処理完了を確認した。
この試行はfailedで保存し、通常実行の遅延や正しさの証拠にしない。
`ROCPROFILER_QUEUE_INTERPOSITION=0` で両方を取り直した結果が上表。
最初のanalysis呼出しのmarker指定漏れと、既存helperのDAG入力不足による解析失敗も保持し、
明示的なROCTX marker指定と必要な二つの解析ツールだけで再解析した。

生成物: `.local-artifacts/phase87/r9700-m3-cause/`。
主要証拠は `profile-comparison.json`、`timing-attribution.json`、`replay-stats.json`、
`classic-comparison.json`、各profile/report/execution。binary hashは直列
`dc11e7555a0facd3d8d4e332298c2bcd3e0a25513ed7c494a528595974b00f2c`、fork
`a620073672d850eab28cc536bbce0e98feb30d16f4758d9e108cfa5be23b1e50`。

前の検証: [R9700 FP8単独の採否](phase87-r9700-fp8-fork.md)

## 完了確認

完了した全GPU実行でservice/performance levelの復元とservice file hashの一致を確認。
本番sourceのSHAは調査前と同じ `7cb87884b44959f815934e6f1fbf4552988af645fb26a4157ba554a1022b6180`。
Markdown linkとdiff検査を実施。新しい本番コード・環境既定値の変更、commit/pushはない。

計画: [M3退行原因の調査](../../../../plans/archive/2026/09/21-30/phase87-r9700-m3-cause.md)
