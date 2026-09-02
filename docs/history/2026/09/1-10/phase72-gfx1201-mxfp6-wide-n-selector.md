# Phase 72: gfx1201 MXFP6 wide-N selector検証

## 結論

2026-09-02にcanonical Radeon AI PRO R9700 exact `gfx1201`で、Phase 70 ID45
`matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4.v2`のN上限を32,768まで検証し、採用した。
selectorはmodel名を見ず、exact target、MXFP6 E3M2 W6A6、M/N/Kだけで選ぶ。従って非公式modelも同じ
operator contractへ一致すれば利用できる一方、model architecture全体の対応を自動的に保証しない。

## 実装

- kernel selectorとprepared low-precision providerの上限を`N<=16384`から`N<=32768`へ同期した。
- `N=32768`はID45、`N=32769`はID25 tiled16へ戻るhost／compile-time contractを追加した。
- evidence runnerへ広幅N matrix、強制指定なしのdefault provider、ID25 control、境界sample oracle、sampled row top-1を追加した。
- M>=17、K>=2048、K%32=0、N>=1024、exact gfx1201以外のselector条件は変更していない。

## operator検証

ROCm 7.14.0、HIP 7.14.60850、AMD clang 23、Code Object V6、wave32の単一exact gfx1201 binaryを使い、
各caseを1 warmup＋3 measuredで実行した。

| M | K | N | ID45中央値 | ID25中央値 | ID45速度比 | ID29中央値 | ID45速度比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 2,048 | 16,384 | 330,362 ns | 1,058,406 ns | 3.2038x | 1,060,087 ns | 3.2089x |
| 17 | 2,048 | 16,385 | 344,723 ns | 1,059,366 ns | 3.0731x | 1,051,646 ns | 3.0507x |
| 128 | 5,120 | 17,408 | 1,137,566 ns | 9,521,695 ns | 8.3702x | 15,027,528 ns | 13.2102x |
| 129 | 5,120 | 17,409 | 1,705,531 ns | 11,227,785 ns | 6.5832x | 16,169,735 ns | 9.4808x |
| 512 | 4,096 | 24,576 | 4,221,863 ns | 44,831,810 ns | 10.6190x | 70,636,055 ns | 16.7310x |
| 128 | 4,096 | 32,000 | 1,528,129 ns | 14,727,323 ns | 9.6375x | 22,714,495 ns | 14.8643x |
| 17 | 2,048 | 32,767 | 546,204 ns | 1,995,052 ns | 3.6526x | 2,041,812 ns | 3.7382x |
| 128 | 4,096 | 32,768 | 1,506,130 ns | 15,190,127 ns | 10.0855x | 23,151,738 ns | 15.3717x |

8 caseすべてで45 sampled outputを独立FP32 oracleと比較し、最大相対誤差は`0.0036457598`、最大絶対誤差は
`1.9578247`だった。判定はabsolute `0.5`かつrelative `0.02`の同時超過だけをFAILとする既存契約で、全点PASSした。
各caseのBF16 output digestと5 sampled row top-1はID25／ID29 controlに一致した。非有限不一致、repeat digest不一致、
dispatch ID不一致、cleanup failureはいずれも0だった。

## 実model検証

Phase 71のQwen3.5-27B MXFP6 artifactを使い、ID45強制環境変数をすべて未設定として512入力、chunk 512、明示FP16 KV、
最大4 output、greedy、ignore EOS、1 warmup＋3 measuredを実行した。prefillは
`384.149649 / 383.170165 / 371.476040 tok/s`、中央値`383.170165 tok/s`、MAD `0.979484 tok/s`だった。
旧上限でID45が選ばれなかったPhase 71中央値`81.746517 tok/s`に対して4.6873倍である。

全sampleの生成tokenは`[23066,23066,23066,23066]`、submission 14,912、kernel dispatch 24,000、
segment／boundary 272、HIP-only、fallback 0、model load 1、request内reload 0だった。residentは
`24,115,002,880` byte、peakは`24,777,018,880` byte、model drop後allocator current 0、retryable cleanup／
durable quarantine 0だった。process終了後の外部確認でも全3 GPUはuse 0%、VRAM allocation 0%だった。

## 数値分類と範囲

量子化recipe、E3M2からE4M3へのexact ingress、E8M0 scale、FP32 accumulation、BF16 RNEは変更せず、既存ID45を
新しいNへ選ぶだけである。ID25／ID29との固定行列出力は測定上bit一致したが、provider間の内部treeが一般入力で常に
bit一致するとは主張せず、Phase 70と同じN1として扱う。output token差は今回観測していない。

採用scopeはexact `gfx1201`、MXFP6 E3M2 W6A6、M>=17、K>=2048、K%32=0、1024<=N<=32768である。
gfx1030、gfx1200、gfx942、未知target、他format、decode、N>32768は対象外で従来providerへ戻る。rollbackはselector上限を
16,384へ戻すか、`SLLM_MXFP6_PREFILL_FORCE_TILED16=1`でID25を選ぶ。

[保存済み計画](../../../../plans/archive/2026/09/1-10/phase72-gfx1201-mxfp6-wide-n-selector.md) /
[数値変更台帳](../../../../compatibility/numerical-output-changes.md) /
[追跡要約](../../../../../ci/matrix/phase72-gfx1201-mxfp6-wide-n-selector-v1.json)
