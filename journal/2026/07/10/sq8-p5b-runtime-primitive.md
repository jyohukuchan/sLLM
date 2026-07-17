# SQ8 P5-B production runtime primitive

日付: 2026-07-10

## 前回の要点

P5-AでQwen3-14Bの7 projection×7個のMについて、実測済みCK dispatchと数値・scaling gateを固定した。ただし実行経路はstandalone componentであり、production runtimeからは使えなかった。

## 今回の変更点

- commit `3477543 Add gfx1201 SQ8 CK runtime primitive`を追加した。
- 既定buildを変えない`rocm-ck-gfx1201` featureを追加し、feature無効時は新ABIが明示的なruntime errorを返すようにした。
- F32 activationからOCP E4M3へのrow×K128 RNE動的量子化と、量子化済みactivationを複数projectionで共有できる`Sq8CkQuantizedActivation`を追加した。
- canonical FP8 weightとF32 block scaleを受けるCK ABScale GEMM、BF16 workspace、同一stream上のBF16→F32変換を追加した。暗黙のsynchronizeは行わない。
- P5-Aの完全な`GetTypeString`と一致数1件を必須にし、4種類のimplementation IDを型付きで返す。未実測のshape/M、backend/device/stream不一致、alias、buffer不足、overflowはfallbackせず拒否する。

## 検証

- 既定build: `ullm-runtime-sys` 139 tests全合格。
- gfx1201 feature: 量子化bit一致、M1 q-shape zero-weight CK起動、BF16→F32変換の3 tests全合格。
- R9700のP4 M=8 q fixture: activation byte exact、scale bit exact、relative L2 `0.001658314`、cosine `0.999998625567`、非有限値`0`。
- 新規warningはなく、既存のanonymous namespace warning 3件のみ。

## 次の行動

P5-CでM=8の狭いQwen3-14B一層runnerへ接続し、QKVとgate/upの共有量子化、4種類全dispatch、referenceより速い一層latencyを確認する。P5-Dで独立CPU oracleと中間・最終tensorを比較する。
