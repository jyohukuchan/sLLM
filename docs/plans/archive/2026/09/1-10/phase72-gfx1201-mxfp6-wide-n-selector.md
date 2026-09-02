# Phase 72: gfx1201 MXFP6 wide-N selector

状態: `完了（2026-09-02）`

## 目的

非公式modelをmodel名で列挙せず、OCP MXFP6 E3M2 W6A6のshapeだけでexact `gfx1201`向けID45を選べる範囲を
`N<=16384`から`N<=32768`へ広げられるか判断する。既存のM/K下限、target、format、accumulation、output、decode、
KV、量子化recipeは変更しない。

## 受入条件

1. `N=16384/16385/17408/17409/24576/32000/32767/32768`を、非整列M/Nを含む独立FP32 oracleで実行する。
2. ID45、従来ID25 tiled16、ID29 col8の出力digest、非有限値、sampled top-1を比較し、同一provider repeatも確認する。
3. selectorとprepared providerの`N=32768`採用、`N=32769`fallbackをhost／compile-time contractで固定する。
4. Qwen3.5-27B MXFP6を強制環境変数なしで実行し、速度、生成token、HIP-only、fallback、VRAM、cleanupを確認する。
5. gfx1030、gfx1200、gfx942、未知target、M<17、K<2048、N<1024、N>32768は従来経路を維持する。

## 判断と結果

受入条件を満たしたため、exact `gfx1201`のMXFP6 ID45 selectorを`1024<=N<=32768`へ拡張した。
8 shapeすべてがPASSし、ID45はID25比`3.0731〜10.6190x`、ID29比`3.0507〜16.7310x`だった。
各shapeの45 sampled output、5 sampled row top-1、BF16 output digestは両controlと一致し、最大相対誤差は
`0.0036457598`、非有限不一致とrepeat不一致は0だった。

Qwen3.5-27Bの512入力／chunk 512／1 warmup＋3 measuredは、強制指定なしで
`384.149649 / 383.170165 / 371.476040 tok/s`、中央値`383.170165 tok/s`となった。Phase 71の旧既定
`81.746517 tok/s`比で4.6873倍である。4 requestすべて生成tokenは`[23066,23066,23066,23066]`、全24,000 dispatchが
HIP、fallback 0、model reload 0、retryable cleanup／durable quarantine 0、resident／peakは
`24,115,002,880 / 24,777,018,880` byteだった。終了後は全GPUのVRAM使用が0へ戻った。

この採用はmodel名に依存しないため、公式・非公式を問わず同じformat／target／shape契約へ適用される。ただし
`N<=32768`なら任意のmodel全体が互換という意味ではなく、model architecture、他operator、GGUF metadata、VRAM収容は
それぞれ既存のfail-closed検証を受ける。`N=32769`以上はID25へ戻す。

[全体計画](../../../../main-plan.md) /
[Phase 72履歴](../../../../../history/2026/09/1-10/phase72-gfx1201-mxfp6-wide-n-selector.md) /
[Phase 72追跡要約](../../../../../../ci/matrix/phase72-gfx1201-mxfp6-wide-n-selector-v1.json)
