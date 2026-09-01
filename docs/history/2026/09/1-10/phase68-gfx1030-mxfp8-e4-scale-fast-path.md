# Phase 68: gfx1030 MXFP8 E4／scale fast path

2026-09-02にcanonical Radeon Pro V620、exact `gfx1030`、ROCm 7.14.0、Code Object V6、wave32で完了した。
対象はOCP MXFP8 E4M3 W8A8 block32/E8M0、FP32 accumulation、BF16 RNE outputである。

## 実装

- 共通`ScalarCodec<E4M3Fn>`へ内部MX value-plane用decodeを追加した。normal codeはFP32 sign／exponent／mantissaを
  直接構成し、signed zeroとsubnormalだけをrare branchへ送る。内部quantizerはNaN blockをE8M0 scale 255＋zero value
  planeへ正規化し、value magnitude 127を生成しない。公開`decode`は従来どおりE4 NaNを扱う。
- `Mxfp8E4Block32::load`は通常値についてE4 exponentとE8M0 exponentを合成し、decode後のscale multiplicationを省いた。
  scale 255、standalone E4 NaN、zero／subnormal、FP32 underflow／overflowは従来演算へfallbackする。
- gfx1030 scoped defaultのID27 MMQはactivation／weight valueに内部helperを使う。scale tile、各blockのscale multiplication、
  FP32 accumulation順、BF16 RNE outputは変えていない。

## 棄却した候補

最初のscale事前乗算案は代表shapeで約10–33%退行したため即時revertした。normal waveを`ballot`で判定する案は
M=512の一部を8–11%改善した一方、M=128 wideを最大約7%悪化させ、weight-only／activation-onlyにも一貫性がなかった。
各valueへstandalone E4 NaN分岐を戻す安全版は旧版相当へ戻ったため、公開codecと内部quantizer契約を分離した。

## correctness

exact gfx1030 codec GPU testはdecode 1,104 code、5 encode boundary set、MX境界`31/32/33/256`をPASSした。
block loadはsigned zero、subnormal、normal、最大有限値、standalone E4 NaNとE8M0 scale
`0/1/118/127/134/254/255`を含む。26-case operator evidenceもPASSし、Phase 67 controlとcase別BF16 digestが一致した。
特殊fixtureはNaN block=`scale 255/value 0`、Inf clamp、最小subnormal、signed zero、最大有限値をactivation／weight双方で確認した。

## Qwen3.5-4B full-model

固定MXFP8 GGUF、FP16 KV、direct pretokenized input、最大4 output、1 warmup＋3 measuredを比較した。

| input | control samples (tok/s) | Phase 68 samples (tok/s) | control median | Phase 68 median |
| ---: | --- | --- | ---: | ---: |
| 512 | 204.6025 / 204.1578 / 202.9946 | 214.6966 / 213.0431 / 212.7907 | 204.1578 | 213.0431 |
| 2,048 | 205.4853 / 207.6376 / 206.9212 | 215.1771 / 211.1313 / 213.0759 | 206.9212 | 213.0759 |

512はthroughput +4.35%／prefill時間-4.17%、2,048は+2.97%／-2.89%だった。residentは
`4,954,035,712` byteで一致し、peakは512で`5,409,973,760`、2,048で`6,220,600,832` byteだった。
全sampleはHIP-only、fallback false、生成token列一致。最終CLI SHA-256は
`af687e0d4562bbf478be537bf6f9582d48a4ef9d2a62a477714063c81ed94c7c`である。

## 結論

gfx1030 MXFP8の残差にはE4 software decodeのrare-case判定costが含まれており、FP32 attention保存や永続展開を追加せずに
約3–4%の実モデル改善を得た。scale処理を消す試みは逆効果で、scale 255 semanticsとscale multiplicationは維持した。
適用は内部MX value-plane契約とexact gfx1030の既存ID27 production scopeに限定する。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/09/1-10/phase68-gfx1030-mxfp8-e4-scale-fast-path.md) /
[追跡要約](../../../../../ci/matrix/phase68-gfx1030-mxfp8-e4-scale-fast-path-v1.json)
