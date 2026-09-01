# Phase 68: gfx1030 MXFP8 E4／scale fast path

状態: `完了・内部MX value-plane fast path採用`

## 目的と固定scope

Phase 67後のcanonical Radeon Pro V620、exact `gfx1030`、ROCm 7.14.0、Code Object V6、wave32、
OCP MXFP8 E4M3 W8A8 block32/E8M0、FP32 accumulation、BF16 RNE outputを固定scopeとした。
Qwen3.5-4B MXFP8 weightを調べた結果、E4 value planeの99.9920294%がnormal、0.00743694%が
subnormal、0.00053364%がsigned zeroで、standalone E4 NaN codeは0だった。E8M0 scaleはvalue 32個に1個で、
固定weightではcode 108–118、scale 255は0だった。runtime quantizerのNaN blockはscale 255＋zero value planeで表す。

## 完了条件と結果

- [x] 公開scalar codecのsigned zero、subnormal、standalone E4 NaN semanticsを維持した。
- [x] 内部MX quantizer専用`decode_mx_value_plane`を共通codecへ追加し、normalをcommon path、zero／subnormalをrare fallbackとした。
- [x] block loadへE4 decode＋E8M0 scaleのcombined-exponent fast pathを追加し、scale 255、E4 NaN、underflow／overflowをreference fallbackへ残した。
- [x] MMQのscale tile、scale multiplication、FP32 accumulation順を変えず、E8M0 scale 255のNaN伝播を維持した。
- [x] 全scalar code、block境界`31/32/33/256`、scale `0/1/118/127/134/254/255`をexact gfx1030 GPUでPASSした。
- [x] 26 operator caseで独立oracle、旧版とのBF16 digest一致、特殊値encoding、HIP-only、fallback falseをPASSした。
- [x] Qwen3.5-4Bの512／2,048 input、FP16 KV、最大4 output、1 warmup＋3 measuredで速度とVRAM不変を確認した。

## 採否

採用したのは内部MX value planeの1分岐decodeと、安全なblock-level combined-exponent decodeである。
standalone E4 NaNを省く契約は内部MX quantizerが生成したvalue planeだけに限定し、公開scalar codecでは省かない。

次の候補は棄却した。

- E4値をFP32へ展開してscaleを事前乗算する案は、代表shapeで約10–33%遅くなった。
- wave ballotでnormal／exceptionalを一括分岐する案は、shape別に改善と退行が逆転した。
- standalone E4 NaN fallbackをMMQの各valueへ戻す案は安全だが、今回の速度差をほぼ消した。公開codecで意味を保ち、内部契約で分離した。

## 最終性能

Qwen3.5-4B MXFP8、direct pretokenized input、FP16 KV、最大4 output、1 warmup＋3 measuredの中央値。

| input | Phase 67 control | Phase 68 | throughput増加 | prefill時間短縮 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 204.1578 tok/s | 213.0431 tok/s | 4.35% | 4.17% |
| 2,048 | 206.9212 tok/s | 213.0759 tok/s | 2.97% | 2.89% |

residentは両方`4,954,035,712` byte、peakは512で`5,409,973,760`、2,048で`6,220,600,832` byteと
control／candidateで一致した。全sampleはHIP-only、fallback false、生成token列一致だった。

## 境界

production効果の主張はexact gfx1030のPhase 67 ID27 scopeと固定4B実モデルに限る。結果をgfx1031–gfx1036、
gfx1201、gfx942、別ROCm tuple、別model、複数GPUへ一般化しない。KV default、quality threshold、sampling、public ABI、
永続BF16／FP32 weight planeは変更しない。

[全体計画](../../../../main-plan.md) /
[matching履歴](../../../../../history/2026/09/1-10/phase68-gfx1030-mxfp8-e4-scale-fast-path.md) /
[追跡要約](../../../../../../ci/matrix/phase68-gfx1030-mxfp8-e4-scale-fast-path-v1.json)
