# Phase 32: native FP8 KV append encode再検証 実装履歴

## 結果

Phase 31後の10k+通常経路でexact `gfx1201`のnative scalar／packed FP8 appendを再設計して測定した。
standalone operatorでは全BF16 bit patternをK/V合わせて一巡し、dynamic/static FP8のpayload byteとF32 scale bitが
software baselineに一致した。native packedは全validation token countで19.59%以上短縮し、10,001 tokenで79.94%、
16,384 tokenで76.85%短縮した。

production traceではcurrent append全体が10,001-token runの0.01523%、16,385-token runの0.01156%にすぎず、当初の固定5%規則で
一度棄却した。その後、2026-08-19のユーザー指示で固定thresholdを担当AIの理由付き裁量判断へ変更して再評価した。
既存kernel、256-thread workgroup、symbol、grid、scale recipe、store、ABIを変えず、exact gfx1201のcompile-time helperだけを
native化できるC1 scalarは、全operator rowの一貫改善、bit exact性、低い保守費用、将来のlow-bit/batching利用価値を総合して採用した。
C2 packedは追加複雑性が利益に見合わないため不採用とした。default KV encodingはFP16、gfx1030 appendはsoftwareのままである。

## Prototype

- B0はproductionと同じ256-thread/token-head row、balanced maximum reduction、F32 scale、software E4M3FN binary searchとした。
- C1はscale後のF32値だけを`__builtin_amdgcn_cvt_pk_fp8_f32(value, value, 0, false)`へ置換し、NaN、Inf、signed zero、
  448 saturationをsoftware contractへ補正した。
- C2は128 threadで隣接2要素を処理し、packed conversionと16-bit storeを使った。K/V maximumとscale recipeはB0と同じである。
- exact gfx1030 probeはB0だけをdispatchし、native candidateを性能claimしなかった。
- gfx1201 device codeには`v_cvt_pk_fp8_f32`が8箇所存在した。probeはproduction registry、force switch、public ABIを変更しない。

Phase 30で旧native candidateがchunk 256を68.69%悪化させた結果は再現しなかった。fresh C1/C2は同じ256 tokenの
dynamic FP8をそれぞれ19.58%／25.73%短縮した。したがって旧結果はnative instruction自体の限界ではなく、
当時のcandidate/code generationまたは候補構造に依存したnegative resultだったと判断する。ただし旧raw candidateを最終sourceへ
残していないため、どの要因が支配的だったかを一つへ断定しない。

## Operator evidence

ROCm 7.14.0、R9700 UUID `GPU-a8e9ddefa2d60f55`、V620 UUID `GPU-76a08c022586fed6`を一度に一台だけ
可視化した。各providerはwarmup 5、measured 31、HIP event median/MADで比較した。

| token | encoding | software median ns | native scalar median ns | 短縮率 |
| ---: | --- | ---: | ---: | ---: |
| 256 | dynamic FP8 | 34,005.217 | 27,347.688 | 19.58% |
| 256 | static FP8 | 33,430.655 | 26,866.438 | 19.64% |
| 10,001 | dynamic FP8 | 519,802.988 | 191,120.997 | 63.23% |
| 10,001 | static FP8 | 488,002.986 | 174,961.001 | 64.15% |
| 16,384 | dynamic FP8 | 868,165.970 | 359,963.000 | 58.54% |
| 16,384 | static FP8 | 811,886.013 | 332,643.002 | 59.03% |
| 16,385 | dynamic FP8 | 863,766.015 | 358,321.995 | 58.52% |
| 16,385 | static FP8 | 808,125.019 | 332,242.996 | 58.89% |

gfx1201は19 token count × dynamic/static × 3 providerにexhaustive dynamic/staticを加えた120 rowでmismatch 0だった。
adopted native scalarの全row最小短縮率は19.40%、最大は65.22%である。native packedは19.59〜82.26%短縮したが、
productionへは採用しない。gfx1030は19 token count × dynamic/staticの
software-only 38 rowを完走した。
exact gfx1201/gfx1030 buildはともにPASSし、gfx1201-only binaryをgfx1030へloadしたnegative testは
`device kernel image is invalid`、exit 1でfail-closedした。operator timingは同一process内のprovider固定順であり、
sample単位のinterleaveではない。ただし全scalar rowの改善幅は19%を超え、production traceでも同じ方向を確認した。

## Full-model Amdahl evidence

production binary SHA-256は`4d20eabf91e9c2c8e6c7117c04ccbb37dcce90e0e899ad078e7f983a72999156`、
Qwen3.5-4B GGUF／lockはPhase 31と同一である。profiler wallは採否に使わず、production JSONの`timing_ns`を分母、
kernel traceのappend family合計を分子にした。decode appendも含めたため、perfect-elimination ceilingとして保守的に大きい。

| case | input / chunk | timing ns | append dispatch | append device ns | share / 完全消去上限 |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | 10,001 / 10,001 × 1 | 29,687,137,652 | 16 | 4,520,428 | 0.015227% |
| F2 | 16,385 / 16,384 + 1 | 64,390,874,518 | 24 | 7,446,067 | 0.011564% |

両caseともgenerated tokenは`[1228, 1228]`、HIP-only、fallback false、cleanup failure 0だった。F1の16 dispatchは
8 full-attention layer × prefill/decode、F2の24 dispatchは8 layer × 2 prefill chunk/decodeと一致する。

## Production採用と判断

C1を`float_to_e4m3fn_fp8_append`として既存FP8 append kernelへ接続した。exact gfx1201だけがcompile-timeに
`__builtin_amdgcn_cvt_pk_fp8_f32`を使い、gfx1030は同じsourceから従来software helperへlowerする。actual code objectは
gfx1201で`v_cvt_pk_fp8_f32` 2命令、gfx1030で0命令だった。gfx1030 final CLI hashはPhase 31と同じである。

production 10,001-token traceのappend 16 dispatchはsoftware 4,520,428 nsからC1 2,191,564 nsへ51.52%短縮した。
full-model差は通常noise以下なのでuser-visible speedupをclaimしない。dynamic/static FP8 production attention oracleは
gfx1201/gfx1030の4 row × 17 case、計68/68 PASSした。candidate full-modelもgfx1201 10,001/16,385 token、gfx1030
10,001 tokenで`[1228, 1228]`、HIP-only、fallback false、cleanup 0だった。

gfx1201 serverはdynamic FP8 KVの10,013-token chat promptでnon-stream/SSEとも1 token `It`を返し、SSE `[DONE]`、
2 requestのfallback 0を確認した。shutdown auditはcurrent/request-state/workspace byte 0、cleanup failure 0だった。

採用理由は、効果の大きさではなく、scope内で一貫して高速、N0 bit exact、既存kernel構造とABIを維持、target分岐がdevice compile内だけ、
revertがhelper call 2箇所で済むという低い保守費用との釣り合いである。C2 packedは128-thread workgroup、packed store、alignment、
odd-tail検証を増やすため、現状のfull-model寄与では採用しない。QKV/RoPE/attention writeとのfusionで独立dispatchを消せる場合だけ再検討する。
Paged Attention、low-bit KV default昇格、TurboQuantはPhase 32の結果から自動的に開始しない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase32-native-fp8-kv-append-revalidation.md)
[bounded summary](../../../../../ci/matrix/phase32-native-fp8-append-summary-v1.json)
[メイン計画](../../../../plans/main-plan.md)
