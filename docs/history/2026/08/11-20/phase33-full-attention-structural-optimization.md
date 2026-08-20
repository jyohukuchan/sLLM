# Phase 33: Full Attention構造最適化

> 状態: 完了（C1/C2限定採用、C3棄却）
> 実施日: 2026-08-19〜2026-08-20

## 結論

long-contextで支配的だったQwen3.5 Full Attentionへ、global scratchや追加combine dispatchを作らない二つの
common providerを実装した。prefillのC2 GQA4 K/V共有はgfx1030/gfx1201共通の限定経路として採用した。
decodeのC1 wave8 KV分割はQK reductionのworst-case error boundが僅かに増えるN2であることを明記し、
大幅な短縮と観測誤差を踏まえた2026-08-20のユーザー承認により限定採用した。C3 gfx1201 matrix innerは、
採用C2 tileとhardware MMA shapeが合わず棄却した。

## Baseline訂正

Phase 32の診断集計はmain prefill dispatchだけを主に数え、native append encodeで1 tokenずつ発生する`M=1`
Full Attentionをattention familyへ合算していなかった。Phase 33 A0のfresh rocprofv3 traceでは次のとおりだった。

| input | full timing | prefill attention | `M=1` attention | attention合計 |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 prompt / 10,001 total | 108.440 s | 18.648 s | 35.275 s | 53.922 s |
| 16,384 prompt / 16,385 total | 230.207 s | 50.532 s | 94.854 s | 145.387 s |

したがってFull Attentionの最適化対象はprefillだけではなく、native append中の逐次`M=1`にも広がった。

## C1: scratch-free decode wave8 KV split

当初計画のmulti-workgroup partial buffer + second combine kernelは、headごとのglobal scratch、arena accounting、
追加launchを必要とする。bounded prototypeでは代わりに256-thread blockを8 wave32へ分け、各waveへ連続した1/8 KV区間を
割り当てた。各waveが`local_max/local_denominator/local_weighted_value`を計算し、workgroup内LDSで区間順に固定mergeする。

- route: exact gfx1030/gfx1201、`M=1`、head dim 256、KV長1,024以上、全4 KV encoding。
- complement: KV長1,023以下、別target/head shapeはPhase 30 providerまたはB0。
- resource: LDS 8,256 byte、VGPR 104、scratch 0、追加dispatch 0。
- publication: outputは単一kernel完了後だけ既存completion boundaryから公開し、KV stateやABIを変更しない。

FP16の21-event中央値は次のとおりだった。

| target | KV case | B0 | C1 | 短縮 |
| --- | --- | ---: | ---: | ---: |
| gfx1201 | 1,024 | 880,170 ns | 374,364 ns | 57.47% |
| gfx1201 | 4,097 mixed | 3,074,314 ns | 1,446,250 ns | 52.96% |
| gfx1201 | 8,193 | 6,101,628 ns | 2,549,697 ns | 58.21% |
| gfx1030 | 1,024 | 1,107,134 ns | 401,244 ns | 63.76% |
| gfx1030 | 4,097 mixed | 4,394,055 ns | 1,575,814 ns | 64.14% |
| gfx1030 | 8,193 | 8,783,950 ns | 3,067,107 ns | 65.08% |

短KV 256付近はV620のDPM/noiseを含めて利益が安定しなかったため、共通thresholdを1,024へ固定した。

### C1数値分類

B0はhead dim 256のQK和を概ね8段のbalanced treeで合成する。C1は各laneが8項を逐次加算してから32-lane treeへ入るため、
依存深さは概ね12段となる。real-number equation、FP32 accumulator、softmax式、BF16 RNEは同じだが、標準的な
worst-case boundは`gamma_8`から`gamma_12`へ僅かに増える。このためC1はN2であり、性能だけでは自動採用しなかった。

観測上はdense signed KV=4,097でFP16最大絶対誤差`2.3841858e-7`、dynamic/static FP8で`4.7683716e-7`、
NVFP4で`1.1641532e-9`だった。KV=1,024のNaN query/+Inf valueも両targetでscalar FP64 oracleと一致した。
生成token差、非決定、causal/GQA違反、fallback、cleanup異常は観測されていない。

ユーザーは2026-08-20にこのN2 tradeoffを承認した。分類自体はN2のままとし、exact gfx1030/gfx1201、`M=1`、
head dim 256、KV長1,024以上、4 encodingだけへproduction routeする。rollbackは`use_decode_wave_split=false`で
Phase 30/B0 complementへ戻す既存の一点を維持する。

## C2: prefill GQA4 K/V共有

採用C2は一つのworkgroupが1 query rowと1 KV headを所有し、同じKV headへmapされる4 query headのQK/softmax/PVを
同時に進める。K/V elementを一度だけdirect decodeして4 headで共有し、各headのcausal key順、online maximum、
denominator、weighted Vは独立に維持する。full score matrix、global scratch、KV mirrorは作らない。

- route: exact gfx1030/gfx1201、`M>=64`、GQA ratio 4、head dim 256、全4 KV encoding。
- complement: `M<=63`、別ratio/head/targetはPhase 30 providerまたはB0。
- resource: LDS 192 byte、VGPR 48、scratch 0、追加dispatch 0。
- grid: `M × kv_heads`。B0の`M × q_heads`からblock数を4分の1へ減らす。

M=64〜257のFP16 device中央値はgfx1201で21.20〜47.43%、gfx1030で38.29〜53.66%短縮した。
prototypeのgfx1201 M=37は8.22%悪化したため、thresholdを64へ固定し、63/64/65の両側を測定した。

gfx1201ではB0 wave providerと同じ32-lane partial + 8 partial固定treeで、全case bit exactのN0だった。
gfx1030では256-thread LDS treeからwave32 + 8 partial treeへ順序が変わるが、加算依存深さは同じ8段で
worst-case boundは非増加であるためN1とした。real-number equation、入力集合、dtype、丸めstageは維持する。
担当AI裁量では、両target/全encodingで大きく一貫した利益、scratch 0、単一共通source、stable complement、
容易なrollbackが保守費用を上回るため採用が妥当である。

## Correctness・routing

evidence runnerを17から29 caseへ拡張し、63/64/65、127/128/129、nonzero start、1,023/1,024/1,025、
4,097 mixed signed、8,193、C1/C2範囲のNaN/+Infを含めた。FP16/dynamic FP8/static FP8/NVFP4 ×
gfx1030/gfx1201 × 29 case、計232/232がscalar FP64 oracle、causal visibility、GQA mapping、dispatch metadata、
no-fallback、terminal cleanupをPASSした。各caseはwarmup 5、HIP event measured 21を持つ。

Phase 33 providerはtarget keyをexact gfx1030/gfx1201へ閉じた。gfx942等は候補symbolがbuildへ存在しても選択せずB0を維持する。
ROCm 7.14、Code Object V6、wave64、`xnack=off`、`sramecc=on`のexact gfx942 compile-onlyもPASSしたが、
これはruntime/full-model evidenceではない。
gfx1201 code objectをV620で開くwrong-target testはexit 1、`requested device gcnArchName does not match exactly`で拒否された。

## C3 matrix innerの棄却

採用C2 tileは`Q_TILE=1 × GQA 4 head = 4 row`である。ROCm 7.14 rocWMMAのFP16 gfx12 builtinは
16×16×16 WMMAを使用し、最小valid MMA dimは16である。同じtileにmatrix innerだけを入れるには12 rowをpaddingするか、
Q_TILE=4へ変えて別のquery-row layout/mask/resource contractを作る必要がある。前者は25% row utilization、後者はC3の
「C2と同じtileでinner mathだけ」というscopeを越える。最終C2 code objectにWMMA/MFMA instructionは0であり、
matrix providerとは呼ばない。別Q_TILE=4 providerを検討する場合は新Phaseで独立に判断する。

llama.cppはprovider topologyのfacts-only比較に使い、source codeはreuseしていない。新しいprovenance importはない。

## Full model・API

| case | B0 | C1+C2 | 全体短縮 | token/audit |
| --- | ---: | ---: | ---: | --- |
| R9700 FP16、10,000 prompt、paired direct | 105.800 s | 75.162 s | 28.96% | 1228、HIP-only、fallback false、cleanup 0 |
| V620 FP16、4,108 prompt | 39.097 s | 36.500 s | 6.64% | 1228、HIP-only、fallback false、cleanup 0 |
| V620 FP8、4,108 prompt | 39.304 s | 36.669 s | 6.71% | 1228、HIP-only、fallback false、cleanup 0 |

R9700 10,000-prompt profileではattention aggregateが53.922秒から23.101秒へ57.16%、全kernelが
88.677秒から57.713秒へ短縮した。V620はattention外のprojection等が相対的に厚いため、operator利益のwall転化率が小さい。
16,384-prompt diagnosticはC1 LDS最終trim前でも230.207秒から150.900秒へ34.45%短縮した。

dynamic FP8のgfx1201 production serverは10,012 prompt tokenからnon-stream/SSEとも1 token `It`、usage 10,012+1、
terminal `[DONE]`を返した。10k requestを1秒でdisconnectしたauditは`cancelled`となり、直後のsmall recoveryは`Hello`を返した。
graceful shutdown後はcurrent/request-state/workspace byte、retryable cleanup、durable quarantineがすべて0だった。

## 承認後のfinal identity再検証

承認後に同じ最終sourceからtarget別release binaryを再buildした。binary SHA-256はbounded summaryのidentityと一致した。

- gfx1030/gfx1201 × FP16/dynamic FP8/static FP8/NVFP4 × 29 caseは232/232 PASS。最大絶対誤差、fallback、cleanupは
  承認前の値から変わらなかった。
- R9700 FP16 10,000 promptは75.553秒、V620 FP16 4,108 promptは36.432秒でPASSし、両方ともtoken 1228、
  HIP-only、fallback false、cleanup 0だった。
- R9700 dynamic FP8 APIは10,012-token non-stream/SSE、`[DONE]`、disconnect=`cancelled`、直後のrecovery、
  graceful shutdownを再度PASSした。shutdown current/request/workspace byteとcleanupはすべて0だった。
- gfx1201 binaryをV620へloadするwrong-target caseはexit 1でfail-closedした。
- bounded summaryを`COMPLETE`へ更新し、C1/C2を限定採用、C3を棄却としてPhase 33を完了した。

[対応するarchive plan](../../../../plans/archive/2026/08/11-20/phase33-full-attention-structural-optimization.md)
[bounded summary](../../../../../ci/matrix/phase33-full-attention-summary-v1.json)
[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../plans/main-plan.md)
