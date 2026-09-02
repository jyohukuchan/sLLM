# Phase 75: gfx1030 MXFP8先行・MXFP6共通half2最適化

## 結論

2026-09-03にcanonical Radeon Pro V620、exact `gfx1030`でPhase 75を完了した。MXFP8でtile／K staging／
half2 dot／scale／output mappingの共通software-MMQ骨格を先行評価し、128x64 output tile、K32、double bufferの
ID55を既存Phase 67 shapeへ限定採用した。次に同じscheduleをscalar E3M2 ingressのMXFP6 ID56へ移植して効果を
分離し、最後に4 value／3 byteを一度だけ読むpacked ingress ID57を追加して既存Phase 74 shapeへ限定採用した。

同一最終binaryのQwen3.5-4B、3 warmup＋10 measured、control/candidate/controlで、MXFP8 ID55は512／2,048入力を
`993.6765 / 1,104.1643 tok/s`、MXFP6 ID57は`1,008.7235 / 1,095.3894 tok/s`とした。候補はそれぞれ両control比
`3.805〜3.875x / 4.380〜4.403x`、`2.622x / 2.961x`で、生成token、VRAM、dispatch、HIP-only、fallbackなし、
cleanupを維持した。Qwen3.5-27B MXFP6も強制指定なしで`157.7535 tok/s`をPASSし、model名をselectorへ入れていない。

decode、attention、GDN、KV、gfx1201 provider、quantization recipe、resident layoutは変更していない。MXFP8 ID41と
MXFP6 ID47は明示rollbackとして残し、ID56は共通scheduleの寄与を示すbenchmark candidateとして残した。

## 固定identity

- 開始時Git HEAD: `097245a9d1841ddd039a20fbdc0caa0caa1db87b`。測定はdirty draft sourceでありrelease provenanceではない。
- GPU: Radeon Pro V620、UUID `GPU-76a08c022586fed6`、BDF `0000:03:00.0`、exact `gfx1030`、wave32。
- software: ROCm 7.14.0、HIP 7.14.60850、AMD clang 23、Code Object V6。
- 最終CLI SHA-256: `d67a8f3742ed705cee60adb8f5f8dc4e3f8797f17e87e4aa91665b01ccfc69b8`。
- operator runner SHA-256: `688a54ebb2e4f5a6066af65082c2a9ee403ca50e8efccd479821283e9fe5c13d`。
- 抽出gfx1030 matmul code object SHA-256:
  `2b67fc2a1f535a2887dbf4fe243b072ac3c6c0e25b114709842604d67cdfea35`。
- primary modelは固定Qwen3.5-4B。MXFP8／MXFP6 derived lock fingerprintは
  `sha256:f253d9f47603d84718b4fdb898b434e493d732b52838ba9abfdfafe73a5d076f` /
  `sha256:d0ff2e1de9d87dddddcde8f85ef305bbf21a06d5f7586d077ba1178580a0264e`。
- 補助modelはQwen3.5-27B revision `fc05daec18b0a78c049392ed2e771dde82bdf654`、derived MXFP6 lock
  `sha256:d1142468252af487d52ebf72a29a4bb62487a635c174e709bebd73b0c337a82c`。

## 実装

`FormatIngress`、`TilePolicy`、`KStagePolicy`をcompile-time policyとして分離し、共通bodyにhalf2 dot2、FP32
accumulator、K32ごとのE8M0 scale適用、BF16 RNE store、M/N tailを置いた。各scale blockの寄与をscale適用前に
跨いで合算せず、persistent FP16／BF16／FP32 weightやrequest全体の展開activationは追加していない。

MXFP8 ingressにはE4M3FN 256 codeからexact FP16 bitsへのconstexpr変換を追加した。signed zero、subnormal、最大有限、
NaN classを含み、weight／activationの両operand、4 laneでdevice oracleを通した。MXFP6は最初に既存scalar extractionを
同じscheduleへ接続し、その後だけ3 byteから4 codeを一括抽出するbounded 24-bit group loadへ置き換えた。

production selectorは次のdimension-only scopeである。

- ID55 MXFP8: exact `gfx1030`、`M>=128`、`K>=2048`、`K%32==0`、かつ`2560<=N<=16384`、または
  `M>=512 && N==1024`。rollbackは`SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69=vector32`。
- ID57 MXFP6: exact `gfx1030`、`M>=128`、`K>=2048`、`K%32==0`、`1024<=N<=32768`。rollbackは
  `SLLM_MXFP6_PREFILL_FORCE_PHASE74=gfx1030-half2-32x32`。

M=1 decode、scope外shape、unknown targetは既存経路を維持する。実行失敗後のsilent fallbackは追加していない。

## P75-B: MXFP8共通骨格の評価

### geometryとK depth

production-shape operatorの各case中央値合計は次のとおりだった。全candidateは独立FP32 oracle、repeat、tail、
HIP-only、fallbackなし、cleanupをPASSした。

| ID | tile / K | 合計中央値 | resource（LDS / SGPR / VGPR） | 判断 |
| ---: | --- | ---: | --- | --- |
| 41 | 旧col8 control | 305,836,907 ns | 既存 | rollback維持 |
| 49 | 32x32 / K32 | 226,947,773 ns | 4,352 B / 32 / 74 | benchmark-only |
| 50 | 64x64 / K32 | 177,030,683 ns | 8,704 B / 34 / 108 | benchmark-only |
| 51 | 128x32 / K32 | 179,486,191 ns | 10,880 B / 32 / 103 | benchmark-only |
| 52 | 128x64 / K32 | 165,793,648 ns | 13,056 B / 34 / 156 | geometry勝者 |
| 53 | 128x64 / K64 | 192,662,763 ns | 26,112 B / 32 / 157 | 棄却 |
| 54 | 128x64 / K128 | 202,643,492 ns | 52,224 B / 34 / 186 | 棄却 |
| 55 | 128x64 / K32 double | 164,974,585 ns | 26,112 B / 38 / 156 | scoped default採用 |

K256は推定LDS 104,448 byteがdevice workgroup上限を超えるため、危険なlaunchを行わずinstantiation前に棄却した。
ID49〜55は全てprivate 0、SGPR/VGPR spill 0、wave32、WG256である。ID55のstatic
`v_dot2_f32_f16`は512命令だった。

### 数値

ID55はID41からaccumulation treeを変えるN1候補で、28 operator case中18 caseのBF16 digestが変化した。
独立FP32 oracleへの最大observed relative errorは`0.003891051`、nonfinite mismatchは0、10 repeatは決定的だった。
全256 E4M3FN code × 4 lane × weight／activation permutationのdevice oracleもPASSした。

## P75-C: 共通scheduleのMXFP6移植

ID56 `matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.scalar.v1`はID55と同じ128x64／K32 double
scheduleを使い、ingressだけをID47相当のscalar E3M2 extractionとした。10-case operator合計中央値はID47
`73,872,054 ns`からID56 `52,677,669 ns`へ1.4023倍改善した。4B draft 1+3では次のように共通scheduleの寄与を
分離できた。

| input | ID47 | ID56 | speedup |
| ---: | ---: | ---: | ---: |
| 512 | 387.9054 tok/s | 715.1196 tok/s | 1.8435x |
| 2,048 | 373.0800 tok/s | 777.5578 tok/s | 2.0842x |

ID56はLDS 26,112 B、SGPR/VGPR 38/151、private/spill 0、WG256、static dot2 512である。tree変更のため
ID47比N1だが、全64 E3M2 code、全packed lane、特殊値、tail、独立FP32 oracleをPASSし、最大relative error
`0.003875792`、nonfinite mismatch 0だった。

## P75-D: MXFP6固有packed ingress

ID57 `matmul.mxfp6.w6a6.gfx1030.half2.128x64.k32d.pack4.v1`はscheduleを固定したまま、各4 valueを
同じ3 byteから一度だけload・展開する。operator合計中央値はID56から1.1455倍、4B draftは512／2,048で
`715.1196→1,009.0863`／`777.5578→1,100.4919 tok/s`、各1.4111／1.4153倍となった。

ID57はLDS 26,112 B、SGPR/VGPR 40/151、private/spill 0、WG256、static dot2 512である。ID56との全10
operator output digestが一致するN0変更で、row終端、K32境界、全64 code混在、repeatをPASSした。この結果から、
共通scheduleだけでなくMXFP6固有のpacked ingressにも独立した大きな改善余地があったと判断した。

## 最終operator

同一runner、1 warmup＋10 repeatの最終値は次のとおり。いずれもfallback false、cleanup 0である。

| format | ID | case | 合計中央値 | 対control | 最大relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP8 | 41 | 28 | 301,622,480 ns | 1.000x | 0.003891051 |
| MXFP8 | 55 | 28 | 155,490,441 ns | 1.9398x | 0.003891051 |
| MXFP6 | 47 | 10 | 73,872,054 ns | 1.000x | 0.003875792 |
| MXFP6 | 56 | 10 | 52,677,669 ns | 1.4023x | 0.003875792 |
| MXFP6 | 57 | 10 | 45,985,831 ns | 1.6064x | 0.003875792 |

## Qwen3.5-4B最終control/candidate/control

direct pretokenized、token ID `23066`反復、FP16 KV、最大4 output、greedy、ignore EOS、single request、
3 warmup＋10 measuredで、各formatをcontrol A、candidate、control Bの順に完全別processで測定した。

| format | input | control A | candidate | control B | candidate MAD | 両control比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP8 | 512 | 261.1472 | 993.6765 | 256.4292 | 0.6585 | 3.8050〜3.8751x |
| MXFP8 | 2,048 | 252.1178 | 1,104.1643 | 250.7527 | 1.0542 | 4.3796〜4.4034x |
| MXFP6 | 512 | 384.6954 | 1,008.7235 | 384.7285 | 0.3676 | 2.6219〜2.6221x |
| MXFP6 | 2,048 | 369.9150 | 1,095.3894 | 369.9914 | 0.2748 | 2.9606〜2.9612x |

MXFP8 candidateのprefill中央値は512／2,048で`515,300,543 / 1,856,568,980 ns`、E2Eは
`745,708,984 / 2,099,585,281 ns`。residentは`4,954,035,712` B、peakは`5,292,664,320 / 6,153,623,040` Bで
同formatのcontrolと一致した。MXFP6 candidateはprefill `507,675,331 / 1,869,700,137 ns`、E2E
`678,295,620 / 2,056,465,396 ns`、resident `4,061,763,072` B、peak
`4,400,391,680 / 5,261,350,400` Bでcontrolと一致した。

各processはsubmission 24,336、kernel dispatch 39,104、segment／boundary 468で一致した。全sampleの生成tokenは
`[23066,23066,23066,23066]`、model loadはprocessごとに1回、request内再load 0、HIP-only、fallback false、
retryable cleanup／durable quarantine 0だった。

Phase 74のfixed llama.cpp Q6_K参照値は512／2,048で`2,077.47 / 2,061.67 tok/s`であり、最終MXFP6との差は
約2.060／1.882倍まで縮小した。ただしQ6_KとMXFP6 W6A6は数値形式・activation処理が異なるため、同一kernelの
strict A/Bや品質比較ではない。

## profileと残差

512-inputの同じrequestをrocprofv3で取得した。

| format | prefill matrix control | prefill matrix candidate | decode matrix | linear column state | prefill attention | activation quantizer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP8 | 88.70% | 62.91% | 25.37% | 3.17% | 1.79% | 1.06% |
| MXFP6 | 87.16% | 71.76% | 16.22% | 3.69% | 2.05% | 1.28% |

candidateでdecode比率が相対的に上がったのはprefillを短縮した結果であり、Phase 75はdecodeを変更していない。
prefill matrixはなお最大のcandidate-side kernel shareなので、残差をattention／KV律速へ読み替えない。次の共通
prefill最適化余地は残るが、本Phaseのscopeを追加拡張しない。

## Qwen3.5-27B補助行

強制環境変数を外したproduction default、512 input／chunk 512、1 warmup＋3 measuredで中央値
`157.753458 tok/s`、MAD `0.018965`、prefill `3,245,570,703 ns`だった。Phase 71 gfx1030旧既定
`34.298907 tok/s`比4.5994倍である。resident／peakは`24,115,002,880 / 24,777,018,880` B、全生成token一致、
HIP-only、fallbackなし、cleanup 0だった。ID57 selectorはmodel名ではなくM/N/Kとtarget／formatだけを見るため、
Qwen3.5-4B専用化していないことを補助確認した。この1行から任意model architectureやMoE対応へ一般化しない。

## 検証

- Rust runner test: release 16/16 PASS。
- native public runtime CTest: 5/5 PASS。
- exact gfx1030 codec GPU test: 全256 E4M3FN code × 4 lane × 2 operand permutation、全64 E3M2 codeとpacked laneをPASS。
- final operator: ID41／55の28 case、ID47／56／57の10 caseを各10 repeatでPASS。
- full model: 4Bの両format 512／2,048 control/candidate/control、27B MXFP6 512 defaultをPASS。
- target別compile-only: exact gfx1201 wave32、logical gfx942／`gfx942:sramecc+:xnack-` wave64をPASS。実機性能は主張しない。
- `cargo fmt --check`、`git diff --check`: PASS。

主要なGit-excluded evidence rootは`/home/homelab1/.local/share/sllm-evidence/phase75`。`baseline`、`p75-b`、
`p75-cd`、`final-operator`、`final-c-c-c`、`final-profile-mxfp8`、`final-profile-mxfp6`、`final-27b`を保持した。

[全体計画](../../../../plans/main-plan.md) /
[保存済み計画](../../../../plans/archive/2026/09/1-10/phase75-gfx1030-mxfp8-first-shared-half2-optimization.md) /
[追跡要約](../../../../../ci/matrix/phase75-gfx1030-mxfp8-mxfp6-shared-half2-v1.json)
