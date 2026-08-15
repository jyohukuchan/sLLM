# Phase 15O FP8/NVFP4 model量子化path最適化履歴

## 2026-08-15: 実装・実機検証・closeout

### 採用した実装

- FP8はsidecar、OCP/FNUZ encoding、per-row scale、W8A8 hipBLASLt、FP32 accumulation、BF16 outputを変更せず、
  `sllm_matmul_bf16_to_fp8_outer_v2`を既定にした。256要素shared reductionを8個のwave最大値へ縮小し、even Kでは
  HIPのnative FP8 pair conversionを使う。NaN/Inf、zero、odd Kは既存規則を維持する。
- NVFP4はM=1をprovider ID 8 `matmul.nvfp4.block16.decode.packed_dequant.v1`、M>1をprovider ID 9
  `matmul.nvfp4.block16.prefill.row8_tiled256.v2`へ分離した。prefillはK=256のpacked weight tileをLDS内だけで
  decodeし、最大8 M rowで共有する。resident weight、sidecar bytes、scale contract、public ABIは変更していない。
- 比較専用に`SLLM_FP8_QUANT_FORCE_BASELINE=1`と`SLLM_NVFP4_FORCE_BASELINE=1`を用意した。後者はprovider ID 10で
  Phase 15 kernelを明示する。いずれも暗黙fallbackではなく、通常dispatchはcandidateへ固定される。
- NVFP4 decodeのwaveあたり複数N列とscale broadcast候補はR9700/V620で改善しなかったため棄却し、device kernelを
  Phase 15実装のまま維持した。activation共有、producer fusion、hipBLASLt solution tuningは、今回の最有力candidate
  より先にPhase完了を妨げる根拠がなかったため将来のbounded follow-upとした。
- reader-onlyのvLLM/SGLangからは技術的事実だけを記録し、実装表現をcopy/adaptしていない。llama.cppを含む外部sourceの
  新規直接reuseもなく、実装は既存sLLM kernel、AMD/HIP API、独立oracleをbasisとした。

### Operator O2

各caseは3 warmup＋10 measured。単位はkernel全体のmicrosecond、中央値である。FP8はcandidate→baselineの逆順、
NVFP4はcandidate/baselineを同一最終binaryの環境選択で比較した。MADとp10/p90も保存して確認し、採用caseの改善幅は
target/case固有のばらつきを十分に越えた。

| target/lane | M,K,N | baseline median | candidate median | 変化 |
| --- | --- | ---: | ---: | ---: |
| R9700 FP8 decode | 1,2560,9216 | 88.522 | 81.901 | -7.48% |
| R9700 FP8 decode | 1,9216,2560 | 178.402 | 127.282 | -28.65% |
| R9700 FP8 prefill | 32,2560,9216 | 90.562 | 83.061 | -8.28% |
| R9700 FP8 prefill | 32,9216,2560 | 180.723 | 127.663 | -29.36% |
| R9700 NVFP4 prefill | 32,2048,6144 | 3293.374 | 1333.414 | -59.51% |
| R9700 NVFP4 prefill | 32,6144,2048 | 3127.693 | 1273.174 | -59.29% |
| V620 NVFP4 prefill | 32,2048,6144 | 2496.428 | 1081.353 | -56.68% |
| V620 NVFP4 prefill | 32,6144,2048 | 2246.765 | 1096.093 | -51.21% |

NVFP4 decodeは採用kernelとbaseline device symbolが同じで、V620の中央値差は0.1%未満だった。R9700 decodeには
単発outlierがあり、candidate改善の証拠には使っていない。production/non-aligned oracleはM=1/M>1、K/N 15/16/17、
31/32/33を両exact targetでPASSし、最大relative errorは`0.00388`、fallbackなし、cleanup 0だった。

### Full-model O2とprovider状態

同一resident modelの32 prompt/32 generation、3 warmup＋10 measuredで比較した。

| target/model/path | prefill tok/s | decode tok/s | E2E ms | TTFT ms | baselineからの要点 |
| --- | ---: | ---: | ---: | ---: | --- |
| R9700/Qwen3.5-4B FP8 candidate | 514.260 | 35.342 | 957.639 | 69.083 | prefill +5.89%、decode +10.69%、E2E -9.27% |
| R9700/Qwen3.5-2B NVFP4 candidate | 205.481 | 51.660 | 765.588 | 159.876 | prefill +135.97%、decode同等、E2E -21.61% |
| V620/Qwen3.5-2B NVFP4 candidate | 174.454 | 41.563 | 937.517 | 187.535 | prefill +78.93%、decode同等、E2E -13.46% |

FP8 baselineはprefill `485.645`、decode `31.929` tok/s、E2E `1055.499` ms、TTFT `72.795` msの
counterbalanced平均である。NVFP4 baselineはR9700が`87.078/51.683 tok/s`、`976.581/371.600 ms`、V620が
`97.496/41.512 tok/s`、`1083.373/332.342 ms`だった。resident/peakはFP8
`4,847,029,760/5,044,359,040` byte、NVFP4 `1,790,406,056/1,891,213,096` byteで前後不変だった。

- FP8 Qwen3.5-4Bはtop-1全一致、最大KLD `0.023937316136398896`で既存budget内であり、R9700 nativeを
  `opt-in production`に維持する。historical BF16 controlとの差はprefill約3.3%、decode約4.6%まで縮小したが、
  memory providerとして明示opt-inのままにする。V620 emulationは`correctness-only`を維持する。
- NVFP4 Qwen3.5-2Bはtop-1全一致だが最大KLD `0.26375229966155406`が`0.05`を超えるため、性能改善後も両targetを
  `correctness-only opt-in`とする。R9700ではBF16比のprefill gapがなお大きく、decodeも既存約20〜22% gapを残す。
- 新しいMI300X VMがないため、exact `gfx942`へPhase 15O candidateを有効化せず、Phase 12の既存provider状態を変更しない。

### 回帰・監査

- R9700のFP8/NVFP4最終candidateでfixed/Unicode/stop、連続request、OpenAI non-stream/SSEと`[DONE]`、shutdownを
  PASSした。fallback false、final current/request/workspace allocation 0、quarantine 0だった。
- exact `gfx1030`/`gfx1201`のtarget別release build、FP8/NVFP4 operator、accuracy、`cargo test --workspace`、
  H3 public-runtime contract 25 tests、JSON manifest/matrix/Markdown link validation、`git diff --check`をPASSした。
  dirty treeは並行するPhase 12変更を含むため、clean checkoutを要求するH3 runnerそのものではなくexact target cargo buildと
  runner契約testでcompile範囲を確認した。
- integration reviewで、新FP8量子化kernelが未検証のFNUZ/gfx942にも選ばれるtarget guard不備を見つけた。FNUZは
  Phase 12のv1を維持し、検証済みOCP pathだけv2を選ぶよう修正した。findingだけのfocused re-reviewでexact RDNA build、
  両GPUoperator、workspace tests、H3契約を再実行してPASSした。Phase 15Oを完了し、Phase 16 KV cache量子化を開始可能とする。

## 2026-08-15: 性能candidate採用基準の見直し

- ユーザー承認により、一律の「改善3%以上、guard退行3%以内」という選択目安を削除した。既存Phase 8/9と
  共通RDNA性能bridgeに合わせ、target/case別の反復測定でnoise envelopeを越える改善と非退行を判断する。
- 性能判断を、実装candidateの採用、lane全体でのBF16 gap回収、providerのdefault昇格へ分離した。
  小さくても再現する累積改善を採用可能にする一方、大きな残存gapを小改善だけで実用化済みと扱わない。
- O1の3 measuredはscreeningに限定し、最終採否はO2のbaseline/candidate各10 measured、実行順の反転または
  counterbalance、median/MAD/p10/p90/driftを使う。releaseの性能hard thresholdは本Phaseで新設しない。
- NVFP4は既存accuracy budget超過が解消されない限り、性能candidateを採用しても`correctness-only opt-in`から
  defaultへ昇格しない。

## 2026-08-15: 詳細計画作成

- ユーザーの明示指示により、Phase 16 KV cache FP8/NVFP4の前にmodel本体のFP8/NVFP4最適化を行う
  Phase 15Oを新設した。decode M=1とprefill M>1を別provider、別計測lane、別採用判断にする。
- FP8の主要候補をdynamic activation量子化の重複除去・producer fusion、M=1専用path、M>1 hipBLASLt
  solution選択とした。NVFP4はdecode専用packed GEMVと、packed residentを維持するprefill tiled GEMMを分けた。
- local primary targetはR9700 exact `gfx1201`とV620 exact `gfx1030`とする。MI300X旧VMは削除済みのため、
  新しいgfx942 candidateを有効化する場合だけ、新規VMのexact tupleで再検証する。
- correctness、fail-closed、accuracy threshold、resident削減をhard contractとして維持する。性能目標はcandidateの
  採用目安であり、BF16超えをPhase完了の新しいhard gateにはしない。
- この時点では計画のみを作成し、source implementation、GPU benchmark、provider stateは変更していない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase15o-model-quant-path-optimization.md)
