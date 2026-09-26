# Phase 87 段階7 C1: input_rmsnorm + FP8 per-row producer融合

2026-09-22〜24。段階7の候補C1（FP8 per-rowのproducer融合、着手時上限137 node）を評価し、
C2（NVFP4 block16＋tensor scale）を追加評価した。C3（MXFP8 KV前処理）は上限から対象外とした。
[段階7計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階7の着手時上限と候補2026-09-22)の
上限・打切り線・候補定義は着手時に確定済みで、本作業単位では再計算しない。

## step 1: 代表1変種のコンパイルと資源記録

AGENTS.mdの「Kernel implementation and fusion」に従い、作業単位の冒頭で代表1変種をコンパイルし、
VGPR・LDS・occupancyを分解版の値と並べて記録する。資源が破綻する場合はここであきらめて設計を変える。

### 対象

代表変種は **`input_rmsnorm` + FP8 per-row量子化**。Qwen3.8のgraphでは、residual fusion有効時に
`input_rmsnorm`（FullAttention層、layer 3/7/…/63）は直前の`mlp_residual_add`と対になり
`ResidualRmsNorm`としてloweringされる。よって対応する融合カーネルは
`sllm_rmsnorm_residual_prequant_fp8_v1`である（layer 0の`input_rmsnorm`はLinearAttention層で
b/aがBF16を読むためC1対象外）。分解版の対照は、段階6採用後の現行経路である
生産RMSNorm `sllm_rmsnorm_residual_fused_wave32_v1` ＋ 生産量子化 `sllm_matmul_bf16_to_fp8_outer_v2`。

### 方法

- probe: `.local-artifacts/phase87/stage7/occupancy_probe.hip.cpp`（production TUをincludeして同一コンパイラ条件でリンク）
- lowp archive: `/tmp/s7-lowp-gfx1030/lowp/libsllm_lowp.a`, `/tmp/s7-lowp-gfx1201/lowp/libsllm_lowp.a`
- VGPR／LDS／SGPR／scratch: `rocprofv3 --kernel-trace` を probe 実行に重ね、kernel_trace.csvから取得
- occupancy: `hipOccupancyMaxActiveBlocksPerMultiprocessor`（理論値。register／shared memory／MP当りスレッド数から算出）
- GPU: V620 `ROCR_VISIBLE_DEVICES=GPU-76a08c022586fed6`（gfx1030）、R9700 `GPU-a8e9ddefa2d60f55`（gfx1201）
- shapeは代表的なdecode形状 `m1k5120`（M=1, K=5120）
- 証拠: `.local-artifacts/phase87/stage7/resources/`（gitignored）

### 資源（両GPUで同一値）

| kernel | 役割 | WG | VGPR | accum VGPR | SGPR | LDS [B] | scratch [B] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sllm_rmsnorm_residual_fused_wave32_v1` | 分解版 producer | 256 | 16 | 0 | 128 | 512 | 0 |
| `sllm_matmul_bf16_to_fp8_outer_v2` | 分解版 量子化 | 256 | 24 | 0 | 128 | 512 | 0 |
| `sllm_rmsnorm_residual_prequant_fp8_v1` | **融合版** | 1024 | 24 | 0 | 128 | 512 | 0 |

### 理論occupancy

`max_threads_per_mp=2048`、`wavefront_size=32`（gfx1030は`mp_count=36`、gfx1201は`mp_count=32`、MP=WGP）。

| kernel | block | blocks/MP | 理論occupancy |
| --- | ---: | ---: | ---: |
| `sllm_rmsnorm_residual_fused_wave32_v1` | 256 | 8 | 100% |
| `sllm_matmul_bf16_to_fp8_outer_v2` | 256 | 8 | 100% |
| `sllm_rmsnorm_residual_prequant_fp8_v1` | 1024 | 2 | 100% |

### 判定

**資源は破綻していない。** 融合版のVGPR 24は、分解版の生産16と量子化24の最大値と等しく、
LDSは512 Bで分解版の両kernelと同一、SGPRとaccum VGPRも増えていない。occupancyは分解版・融合版とも
100%で低下しない。WGが256から1024へ変わるが、MP当りスレッド数はいずれも2048で頭打ちにならず、
registerもshared memoryも制約にならない。したがって設計変更は不要で、step 2以降へ進む。

## 現状の注記

- 融合カーネル`native/hip/src/rmsnorm_kernel.hip.cpp`の`sllm_rmsnorm_residual_prequant_fp8_v1`／
  `..._nvfp4_v1`、および`native/hip/src/elementwise_kernel.hip.cpp`の融合カーネルは、
  本作業単位より前のセッションで実装・両GPUでbit一致確認済みである
  （対応するprobeは`.local-artifacts/phase87/stage7/`）。
  したがってstep 1は「新規に書いてから資源を測る」ではなく「既存融合版をコンパイルして資源を測る」で完了した。
- [段階7の破棄記録](phase87-stage7.md)はconsumer側（projection pack）の試行の破棄を記録した09:35時点の文書であり、
  producer側融合カーネル着手前を前提としている。本ファイルが以降の記録である。
- native側の実行時統合（`public_runtime.hip.cpp`のprepare/execute dispatch、
  `graph_span_runtime.inc`のcapture gate）は2026-09-23に接続し、以下で検証した。

計画: [Phase 87計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)

## 2026-09-23 実行時接続の途中結果

producer出力とmatmulのprequant values/scalesをnative runtimeへ接続した。Rustの動的descriptor再構築で
NVFP4 input-global scaleが落ちる箇所も修正した。これらは作業ツリー内の変更であり、段階7の採用・完了は未確定。

- V620/R9700のproducer単体probeでは、FP8/NVFP4のcodes・scale・残差が分解版とbit一致した。
  M=17、K=5120/17408も追加確認し、両GPUでmismatch 0。原票は
  `.local-artifacts/phase87/stage7/m17/`。
- M=17の短い実モデル入力で、C1+C2を同時に有効にした初期候補はbaselineの最初のEOS token
  `248046`から分岐した。C2だけを有効にした診断でも分岐し、M=1のdecode graph captureで
  `execution resource is busy`となった。C1だけの診断はbaselineと同じEOS tokenで完了した。
- C1だけの通常8192/128、V620、MTPなし、warmup 1＋measured 3は生成token digest
  `c9c0b4ee401b11544fe0faaec15d882cb2491c31a460a32a89dce28e437bcc0e`が段階6と一致し、
  decode中央値17.0592 token/s（段階6の16.8139から約+1.46%）、prefill中央値36.057秒だった。
  `selected_backend=hip`、fallbackなし、cleanup 0を確認した。これはC1の一部を単独で測った値であり、
  両GPU・MTP・全対象を含む採用証拠ではない。原票は
  `.local-artifacts/phase87/stage7/normal-gfx1030-fp8only.json`。
- 同じC1単独診断をtarget専用ビルドのR9700でも通常8192/128、MTPなし、warmup 1＋measured 3で実行。
  生成token digest `75d36def8ff45d155373ebb05b885d0e4f0e7197ba9adb2123adfb1fe63b189d`は
  段階6と一致し、decode中央値21.7836 token/s（段階6の21.4663から約+1.48%）、
  prefill中央値14.872秒だった。HIPのみ、fallbackなし、cleanup 0。原票は
  `.local-artifacts/phase87/stage7/normal-gfx1201-fp8only-isolated.json`。
- 同条件のMTPあり・幅2も両GPUでPASS。V620は段階6の30.7175→31.0362 token/s（+1.04%）、
  R9700は35.6270→36.0893 token/s（+1.30%）。両GPUとも生成token digestは段階6と一致し、
  V620の受理77/提案101、R9700の受理75/提案105も一致した。native kernel node/replayは
  MTPなしで1170→1082、ありで1308→1220（各88減）。原票は
  `.local-artifacts/phase87/stage7/normal-gfx1030-fp8only-mtp.json`と
  `.local-artifacts/phase87/stage7/normal-gfx1201-fp8only-mtp.json`。
- 一度試した`GraphBuilder`のM<=3静的gateは撤去した。実際のQwenグラフはprefill chunk容量で一度作って
  decodeにも再利用するため、そのgateではdecodeの融合も無効になる。gate付き8192/128の数値一致・速度は
  段階7の採否へ使用しない。

C2の実モデル差分とpack captureエラーを切り分けている。direct lowp prequant matmulのM=1/17は
両GPUでbit一致し、M=1 graph replayもPASSしたため、上位のscale/graph受け渡しとpack lifecycleを調査する。

### C2の差分原因（2026-09-23追記）

モデルのNVFP4 `input_global_scale` source planeはencoding用の値であり、既存weight uploadは
`read_f32_reciprocal`で逆数をGPU常駐planeへ置く。初期producer graphはsourceの生bitsを
descriptorへ渡したため、既存matmul quantizerの常駐値と異なっていた。例えばlayer 0 gate/upの
sourceは836、producerが使うべき値はFP32の`1/836`である。単体probeと公開C ABI probeは
対照・候補に同じscaleを渡していたため、その局所比較だけではこの接続差を検出できなかった。
graph側を常駐値と同じFP32逆数へ修正し、packのproducer scale照合も逆数を比較するようにした。
gfx1030の17/17実モデルは修正後、baselineと同じ最初のEOS token `248046`へ復帰し、
fallbackなし・cleanup 0でPASSした。通常8192/128の性能・MTP・gfx1201は続けて評価する。

別件のcapture開始時busyはpack内の並列分岐が原因ではなかった。prequant packは物理dispatch 2本・
pack内quantize 0回だが、Rustの`ExecutionAuditAccumulator`が従来の3本・quantize 1回を必須にしていた。
最初のeager decodeでpack ownerをfinalizeした後、監査エラーがterminal token selectorのfinalizeを飛ばし、
queueのactive submissionが1件残った。共通Busy表示で元の監査エラーが覆われていた。
registry診断は`active=1, pack=0, selector=1`、native診断はcompletion mode切替時の
`queue completion mode cannot change while submissions are active`だった。監査をprequant exact symbolでは
2本／0回として修正し、段階6の並列captureも復元。診断用のエラー表示・registry走査はsourceから撤去した。

逆数・監査・並列capture修正後のV620通常8192/128、MTPなし、1 warmup＋3 measuredは
生成token digestが段階6と一致し、HIPのみ、fallbackなし、cleanup 0でPASS。
decode中央値は17.2036 token/s、TPOT 58.1273 ms、native graph kernel nodeは1170→970。
C1単独17.0592 token/sとの増分はTPOT約0.4921 ms（約0.85%）で、C2の採用基準0.5948 msに未達。
この時点でC2をV620既定へ採用しない。原票は
`.local-artifacts/phase87/stage7/normal-gfx1030-forkrestored-off.json`。
同じC1+C2候補のR9700通常8192/128、MTPなし、1 warmup＋3 measuredも
生成token digestが段階6と一致し、HIPのみ、fallbackなし、cleanup 0でPASSした。
decode中央値21.9810 token/s、TPOT 45.4937 ms、native graph kernel nodeは1170→970。
C1単独21.7836 token/sとの増分はTPOT約0.4123 ms（約0.91%）で、R9700の採用基準
0.4662 msに未達。原票は`.local-artifacts/phase87/stage7/normal-gfx1201-both-off.json`。
この時点では**C2を両GPUで既定不採用**とし、Qwen3.8 graphのEncoding-B producer retypeは有効化しなかった（2026-09-24に採用へ変更、下記）。
直接呼ぶ融合kernel／公開C ABI経路と分解版のbit一致probeは診断用に保持するが、
通常モデルのselectorは従来BF16 producer＋量子化経路を維持する。

### C1の評価範囲と非採用箇所

採用候補のFP8 producer融合は、FullAttention `input_rmsnorm`のq/k/v 48量子化node、
FP8-MLPの`post_attention_rmsnorm` 16 nodeと`mlp_silu_mul` 8 node、
FullAttentionの`sigmoid_mul` 16 node、計88 nodeを取り除く。モデルのgraph実測でも
MTPなし1170→1082、あり1308→1220と各88 node減った。

着手時C1の残る49 node（`linear_attention_state`→GDN out 48、`final_rmsnorm`→lm_head 1）は
今回の融合から除外する。前者は既存GDN kernelがvalue headごとに別blockへ分かれ、
FP8 per-row scaleの全head amaxを一つのproducer kernel内で求められない。
全headを再設計するかblock間同期を導入しないまま2 kernelを連ねてもnode削減にならない。
後者は最終行だけをmatmulへ渡す現行aliasが、値plane末尾にまとめたFP8 row-scale planeの
最終行scaleを表せない。1 nodeの理想上限も単独の1%採用基準に届かない。
これらは完成した融合として数えず、今回のC1実測88 nodeの範囲だけで採否を判断する。

C3（MXFP8 KV前処理）は着手時計算どおり理想上限でも両GPUの1%基準に届かないため
実装対象外とした。分解版KV経路を維持する。

## 2026-09-24 C1既定候補の最終通常測定

C2を既定selectorから外した最終sourceをtarget専用ディレクトリで各gfx向けにbuildし、
通常8192入力／128出力、1 warmup＋3 measured、MXFP8 E4 KV、固定GPU samplingで測定した。
各runは`state=PASS`、HIPのみ、fallbackなし、cleanup zero。MTPありの受理／提案数は
V620 77/101、R9700 75/105で段階6と同じ。4構成すべて生成token SHA-256が段階6と一致した。

| GPU | MTP | 段階6 token/s | C1 token/s | 改善 | C1 TPOT ms | C1 prefill ms | 証拠 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| V620 gfx1030 | なし | 16.813859 | 17.085616 | +1.616% | 58.528764 | 36024.436 | `.local-artifacts/phase87/stage7/final-gfx1030-off.json` |
| V620 gfx1030 | あり | 30.717491 | 31.035159 | +1.034% | 32.221520 | 38028.840 | `.local-artifacts/phase87/stage7/final-gfx1030-mtp.json` |
| R9700 gfx1201 | なし | 21.4663 | 21.768134 | +1.406% | 45.938711 | 14887.587 | `.local-artifacts/phase87/stage7/final-gfx1201-off.json` |
| R9700 gfx1201 | あり | 35.6270 | 36.073469 | +1.253% | 27.721204 | 15215.443 | `.local-artifacts/phase87/stage7/final-gfx1201-mtp.json` |

同じ段階6対照のprefill中央値比は順にV620 −0.916%／−0.992%、
R9700 −0.196%／+0.121%。最後の+0.121%は小さく、通常測定の揺れの範囲で、
この4条件に目立つprefill退行はなかった。M=1/3/17のproducer単体も境界とbitwiseを確認した。

最終benchmark binary SHA-256はgfx1030
`291beea1405c5b96ba9739a0d70750c58ff31fbd791f0fb2cdc8e210d77a4321`、
gfx1201 `2eb067e4dde606230025c84e635d2ae3ad10651d363d746c822114d9a0ee0a3c`。
buildはそれぞれ`.local-artifacts/phase87/stage7/build/gfx1030/`と`build/gfx1201/`で行い、
別targetのnative objectが混入しないよう分離した。未コミット作業ツリーなので架空のcommit IDは割り当てない。

Rust core test 672件PASS（24 ignored）、HIP test 163件PASS、`cargo clippy -p sllm-core -p sllm-hip --all-targets -- -D warnings`、
`cargo fmt --all`、C++ format、CI JSON manifest検証、`git diff --check`がPASSした。
kernel単体・公開ABI probeは両GPUでbitwise mismatch 0。非整列K=17/31/33/5119/5121、
M=1/17とscale 1/19/60.75/74/504/836を含む。raw原票は`.local-artifacts/phase87/stage7/`。

V620の固定128位置・全有効語彙の品質比較は、段階6と同じwhole-graph forced-token captureで取り直した。
FP32 logit dumpのSHA-256は段階6保存値と同一の
`6fee4447aa5e8451125f1083f35a65e5234af80cf45a3076bd78ea2093cf9b9a`、
llama.cpp BF16基準の平均KLDも`0.026570786734159246`で差0、top-1一致率`0.921875`。
原票は`.local-artifacts/phase87/stage7/quality-gfx1030-forced/`と
`quality-gfx1030-forced-vs-bf16.json`。初回の`qwen38_kld_sllm.py --chunk-size 2048`は
prefill/decodeの分割とlogit取得精度がこのwhole-graph captureと異なり、平均KLD `0.03323946`を示した。
取得経路が異なるため、その値を段階6との差またはC1の数値変更と解釈しない。

R9700も同じwhole-graph forced-token captureで再取得し、以前の段階6相当baselineと全128位置の
FP32 logit dump SHA-256が同一の
`1de799171184ce565f271fb6bf50827ea6e9c517bad8995716d5fcd35fe45a53`だった。
BF16基準の平均KLDは`0.023446019680224913`、top-1一致率`0.9453125`で前回と差0。
原票は`.local-artifacts/phase87/stage7/quality-gfx1201-forced/`と
`quality-gfx1201-forced-vs-bf16.json`。別経路の`--chunk-size 2048`測定値`0.02848835`は
同条件の段階6差として用いない。

### C1単体採用判定

同一processで分解版／producer融合版をAB/BA交互に各7 round測定した。
FullAttention input RMSNorm 48、FP8-MLP post RMSNorm 16、SiLU 8、FullAttention sigmoid 16の
**実際の構成88 node**を、それぞれ本番のM1/K5120、M1/K17408、M1/K6144で構成した。
分解版は176 kernel、融合版は88 kernelを同じstreamへ並べ、HIP graphへcaptureして100 replay/sampleで測った。
全familyでquantized codeと行scaleはbitwise一致し、cleanup failure 0。

| GPU | graph分解版 ms/replay | graph融合版 ms/replay | 短縮 ms/replay | 7 round最小短縮 | 採用基準 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620 gfx1030 | 1.772490 | 0.929949 | **0.842541** | 0.831881 | 0.5948 |
| R9700 gfx1201 | 1.489460 | 0.784869 | **0.704588** | 0.704355 | 0.4662 |

両GPUともAB/BA全roundで融合版が速く、通常TPOTの1%基準を超えた。
stream直列launchの対照でもV620 0.950777、R9700 0.781961 ms/replay短縮した。
原票は`.local-artifacts/phase87/stage7/c1-perf-gfx1030.jsonl`と`c1-perf-gfx1201.jsonl`。
probe sourceは`native/hip/tests/phase87_s7_c1_perf_probe.hip.cpp`（SHA-256
`ded24f745db2e07d70a9ac26a6b59428231f903ad6f70337c13c72601690ff42`）。
両gfx専用binary SHA-256は順に`b2a7fcabd2cd6d6e8162e9fd41bbb36e2e6d625137d14148633b114ad6dad423`、
`dc7c167625752fe8eed04463d518705ba1c5cfe550ca3bf92427cade15aeb841`。

この時点では段階7のC1を88 node範囲で採用した。C1残り49 nodeとC3は上記の除外理由を維持する。
C2は以下の同一process計測とユーザー決定を受けて採用へ変更した。

### C2の同一process単体計測（2026-09-24、採否の再検討）

C2の不採用判断は、別processで測った通常計測どうしの差（C1+C2とC1単独）に基づいていた。
採用基準は同一processのAB/BAであるため、C1と同じ方式で測り直した。
対象は実構成の112 node（NVFP4 MLPの`post_attention_rmsnorm` 56、M1/K5120と、`mlp_silu_mul` 56、M1/K17408）。
分解版はproducer＋lowp NVFP4量子化の224 kernel、融合版は112 kernelで、HIP graphへcaptureし100 replay/sampleを7 roundずつ測った。
両GPUとも、codeとE4M3 block scaleはbitwiseで一致した（input global scaleは1/836と1/74の2通り）。
全roundで融合版が速かった。

| GPU | 分解版 ms | 融合版 ms | 短縮 ms | 通常TPOT比 | producer単体 ms | 量子化単体 ms | 採用基準 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 | 1.0913 | 0.8281 | 0.263 | 0.45% | 0.7047 | 0.3911 | 0.5948 |
| R9700 | 1.1059 | 0.7809 | 0.325 | 0.71% | 0.7769 | 0.3335 | 0.4662 |

融合版の短縮は、量子化kernelを丸ごと除いた場合の上限（量子化単体の時間）に近い。
R9700では融合版がproducer単体とほぼ同じ時間で、すでに上限に達している。
V620でも、融合kernelをproducer単体の速さまで詰めた場合の上限は0.391 ms（0.67%）である。
したがって、このkernelを調整しても両GPUとも1%基準には届かない。
通常計測どうしの差（V620 0.49、R9700 0.41 ms）は単体計測より大きいが、これも1%基準に満たない。
**2026-09-24のユーザー決定でC2を採用した。** 1%基準には届かないが、N0で全roundが改善し、実装は完了済みで、
無効のまま経路を残さないためである。`producer_activation`の`EncodingAOrB`がNVFP4 block16も選ぶように戻し、
「C2無効」を前提にしていたRust test 2件を採用後の型・scale・workspace sizeへ更新した
（sllm-core 672 PASS、clippy、fmt、local h0 628 PASS）。
速度の根拠は、前記の通常計測（MTPなし、C1+C2でV620 17.2036、R9700 21.9810 tok/s、生成token列は段階6と一致）と、この単体計測である。
MTPありでC2を有効にした実行は、ユーザー指示により再計測していない。
probeは`.local-artifacts/phase87/stage7/c2-perf/`（sourceと両gfxの結果）。

採用後の`producer_activation`は`EncodingAOrB`でNVFP4 block16を選び、NVFP4融合kernel、
公開ABIのkernel ID（`SLLM_HIP_RESIDUAL_RMSNORM_KERNEL_ID_PREQUANT_NVFP4_V1`、
`SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_PREQUANT_NVFP4_V1`）、graphの型変換、scaleの受け渡し、
packのprequant受け入れを通常経路で使う。
Phase 87全体の段階3・4・9などは別作業単位であり、ここでは完了扱いしない。

計画: [Phase 87単一要求NVFP4計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。
