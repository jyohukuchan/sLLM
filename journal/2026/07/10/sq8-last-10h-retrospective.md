# SQ8直近約10時間の実装監査

日付: 2026-07-10

## 目的

2026-07-09夕方から2026-07-10未明までのSQ8実装について、なぜ期待した結果へ届かなかったか、不可避だった部分と改善できた部分を切り分ける。

監査対象は、期間中のgit履歴、journal、Qwen3-14B-FP8の元checkpoint、生成したsidecar manifest、SQ8 kernel、model loop、benchmark結果である。ソースコードは変更していない。

## 結論

今回の作業は全面的な失敗ではない。以下の接続基盤は実装できている。

- SQ8 sidecarのresident load
- runtime dispatchとfallback telemetry
- 40層、7 projectionのSQ8接続
- 複数requestをprojection batchへ渡す経路
- 層間residualのD2D handoff
- 計測結果、gate、比較用metadataの基盤

一方、正しさと性能の中心目標は達成していない。主因は次の2点である。

1. Qwen3-14B-FP8の元checkpointにある`weight_scale_inv`を適用せず、raw FP8値をそのまま再量子化したため、uLLM側sidecarが数学的に元モデルと異なる。
2. batch kernelが重みをbatchごとに読み直すscalar W8A16 matvecであり、vLLMが使うdynamic W8A8 block-scaled GEMMのような重み再利用やmatrix-core実行をしていない。

R9700向けの高性能block-scaled FP8 GEMMと実serving loopを10時間で完成させるのは現実的ではなかった。しかし、上の2点は最初の1〜2時間でgolden testとcomponent microbenchmarkを置けば検出できた。したがって「全部仕方なかった」ではなく、難しい目標に対して検証順序と停止条件が弱かったと判断する。

## 1. 最重要の正しさ問題

対象checkpointの`config.json`は以下を宣言している。

- `quant_method: fp8`
- `fmt: e4m3`
- `activation_scheme: dynamic`
- `weight_block_size: [128, 128]`

checkpointには280個のF8 projection weightと、対応する280個のBF16 `*.weight_scale_inv`がある。しかし、現builderはF8 tensorを`to(torch.float32)`で数値化してrow/row-block形式へ再量子化し、対応する`weight_scale_inv`を読んでいない。full sidecar manifestでも、この280個のscale tensorは`not_fp8_target_family`としてpassthrough扱いになっている。

layer 0の`q_proj`を抜き取り、正しい復元値`raw_fp8 * source_block_scale`とartifact復号値を比較した結果は以下だった。

- 正しい復元値のabsmax: `0.111328`
- artifact復号値のabsmax: `386.286`
- relative L2 error: `3297.44`
- mean absolute ratio: `3289.41`
- cosine similarity: `0.998495`

方向はある程度残るが、絶対scaleが約3,000倍ずれている。このsidecarは同じcheckpointから作られていても、同じQwen3-14B-FP8モデルではない。

この問題により、2026-07-09/10のuLLM Qwen3-14B-FP8 same-model比較行は、接続実験としては残せるが、品質・性能比較の根拠には使えない。vLLM側のbaselineは元checkpointを正しく読むため、baselineとしては有効である。

prompt suiteが`准准`を出している一方で、同じsummaryをreference/candidateの両方へ渡したself-behavioral guardが`passed=true`になっていた。`verified=true`も有限値、counter、内部sample consistencyを示すだけで、元モデルとの一致を示していない。

### 防止策

- full modelへ進む前に、F8 sourceの1 blockをCPUで復元するgolden testを置く。
- 1 tensor、1 linear layer、1 decoder layerの順でPyTorchまたはvLLM oracleと比較する。
- source scaleを表現できないartifact schemaでは、同一モデル比較を開始しない。
- 異常な生成文字列が出た場合、output healthを無効のままpromotionしない。

## 2. 性能が伸びない構造的理由

現`ullm_sq_fp8_matvec_batch_f32_kernel`は次の構造である。

- `grid.x = output row`
- `grid.y = batch index`
- 各batchが同じweight rowを独立に全量読み直す
- FP8 weightをscalarでF32へ変換する
- F32 activationとの積をshared memoryでreductionする
- WMMA/MFMA、weight tileの共有、vectorized loadを使わない

つまり、API上はbatchでも、計算は複数GEMVを同時にlaunchしているだけである。batchが増えるたびにweight trafficも同じ比率で増えるため、総tok/sが伸びない。

no-host-staging結果は以下だった。

| batch | decode tok/s | active weight payloadから見た最低読出し帯域 |
| --- | ---: | ---: |
| 2 | 16.4745 | 217.7 GB/s |
| 4 | 16.7037 | 220.7 GB/s |
| 8 | 16.6594 | 220.1 GB/s |

帯域がほぼ一定で、batchによるweight再利用がないことと一致する。

一方、vLLMのログは`TritonFp8BlockScaledMMKernel`、`norm_quant`、`act_quant`を選択している。vLLMはdynamic activation quantizationを含むW8A8 block-scaled FP8 matrix multiply、uLLMはBF16/F32 activationを使うW8A16 scalar direct-dequant matvecであり、名称がどちらもFP8でも計算契約が異なる。

現SQ8 specがactivationを`bf16_or_f32`としているため、既存のFP8×FP8 WMMA probeへそのまま接続することもできない。vLLM級のbatch scalingを目標にするなら、activation quantizationとtiled FP8 GEMMを含めて設計する必要がある。

## 3. Prefillとserving経路

prefillはprompt tokenを一つの大きなGEMMへまとめず、timestepごとに複数requestをbatch化して40層を実行する。

prompt length 16、decode 8 stepsでは、SQ8 projection callは次の内訳になる。

- prefill: `40 layers * 7 projections * 16 timesteps = 4480`
- decode: `40 layers * 7 projections * 8 steps = 2240`
- 合計: `6720`

`6720/6720`はfallbackなしのcoverageを証明するが、効率は証明しない。Q/K/Vとgate/upも別launchで、実kernelを伴うfusionはまだない。

最新offline-serving CLIは既存mixed-request smokeを呼び、stdoutを再parseしてreport化するwrapperである。実際のserving parityに必要な以下はまだ揃っていない。

- lm_head、sampling、次tokenのembedding feedback
- prompt tokenをまとめるprefill
- schedulerが選ぶready batch
- EOSとrequest別完了時刻
- continuous/online arrival
- request単位latency

したがって、現結果は40層projection経路の接続・計時には使えるが、実生成servingの比較には使えない。

## 4. Host stagingの評価

D2H readと同期をなくし、層間をD2Dへ変えた初期作業は必要だった。しかし、その後のH2D write削減は主ボトルネックではなかった。

- host writes: `72/120/216 -> 24 -> 0`
- post-pack decode: `16.60 / 16.78 / 16.69 tok/s`
- zero-host decode: `16.47 / 16.70 / 16.66 tok/s`

writeを0にしても性能が変わらないため、最初のA/B計測後はkernel microbenchmarkへ優先順位を移すべきだった。

## 5. 作業履歴から見た問題

約17:00〜03:00に85前後のcommitがあり、loader、dispatch、stack batch、telemetry、gate、serving reportまで広く実装した。大まかな流れは次の通りだった。

- 17:00〜20:37: loader、dispatch、計測、Qwen3 thin/full sidecar、same-model比較
- 20:45〜22:42: grouped requestを実projection batchへ接続し、40層7 projectionまで拡張
- 22:53〜00:17: host staging削減とb2/b4/b8測定。00:10ごろにはflat scalingが明確
- 00:23〜01:35: harness、gate、metadata、実kernelのないfused descriptor catalog
- 01:39〜02:20: host stagingを0まで削減したが性能は不変
- 02:33〜02:58: serving parser、contract、gate、CLI wrapper

この期間、SQ8 HIPRTC kernel本体はファイル分割以外のアルゴリズム変更がなく、representative shapeのmicrobenchmark、profiler、roofline分析も確認できなかった。

個々のAI作業は「loaderが通る」「counterが増える」「schemaが通る」のような局所的な完了条件を満たした。しかし、最上位の固定acceptance criteriaが弱かったため、検証しやすい統合作業へ進み、元重みの正しさとbatch scalingという核心が後回しになった。

## 6. 仕方なかった部分

- AQ4中心のruntimeへresident SQ8 loader、dispatch、D2D bufferを接続する基盤作業
- selected-layer pathとstack pathでmaterialization方法が異なることの調査
- gfx1201向けの高性能block-scaled FP8 GEMMを成熟したvLLM実装と同等にする難しさ
- smoke中心のruntimeから実serving loopへ移すためのscheduler/prefill/generation再設計
- AQ4とFP8でexact top-1が一致しないこと自体

production品質のvLLM parityを10時間で完成させる目標は現実的ではなかった。

## 7. 改善できた部分

- `weight_scale_inv`を無視したままfull sidecarを作り、same-modelとして扱った。
- 1 tensorのsource reconstruction testを置かなかった。
- scalar W8A16 GEMVでW8A8 matrix-core実装とのperformance parityを測った。
- representative shapeのcomponent benchmarkより先に40層統合へ進んだ。
- flat scaling判明後も、kernel profilingではなくstaging、gate、descriptor、CLIへ時間を使った。
- self-reference guardとoutput health無効の結果を品質確認として扱った。
- 実装を伴わないR9700/fused descriptorをmilestoneとして数えた。
- timestep serial prefillとpredetermined decode IDを、実servingに近い比較へ広げすぎた。

## 8. 次の実装順序

### Phase 0: 目標と停止条件を固定する

- 対象をQwen3-14B-FP8、gfx1201/R9700、固定projection shape、M=`1,2,4,8,16,32,128`に絞る。
- 正しさgate: source block、1 linear、1 decoder layerがoracleと一致する。
- 性能gate: b2〜b8で総throughputが明確に伸びる。flatならfull model作業を止めてprofileする。

### Phase 1: 正しいartifactを作る

- F8 payloadと128×128 `weight_scale_inv`を保持する。
- artifact schemaを2D block scaleへ拡張するか、正しくdequantizeしてから目的形式へrequantizeする。
- raw F8値だけを再量子化しない。
- 実checkpoint tensorを使うgolden testを必須にする。

### Phase 2: component kernelを先に成立させる

- per-token/per-block activation FP8 quantizationを実装する。
- M=1は専用GEMV、M>=2または4はweight tileを共有するFP8 GEMMへ分ける。
- hipBLASLt、Composable Kernel、rocWMMAなどの利用可能性を先に比較する。
- direct HIPが必要なら、FP8×FP8 tiled kernelとoffline weight prepackをcomponentだけで検証する。
- rocprofv3でeffective bandwidth、occupancy、launch count、matrix instructionを確認する。

### Phase 3: model経路を効率化する

- RMSNormとactivation quantizationを統合する。
- shared inputを一度だけquantizeし、QKVとgate/upをpack/fuseする。
- prefillを全prompt tokenまたはchunk単位のGEMMへ変える。

### Phase 4: 段階的に統合する

- 1 full decoder layerをoracleと比較する。
- 40層D2Dへ拡張し、lm_headとoutput healthを通す。
- 実生成loopとschedulerを接続する。
- 最後にb1/b2/b4/b8のvLLM比較、gate、serving reportを整える。

## 9. 停止条件

- source tensor goldenが不一致ならfull artifactを作らない。
- 異常出力またはoutput health未評価ならthroughput結果をpromotionしない。
- b2〜b8がflatなら、stagingやschema作業を増やさずkernelをprofileする。
- descriptorだけで実kernelがなければ、性能milestoneとして数えない。
- typed reportがない段階でstdout parserを比較契約の中心にしない。

## 10. 再利用するもの

維持する価値があるもの:

- `sq_runtime` resident abstraction
- D2D layer handoff
- scheduler batch接続
- fallback/counter telemetry
- benchmark result schemaとgateの骨格

位置づけを下げるもの:

- 現`SQ8_0` batch matvecは性能kernelではなく、reference/correctness kernelとする。
- fused descriptor catalogはroadmapであり、実装済み扱いにしない。
- 2026-07-09/10のuLLM Qwen3-14B-FP8 same-model行は接続実験として隔離する。

次の着手点はserving wrapperではなく、`weight_scale_inv`を反映するartifact schema/builderとsource golden testである。その後に、実shapeのW8A8 block-scaled component kernelを成立させる。
