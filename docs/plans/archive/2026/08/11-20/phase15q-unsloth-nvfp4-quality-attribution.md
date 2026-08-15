# Phase 15Q: Unsloth NVFP4品質要因の切り分け

> 状態: complete
> 作成日: 2026-08-15

## 目的

Phase 15/15Oで観測したNVFP4 full-model最大KLD `0.26375229966155406`が、sLLMの単純なweight
min-max量子化、scale/packing/runtime解釈、またはE2M1 block-16という数値表現の実用上の限界のどれに主に由来するかを、
一変数ずつ変える比較で切り分ける。Unslothが公開するNVFP4 checkpointのweight payloadを同一sLLM runtimeへ取り込み、
同じBF16 source、同じ量子化対象tensor、同じBF16 activation、同じprompt/logit位置で比較する。

本Phaseは品質要因の特定が目的であり、NVFP4を直ちにdefaultへ昇格するPhaseではない。結果からconverter改善で既存budgetを
満たせる見込みが得られた場合だけ、その最小candidateを実装・検証する。絶対的な数学上の「FP4限界」を証明したとは表記せず、
固定したmodel、target set、calibration/evaluation setで観測した実用上のformat/configuration ceilingとして結論する。

Phase 16 KV cache FP8/NVFP4は、本Phaseのcloseoutまたはユーザーによる明示的な順序変更まで開始しない。

## 固定する外部artifact候補

### Unsloth NVFP4 checkpoint

- repository: [`unsloth/gemma-4-12b-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4)
- revision: `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`
- `model.safetensors`: `9,304,966,064` byte、SHA-256
  `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`
- license表示: Apache-2.0
- header観測: 1,389 tensor。BF16 629、F32 288、F8_E4M3 328、U8 144。
- MLP 48 layer × gate/up/downの144 weightは`weight_packed` U8、`weight_scale` E4M3、
  `weight_global_scale` F32を持つ。configはweight observerを`imatrix_mse`、group size 16、static actorderと記録する。

### 対応するBF16 source候補

- repository: [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
- revision候補: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- `model.safetensors`: `23,919,549,408` byte、SHA-256
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`
- Unsloth revisionより後のmetadata-only commitを含むため、量子化sourceと同一weightであることを、両artifactの非量子化
  BF16 tensor、config、tokenizer identityから実測する。同一性を確認できなければ、Google repository historyから一致する
  revisionを固定するまでaccuracy attributionへ進まない。

artifactはmodel cache外へ複製せず、repositoryへmodel、slice、raw logits、profileを追加しない。revision、file size、完全hash、
header/catalog hash、license、取得日時をmodel lockとhistoryへ残す。

## 重要な比較境界

Unsloth checkpointはsLLMの現行weight-only NVFP4と同じ実行構成ではない。公開configでは次を組み合わせる。

- MLP: weight NVFP4に加えてinput activationもdynamic/local block-16 NVFP4。
- self-attention projection: FP8 W8A8。
- KV cache: static FP8。
- vision/audio/lm_head等: ignoreまたはBF16。

このcheckpointをそのまま実行してBF16またはsLLM NVFP4と比較すると、weight量子化algorithm、activation dtype、attention、KV、
source modelの差が混ざる。したがってprimary attributionではUnslothのMLP weight payloadだけを取り出し、BF16 activation、
BF16 attention、FP16 KV、既存sLLM NVFP4 packed-dequant providerへ接続する。Unsloth mixed checkpointのend-to-end実行は、
primary結論後のsecondary caseとする。

## 比較variant

| ID | source/weight | activation/attention/KV | 用途 |
| --- | --- | --- | --- |
| `B0` | exact Gemma 4 12B-it BF16 | BF16 / BF16 / FP16 | logitsとlayer出力の基準 |
| `S0` | 同じBF16 sourceを現行sLLM min-maxでMLP 144 tensorだけNVFP4化 | BF16 / BF16 / FP16 | 現行algorithm control |
| `U0` | UnslothのMLP 144 packed weight/scale/global-scaleをimport | BF16 / BF16 / FP16 | 同じformatでalgorithmだけ変更するprimary candidate |
| `O0` | 同じE2M1/block-16/E4M3/F32 contractでscaleをbounded search | BF16 / BF16 / FP16 | 同一format内の経験的上限候補 |
| `M0` | Unsloth公開mixed checkpointを可能な範囲で忠実に実行 | NVFP4 / FP8 / FP8 | activation/attention/KV相互作用のsecondary比較 |

`S0`と`U0`は量子化対象、source weight、非量子化weight、token IDs、provider、accumulation、output dtypeを一致させる。
`M0`を`U0`の代替にせず、未実装のW4A4やFP8 KVを暗黙にBF16へ変換した結果を「Unsloth model実行」と呼ばない。

## 受入条件

### correctness/security blocker

1. Unsloth/baseの完全revision、artifact hash、header/catalog、licenseとsource同一性をfail-closedに固定する。
2. `weight_packed` nibble順、logical shape、K-axis block 16、E2M1 code、E4M3 scale、global scaleの式を独立decoderで確認する。
   compressed-tensorsのfield名が似ているだけでsLLM sidecar v1と同一と推測しない。
3. Unsloth payloadを独立CPU decoderとsLLM GPU providerへ入力し、production shape、K/N 15/16/17・31/32/33、odd tail、
   zero、scale極値を照合する。CPU fallback、全weight BF16 materialization、別provider fallbackをPASSにしない。
4. `B0/S0/U0/O0`で同一model source、MLP 144 tensor、prompt token IDs、logit位置、BF16 activation/attention、FP16 KVを
   固定する。source不一致またはtarget-set不一致のrunからalgorithm/format結論を出さない。
5. 量子化対象外tensorはBF16 sourceとbyte-identical、量子化tensorはsource rangeとimport rangeを完全hashで結び付ける。
6. tensor/layer/full-modelの結果はmaxだけでなくmedian、p90、分布、top-1一致、非finite、最悪case IDを保持する。
7. 既存KLD budget `0.05`を緩めない。品質改善candidateの採用とproviderのproduction昇格を別判断にする。
8. 最終candidateでfixed/Unicode/stop、連続request、OpenAI non-stream/SSE、fallback、cleanupを回帰する。

### attribution判定

- `U0`が`S0`より同一logit位置で一貫して低KLD・高top-1一致となり、layer/tensor誤差も同じ方向なら、
  **現行converter/scale選択が主要因**と判定する。`0.05`を満たすかと改善方向は分けて記録する。
- `O0`が`S0/U0`より改善する場合も、同じ数値型で改善余地があるため**量子化algorithm/calibration余地あり**とする。
- `U0/O0`がいずれも改善せず、format oracle、source identity、runtime decodeがPASSした場合は、固定したGemma 4 12B-itと
  weight-only target setについて**NVFP4 format/configuration ceilingの寄与が大きい**と判定する。数学的な型限界とは呼ばない。
- `U0`の独立decoderとGPU出力が一致しない、またはpacking/scale contractがsLLM sidecarと異なる場合は、
  **format mapping/runtime defect**を先に修正し、algorithm比較を無効として再取得する。
- layer/tensorごとに結論が分かれる場合は**mixed**とし、gate/up/down、layer位置、outlier blockを記録する。全tensorへ一律の
  algorithmを強制せず、必要なら高感度tensorだけBF16/FP8へ残すmixed-precision follow-upを提案する。

決定論的logit比較には性能benchmarkのnoise floorを流用しない。run再現性をbyte/hashで確認し、製品上のmaterialityは
既存KLD budget、top-1一致、worst-case改善、target set全体の一貫性で判断する。

## 実装・検証順序

### P15Q-A0: artifact/source lockとformat inventory

- cache容量を事前確認し、BF16約23.9 GB、Unsloth約9.3 GB、tokenizer/configを固定revisionで取得する。
- LFS SHA-256、size、safetensors header/catalog、全tensor rangeを検証する。model payloadをGitへ追加しない。
- 両checkpointで同名かつ両方BF16の349 source tensor、config、tokenizerを比較してsource ancestryを確定する。Unsloth側の
  残りBF16 entryはattention/input scale等の量子化metadataであり、source tensor数へ混ぜない。weightが異なる場合は
  repository historyから一致revisionを探し、見つからなければfull-model algorithm attributionをblockする。
- Unsloth 144 MLP tensorについてpacked/scale/global-scale、shape、offset、alignment、input scaleをinventory化する。
  attention FP8とKV FP8はsecondary laneとして別表に分ける。

### P15Q-A1: 独立decoderとbounded importer

- repository外artifactからheader/rangeをpositional readし、必要tensorだけを読むcompressed-tensors importerを作る。
- E2M1 packed byte、E4M3 block scale、F32 global scaleを独立CPUでdecodeし、元BF16 weightに対するMSE、MAE、cosine、SQNR、
  zero/saturation率、block scale分布を144 tensorすべてで算出する。
- 同じpayloadを既存`sllm-nvfp4-sidecar-v1`へlosslessに包めるか検証する。metadata/packingが異なる場合は既存schemaを
  過負荷にせず、明示的なimport schema/versionを追加する。
- layer 0/中間/最終のgate/up/downとnon-aligned synthetic shapeをGPU providerへ通し、CPU decoder由来FP32 oracleと照合する。

### P15Q-A2: matched S0/U0/O0作成

- exact BF16 sourceのMLP 144 tensorだけを現行converterで量子化して`S0`を作る。Phase 15の186 tensor policyや別modelの
  sidecarをそのまま比較に使わない。
- Unsloth payloadだけを同じtarget setへbindする`U0`を作り、input global scaleはprimary W4A16 laneでは適用せず、
  provenanceとして保持する。attention/embedding/lm_headはexact BF16 sourceを使用する。
- `O0`は同じformatを固定し、weight MSEと代表BF16 activationに対するlinear output MSEを目的関数とするbounded scale searchを
  比較する。探索範囲、calibration token IDs、seed、反復数を固定し、評価setをtuningへ使わない。
- 各variantのartifact bytes、target set、resident bytes、converter/importer identityを記録し、全weight展開を禁止する。

### P15Q-A3: tensor・layer感度の切り分け

- 48 layer × gate/up/downで`S0/U0/O0`のweight誤差を比較し、差が大きいblockとscale outlierを特定する。
- 固定calibration promptから各MLP入力BF16 activationを取得し、同じ入力でlinear出力、Silu/GELU後、down projection後の
  MSE、cosine、max relative errorを比較する。
- layer単位の単独差し替えと累積差し替えでlogit KLDの増加位置を測る。全48 layerのfull generationを総当たりせず、
  layer-output全件と、worst/median/bestおよび先頭/中間/最終のbounded full-model差し替えを使う。
- `imatrix_mse`の効果が特定projection/layerだけに集中する場合は、全NVFP4とmixed BF16/FP8保持案を分ける。

### P15Q-A4: full-model attribution

- text-only primary setとして日本語、英語、code、math、短文、長文を含む固定prompt/token manifestを作る。
  最低32 promptについて複数teacher-forced位置を比較し、prompt hashと全位置IDを固定する。
- `B0/S0/U0/O0`の同じ位置でlogit KLD、top-1、top-k overlap、logit max errorを取得し、median/p90/maxとworst caseを記録する。
- greedy fixed/Unicode/stop generationはreference token、最初のdivergence位置、finish reasonを比較する。
- primary結論後、必要なW4A4/FP8 attention/FP8 KV supportを実装する価値がある場合だけ`M0`を別providerとして試す。
  `M0`の差はactivation/attention/KV interactionであり、weight algorithmのprimary結果へ混ぜない。

### P15Q-A5: 採否、文書同期、closeout

- attribution表へsource identity、format mapping、S0/U0/O0比較、layer感度、full KLD、top-1、resident/peakをまとめる。
- converter改善が主要因なら、最小のscale/calibration candidateを採用または理由付きで棄却する。model-specific calibration
  artifactが必要ならmodel lock、GGUF Phase 19への格納、ホビーユーザー向け変換時間/容量も記録する。
- format ceiling寄与が大きい場合は、NVFP4を`correctness-only opt-in`に維持し、sensitive tensorをBF16/FP8へ残す案、
  別group/format、W4A4の有無を独立follow-upとして提示する。
- runtime、model lock、compatibility、provenance、main plan、historyを同期し、1回のintegration reviewとfindingだけの
  focused re-reviewを行う。本planをarchiveしてからPhase 16を開始可能にする。

## 実施結果

- 固定したBF16/Unsloth artifactの完全hashとheader/catalogを検証し、両artifactに共通する349 BF16 source tensorが
  byte-identicalであることを確認した。Unsloth側の残りBF16 entryは量子化metadataで、source tensorへ数えていない。
- 独立decoderはE2M1全16 code、E4M3、nearest-even tie、zero block、non-aligned境界を確認した。Unslothの
  `weight_global_scale`はreciprocalであり、sLLM sidecarの乗算scaleへ`1 / weight_global_scale`でlosslessにimportした。
- 144 MLP tensorのsampled weight MSEではU0がS0を改善したtensorは0件で、U0/S0比medianは`1.3933`だった。
  O0は120/144 tensorで改善したが、その改善はfull-model品質へ一貫して伝播しなかった。
- exact `gfx1201`/`gfx1030`のoperator境界caseはすべてHIP dispatch、fallbackなしでPASSし、最大relative errorは
  `0.0036375308`だった。32 fixed prompt・96位置のfull-model結果は次の通りである。

| target | variant | KLD median / p90 / max | top-1一致 |
| --- | --- | --- | ---: |
| R9700 `gfx1201` | S0 | `0.3315 / 3.4727 / 11.7972` | `61.46%` |
| R9700 `gfx1201` | U0 | `0.1619 / 2.3621 / 9.1781` | `79.17%` |
| R9700 `gfx1201` | O0 | `0.2880 / 2.1219 / 14.4025` | `65.63%` |
| V620 `gfx1030` | S0 | `0.3715 / 3.5324 / 5.1655` | `62.50%` |
| V620 `gfx1030` | U0 | `0.1736 / 1.9045 / 7.5777` | `76.04%` |
| V620 `gfx1030` | O0 | `0.3433 / 2.4327 / 6.4180` | `69.79%` |

- U0がS0より低KLDだった位置はR9700 66/96、V620 61/96だった。R9700のlayer単独差し替えではU0が特に
  layer 0/1/47を改善したが、選択6 layerの累積U0はmax KLD `12.5620`となり、改善はlayer/prompt依存だった。
- 3 fixed greedy caseの最初のdivergence位置はS0が`[なし, 7, なし]`、U0/O0が`[0, 1, なし]`であり、
  median KLDの改善は生成trajectoryの一貫改善を意味しなかった。
- よって、activation-aware algorithmの寄与はmaterialだが一様ではなく、同じformat/configurationのceilingも残る
  `mixed`と判定した。既存KLD budget `0.05`は変更せず、S0/U0/O0をdefaultまたはproductionへ採用しない。
  NVFP4は両targetで`correctness-only opt-in`を維持する。M0はW4A4/attention W8A8/KV FP8の未実装差が混ざるため、
  primary attribution後の必須caseにはせず実行していない。
- B0 residentは`23,814,729,316` byte、candidate residentは`11,605,373,092` byteだった。両targetのfull runは
  fallbackなし、nonfinite 0、cleanup 0で完了した。model、sidecar、raw reportはrepositoryへ追加していない。
- integration reviewでCI Rust toolchainが`if let` chainを受理しない互換性findingを検出し、同じ意味のnested構文へ修正した。
  focused re-reviewとworkspace、manifest、Markdown、exact target checkを通してcloseoutした。

## 計測matrix

| level | variant | case | 主指標 |
| --- | --- | --- | --- |
| format | Unsloth payload/independent decoder | 全code、block/tensor scale、odd/non-aligned tail | byte、decode値、scale式 |
| tensor | S0/U0/O0 | 144 MLP tensor、全block | MSE、cosine、SQNR、zero/saturation |
| operator | S0/U0/O0 | layer 0/mid/final gate/up/down、production M/K/N | output error、provider、fallback |
| layer | S0/U0/O0 | 48 layer MLP入力と出力 | layer error、累積logit KLD |
| full logits | B0/S0/U0/O0 | 32+ fixed prompt、複数位置 | KLD median/p90/max、top-1/top-k |
| generation | 最終candidate | fixed/Unicode/stop、連続request、OpenAI SSE | divergence、finish、cleanup |
| secondary | M0 | primary結論後の代表case | W4A4/FP8 attention/KV相互作用 |

## 非対象

- Unsloth/vLLM/CUTLASS kernelの移植、NVIDIA性能値のAMDへの転用。
- 量子化artifact、raw logits、calibration corpus、model sliceのGit追跡。
- vision/audio full-model、MoE、multi-GPU、KV量子化のproduction実装。
- KLD thresholdの緩和、Unsloth model cardのbenchmark値をsLLM PASSとして再利用すること。
- Phase 19より前のユーザー向け形式のGGUF全面移行。調査用importerは最終GGUF設計へ渡せるmetadataを保持する。

## 停止・再計画条件

- exact BF16 source同一性を確定できない場合、full-model algorithm対format結論を出さずsource lock調査へ戻る。
- packing/scale contractが現行NVFP4と異なる場合、raw reinterpretを止め、別encoding/import schemaとして再計画する。
- 12B BF16 controlとUnsloth artifactを同時保持できない場合、hash固定した逐次resident切替へ変更し、異なるsource/runを混ぜない。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索を止めて再計画する。
- timeout、crash、CPU fallback、zero case、source mismatch、非finite logitsをPASSにしない。

[対応する履歴](../../../../../history/2026/08/11-20/phase15q-unsloth-nvfp4-quality-attribution.md)
