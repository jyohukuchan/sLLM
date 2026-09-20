# Phase 87: Qwen3.8 NVFP4の単一要求decode最適化とW×A16の廃止

## 状態

- 段階0・WU0・WU1・WU1.1・WU-C1完了（WU0再計測・WU1/WU1.1完了は2026-09-20）。WU2以降は未着手。[Phase 76〜88計画](../1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)のPhase 87を置き換える。
- 両GPU・MTPなし／ありの通常速度、kernel時間、read-request counter、形状別copy、R9700の2 KV形式のKLD、
  W×A16利用箇所と過去の棄却候補の棚卸しを完了した。
  [段階0履歴](../../../../../history/2026/09/11-20/phase87-stage0.md)と
  [結果JSON](../../../../../history/2026/09/11-20/phase87-stage0-results.json)を証拠の正本とする。
- 2026-09-19のレビュー補正で、当初のcounter／copy指標が行列積の余地を過小評価していたことを確認し、
  重みの論理byte数で優先順位を訂正した（[段階0履歴のレビュー補正](../../../../../history/2026/09/11-20/phase87-stage0.md#レビュー補正2026-09-19)）。
  その後WU0で一律90%仮定をread実測へ置き換えた。2026-09-20にクロック状態を補正して再計測し、
  V620 attentionの余地12.41、R9700 attention 2.79、R9700 FP8の正の形状別余地4.08 ms/tokenを得た。
  WU1でV620のGQA共有を採用し、MTPなし+8.09%／あり+6.88%を確認した。R9700は現行維持。
  WU1.1では新採用基準に基づき、両GPUのlong contextへsplit128を採用した。WU-C1で不要な切替を削除し、既定出力のN0を確認した。
  次にWU2のR9700 FP8、融合等を扱う。
  下記の段階番号の並びを優先順位とはしない。着手順は「作業単位」の節（WU0→WU1→WU1.1→WU-C1→WU2）に従う。
- GPU空白時間（通常計測でMTPなしV620約5.8、R9700約2.7 ms/token）は、kernelごとの数µsの隙間と
  tokenごとのhost往復から成る。2026-09-19のユーザー決定で、decode 1段全体のHIP graph化と、
  MTPなし・ありのサンプリング経路のCPU-GPU間通信削減を段階5としてPhase 87に追加した。

## 方針の前提（2026-09-19ユーザー決定）

ユーザーが`README.md`に書いた方針（ハードウェア特性とユースケースの検討結果）に従う。要点は次のとおり。

- 対応する重み／活性値の組は BF16/BF16、FP8/FP8、MXFP8/MXFP8、MXFP6/MXFP6、MXFP4/MXFP6、NVFP4/NVFP4（EXL3は未定）。
  活性値だけBF16の低精度経路（**NVFP4 W4A16、MXFP8 W8A16、MXFP6 W6A16**）は廃止する。
- MXFP4の活性値はMXFP6（W4A6）とし、2026-09-03の「MXFP4はW4A8のみ」決定を置き換える。
- coding agent等のagent的な用途を重視し、過度な量子化や遅いハードウェアには対応しない。
  高バッチ時の合計生成速度で有利な、ネイティブ対応のある形式（MXFP／NVFP4／FP8）を優先する。
- Qwen3.8では基本的にNVFP4だけを考える。Phase 87は、本番のQwen3.8-27B NVFP4（Unsloth、混合精度）の
  単一要求decodeを対象とし、その中の**FP8部分（重みの約40%）もPhase 87の範囲に含める**。
- MTP companionをNVFP4化してMXFP6と比べ、採用率が5%以上落ちる場合はMXFP6も含める。
- 2026-09-19の段階0実測でV620のdecode attentionの寄与が大きいと確認し、ユーザーは
  「attentionも候補に含め、帯域・時間から優先順位を決める」と承認した。
  decode attentionを後続最適化の候補に追加する。段階0では計測・順位付けまで行う。

- 2026-09-19のユーザー決定で、decode 1段全体のHIP graph化と、MTPなし・ありの両経路で
  サンプリング経路のCPU-GPU間通信時間の削減をPhase 87の範囲に加える（段階5）。

旧Phase 87の項目のうち、MXFP8／MXFP6本体のdecode残差とMXFP4の新形式（W4A6）はPhase 87から外し、
番号を割り当てない後続項目とする（下記「対象外」）。

## 現状の数字（出発点）

Qwen3.8 NVFP4の重みは、NVFP4 W4A4（MLPのgate/up/down、64層中56層、パラメータの55.7%）、
FP8 W8A8（重みは出力チャネルごと、活性値はtokenごとの動的scale。attention、GDNの投影、最後の8層のMLP、
lm_head。39.5%）、BF16（embedding、GDNのA/B。4.8%）の混合である。

MTPなしdecodeの実測（Phase 83.5最終、8192/128）と、1 tokenで読む重み約19GBからの実効帯域:

| GPU | decode | 実効帯域 | ピーク比 |
| --- | ---: | ---: | ---: |
| R9700 gfx1201 | 19.1 tok/s | 約364GB/s | 640GB/sの約57% |
| V620 gfx1030 | 14.3 tok/s | 約273GB/s | 512GB/sの約53% |

2026-09-19の段階0で、現行Qwen3.8のM=1 decodeは既にNVFP4 W4A4であることをsourceと
R9700の実行audit（ID84）で確認した。計画作成時の「現行はW4A16」という前提は誤りだった。
同日のユーザー指示により現行W4A4を基準とし、不要なW4A16→W4A4の移行比較は省く。
W×A16の廃止対象は、別途棚卸しする旧sidecar互換経路・公開ABI・MX A16 opt-in等である。

## 進め方の原則

- **計測で優先度を決める。** 作業の順番は段階0の内訳から決め、上の項目の並び順では決めない。
- **1作業単位＝1仮説。** 着手前に、対象カーネルと形状、段階0のデータに基づく仮説、到達可能帯域から計算した
  改善の上限を書く。候補は1作業単位あたり3つまでとし、上限の半分に届かなければ打ち切り、効かなかった理由を記録する。
  打ち切り線は探索を続けるかどうかの判断であり、候補の採否は次の採用基準で別に決める（2026-09-20ユーザー決定）。
- **採用基準（候補ごと・GPUごと）。** 次をすべて満たす候補は、打ち切り線に届かなくても採用する。
  1. 正しさ: 独立oracle、finite、repeat、cleanupがPASSする。
  2. 数値分類: N0／N1は数値・出力影響変更台帳の規則どおり自動承認し、N2はユーザーが判断する。
  3. 速度: 同一processのAB／BAの全roundで改善の向きが揃い、本番形状の短縮が通常計測のTPOTの1%以上。
     モデル全体の1 warmup＋3 measuredには約0.5%の揺れがあるため、採否は単体ベンチの短縮量で判断し、
     モデル全体の速度は記録する。
  4. 他のcontext長・M・対象GPUで遅くならない（対象を限定して採用してよい）。
  元に戻すための環境変数は要求しない（Gitで戻せるため）。比較のcontrolに必要な切替だけを置いてよい。
- **毎回同じ物差しで測る。** 作業単位ごとに、本番形状の単体ベンチ（両GPU）、独立FP32 oracle、
  モデル全体のdecode tok/s（MTPなし、代表8192/128、1 warmup＋3 measured）を取る。
- **作業単位ごとにレビューする。** Codexへは作業単位ずつ渡し、結果を確認してから次へ進む。一括の丸投げはしない。
- **過去の棄却案を繰り返さない。** Phase 78〜85で試して棄却した候補（`native/hip/tests/phase78_*`の各probe、
  履歴の不採用一覧）を段階0で棚卸しし、同じ案は前提が変わった場合だけ再検討する。
- 新しい速度下限は設けない。数値が変わる変更は数値・出力影響変更台帳へ記録する。

## 作業単位（着手順、2026-09-19）

段階番号とは別に、段階0の補正後の優先順位で作業単位を並べる。Codexへは1単位ずつ渡し、結果を確認してから次を渡す。
最初に渡すのはWU0とWU1だけとする。各単位の共通条件は「進め方の原則」に従う。

### WU0: 読み出し専用の帯域上限（計測のみ）

- **完了（2026-09-19）**。両GPUで同じ73 payload、copy/read各3起動、各payload 3 warmup＋9 measuredを確認した。
  [WU0履歴](../../../../../history/2026/09/11-20/phase87-wu0-read-bandwidth.md)と
  [結果JSON](../../../../../history/2026/09/11-20/phase87-wu0-read-bandwidth-results.json)を正本とする。
  2026-09-20に、計測前の連続warmup（300 ms）がなくクロックが上がり切る前に測っていた点を補正して再計測した
  （[再計測](../../../../../history/2026/09/11-20/phase87-wu0-read-bandwidth.md#再計測-クロック状態の補正2026-09-20)、
  [再計測結果JSON](../../../../../history/2026/09/11-20/phase87-wu0-read-bandwidth-warm-results.json)）。
  WU1のV620は余地12.4143／半分6.2072 ms/token、WU2のR9700 FP8は正の形状別余地の合計4.0813／半分2.0406 ms/token。
  読み出し専用kernelによる参照値であり、演算・復号の費用を含まない。

- 目的: 補正で仮定した「ピークの90%」を実測値に置き換え、WU1以降の上限と打ち切り線を固定する。
- 方法: 段階0の`native/lowp/tests/phase87_copy_bandwidth.hip.cpp`と同じ予熱・巡回（1 MiB以上は512 MiB以上の領域を巡回、
  Infinity Cacheに収まらない条件）で、書き込みをほぼ伴わない読み出し専用kernel（`uint4`読み出し→register累積→
  block当たり1値だけ書く）を追加する。copyとの比較のため同じpayload一覧（行列積の実K×N、attentionのKV約17.3 MB、
  非整列境界を含む）で両GPUを測る。
- 出力: payloadごとの読み出しGB/s。段階0の結果JSONと同じ行に並べ、「余地」の再計算表を段階0履歴へ追記する。
- 参考値: 段階0でV620のFP8 `K5120,N17408`は465 GB/s（91%）、lm_headは487 GB/s（95%）に達しており、
  V620の上限は90%以上と見込む。

### WU1: V620 MXFP8 E4 decode attention stage1

- **完了（2026-09-20）**。C1 GQA共有（tile8）をV620のM1〜3へ採用。context 8256でstage1を
  6.3022 ms/token短縮し、6.2072の打ち切り線を超えた。モデル全体はMTPなし14.295→15.451 tok/s、
  あり25.430→27.181 tok/s。C2/C3とR9700は打ち切り線未達で不採用。C1 tile16の退行も記録した。
  両GPUの独立oracle・境界・公開APIとモデル比較がPASS、全4モデル構成の生成列は前後一致。
  [WU1履歴](../../../../../history/2026/09/11-20/phase87-wu1-attention.md)と
  [集約結果](../../../../../history/2026/09/11-20/phase87-wu1-attention-results.json)を正本とする。

- 対象: `native/hip/src/causal_attention_kernel.hip.cpp`の
  `causal_attention_decode_wave_split_staged_stage1_kernel<true, 6, 32>`（ID93 staged32、既定）。
  Qwen3.8 full attention 16層、q_heads 24、kv_heads 4（GQA 6）、head_dim 256、MXFP8 E4 KV、M=1〜3。
  V620・MTPなしで13.18 ms/token（1層1回約0.82 ms、KV約17.3 MBを約21 GB/s）。
- 仮説: 帯域ではなく次の二つが律速している。
  1. GQAの重複: 1 blockが1 query headだけを担当し、同じK/Vを6 query headが別々に読み、gfx1030では
     MXFP8のsoftware decodeも6回行う（counter上の読み出し量はcacheで1倍に見えるが、decode演算は6倍）。
  2. keyごとの直列依存: 1 wave32が約256 keyを1個ずつ処理し、keyごとに内積→5段の`__shfl_down`→lane 0の
     `expf`→broadcast→V累積が直列に並ぶ。blockは1層768個、`__launch_bounds__(32, 1)`で遅延を隠せない。
- 候補（最大3）:
  - C1: GQA共有。1 blockがkv head 1個の6 query headを担当し、K/Vの読み出しとdecodeを1回にして6本の内積と
    6組のonline softmaxを保持する。splitとstage2の形は維持する。
  - C2: 複数key処理。1反復で4〜8 keyを扱い、laneをkey方向にも割り当てて、softmax更新をkeyのまとまり単位にする。
    C1と組み合わせた版を含めてよい。
  - C3: splitとoccupancyの調整（splits 32→64／128、block内wave数）。安価な対照として測る。
- 上限と打ち切り: WU0再計測のV620 read参照はcontext平均8256で364.1 GB/s。
  unique KV 17,436,672 byte×16層の参照時間0.7662 ms/tokenに対し、現行stage1は13.1805 ms/token。
  余地12.4143、その半分**6.2072 ms/token（約6.21）**を打ち切り線とする。
  最良候補の短縮がこれに届かなければ打ち切り、理由を記録する。演算・復号を省いた条件付き参照であり、物理限界ではない。
- 数値: 計画時はC1／C2ともN1相当を予想した。C1は実装で累積順を維持できたためN0とし、controlとのbitwise一致も確認する。独立FP32 oracle、context長1023／1024／1025／8192／8193、M=1〜3、
  全出力finite・repeatを確認し、数値・出力影響変更台帳へ記録する。control（現行ID93）との同一process AB／BA比較で速度を測る。
- 既存の試行との関係: Phase 83.5のstaged32 2-lane expf案（棄却、改善なし）と、R9700での`decode_scaled` loader
  （棄却）を繰り返さない。GQA共有のdecode（Phase 82のGQA6 decode P64）はFP16 KVだけで、MXFP8 E4では未試行。
  prefill側のQ8/W16はGQA共有とowner-lane softmaxを採用済みで、その構造を参考にできる。
  llama.cppの`fattn-vec`（GQA、複数key）も参考・流用してよい。流用する場合は取り込み記録に残す。
- 範囲: V620を主対象とし、採用候補はR9700でも同じ条件で測る（WU0再計測の参照で余地2.7869、半分1.3935 ms/token）。
  target別の採否を許す。
- 計測: 単体ベンチ（両GPU）と、MTPなし・あり8192/128のdecode tok/s（1 warmup＋3 measured）。
- 変更してよいファイル: `native/hip/src/causal_attention_kernel.hip.cpp`、
  `native/hip/src/causal_attention_kernel_internal.hpp`、`native/hip/src/causal_attention_runtime.inc`、
  新規のscratch probe（`native/hip/tests/phase87_*`）と、それに伴うCI hash manifest。
  C1のphysical grid変更に合わせ、Rust metadata validatorと既存host/public GPUテスト・fake HIP宣言も同期する。

### WU1.1: decode attentionの組合せと採用基準による再判定

- **完了（2026-09-20）**。両GPUともKV長8192以上・M1〜3にsplit128をN1として採用。
  V620はC1 GQA共有との組合せ、R9700は通常wave。short context／M4はsplit32を維持する。
  同一processのstage1+stage2短縮はV620 1.196848、R9700 1.157152 ms/token（通常TPOT比1.849%／2.221%）。
  long context全形状・全roundで改善し、独立oracle・公開API・モデル検証もPASS。
  モデル実測はV620 MTPなし15.451→15.831／あり27.181→28.883 tok/s、
  R9700なし19.192→19.589／あり35.130→33.932 tok/s。N1による生成列・受理数の差を記録し、
  R9700 MTPありの低下も含めて、単体による採否とモデル実測を分けた。
  [WU1.1履歴](../../../../../history/2026/09/11-20/phase87-wu1-1-attention.md)と
  [集約結果](../../../../../history/2026/09/11-20/phase87-wu1-1-attention-results.json)を正本とする。

WU1は候補を単独で比べたため、仕組みの異なる候補の組合せを評価していない。上の採用基準で次の2点を扱う。

- **V620 C1×C3**: C1（GQA共有）は1層のblock数を768から128（4 KV head×32 split）へ減らし、72 CUのV620で
  並列度が不足している可能性がある。C3（split増）は単独で5.01 ms/tokenを短縮しており、仕組みが補完的である。
  - 候補（最大3）: C1 tile8＋split64、C1 tile8＋split128、context長に応じたsplit数の選択（前2者の結果で必要な場合のみ）。
  - controlはWU1完了時のC1（当時の比較用切替はWU-C1で削除済み）。
  - 上限と打ち切り: WU1後のC1 stage1は0.336 ms/層、16層で約5.38 ms/token。WU0再計測の参照0.766 ms/tokenとの差
    約4.61 ms/tokenを余地とし、その半分約2.31 ms/tokenを打ち切り線とする。
- **R9700 C3**: WU1のsplit128は1.1725 ms/token（打ち切り線1.3935未達）だが、TPOT約52 msの約2.2%で
  採用基準の速度条件を満たす見込み。split64／128／256を比べて最良を採否判定する（C1はR9700で効果がなかったため組み合わせない）。
- 数値: split数の変更はpartitionとstage2の合成順を変えるため、controlとのbitwise一致を前提にしない。
  各partitionの直列和が短くなりstage2の合成が増える変更として、誤差boundが非増加であることを解析で示せればN1とする。
  示せない場合はN2としてユーザーへ提示する。独立oracle、context長1023／1024／1025／8192／8193／8256、M=1〜3を確認する。
- 計測: WU1と同じ（300 msの連続warmup、同一process AB／BA 3 round×9 sample、stage1とstage2を分けて記録）。
  採用した場合は両GPUのMTPなし・ありの通常8192/128を測り、数値・出力影響変更台帳へ記録する。
- 変更してよいファイル: WU1と同じnative 3ファイル、`crates/sllm-hip/src/kv_state.rs`のmetadata検査、
  対応するhost／public GPUテストとfake宣言、`native/hip/tests/phase87_wu1_*`、CI hash manifest。

### WU-C1: 直近作業の不要な切替と実験経路の削除（WU1.1の後、WU2の前）

- **完了（2026-09-20）**。下記8環境変数と旧CLI scale flags、専用実験経路・診断統計・activation CPU参照を削除。
  両target release/host/GPU/evidence/public API、3種のH3 validatorsがPASS。CI hashの参照連鎖は2 iterationで収束した。
  WU1.1最終binaryとの両GPU MTPなし8192/128（1 warmup＋1 measured）は生成token列が完全一致（N0）。
  [WU-C1履歴](../../../../../history/2026/09/11-20/phase87-wu-c1-cleanup.md)と
  [集約結果](../../../../../history/2026/09/11-20/phase87-wu-c1-results.json)に証拠を記録した。既存履歴は変更していない。

採用基準で環境変数による元戻しを要求しないことにした（2026-09-20）。直近の作業で追加された切替と、
不採用にした実験だけのコード経路を削除し、既定の挙動を1つにする。**既定経路の数値・出力は変えない（N0）。**

- 削除する環境変数（読み出し箇所、それだけで到達する分岐、テスト・evidence tool・microbenchの対応箇所を含む）:

  | 環境変数 | 由来 | 削除後 |
  | --- | --- | --- |
  | `SLLM_CAUSAL_ATTENTION_DECODE_GQA_SHARED` | WU1 | V620のGQA共有を無条件に適用（採用scopeは維持） |
  | `SLLM_CAUSAL_ATTENTION_DECODE_SPLIT128` | WU1.1 | KV長8192以上・M1〜3のsplit128を無条件に適用 |
  | `SLLM_MXFP8_ACTIVATION_LEGACY_FLOOR_SCALE` | scale選択 | MXFP8活性値は飽和しない規則のみ |
  | `SLLM_NVFP4_ACTIVATION_BEST_OF_TWO_SCALE` | scale選択（不採用） | 2候補選択のkernel経路・CPU参照ごと削除 |
  | `SLLM_MXFP4_ACTIVATION_BEST_OF_TWO_SCALE` | scale選択（不採用） | 同上 |
  | `SLLM_MXFP6_ACTIVATION_NO_CLIP_SCALE` | scale選択（不採用） | MXFP6活性値は従来の規則のみ |
  | `SLLM_KV_MXFP8_NO_CLIP_SCALE` | scale選択（不採用） | KVは従来の規則のみ |
  | `SLLM_NVFP4_ACTIVATION_SATURATION_STATS` | scale選択の診断 | 統計採取ごと削除 |

- 変換CLI（`sllm-convert-gguf`、`sllm-convert-qwen38-mx`、`sllm-convert-qwen38-mtp`）の`--legacy-floor-scale`と、
  既定になった規則を明示するだけの旧フラグ（`--mxfp8-no-clipping-scale`、`--mxfp6-no-clipping-scale`等）を削除する。
  旧フラグの指定は未知の引数としてエラーにしてよい。
- 残すもの: `SLLM_PHASE87_PROFILE`（ベンチ用binaryの計測区間マーカーで、推論経路に入らない）。
  従来の床関数そのもの（KV、MXFP6活性値、MXFP4で使用中）は削除しない。削除するのは切替だけである。
- 対象外: WU1.1より前からある約200個の`SLLM_*`環境変数（attentionのopt-in・HOLD・`FORCE_BASELINE`系等）。
  棚卸しと整理は別の作業単位とし、ここでは触らない。
- 検証:
  - 両targetのrelease build（`-ffp-contract=off`）、Rust test（`sllm-hip`の`kv_state`、変換CLI、evidence tool）、
    native host selector test、lowpのhost／GPUテスト（`low_precision_block_codec_gpu_test`等）。
  - attentionの公開API GPUテスト（両GPU、1023/1024/1025/8191/8192/8193/8256×M1〜4）。
  - N0の確認として、WU1.1最終binaryと削除後binaryで、両GPUのMTPなし8192/128（1 warmup＋1 measured）の
    生成token列が一致すること。
  - CI hash manifestを更新する。`rmsnorm-h3-compile-v1.json`は`hip-runtime-compile-v1.json`自体のhashを持つため、
    後者を更新した後に前者も更新し、変化がなくなるまで繰り返す（WU1とWU1.1の両方でこの連鎖が古いまま残っていた）。
    `validate_h3_contracts.py`、`validate_h3_public_runtime_contracts.py`、`validate_rmsnorm_h3_contracts.py --non-strict-local`
    等のvalidatorでPASSを確認する。
- 文書: 数値・出力影響変更台帳の「rollback」「比較control」欄にある環境変数の記述を、Gitでの差し戻しへ書き換える。
  `docs/architecture/runtime.md`等の現行説明から削除した切替を除く。履歴文書（`docs/history`）は当時の記録として変更しない。
- 変更してよいファイル: 上の環境変数・CLIフラグを参照するsource、test、evidence tool、microbench、CI hash manifest、
  数値・出力影響変更台帳、`docs/architecture/runtime.md`、この計画。

### WU2: R9700 FP8 W8A8 decode projection（WU-C1の後に渡す）

- 対象: gfx1201のhipBLASLt M=1（MT16x16x3系）。MTPなしの形状別（ms/token、論理byte数による帯域）は次のとおり。

  | 形状 | 回数/token | ms/token | GB/s（640比） |
  | --- | ---: | ---: | ---: |
  | K5120,N10240 | 48 | 5.448 | 462（72%） |
  | K6144,N5120 | 64 | 4.651 | 433（68%） |
  | K5120,N17408 | 16 | 3.373 | 423（66%） |
  | K5120,N6144 | 48 | 3.064 | 493（77%） |
  | K5120,N12288 | 16 | 2.259 | 446（70%） |
  | K17408,N5120 | 8 | 1.293 | 551（86%） |
  | K5120,N1024 | 32 | 0.691 | 243（38%） |
  | lm_head K5120,N248320（MT64x64） | 1 | 2.324 | 547（85%） |

  別kernelの活性値量子化`sllm_matmul_bf16_to_fp8_outer_v2`が185回/token、1.64 ms/token。
- 仮説: M=1にGEMMのtile（MT16x16）を使うため、読み出しの並列度とcache効率が不足している。
  gfx1201のnative FP8変換を使う専用のM=1 GEMVなら、V620の自前FP8 kernelと同程度（85〜90%）に届く。
- 候補（最大3）:
  - C1: native FP8変換（gfx12の`v_cvt_pk_f32_fp8`系）とdword幅の読み出しによるM=1専用GEMV。
    Phase 83.5で棄却されたgfx1030 ID92 bodyの移植（small-M、約2.2倍遅い）とは別構造にする。
  - C2: C1に活性値量子化を融合する。各blockがK要素（最大17,408個）のamaxをL2から冗長に計算し、現行quantizerと
    同じscale式で量子化する。別kernelの185回/tokenの起動をなくす。
  - hipBLASLtのrank／algorithmの再選択は、Phase 82で再現しなかったため候補にしない。
- 上限と打ち切り: WU0再計測の形状別read実測と論理weight bytesで90%仮定を置き換えた。
  正の形状別余地の合計は**4.0813 ms/token**、打ち切り線はその半分**2.0406 ms/token（約2.04）**。
  主な余地は`K5120,N10240` 1.009、`K5120,N17408` 0.978、`K6144,N5120` 0.850、`K5120,N12288` 0.525 ms/token。
  半分に届かなければ打ち切る。
  活性値量子化の融合による追加利益は、このweight読み出しの参照値に含めない。
- 数値: C1は累積順が変わるためN1相当。C2は現行quantizerと同じscale・丸めならN0を目指す。
  M=1〜3（MTP verify）、非整列K／Nで独立FP32 oracleを確認する。

### 後続の作業単位（WU2の後に具体化する）

- V620 FP8の残差（主に`K6144,N5120`の約0.68 ms/token）、NVFP4 W4A4 decode（段階1。WU0再計測の余地は
  V620約0.83、R9700約1.41 ms/token）、
  活性値量子化・RMSNorm等の融合、段階5のgraph化・サンプリング経路の通信削減、段階3のMTP companion、段階4のW×A16廃止。
  順番はWU0〜WU2の結果で決め直す。

## 作業段階

### 段階0: 計測と棚卸し

1. **decode 1 tokenの内訳**（両GPU、MTPなしとMTPあり）: カーネルごとの時間、読んだbyte数、実効帯域を取り、
   形状ごとの到達可能帯域（単純なcopyカーネルで実測）と並べる。行列積（NVFP4、FP8）、GDN、attention、
   活性値量子化、RMSNorm等、起動・同期の待ちを分けて集計する。
2. **現行W4A4の基準**: M=1のNVFP4部分が既存W4A4 decodeカーネルで実行されることを確認し、速度と
   KLD（Qwen3.8 KLDコーパス、R9700、chunk32、FP16 KVと既定のMXFP8 E4 KV）を記録する。
   現行がW4A4と確認されたため、2026-09-19ユーザー指示で不要なW4A16への切替・移行比較は省く。
3. **優先順位表**: 「時間の割合 ×（到達可能帯域 − 実効帯域）」で並べ、以降の作業単位を決める。
   decode attentionは2026-09-19に対象への追加をユーザー承認済み。
   その他の行列積以外（起動・同期、GDN等）が大きい場合は、Phase 87に含めるかをユーザーに確認する。
4. **W×A16の利用箇所の棚卸し**: NVFP4関連の各経路（Qwen3.8、Gemma 4 12B Unsloth NVFP4、Gemma 4 26B-A4B MoE、
  nvidia Gemma 4 31B NVFP4等）を現行の実行形式ごとに区別し、残るW4A16とMX A16 opt-in（Phase 85のID101／102）の全利用箇所を列挙し、
   各配布物がW4A4に必要な活性値のglobal scaleを持つかを確認する（Gemma 4 12B Unslothは144個を持つことを確認済み）。

### 段階1: NVFP4 W4A4のM=1 decode

- 段階0の結果に基づき、W4A4の行列ベクトル積と、M=1の活性値量子化（行列ベクトル積の前処理への融合を含む）を最適化する。
- M=2〜3（MTP verify）の形状も同じ作業単位で扱う。
- 現行のW4A4既定を維持する。最適化で数値が変わる場合は、その変更内容に基づいて数値分類を記録する。

### 段階2: FP8 W8A8のdecode

- gfx1201（hipBLASLtのnative FP8）とgfx1030（software経路、Phase 78のID68／71／75／82等）について、
  tokenごとの活性値量子化と行列ベクトル積を段階0の優先順位に従って最適化する。lm_headを含む。

### 段階3: MTP companionの形式

- MTP companionをNVFP4 W4A4（重みはsLLMの量子化器、活性値のglobal scaleは較正で決める）とMXFP6 W6A6で作り、
  [MTP採用率ベンチマーク](../../../../../development/mtp-acceptance-benchmark.md)の主指標M1（期待受理率、両GPU）で比べる。
  あわせて実効decode速度も記録する。
- NVFP4の採用率がMXFP6より5%以上低い場合は、MXFP6 companionも対応に含める。
  「5%」は期待受理率の差5ポイントと解釈する（相対5%の場合は着手前にユーザーへ確認する）。
- 活性値global scaleの較正に使う入力は、ベンチマークの評価入力と重ならないものにする。

### 段階4: W×A16の廃止と契約の整理

- NVFP4 W4A16、MXFP8 W8A16、MXFP6 W6A16の実行経路・provider・selector・evidence toolを削除する。
  段階0の棚卸しで挙がった全モデルがW4A4等で動くことを、両GPUで確認してから削除する。
- lowpの公開C APIのMXFP4契約をW4A8 v1からW4A6へ書き換える（実装は後続。planは未対応として拒否のまま）。
- main-plan、`docs/architecture/runtime.md`、数値変更台帳を更新する。

### 段階5: decode実行制御（graph化とサンプリング経路の通信削減）

段階0の補正で、GPU空白はkernel 1,171個/token（MTPなし）ごとの2〜10 µsの隙間と、tokenごとのhost往復
（sampler後の読み戻し、約0.5 msのhost処理、同期的な`hipMemcpyAsync`、次tokenの送り直し）から成ると分かった。
MTPありでは受理判定の読み戻しと、一部受理時のrestore＋replayの起動がhost主導である。

- **decode 1段全体のHIP graph化**: MTPなしの1 token、MTPありのdraft＋verify（＋状態選択）の1段を、
  それぞれ一つのgraphとして再生する。現行の部分的なgraph span（hipBLASLt等）を包含する。
  位置・KV長・attention分割等のtokenごとに変わる値は、device上の値またはgraphの再instantiateを要しない
  引数更新で渡す。
- **サンプリング経路の通信削減（MTPなし）**: sampler結果をdeviceのtoken bufferに置き、次段のembeddingが
  直接読む。hostへのtoken読み戻しは非同期にし、停止判定は1段遅れで行う（余分な1段は捨てる）。
  hostの送り直しと、次段起動前の同期待ちをなくす。
- **サンプリング経路の通信削減（MTPあり）**: 既存のdevice上の受理判定（`speculative_device.rs`）の結果を、
  位置・KV長・次draftの入力（受理位置のtokenとhidden）としてdevice上で次段へ渡す。
  一部受理時のrestore＋replayの扱い（例: verify各位置のGDN状態を保持して選ぶ）は、着手時に候補を比較して決める。
  停止tokenが受理範囲の途中に現れる場合は1ブロック遅れで判定し、余分を捨てる。
- 固定device sampling（現行ベンチの経路）を対象とする。penalty・logit_bias・DRY等のhostが語彙長の配列を
  作る経路と、grammar等のtoken依存の制約は、従来の同期経路を維持してよい（device化は後続項目）。
- 生成token列は変更前と一致させる（数値分類N0）。1段遅れの停止判定で余分に計算した段は、出力・KV・
  状態へ公開しない。
- 各作業単位で、profileのGPU空白時間（kernel間の隙間とtokenごとの往復を分けて）と、通常計測の
  decode tok/s（MTPなし・あり、両GPU）を段階0と同じ条件で記録する。

## 受入条件

1. Qwen3.8 NVFP4の単一要求decodeで、NVFP4部分がW4A4、FP8部分がW8A8で動き、W×A16の経路がコードに残っていない。
2. 変更した各カーネルが、両GPUの独立FP32 oracleと境界ケース（非整列K／N、M=1〜3）でPASSする。
3. decode速度（MTPなし・あり）とKLDを、段階0の基準と同じ条件で記録する。速度下限は設けない。
4. MTP companionの形式を、M1の比較結果とルールに基づいて決め、記録する。
5. 段階0で挙がった全モデルが、W×A16廃止後も両GPUで動作する。
6. 各作業単位の仮説・候補・結果（不採用を含む）が履歴に残っている。
7. MTPなし・ありの固定device samplingのdecodeで、1段全体がgraphとして再生され、次段の起動がhostへの
   token読み戻しを待たない。生成token列は変更前と一致し、GPU空白時間とdecode速度を段階0と同じ条件で記録する。

## 対象外（後続項目）

- MXFP8／MXFP6の本体（Qwen3.5、Qwen3.8のMX GGUF）のdecode残差。
- MXFP4 W4A6の実装（Phase 87では契約の書き換えだけを行う）。
- MXFP6 KV等、READMEにあるKV形式の追加。
- EXL3の導入（流用方針の決定が先に必要）。
- リクエストバッチ処理（Phase 88）。

## 実行時の注意

- Codexは既定のOpenAI profileで動かす（2026-09-19、週間制限の回復によりdeepseek-flashは当面不要）。
  subagentの選択はAGENTS.mdのprofile別規則に従う。
- V620を使う間はローカルQwenサービスを止め、R9700はservice lease手順に従う。

段階0履歴: [計測・棚卸しと優先順位](../../../../../history/2026/09/11-20/phase87-stage0.md)

WU0履歴: [読み出し専用帯域と打ち切り線](../../../../../history/2026/09/11-20/phase87-wu0-read-bandwidth.md)

WU1履歴: [MXFP8 E4 decode attentionのGQA共有](../../../../../history/2026/09/11-20/phase87-wu1-attention.md)

WU1.1履歴: [split増の組合せと新基準での採否](../../../../../history/2026/09/11-20/phase87-wu1-1-attention.md)

WU-C1履歴: [不要な切替と実験経路の削除](../../../../../history/2026/09/11-20/phase87-wu-c1-cleanup.md)
