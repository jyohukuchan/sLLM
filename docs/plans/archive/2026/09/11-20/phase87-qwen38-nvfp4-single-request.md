# Phase 87: Qwen3.8 NVFP4の単一要求decode最適化とW×A16の廃止

## 状態

- **Phase 87は2026-09-26のユーザー決定で完了した。** 段階8はbacklogへ移し、llama.cppとの比較で見つかったkernel本体の差は[Phase 88](../../../../active/2026/09/21-30/phase88-qwen38-llama-kernel-parity.md)で扱う。
- 段階0・WU0・WU1・WU1.1・WU-C1・WU2・WU-D1〜D3・段階2（WU-2Vを含む）・段階5・段階6・段階7・段階9・段階4（WU-4Rを含む）・WU-P1・段階3（WU-3S）完了。段階3はMXFP6 MTP sidecarを退役させ、測定上のTPOT低下を確認したうえで、2026-09-24のユーザー決定によりNVFP4 MTP companionを既定化して閉じた。段階7はproducer融合のC1を88 node/replay、C2を112 node/replayへ採用した（2026-09-24）。
  2026-09-22に別途あった「consumer側へ量子化を取り込む」試行は段階7の対象ではなく破棄済みで、producer融合の評価結果ではない（[記録](../../../../../history/2026/09/21-30/phase87-stage7.md)）。[Phase 76〜88計画](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)のPhase 87を置き換える。
- WU-3Pは2026-09-25完了。段階3後に残ったNVFP4 MTP prefixのprefill退行を、実shape限定の既存DP4A／WMMA選択で改善した。8192/128のprefillはV620 43.515→36.198秒、R9700 20.746→14.916秒（[計画](../../../../archive/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)、[履歴](../../../../../history/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)）。
- 段階1は2026-09-25完了。V620のNVFP4 W4A4 M=1 2形状へSGPR基点hoistをN0で採用し、通常8192/128のMTPなしTPOTは58.069→56.990 ms（1.857%短縮）、MTPありは同等。R9700 M=1の同hoistは全round退行、M=2向けsplit-KもV620で約28%退行したため、R9700 M=1と両GPU M=2〜3は現行を維持する（[履歴](../../../../../history/2026/09/21-30/phase87-stage1.md)）。
- 段階9のMTP draft専用98,304語彙headは2026-09-24に採用した。両GPUのTier A 26条件、同一process AB/BAの全roundで1%以上短縮、MTPなし不変を確認した（[履歴](../../../../../history/2026/09/21-30/phase87-stage9.md)）。
- WU-P1では両GPUのPaged Attention probeが数値・単体性能の受入を通過した。2026-09-24のユーザー指示で、本移行を段階10、attention kernelの最適化を段階11としてPhase 87へ統合した（段階10の詳細は[本移行計画](../../../../../plans/archive/2026/09/21-30/paged-kv-full-migration.md)）。
- 段階12は完了。幅2／3／4選択と両GPUの機能受入、幅4の重大退行防止を確認した。2026-09-25のユーザー決定で既定幅は両GPUとも2に固定し、追加の長時間採否ベンチマークを終了した（[履歴](../../../../../history/2026/09/21-30/phase87-stage12.md)）。段階10・11の元の範囲は維持する。
- 段階10は2026-09-25完了。exact `gfx1030`／`gfx1201`をPaged KVへ統一し、最終連続KV対照の10%条件、公開GPU数値、V2 checkpoint、旧VMM／resident退役を確認した。`gfx942`は明示未対応。詳細とartifact不足の証拠範囲は[履歴](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)に記録した。
- 段階11は2026-09-25に実装・採否を完了した。Paged decodeのM=3／KV8192以上でC1の行間KV共有を両GPUへ限定採用し、1 warmup＋2 measuredのTPOTはV620で2.10%、R9700で4.32%短縮。C2のR9700 GQA共有・V620 M1候補は速度条件未達、C3のdot2は退行、FP8 WMMAは品質非増加を満たさず不採用とした（[履歴](../../../../../history/2026/09/21-30/phase87-stage11.md)）。
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
  WU2でR9700 FP8 W8A8のM1へnative dot4 GEMVを採用した。後続は下記の作業単位を結果に基づいて具体化する。
  下記の段階番号の並びを優先順位とはしない。着手順は「作業単位」の節（WU0→WU1→WU1.1→WU-C1→WU2→WU-D1〜D3で完了）に従う。
  D系統は2026-09-20に打ち切り、段階5は2026-09-21に完了（MTPなしの退行も解消）。段階6は2026-09-21に完了。V620のNVFP4 gate/upだけを並列captureへ採用した。
- GPU空白時間（2026-09-20のWU2後に再計測: 通常計測でMTPなしV620約4.9、R9700約3.0 ms/token、
  割合は7.8%／6.5%。[再計測](../../../../../history/2026/09/11-20/phase87-idle-recheck.md)）は、kernelごとの数µsの隙間と
  tokenごとのhost往復から成る。2026-09-19のユーザー決定で、decode 1段全体のHIP graph化と、
  MTPなし・ありのサンプリング経路のCPU-GPU間通信削減を段階5としてPhase 87に追加した。
  段階5の結果、host往復は消えたが空白の大半はgraph内のkernel間dispatch固定費と判明したため、
  独立nodeの並列表現を段階6として追加した（2026-09-21）。

## 方針の前提（2026-09-19ユーザー決定）

ユーザーが`README.md`に書いた方針（ハードウェア特性とユースケースの検討結果）に従う。要点は次のとおり。

- 対応する重み／活性値の組は BF16/BF16、FP8/FP8、MXFP8/MXFP8、MXFP6/MXFP6、MXFP4/MXFP6、NVFP4/NVFP4（EXL3は未定）。
  活性値だけBF16の低精度経路（**NVFP4 W4A16、MXFP8 W8A16、MXFP6 W6A16**）は廃止する。
- MXFP4の活性値はMXFP6（W4A6）とし、2026-09-03の「MXFP4はW4A8のみ」決定を置き換える。
- coding agent等のagent的な用途を重視し、過度な量子化や遅いハードウェアには対応しない。
  高バッチ時の合計生成速度で有利な、ネイティブ対応のある形式（MXFP／NVFP4／FP8）を優先する。
- Qwen3.8では基本的にNVFP4だけを考える。Phase 87は、本番のQwen3.8-27B NVFP4（Unsloth、混合精度）の
  単一要求decodeを対象とし、その中の**FP8部分（重みの約40%）もPhase 87の範囲に含める**。
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
  打ち切り線は探索を続けるかどうかの判断であり、候補の採否は[共通ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)で別に決める。
- **採否（2026-09-24から）。** [main-planの変更の採否ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)に従う。
  正しさ（独立oracle、finite、repeat、cleanup）は前提とする。完了済み作業の当時の採否と測定結果は各履歴に残す。
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

WU1は候補を単独で比べたため、仕組みの異なる候補の組合せを評価していない。当時、次の2点を扱った。

- **V620 C1×C3**: C1（GQA共有）は1層のblock数を768から128（4 KV head×32 split）へ減らし、72 CUのV620で
  並列度が不足している可能性がある。C3（split増）は単独で5.01 ms/tokenを短縮しており、仕組みが補完的である。
  - 候補（最大3）: C1 tile8＋split64、C1 tile8＋split128、context長に応じたsplit数の選択（前2者の結果で必要な場合のみ）。
  - controlはWU1完了時のC1（当時の比較用切替はWU-C1で削除済み）。
  - 上限と打ち切り: WU1後のC1 stage1は0.336 ms/層、16層で約5.38 ms/token。WU0再計測の参照0.766 ms/tokenとの差
    約4.61 ms/tokenを余地とし、その半分約2.31 ms/tokenを打ち切り線とする。
- **R9700 C3**: WU1のsplit128は1.1725 ms/token（打ち切り線1.3935未達）だが、TPOT約52 msの約2.2%で
  split64／128／256を比べて採否を判断した（C1はR9700で効果がなかったため組み合わせない）。
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

- **完了（2026-09-20）**。C1/C2は不採用、C3のnative FP8 dot4 GEMVをexact gfx1201、E4M3FN、下記8形状のM1へN1として採用。
  共通quantizerの揺れを除くdot単体の短縮は3.269757 ms/token（通常TPOT比6.4039%）で、全形状の全3roundが正。
  M2〜3・他target／shapeは従来providerを維持する。両GPU各59探索ケース、公開API・graph・host・CI検証がPASS。
  通常8192/128、1 warmup＋3 measuredのR9700はMTPなし+9.40%／あり+1.50%、V620は−0.03%／+0.15%。
  R9700 MTPなしのみtoken位置15で分岐し、他3構成は前後一致。詳細は
  [WU2履歴](../../../../../history/2026/09/11-20/phase87-wu2-fp8.md)と[結果JSON](../../../../../history/2026/09/11-20/phase87-wu2-fp8-results.json)を参照。

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
  - C3: C2のnative FP8 dot4 coreを量子化から分離し、16-byte読み出しと4独立accumulatorのGEMVとする。
    2026-09-20のC1/C2代表形状の回帰を受けた3番目の候補。
  - hipBLASLtのrank／algorithmの再選択は、Phase 82で再現しなかったため候補にしない。
- 上限と打ち切り: WU0再計測の形状別read実測と論理weight bytesで90%仮定を置き換えた。
  正の形状別余地の合計は**4.0813 ms/token**、打ち切り線はその半分**2.0406 ms/token（約2.04）**。
  主な余地は`K5120,N10240` 1.009、`K5120,N17408` 0.978、`K6144,N5120` 0.850、`K5120,N12288` 0.525 ms/token。
  半分に届かなければ打ち切る。
  活性値量子化の融合による追加利益は、このweight読み出しの参照値に含めない。
- 数値: C1は累積順が変わるためN1相当。C2は現行quantizerと同じscale・丸めならN0を目指す。
  M=1〜3（MTP verify）、非整列K／Nで独立FP32 oracleを確認する。

### WU-2V: 段階2のV620 FP8 decode残件（2026-09-25着手）

- **完了（2026-09-25、新規採用なし）**。WU2が完了させたのはR9700のM=1だけで、V620 `gfx1030` のsoftware FP8 W8A8 decodeをここで扱った。
  既定ID82の本番8形状、M=1〜3のMTP verify、token毎のactivation quantizerとlm_headを測定範囲に含める。
  まずM=1 `(K,N)=(6144,5120)`を候補形状とする。WU0のwarm read参照に対する正の余地はこの形状で
  **0.68118 ms/token**、V620 FP8全8形状で**1.05206 ms/token**。全体の探索継続線は**0.52603 ms/token**。
  `(5120,6144)`の余地は0.08785 ms/tokenなので、単なる専用wrapper追加を初手にしない。
- **仮説C1**: 現行ID82のweight column baseはwave内で同じだが、生成ISAでlaneごとのaddress計算が残る可能性がある。
  wave-uniform baseを明示して、LUT、weight load順、FP32累積順、scale、BF16 RNEを変えずにM=1
  `(6144,5120)`のmemory命令／VGPR費用を下げる。最初はtest-onlyの独立候補としてproduction ID82と同一processで比較する。
  生成ISAが既に同等のscalar化をしている場合、または単体改善が形状余地の半分**0.34059 ms/token**に
  届かなければC1探索を止め、理由を記録する。この値も探索線であり採否の自動条件ではない。
  **C1は2026-09-25に打切り**。両経路はbitwise一致・独立oracle最大0 ULPだったが、
  再確認r2のpattern 0 dotは0.089001→0.088681 ms/call（64回換算+0.02048 ms/token）で、
  探索線0.34059に届かずAB-BA-ABの3roundも改善方向が揃わなかった。
  VGPRは170→169のみ。詳細は[WU-2V履歴](../../../../../history/2026/09/21-30/phase87-wu-2v-v620-fp8.md#c1-wave共通weight-baseの明示scalar化)。
- **後続候補の上限**: C1を含めて最大3 family。C2は既存probeのactivation-shared wave4／4 columnを
  現行ID82と同じLUT decodeへ合わせ、M=1 `(6144,5120)`でactivation再読込だけを減らす。
  C3はC2の結果がなお余地を示す場合に限り、ID82 LUTを維持したdirect-wave4 pair2を検討する。
  C2/C3とも既存probeをproduction成果とみなさず、現行ID82に対する数値・資源・速度で判定する。
  既採用のnon-temporal loadとLDS padding、既棄却のID82 prefetch P1、hipBLASLt全形状切替は再提案しない。
  **C2/C3も2026-09-25に不採用**。両者ともpattern 0〜2で現行ID82とbitwise一致・独立oracle最大0 ULPだったが、
  C2はdot約7〜8%退行、C3は約30〜37%退行した。本番sourceは不変。3候補の結果は
  [WU-2V履歴](../../../../../history/2026/09/21-30/phase87-wu-2v-v620-fp8.md)に記録した。
  残る7形状の正のread参照余地は合計0.370885 ms/tokenで、全8形状の探索継続線0.526032を下回る。
  lm_headの余地は0.094098 ms/token。段階7後に残るFP8量子化97 nodeの具体的な融合候補は
  [backlog P3](../../../../backlog.md)へ移動済みである。既存ID92のM=2〜3は維持する。
  以上を含めて段階2はWU2とWU-2Vで完了し、次は段階1へ進む。
- **採否前の証拠**: exact GPU UUID、300 ms継続warmup、同一process AB-BA-AB各9 samples、quantizerと
  matmulの分離値、512 MiB以上の巡回weight pool、VGPR／SGPR／LDS／occupancy／spillを記録する。
  数値は本番8形状×M=1〜3と非整列境界で独立FP32 oracle、finite、controlとのbitwiseまたは分類済み差、
  repeat、guard、cleanupを確認する。採用する場合は公開dispatchと通常8192/128のMTPなし／ありを
  1 warmup＋3 measuredで検証し、他GPU・他shapeの非退行も確認する。単体の探索線は採用条件と混同しない。
  作業結果は[WU-2V履歴](../../../../../history/2026/09/21-30/phase87-wu-2v-v620-fp8.md)へ記録する。

### WU-D1: NVFP4 M=1 decodeの近傍依存の原因調査（段階1のNVFP4最適化より前）

- **完了（2026-09-20）**。両shapeでisolated／staged型は約123〜124 µs、GQA型のKV reader後は約135〜136 µsとなり、
  全AB/BAで約9〜10%の差を再現した。traceでも同方向。GL2C/EA転送量はほぼ同じでEA busy cycleが増えるが、
  MALL／DRAM stall内訳は取得不能で物理機構の単一帰属は未特定。受入条件に従い特定範囲と限界を記録して終了。
  production sourceは不変。詳細は[WU-D1履歴](../../../../../history/2026/09/11-20/phase87-wu-d1.md)と
  [結果JSON](../../../../../history/2026/09/11-20/phase87-wu-d1-results.json)。

2026-09-20のWU2後の再計測で、V620のM=1 NVFP4 decode `sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1`
（grid 139264）が1 callあたり118.2→129.7 µs（約+10%）になった。
切り分けの結果、**原因はbinaryの内容ではなく、同じtoken内で直前に実行されるdecode attention kernelの種類**である。
旧staged32のときだけ速く、GQA共有（採用済み）でもbaseline（`SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`）でも遅い。
条件表と排除済みの原因は[再計測履歴](../../../../../history/2026/09/11-20/phase87-idle-recheck.md#調査-m1のnvfp4-decodeが約10遅くなる条件)にある。

- 仮説: 同一命令列・同一起動設定・同一読み出し量・同一wave数でcycleだけが約9%増え、GL2C hit率も変わらない。
  増えているのはL2 miss後の待ち時間であり、直前のattention kernelが残すDRAM側の状態
  （page/bank、書き戻しの排出、Infinity Cacheの内容）が次のkernelに影響していると考える。
- 手順:
  1. **単体ベンチ**: attentionを含まない単体で、同じ形状（`K5120,N17408`と`K17408,N5120`、M=1、
     本番と同じNVFP4 weight／activation／scale配置）のNVFP4 decodeを測り、本来の1 call時間を出す。
     旧staged32が速くしているのか、他のattentionが遅くしているのかを決める。
  2. **近傍の再現**: 同じ単体ベンチで、NVFP4 kernelの直前にKV読み出しだけを行うkernelを挟み、
     旧staged32（query head単位で6回読む）とGQA共有（kv head単位で1回読む）のアクセス順・分割数を模したとき、
     NVFP4の1 call時間が両条件で分かれるかを確認する。分かれれば直前kernelのアクセス順が原因と確定する。
  3. **stall内訳**: 1と2で差が再現した条件について、`rocprof-compute`等でNVFP4 kernelのmemory stall内訳
     （EA／MALL／DRAM待ち）を両条件で取得する。取得できない指標は「未取得」と記録する。
- 受入条件: 原因を特定して根拠を示すか、特定できない場合はどこまで絞れたかを証拠付きで記録する。
  性能の修正自体はこの作業単位に含めず、原因が分かった時点で別の作業単位として計画する。
- 制約:
  - 採用済みのWU1／WU1.1／WU2を戻さない。production sourceは変更せず、probe・test・計測ツールと履歴だけを扱う。
  - [再計測履歴](../../../../../history/2026/09/11-20/phase87-idle-recheck.md)で既に排除した原因
    （機材変化、クロック・電力、kernel code、起動設定、workspaceの大きさ、割り当て配置、code量・資源、
    ROCTX、メモリ量とcache hit率、kernelの重なり）を再検証しない。
  - 計測条件はV620・MTPなし・8192/128・1 warmup＋1 measuredのprofileを基準にし、
    単体ベンチは300 ms継続warmupと同一process AB／BAを用いる。
  - 新しい環境変数は追加しない（比較の対照に必要な切替のみ可、Gitで差し戻せる）。
- 変更してよいファイル: `native/hip/tests/`または`native/lowp/tests/`の新規probe、`ci/tools/`の計測・集計script、
  `docs/history/2026/09/11-20/`の新規履歴、この計画。CI hash manifestは触る必要があれば更新する。

### WU-D2: 遅延の持続範囲とgfx1201での再現（WU-D1の続き）

- **完了（2026-09-20）**。V620の遅延は32 dispatch後も残り、量子化／RMSNorm／1 MiB copyで回復しなかった。
  R9700は同じprobeで非再現。MALL／DRAM内訳は未取得で、gfx1201のfetch等も0を返し使用不能だった。
  n=2の重み再利用を修正してdecayを再計測し、全条件・数値検査を確認。
  [WU-D2履歴](../../../../../history/2026/09/11-20/phase87-wu-d2.md)と[結果JSON](../../../../../history/2026/09/11-20/phase87-wu-d2-results.json)。

WU-D1は「GQA型のKV読み出しの直後の1 NVFP4 dispatch」で差を再現した。一方、実モデルのtraceでは
1 token内の168 NVFP4 dispatchすべてが遅く、full attentionのない層にも及ぶ。両者を結ぶ持続範囲を測る。

- 手順:
  1. **減衰の測定**: WU-D1のprobeを拡張し、GQA型のKV読み出し1回の後にNVFP4 M=1を連続n回（n=1,2,4,8,16,32）流し、
     何回目で単独時の水準へ戻るかを両shapeで測る。staged型とisolatedを同じ条件の対照にする。
  2. **介在kernelの影響**: 同じprobeで、GQA型の後にNVFP4以外の代表kernel（活性値量子化、RMSNorm相当の軽いkernel、
     一定サイズのcopy）を挟んでからNVFP4を測り、介在処理で回復するかを見る。回復するなら実用的な緩和策の候補になる。
  3. **gfx1201での再現**: 同じprobeをR9700で実行する。再計測ではR9700のNVFP4も約6%遅く、
     gfx1201ではMALL／DRAM系counterや`rocprof-compute`が使える可能性があるため、物理機構の特定を試みる。
     使えない場合は「未取得」と記録し、gfx1030の結論を一般化しない。
- 受入条件: 持続範囲（何dispatch分か）と、gfx1201での再現有無を数値で記録する。物理機構を特定できない場合は、
  WU-D1と同じく到達点と限界を記録して終える。性能修正はここに含めない。
- 制約: WU-D1と同じ（production source不変、採用済みWUを戻さない、排除済み原因の再検証なし、
  300 ms継続warmup＋同一process AB／BA、新しい環境変数を追加しない）。
- 変更してよいファイル: WU-D1と同じ範囲（probe、`ci/tools/`、`docs/history/2026/09/11-20/`、この計画）。

### WU-D3: decode attentionのKV読み出し方式による罰則の緩和（WU-D2の後）

- **完了・3候補とも不採用（2026-09-20）**。C1 tile32とC3先読みはattentionが大幅に退行し、C2 block配置は単体でほぼ中立。
  C2の実モデルはV620 MTPなし+0.344%／あり−0.093%、token一致。当時の打ち切り線0.95 ms/tokenに未達。
  実attention harnessはD1/D2の合成readerの罰則を再現しないという限界を明記した。
  両GPUの通常モデルと境界検査を記録し、本番sourceとbuild cacheを復元。現象の修正は採用していない。
  [WU-D3履歴](../../../../../history/2026/09/11-20/phase87-wu-d3.md)と[結果JSON](../../../../../history/2026/09/11-20/phase87-wu-d3-results.json)。

WU-D1で、GQA型のKV読み出しが後続のNVFP4 M=1を約9〜10%遅くすることが分かった。実モデルでの影響は
V620で約1.9 ms/token（TPOT比約3%）、R9700で約1.0 ms/tokenに相当する。attention自体の短縮
（V620で8.5 ms/token）を保ったまま、この罰則を減らせるかを試す。

- 仮説: KV読み出しの順序・粒度（block当たりのtoken tile幅、KV headとsplitの割り当て、1 waveが触るアドレス間隔）が
  後続kernelの待ち時間を左右する。読み出し総量と数値結果を変えずに、並びだけを変えれば緩和できる。
- 候補（最大3、WU-D2の結果で絞る）:
  - C1: stage1のKV読み出しをtoken方向の連続幅を広げた順序に変える（8-token tile→32/64-token tile等）。
  - C2: split割り当てを変え、同時に動くblockが触るアドレス範囲を近づける／離す。
  - C3: 読み出しと復号の間にprefetch距離を設け、outstanding requestの分布を変える。
- 上限と打ち切り: 上限はNVFP4側の罰則の解消（V620で約1.9 ms/token）。attention自体が遅くなる分は差し引く。
  半分（約0.95 ms/token）に届かなければ打ち切り、理由を記録する。
- 当時の採否と数値分類は[WU-D3履歴](../../../../../history/2026/09/11-20/phase87-wu-d3.md)に残す。
  attentionと後続NVFP4を合わせて測り、他のcontext長・M・GPUの影響も確認した。
- 計測: 単体ベンチ（attention単体とNVFP4を分けて記録し、合計も出す）と、両GPUのMTPなし・ありの通常8192/128。
- 変更してよいファイル: `native/hip/src/causal_attention_kernel.hip.cpp`、`causal_attention_kernel_internal.hpp`、
  `causal_attention_runtime.inc`、`crates/sllm-hip/src/kv_state.rs`のmetadata検査、対応するtest、
  `docs/history/2026/09/11-20/`の新規履歴、CI hash manifest、この計画。
  採用時は数値・出力影響変更台帳へ記録する。

### D系統（WU-D1〜D3）の打ち切り（2026-09-20ユーザー決定）

GQA型のKV読み出しが後続のNVFP4 M=1を約9〜10%遅くする現象について、これ以上の調査・緩和を行わない。

- 到達点: 合成readerで再現（V620、32 dispatch以上持続、軽い介在処理で回復しない）。R9700の合成probeでは非再現。
  EA busy cycleの増加までは観測したが、MALL／DRAMの内訳は両GPUのcounterに存在せず物理機構は未特定。
  緩和3候補はattention側の退行が大きく、最も中立なC2も実モデルで+0.344%／−0.093%だったため採用しなかった。
- 打ち切りの理由: 影響はV620で約1.9 ms/token（TPOT比約3%）、R9700で約1.0 ms/tokenで、残る候補より小さい。
  特定に必要なcounterがROCm 7.14のgfx1030／gfx1201 catalogにない。緩和候補の費用が利益を上回る。
- 再開の条件: 新しい計測手段（MALL／DRAM内訳を取得できるtoolやGPU）が使えるようになった場合、
  または実attentionのdata flowで罰則を再現するharnessが用意できた場合に限り、前提を変えて再検討する。
- 引き継ぎ: 後続のNVFP4最適化では、isolated値ではなく実モデルのGQA先行条件を対照に使う
  （[WU-D1](../../../../../history/2026/09/11-20/phase87-wu-d1.md)の指摘）。

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

- **完了（2026-09-25）**。V620のexact M=1 wide／downにC1を採用し、R9700 M=1はID84、両GPUのM=2〜3はID94を維持する。R9700 hoist wideはGQA先行で0.103121→0.104801 ms/call、V620 M=2 wideのsplit-K=2＋reductionは0.125360→0.160361 ms/callと退行したため、候補を広げず打ち切った（[履歴](../../../../../history/2026/09/21-30/phase87-stage1.md)）。
- 段階0の結果に基づき、W4A4の行列ベクトル積と、M=1の活性値量子化（行列ベクトル積の前処理への融合を含む）を最適化する。
- M=2〜3（MTP verify）の形状も同じ作業単位で扱う。
- 現行のW4A4既定を維持する。最適化で数値が変わる場合は、その変更内容に基づいて数値分類を記録する。
- **2026-09-25着手時のC1仮説と受入条件**: 両GPUの実M=1はID84で、対象は
  `(M,K,N)=(1,5120,17408)`（gate/up、112回/token）と`(1,17408,5120)`（down、56回/token）。
  WU0のwarm read参照に対する正の余地はV620 0.8320、R9700 1.4121 ms/token、探索継続線は
  その半分の0.4160／0.7061 ms/token。C1はweight／scaleのwave共通な列baseを明示的に
  scalar化し、現行のE2M1 byte permute、DP4A／sudot4、scale LUT、2／4-block先読み、
  FP32加算順とBF16 RNEを保つN0候補とする。non-temporal loadとLUT paddingは既実装で再提案しない。
  まず代表shapeをcompileしてVGPR／SGPR／LDS／occupancy／spillとISAのaddress生成を記録する。
- C1は最初test-onlyで両GPUのproduction ID84をcontrolにする。両shapeの全出力bitwise、独立FP32
  oracle、finite、repeat、guard、cleanup、量子化済み入力とtensor scaleの一致を確認する。
  M=2〜3のID94とexact tuple外の選択は変えない。非整列値とM1／M2の境界もselectorで検査する。
  300 ms継続warmup、同一process AB-BA-AB各9 samples、512 MiB以上の巡回weight poolで
  quantizer／matmul／合計を分けて測り、孤立条件と[WU-D1](../../../../../history/2026/09/11-20/phase87-wu-d1.md)の
  GQA型KV reader先行条件を並べる。両条件を同じ条件内のcontrolと候補で交互に測り、近傍依存を効果と混同しない。
  探索線に届かないGPUではC1を打ち切る。採否は探索線と分け、公開経路のMTPなし／あり8192/128
  各1 warmup＋3 measuredと[共通ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)で判断する。
  結果は[段階1履歴](../../../../../history/2026/09/21-30/phase87-stage1.md)へ記録する。
- **C1の採否（2026-09-25）**: 両GPUのM=1両tupleはproduction ID84とbitwise一致、独立oracle最大0 ULP。
  wave共通baseをKループ外へhoistした最終版はV620のWU-D1型GQA先行AB-BA-ABで両shape・全3round改善し、
  112/56回を掛けたdot短縮は合計1.2768 ms/token。通常8192/128のMTPなしTPOTは
  58.068919→56.990410 ms（**1.857%短縮**）、MTPあり30.416029→30.414372 msで同等。
  生成SHA、MTP受理75/103、peak VRAMは前後一致した。共通ルールに従いV620 exact M=1 2 tupleへ採用する。
  R9700のGQA先行合計は−0.098560 ms/tokenで不採用。hoistをR9700のwideへtest-only適用しても全round退行し、
  ID84を維持する。M=2〜3は上記のtest-only候補を評価してID94維持で完了した。
  詳細は[段階1履歴](../../../../../history/2026/09/21-30/phase87-stage1.md#c1-hoist最終結果とv620限定採用)。

### 段階2: FP8 W8A8のdecode

- **完了（2026-09-25）**。R9700 M=1はWU2のID103を採用し、V620はWU-2Vの3候補を数値PASS・速度不採用で終了した。
  V620の他7形状（lm_headを含む）の正のread参照余地は探索継続線未満。残るFP8 producer融合97 nodeは
  段階7完了時にbacklog P3へ移したため、この段階で再実装しない。
- gfx1201（hipBLASLtのnative FP8）とgfx1030（software経路、Phase 78のID68／71／75／82等）について、
  tokenごとの活性値量子化と行列ベクトル積を段階0の優先順位に従って最適化する。lm_headを含む。

### 段階3: MTP companionの形式

- **完了（2026-09-24、WU-3S）**。形式比較後に一度完了を差し戻し、低bit companionの遅延原因、形式整理、受理率込み速度を確認した。最初の通常8192/128の実効速度は、BF16前後平均に対して
  MXFP6がV620 −7.34%／R9700 −7.59%、NVFP4が−0.56%／−11.80%だった。draftが読む重みはBF16の約0.85 GB/stepから
  NVFP4で約0.25 GB/stepへ減るはずで、受理数がほぼ同じV620のMXFP6（75/105、BF16は77/101）でも遅い。
  **WU-3Sの実施項目（2026-09-24に書き直し）**:
  **原因（[記録](../../../../../history/2026/09/21-30/phase87-stage3.md#低bit-companionが遅い原因2026-09-24wu-3s)）**:
  MXFP6はcompanionのM=1 kernel `sllm_mxfp6_w6a6_m1_col2_v1`が読み出し帯域の約2〜3割（95〜161 GB/s）しか出ず、
  BF16 kernel（約500〜625 GB/s）より1回あたり1.1〜1.7倍遅い。NVFP4のkernelはBF16より1 blockあたり約2 ms速く、
  通常計測の遅さは単一prompt自由生成の受理数のばらつき（R9700で66/123）による。
  1. **MXFP6 companion sidecarを廃止する**（種類F。ただし公開opt-inの削除は今回のユーザー委任による明示判断）。
     MTP companion専用のMXFP6変換・manifest読込み・選択経路だけを削除する。既存のMXFP6 sidecarを
     `--mtp-weights`へ渡した場合は、BF16／NVFP4への暗黙fallbackをせず、退役した形式として明確に拒否する。
     CLI／APIの説明、converterのhelp、関連testを同じ作業で更新し、現在MXFP6を明示選択している利用者への互換性変更を履歴へ記録する。
     MXFP6形式そのものと本体モデルの経路、共用kernel `sllm_mxfp6_w6a6_m1_col2_v1`は残す
     （kernelの遅さは[未計画の課題一覧](../../../../backlog.md)のP1）。MXFP8とNVFP4のsidecarは維持する。
  2. **NVFP4経路の非効率の確認結果を記録する**: decodeのprofileでは大きな行列積が読み出し帯域の約75%で動いており
     （gate/upはV620 384、R9700 480 GB/s）、decode側に大きな行列積の非効率は見えない。
     k/vの小さな行列積と、MTP層RMSNorm後の独立した量子化nodeが合計1 blockあたり0.2〜0.3 ms程度残る（一覧のP2）。
     一方、prefix準備のNVFP4行列積は遅く、[backlogのP12](../../../../backlog.md)へ記録する。今回の段階3では直さない。
  3. **ベンチマーク**: 両GPUのTier A 26条件で、BF16とNVFP4のM3（1 blockあたりのdraft／non-draft時間）を測り、
     取得済みのM1と合わせてM4（受理率込みのdecode速度）を導く。単一prompt自由生成は経路の分岐で大きくぶれるため判定に使わず、
     整合の確認として同一processのAB/BAの通常計測を付ける。VRAMの増減も記録する。
  4. **既定採用**: 測定結果を記録したうえで、2026-09-24の追加ユーザー決定を適用する。
     ユーザーは今回のNVFP4 MTP companionの既定採否について共通の性能採否ルールをすべて無視し、
     M4と通常AB/BAのTPOTが共通基準を満たさないこと、prefill／TTFTが低下することを受け入れてNVFP4を既定にするよう指示した。
     draftだけの変更でp/q補正後のtarget分布は維持するため、KLDはこの採否で要求しない。
     この明示的な例外を後続作業へ引き継がない。BF16はベンチマークcontrolとsource／binary rollback経路として残し、
     BF16へ戻す専用CLI flagは追加しない。
  5. 結果を記録して段階3を完了にする。
- **WU-3Sの結果**: MXFP6 companion sidecarの変換・読込みを退役させ、既存sidecarは明示エラーで拒否する。MXFP6本体形式・共用kernelは維持する。Tier A 26条件のM4はNVFP4がBF16比でV620 +1.70%／+1.76%、R9700 +1.75%／+1.61%（BF16／W8A8凍結列）で、各prompt-cluster 95%区間も正だった。一方、同一processの通常8192/128 AB/BAではV620 −0.63%／−0.48%、R9700 −13.28%／−13.27%と両順序でTPOTが悪化した。共通ルールならBF16維持となる結果だが、ユーザーの明示決定によりNVFP4を既定にして段階3を閉じた。詳細なM3内訳・資源・公開経路と既定解決は[履歴](../../../../../history/2026/09/21-30/phase87-stage3.md#wu-3sの最終結果)へ記録した。
- 形式比較の結果（2026-09-24）: sLLM量子化器と評価入力に重ならない較正入力からNVFP4 W4A4 sidecarを作り、
  既存MXFP6 W6A6とBF16を両GPU・Tier A 26条件の固定列M1で比較した。
  NVFP4−MXFP6は−0.80〜−1.01ポイントだった。
  NVFP4−BF16は−1.08〜−1.37ポイントだった。既定採否は上のWU-3Sで判断した。
  通常8192/128の実効decode速度、公開CLI、MTPなしの不変性まで[履歴](../../../../../history/2026/09/21-30/phase87-stage3.md)へ記録した。
- MTP companionをNVFP4 W4A4（重みはsLLMの量子化器、活性値のglobal scaleは較正で決める）とMXFP6 W6A6で作り、
  [MTP採用率ベンチマーク](../../../../../development/mtp-acceptance-benchmark.md)の主指標M1（期待受理率、両GPU）で比べる。
  あわせて実効decode速度も記録する。
- 活性値global scaleの較正に使う入力は、ベンチマークの評価入力と重ならないものにする。

### 段階4: W×A16の廃止と契約の整理

- **完了（2026-09-24、WU-4R後）**。旧A16経路を退役させ、直接W4A4 artifactのQwen3.8／Gemma 4 12B／Gemma 4 26B-A4Bを両GPUで確認。MXFP4公開placeholderはW4A6へ変更した。初回受入後に見つかったNVFP4 W4A16の残存kernel・launcher・variant選択はWU-4Rで削除した（[履歴](../../../../../history/2026/09/21-30/phase87-stage4.md)）。
- NVFP4 W4A16、MXFP8 W8A16、MXFP6 W6A16の実行経路・provider・selector・evidence toolを削除する。
  段階0の棚卸しで挙がった全モデルがW4A4等で動くことを、両GPUで確認してから削除する。
- lowpの公開C APIのMXFP4契約をW4A8 v1からW4A6へ書き換える（実装は後続。planは未対応として拒否のまま）。
- main-plan、`docs/architecture/runtime.md`、数値変更台帳を更新する。

#### WU-4R: NVFP4 W4A16の残存コードの削除（段階4の残件、2026-09-24追加）

**完了（2026-09-24）**。下記の受入条件を確認し、段階4を完了に戻した。

**背景**: 段階4では、MXFP8 W8A16／MXFP6 W6A16はkernelまで削除したが、NVFP4 W4A16は「planから選択されない」状態に
しただけで、次のコードが残っている。リポジトリ全体で呼び出し元は0件であり、kernelだけがbinaryへ入っている。

| 残っているもの | 場所 |
| --- | --- |
| `sllm_matmul_kernel::launch_nvfp4`（BF16 activation × NVFP4 weight） | `native/lowp/src/lowp_kernel.hip.cpp`（定義）、`native/lowp/include/lowp/detail/lowp_kernel_internal.hpp`（宣言） |
| kernel `sllm_matmul_nvfp4_block16_packed_dequant_v1` | `native/lowp/src/lowp_kernel.hip.cpp` |
| kernel `sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2` | 同上 |
| `select_nvfp4_variant`と、その中の環境変数`SLLM_NVFP4_FORCE_BASELINE` | `lowp_kernel_internal.hpp` |
| `KernelVariant::Nvfp4DecodePackedDequant`（8）、`Nvfp4PrefillRow8Tiled256`（9）、`Nvfp4BaselinePackedDequant`（10）と、それらを参照する名前・symbol・ID表の分岐 | `lowp_kernel_internal.hpp`（2,700〜3,140行付近の3つの表を含む） |

**作業**
1. 上表の関数・kernel・選択関数・環境変数読み出しを削除する。kernel専用の`__device__` helperで、他のkernelが使わなくなるものも削除する。
   W4A4の経路（`launch_nvfp4_quantize`、`launch_nvfp4_w4a4*`、`sllm_matmul_nvfp4_w4a4_*`、
   `sllm_matmul_nvfp4_block16_to_fp16_staging_v1`、`sllm_matmul_nvfp4_tensor_scale_epilogue_v1`）は変更しない。
   削除前に、各helperとstaging／epilogue kernelがW4A4経路から使われていることを確認する。
2. `KernelVariant`の8／9／10は数値を再利用しない。MXFP8／MXFP6のA16（101／102）と同じく、
   監査記録を読むための欠番（tombstone）として名前付きで残し、実行・選択・名前表から実行可能な分岐を外す。
3. 呼び出し元がないことを`rg`で再確認し、`native/`・`crates/`・`ci/`に旧symbol名が実行経路として残っていないことを確かめる
   （履歴文書・過去evidenceの記述は変更しない）。
4. 他に同じ状態（呼び出し元0件のA16 kernel・launcher・環境変数）が残っていないかを`native/lowp`と`native/hip/src`で点検し、
   見つかれば同じ扱いで削除して記録する。

**受入条件（着手前に固定）**
1. 上表のsymbolが`native/`と`crates/`の実行コードから消え、tombstoneだけが残る。W×A16を実行するkernelがbinaryに含まれない。
2. 両GPU（exact `gfx1030`／`gfx1201`）でlowpとHIP runtimeをbuildし、lowpのhost／GPU test、`sllm-core`・`sllm-hip`のtestがPASSする。
3. W4A4の既定経路が変わらないこと: Qwen3.8通常8192/128のMTPなしを両GPUで0 warmup＋1 measured実行し、
   生成token SHA-256が段階9の値（V620 `c9c0b4ee…`、R9700 `75d36def…`）と一致する。速度の再計測は不要。
4. `validate_cpp.py --mode format`、`cargo fmt`、clippy（`-D warnings`）、CI hash連鎖（`hip-runtime-compile`→`rmsnorm-h3`）の更新、
   `validate_json_manifests.py`、local h0がPASSする。
5. [段階4履歴](../../../../../history/2026/09/21-30/phase87-stage4.md)の「旧W4A16の実行可能kernel launcherは現行provider planから選択されない」を、
   削除した事実へ書き換え、段階4を完了にする。

数値分類はN0（実行経路の変更なし）。この後にWU-P1へ進む。

### 段階5: decode実行制御（graph化とサンプリング経路の通信削減）

2026-09-21完了。targetの1 tokenとMTPのdraft／verify／状態選択を、それぞれ一つのHIP graphへ接続した。
固定device samplingのeligibleなfresh requestで自動有効化し、次graphをreadback待ちより先にenqueueする。
両GPU・MTPなし／ありの8192/128、1 warmup＋3 measuredで全生成tokenが変更前と一致した。
停止・予算・context末尾・正常解放と、marker付きprofileによる非同期性／再instantiateなしを確認した。
decode中央値はV620 15.6762／29.4250、R9700 20.8622／34.6104 tok/s（MTPなし／あり）。
MTPありは直前baseline比+2.00%／+1.69%、MTPなしは-0.61%／-2.56%。速度下限は追加しない。
実装方式、適用条件、GPU空白の分解と証拠は[段階5履歴](../../../../../history/2026/09/11-20/phase87-stage5.md)へ記録した。

同日の追加依頼でMTPなしの速度低下を調査し、共通の状態コピー・attention定数最適化・既知budget終端の
不要replayを修正した。V620 16.1868／30.1174、R9700 21.4909／35.6390 tok/sとなり、
両GPUで低下を解消した。全12ケースのN0・停止・容量境界と最終traceを確認済み。
MTP有無で全体経路を分ける必要はなかった。[追加修正履歴](../../../../../history/2026/09/21-30/phase87-mtp-off-regression.md)。

段階0の補正で、GPU空白はkernel 1,171個/token（MTPなし）ごとの2〜10 µsの隙間と、tokenごとのhost往復
（sampler後の読み戻し、約0.5 msのhost処理、同期的な`hipMemcpyAsync`、次tokenの送り直し）から成ると分かった。
MTPありでは受理判定の読み戻しと、一部受理時のrestore＋replayの起動がhost主導である。

- **decode 1段全体のHIP graph化**: MTPなしの1 token、MTPありのdraft＋verify（＋状態選択）の1段を、
  それぞれ一つのgraphとして再生する。現行の部分的なgraph span（hipBLASLt等）を包含する。
  位置・KV長・attention分割等のtokenごとに変わる値は、device上の値またはgraphの再instantiateを要しない
  引数更新で渡す。
- **サンプリング経路の通信削減（MTPなし）**: sampler結果をdeviceのtoken bufferに置き、次段のembeddingが
  直接読む。hostへのtoken読み戻しは非同期にし、停止判定は1段遅れで行う（余分な1段は捨てる）。
  残り1出力と確定したbudget終端だけは、不要な後続replayを予約しない。
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

### 段階6: graph内の並列表現によるkernel間空白の削減（段階5の続き、2026-09-21追加）

- **完了（2026-09-21）**。両GPU・MTP有無のDAG／critical pathと、96条件の単体probeを確認した。
  exact gfx1030のNVFP4 gate/up（M=1／3、56組）だけをforkし、他の辺を保持する。
  通常8192/128の1 warmup＋3 measuredで、V620はMTPなし16.146946→16.708786、
  あり30.176041→30.695941 token/s（+3.48%／+1.72%）。通常とprofileのtoken列一致、独立oracle、
  cleanup、async replay契約、CI hash連鎖を確認した。R9700は単体で退行するため直列のまま。
  初回はFP8 GDNも含む候補をN0不一致で棄却したが、追加調査で再現せず、ユーザー判断でその結果を採否から除外した。
  [再評価](../../../../../history/2026/09/21-30/phase87-fp8-fork-investigation.md)によりFP8 M1だけを追加採用。
  単体全18roundで改善し、通常MTPなしはNVFP4-only比16.686732→16.813859 token/s（+0.76%）。
  FP8 M3は退行roundがあり維持、MTPありは30.719020→30.717491 token/sで同等。
  固定128位置のBF16比mean KLDは0.026571、変更前との差0。V620のHIP signal停止は既存classic設定で回避し、R9700は従来設定を維持する。
  理想上限の半分へは届かず探索を終了し、単体の全roundで通常TPOT比1%以上を満たす対象だけを採用する。
  詳細・失敗した試行・profiling回避設定は[段階6履歴](../../../../../history/2026/09/21-30/phase87-stage6.md)。

段階5でhost往復は実質消えた（graph間隔0.03 ms/token）。しかし**graph内のkernel間の隙間が、以前「空白」と
呼んでいた時間のほぼ全部を占める**ことが分かった。MTPなしでV620 7.978／R9700 6.623 ms/token（当時1,218 kernel node/token、
1 nodeあたり5.4〜6.6 µs）で、profileの`idle_outside_hip_api`7.997／6.678とほぼ一致する。
これはhostの遅延ではなくdispatchごとの固定費であり、graph化では取り除けない。
残る手はnode数を減らす（融合）か、**独立nodeを並列な枝として表し、dispatchの固定費を重ねる**かである。
本段階は後者を先に評価する。融合は変更が大きいので、並列化が効かない場合の後続候補とする。

- 仮説: decode 1段のnodeの多くは依存していない（例: 同一層のq/k/v投影、MLPのgate/up、GDNのin_proj束、
  KV書き込みとstage2、sampling末尾）。これらを直列の鎖ではなく並列枝としてgraphへ入れると、
  command processorが次のdispatchを前のdrainと重ねられ、隙間の一部が隠れる。
- 手順:
  0. **現状の再計測と構造分析**: 段階5とMTPなし修正後の最終binaryで、kernel node数・隙間・
     `intra_graph_gap`を両GPU・MTP有無で測り直す（MTPなしは状態選択48 nodeを除いた現行値を使う）。
     capture済みgraphから依存DAGを作り、critical pathの合計kernel時間と、並列に置ける最大幅・
     各枝の所要時間を出す。**改善の上限は「現在の合計時間 − critical pathの合計kernel時間 − critical path上の隙間」**として算出する。
  1. **単体probeで並列枝の効果を確認**: 1本の鎖のN kernelと、K本の並列枝に分けた同じN kernelを
     同一graphで比較し、1 kernelあたりの隙間がKでどう変わるかを両GPUで測る。
     枝数、kernelの長さ（1／5／20 µs相当）、使用queue数を変えた表を作る。
     ここで隙間が縮まなければ、この方式は成立しないので段階6を打ち切り、融合へ切り替える。
  2. **decode graphの再構成**: 手順1で効果が出た構成に合わせ、capture時の依存関係を実際の
     データ依存だけに絞り、独立nodeを別枝にする。対象候補は手順0のDAGから選び、最大3箇所に限る。
  3. **計測**: 段階0と同じ条件（両GPU、MTPなし／あり、8192/128、1 warmup＋3 measured）と、
     profileのkernel間隙間・intra_graph_gapを記録する。
- 上限と打ち切り: 上限は手順0で算出する。打ち切り線はその半分とし、届かなければ理由を記録して段階6を終える。
  当時の採否と数値分類は[段階6履歴](../../../../../history/2026/09/21-30/phase87-stage6.md)に残す。
- 数値と正しさ:
  - 並列化は独立kernelの実行順だけを変えるので、各kernelの入力・演算順は不変でN0を目指す。
    依存を1つでも取り違えると競合で結果が変わるため、**依存関係の根拠を各枝について明記**し、
    生成token列の一致と既存の独立oracleで確認する。
  - 同じbufferを読む枝と書く枝の分離、KV／GDN状態の公開順、停止・破棄replayの扱いは段階5の契約を維持する。
  - 非決定性（実行順による結果差）が出た場合は即座に候補を棄却する。
- 制約:
  - 新しい環境変数を追加しない。capture対象の適用条件（固定device sampling、eligibleなfresh request）は変えない。
  - graphのinstantiate回数を増やさない。再生中の再instantiate・host確保は0を維持する。
  - queue数を増やす場合は、既存のqueue所有・cleanup契約と公開C APIの制約に収める。
- 変更してよいファイル: `native/hip/src/decode_graph_capture_internal.hpp`、`graph_span_runtime.inc`、
  関連するnative runtime、`crates/sllm-core/src/decode_control.rs`／`decode_replay.rs`／capture系、
  `crates/sllm-hip/src/graph_span.rs`、新規probeとtest、`docs/history/2026/09/21-30/`の新規履歴、
  CI hash manifest、この計画。commit前に`validate_cpp.py --mode format`、`cargo fmt`、clippy、
  CI hash更新（`hip-runtime-compile`→`rmsnorm-h3`の連鎖が収束するまで）を必ず通す。

### 段階9: MTP draft専用の縮小語彙lm_head（2026-09-23追加）

- **完了（2026-09-24）**。98,304語彙を採用し、両GPUの実M1と通常8192/128のAB/BA、MTPなし対照をPASS（[履歴](../../../../../history/2026/09/21-30/phase87-stage9.md)）。
- **目的**: MTPのdraftだけ、頻度上位N行のFP8 lm_headを使う。verifyのtarget headとp/q受理規則は変えない。
  draftのlm_headは1回あたりV620 2.668／R9700 2.013 msで、幅2では確定tokenあたり通常TPOTの約6.6%／5.8%を占める。
- **見込み**（[探索記録](../../../../../history/2026/09/21-30/phase87-mtp-proposal-shortlist.md)）:
  N=98,304で期待受理率（M1）−0.17 pt、draft lm_headは両GPUで行数に比例して短縮し、
  正味は約+3.8%（V620）／+3.2%（R9700）。追加VRAMは約503 MB（MTP有効時のみ）。第一候補はN=98,304とする。
- **当時の打切り線**: 見込みの半分（正味V620 +1.9%／R9700 +1.6%）。採否の測定結果は[段階9履歴](../../../../../history/2026/09/21-30/phase87-stage9.md)に残す。

**語彙集合の作り方（2026-09-23ユーザー決定: 生成手順だけを置く）**

語彙集合そのもの（token ID一覧やmask）はGitで追跡しない。追跡するのは生成toolと手順、
入力corpusの固定情報、生成物のSHA-256だけとし、生成物はMTP companionのsidecarと同じくmodel側のlocal artifactとして置く。

1. 入力は、dataset名・revision（commit）・file・SHA-256を固定したcorpus manifestで指定する。
   mtp-bench-v1の入力（sLLMリポジトリの文書・コードと作成済みcorpus）とは重ねない。
2. model lockの`tokenizer.json`（SHA-256を照合）で各domainをtoken化し、domainごとの相対頻度を等重みで足す。
3. 得点の降順（同点は小さいID優先）で上位Nを取り、special tokenを必ず加え、**語彙ID順に並べる**。
   ID順に並べることで、S内の各logitが全語彙headと同一になり、selectorの同値規則も保たれる。
4. 生成物のSHA-256、N、corpus manifestのSHA-256を記録する。同じ入力から同じbytesが出ることを確認する。

corpusの候補は、FineWeb（英語web）、日本語・中国語Wikipedia、`reference/`配下のsourceと、
実利用のcoding agent sessionである[SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat)（ODC-By、gated）。
探索で使ったJParaCrawlは研究目的の利用条件があるため、本番用の順位では使わない。
SWE-chatを使う場合は、sLLMリポジトリ由来のsessionが含まれていないことを確認して除く。

**実装**

1. load時、MTP有効ならFP8 lm_headのS行と行scaleをID順に別bufferへ集める。全語彙headは変更しない。
2. draft graphのlm_headをN行のheadへ差し替え、fixed K20 selectorを`vocab=N`で動かし、
   support recordを書く前に行番号を語彙IDへ戻す。verify側のsupport record形式は変えない。
3. R9700はID103の許可形状へ`K5120,N`を追加する。V620の`dword8_wave4col32`は任意Nを扱える。
   縮小headの時間は探索で両GPUとも実測済み（98,304行でV620 1.089、R9700 0.804 ms）。

**受入条件（着手前に固定）**

1. 実runのM1（mtp-bench-v1、両GPU、Tier A 26条件）が探索の反実仮想値と一致する。
   差が出た行は、draft top-1がS外にありblock進行が変わった行かどうかで説明する。
2. 通常計測（8192入力／128出力、warmup 1＋measured 3）のMTPありで同一process AB/BAを記録し、
   MTPなしは経路が変わらないことを確認する。当時の採否条件と結果は[段階9履歴](../../../../../history/2026/09/21-30/phase87-stage9.md)に残す。
3. 数値分類: target分布は不変、固定seedの生成token列はdraft変更により変わりうる。N1として記録する。
4. 追加VRAM、生成物のSHA-256、corpus manifestを履歴へ記録する。
5. commit前に`validate_cpp.py --mode format`、`cargo fmt`、clippy、CI hash連鎖の更新を通す。

### WU-P1: Paged Attentionの試作と軽い検証（2026-09-24追加）

**完了（2026-09-24）**。[最終結果](../../../../../history/2026/09/21-30/phase87-wu-p1-paged-attention.md):
exact V620／R9700の各27条件でbitwise・独立oracleをPASSし、decode／prefillの全AB/BA/AB roundが
kernel単体の増加10%未満だった。最悪増加はV620 +1.345%／+8.194%、R9700 −16.346%／+3.290%。
本番source・公開ABI・既定経路は変更せず、[本移行計画](../../../../../plans/archive/2026/09/21-30/paged-kv-full-migration.md)を作成した。

**方針（2026-09-24ユーザー決定）**: 将来の拡張性（1M以上のcontext、8〜16並列、複数GPU、READMEの対象ハードウェア）を考え、
vAttention（HIP VMMによるvirtual-contiguous KV）を完全に廃止し、Paged Attentionへ完全移行する。
ただし、pagedを最適化しても性能低下が10%以上になる場合は再考する。
判断の経緯は[paged移行の検討記録](../../../../../history/2026/09/21-30/kv-paged-migration-decision.md)にある。

- **時期**: 段階9と段階4（残件WU-4Rを含む）の完了後。attention kernelとKVはPhase 87の残りの段階が触らないため、他の段階と衝突しない。
- **範囲**: probeだけで行い、本番source・公開ABI・既定経路は変更しない。
  1. block 128 tokenのpaged decode kernel（M=1〜3）を試作する。
     decodeのsplit区間とblockの境界を揃え、block tableの参照は区間ごとに1回とする。
  2. prefillのpaged kernelを、現行の代表providerと同じ形状で試作する。
  3. KV形式は現行既定のMXFP8 E4とし、Qwen3.8の形状（head dim 256、GQA 6）を使う。
- **比較**: 両GPUで、現行の連続KV kernelと同一processのAB/BAで比べる。
  KV長は1023／1024／1025／8192／8193／65536を含め、blockと区間の境界の前後を覆う。
- **判定（着手前に固定）**:
  1. 数値: 現行kernelと同じ加算順・丸めにしてbitwise一致を目標とし、無理ならN1として理由を記録する。
  2. 性能: attention kernel単体の時間の増加が、decodeとprefillのそれぞれで10%未満であること。
     decodeのattentionはTPOTの一部（V620のMTPなしで約4.8/58 ms）しかなく、TPOTで判定すると基準が緩すぎるため、
     単体の時間で判定する。モデル全体のTPOTとprefill時間への換算は記録用とする。
  3. 10%以上の場合は、原因（表の参照、block境界、prefill tile）を切り分けてユーザーへ報告し、移行方針を再考する。
- **問題がなければ**: 本移行を独立した作業として計画する。その計画に次を含める。
  - block単位のKV pool、参照カウント付きのprefix共有と分岐、graph capture中のblock table更新
  - VMM provider、page共有と末尾COW、growのtransaction、V620の65,536境界、R9700の全量確保の特例の削除
  - 公開C ABIのKV memory kindと[KV memory方式の決定](../../../../../architecture/kv-memory.md)の書き換え

  連続KVの経路は、移行の作業単位の中で最終比較を終えるまでだけ残す。実行時の切替は作らない。

### 段階10: Paged KV／Attentionへの本移行（2026-09-24、Phase 87へ統合）

WU-P1の合格を受け、vAttention（HIP VMM）を廃止してPaged KVへ完全移行する。
範囲・実装の順序・受入条件は[本移行計画](../../../../../plans/archive/2026/09/21-30/paged-kv-full-migration.md)を詳細計画とし、ここでは要点だけを書く。

**2026-09-25完了**。実装・検証・対象外の詳細は[履歴](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)。

- 本番のdecode（段階12で追加した幅3／4のverifyを含むM=1〜5）とprefillをblock単位のKV poolへ接続し、prefix共有・分岐・末尾COW・cancel・graph replay前のblock table更新を揃える。
- 現行の計算順序と丸めを保ち、現行連続KVとの最終比較（kernel単体の10%条件の再確認、TPOT・TTFT・peak VRAM）の後に、
  VMM provider、page共有・末尾COW、grow transaction、V620の65,536境界、R9700の全量resident特例を撤去する。
- 性能の改善は段階11で行う。段階10はbitwiseの対照を保つため、計算の構造を変えない。

### 段階11: attention kernelの最適化（2026-09-24追加）

**2026-09-25実装・採否完了**。C1をM=3／KV8192以上のexact両GPUで採用し、C2/C3候補は採否ルールにより不採用。数値・モデル速度・VRAM・残る制約は[段階11履歴](../../../../../history/2026/09/21-30/phase87-stage11.md)を正とする。

**根拠**: [改善余地の分析](../../../../../history/2026/09/21-30/phase87-attention-headroom.md)。
現行のdecode attentionは、KVの論理byte数に対する実効帯域がread参照値のV620約19%、R9700約24%にとどまる。
prefillは両GPUとも1〜3 TFLOP/sで、行列演算命令を使っていない。構造上の原因として次の2つを確認した。

- query行ごとにKVを読み直している（両GPU）。MTP verifyのM=3はM=1の2.4〜2.5倍の時間になる。
- R9700の本番decodeはGQAを共有せず、1本のKV headを6つのquery headが別々に読む。

**時期**: 段階10の完了後。段階10で書き直したpaged kernelの上で行い、kernelを二度書き直さない。

**着手時に行うこと**
1. 段階10後の本番kernelを基準に、上限と打切り線（上限の半分）を計算し直す。分析時点の上限の目安は、
   decode M=1がV620約3.4／R9700約1.9 ms/token、MTP verify M=3がV620約3.7／R9700約2.8 ms/token（確定tokenあたり）、
   prefillがTTFTのV620約2割／R9700約3割である。R9700の値は段階10でpaged候補の読み方が入る分だけ小さくなる。
2. V620のdecode M=1が1 keyずつの逆量子化とonline softmaxの更新で律速しているという推定を、profilerで確かめる。
3. AGENTS.mdのkernel融合方針に従い、代表1変種をcompileしてVGPR・LDS・occupancyを記録する。

**候補（作業単位ごとに3つまで）**
1. **C1: decodeでquery行の間のKV共有（両GPU、MTP verify向け）。** M=2〜3の行が同じKV tileを一度だけ読み、
   各行のQK・softmax・value累積は現行と同じ順序で行う。N0を目標とする。
2. **C2: R9700 decodeのGQA共有と、V620 decode M=1の内側loopの見直し。** R9700はV620のWU1と同じく6つのquery headでKVを共有する。
   V620は、複数keyをまとめて逆量子化・内積し、softmaxの更新を塊単位にする。加算順が変わる場合はN1として記録する。
3. **C3: prefillを行列演算命令で書き直す（FlashAttention型）。** R9700はWMMA、V620はpacked FP16のdot命令を使う。
   加算順が変わるためN1の候補とし、別の作業単位にしてよい。

**採否**
- decodeの候補は、[main-planの変更の採否ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)で判定する。
  数値差は台帳へ分類して記録し、N0／N1の分類だけで採用を自動承認しない。
  C1はMTPあり、C2はMTPなし・ありの両方で測る。
- prefillの候補は、8192入力のprefill時間（TTFT）を同一processのAB/BAで比較し、decodeの副作用も含めて同じ共通ルールで判定する。
  KV長65536での効果も記録する。
- 実行時の切替は作らない。exact-shapeのgateは、既存のproviderと同じく実際に使う形状に限る。

### 段階12: MTP幅3・4への対応と最適化（2026-09-25追加）

**目的**: MTPの幅（1回に出す候補の数）を今の2から3・4へ広げ、1回あたりの確定token数を増やす。
DFlash2は[main-plan](../../../../main-plan.md)のとおりEXL3の後に検討し、この段階では扱わない。
段階1の後に行い、段階10・11より前に置く（2026-09-25ユーザー指示）。

**背景と見込み**
- 今の幅2では1回あたりの確定token数は最大3で、実測は約2.4（段ごとの受理率a₁≈0.81、a₂≈0.72）。
  3段目の受理率を約0.65とすると幅3で約2.77（+16%）、4段目を約0.6とすると幅4で約3.0（+25%）の見込み。
  Cinferenceの公開値（Qwen3.8 NVFP4、MTP3、Code）では1回あたり3.29 token。
- 増える費用は、draftの1 stepあたり約3 ms（NVFP4 companion、V620）と、verifyのMが3から4・5へ増える分。
  attentionはquery行ごとにKVを読み直すため、verifyの費用がMにほぼ比例して増える（段階11の分析）。
  改善なしでは得の多くが消える可能性がある。

**位置づけ（2026-09-25ユーザー指示）**: 幅3・4を選べることは、[採否ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)の
種類D（機能の追加）として**必ず実装して残す**。既定の幅をどうするか（WU-12D）とは切り離し、幅3・4が既定にならなくても、
利用者が明示的に選べる機能として維持する。「使わない経路を残さない」整理の対象にもしない。
WU-12Aの完了は機能の正しさだけで判断し、速度や探索の打切り線で中止しない。

**WU-12A: 幅を2・3・4で選べるようにする（機能、必須）**

**完了（2026-09-25）**。両GPUで既定幅2の回帰一致、幅3・4の固定seed反復、幅4の独立p/q残差oracle、
全幅のstop・1 token／幅＋1予算・32-token context末尾とcleanupを確認した。
幅2を既定に保ち、幅3・4の明示選択を残す。後続は幅4の重大退行を防ぐ確認に絞る
（[履歴](../../../../../history/2026/09/21-30/phase87-stage12.md)）。

1. `QWEN38_MTP_DRAFT_WIDTH`（従来は2に固定）を設定値にし、CLI（従来は幅0か2だけを受け付けた）とserverで2・3・4を選べるようにする。
   既定は2のまま維持し、幅3・4の明示選択を残す。
2. 幅固定のdraft／verify graph capture、候補ごとのsupport record、workspace、device制御記録、
   verify行（幅＋1行）のKV書き戻しとGDN状態の保持・巻き戻しを、幅3・4で動くようにする。
3. 両GPUで、幅2の出力が変更前と一致すること、幅3・4で固定seedの生成が決定的で、target単独と同じ分布の規則
   （p/q受理と残差からの引き直し）を守ることを確認する。途中停止・予算・context末尾・cleanupも幅ごとに確認する。

**段階12の受入を簡素化する最終ユーザー決定（2026-09-25）**: 既定幅はGPU共通で2に固定し、
幅3・4は明示選択の機能として残す。GPU別の既定幅や、既定採否のための追加の26条件M3／M4・
複数prompt AB/BAは行わない。既に取得した詳細ベンチマークは[履歴](../../../../../history/2026/09/21-30/phase87-stage12.md)へ残す。
簡素化するのは段階12だけであり、段階10・11の範囲は維持する。

**WU-12B: 取得済み測定と重大退行の原因確認**

- `mtp-bench-v2`のTier A 26条件M1は両GPU・全幅で完了。M3／M4はR9700の幅2・3・4と
  V620の幅2・3を取得した。R9700幅4の未最適化M4は幅2比`−89.57%`で、
  M=5のNVFP4 MLPがID94からID59 row8へ落ちることをprofileで確認した。
- V620幅4の未最適化26条件M3は追加しない。candidateのR9700幅4・26条件M3も
  ユーザー指示時点で中止した原票として区別し、採否証拠へ含めない。

**WU-12C: 幅4の深刻な退行を解消する**

**完了（2026-09-25）**。両GPUのC1数値oracle／bitwise対照と単体比較がPASSし、
本番serverの短い64出力要求で幅4／幅2の時間比はR9700 `1.075`、V620 `1.260`だった。
どちらも2倍以下で、HIP-only・fallbackなし・cleanup・service復元を確認した。

- 実際に使う`M=5`、`(K,N)=(5120,17408)`／`(17408,5120)`だけを既存ID94の共有算術へ接続する。
  両exact GPUで独立数値oracle、現行row8とのbitwise一致、全roundの単体改善、M4/M6境界、
  HIP-only、fallbackなし、cleanupを確認する。他の幅とMTPなしのselectorは変更しない。
- productionの短い同一入力・固定seed・64出力要求を、幅2と幅4それぞれで両GPUのserverへ
  1 warmup＋2 measured流す。各GPUで幅4の要求時間中央値が幅2の**2倍以下**、
  64 token完走、NVFP4 companion一致、HIP-only、fallbackなし、cleanup・service復元を受入線とする。
  これは過度な遅さの防止であり、幅4を既定採用する速度判定ではない。

**WU-12D: 共通の既定幅と段階の完了**

**完了（2026-09-25）**。GPU共通の既定幅2、明示選択の幅3・4で段階12を閉じた。
短い要求での重大退行防止を受入とし、残る26条件M3と複数prompt AB/BAは
ユーザー決定により段階12の受入から除外した。段階10・11の範囲は維持する。

- 両GPUとも既定幅2を維持し、幅3・4は明示選択として残す。GPU別の既定値分岐を作らない。
- WU-12Aの機能試験とWU-12Cの重大退行防止を照合し、結果・省略した測定・残る制約を履歴に記録して段階12を閉じる。

## 今後の順序（2026-09-22整理）

段階5・段階6と[vllm-mxfp4の分析](../../../../../history/2026/09/21-30/vllm-mxfp4-optimization-analysis.md)を踏まえ、
残りの作業を効果の大きい順に並べ替える。基準は段階6採用後の通常計測
（V620 16.8134／30.6959、R9700 21.4513／35.6294 tok/s。MTPなし／あり）とする。

MTPなし1 tokenの残りの時間の内訳（段階6後のprofile、ms/token）は次のとおりで、
**最大の塊はkernel間のdispatch固定費**である。段階6の並列枝はV620で約1.3 ms減らしたが上限には届かず、
graph化でも取れないことは段階5で確認済みである。したがって次はnode数そのものを減らす。

| 対象 | V620 | R9700 |
| --- | ---: | ---: |
| graph内のkernel間gap | 6.09 | 5.19 |
| FP8行列積（WU0参照比の余地） | 25.0（1.05） | 18.5（WU2で大半を回収済み） |
| NVFP4行列積（同） | 21.5（0.83） | 17.2（1.41） |
| 活性値量子化＋RMSNorm等 | 約4.1 | 約3.1 |
| full attention | 4.8 | 2.2 |

### 順序と根拠

1. **段階7: 活性値量子化を前段producerへ融合（完了。C1の88 nodeとC2の112 nodeを採用）**
   - 1 tokenあたり量子化185回、RMSNorm・residual・SiLU等の軽いkernelも多数あり、
     これらをproducerへ畳み込めばnodeとgapを同時に減らせる。1,170 nodeのうち削減余地が最も大きい。
   - vllm-mxfp4も同じ結論に達しており、decodeの128箇所で2〜3 kernelを1個へ集約し、
     producerが`(codes, scale)`をconsumerへ直接渡す契約にしている。相手の計測では量子化は
     実処理2.2 µsに対しdispatch 4.7 µsで、dispatch側が支配的だった。
   - sLLM側の上限は「削減できるnode数×1 nodeあたりのgap（V620約5.2／R9700約4.4 µs）＋量子化kernel自体の時間」。
     着手時に対象families（FP8 per-row、NVFP4 block16＋tensor scale、MXFP8 KV前処理）ごとに上限を算出し、半分を打切り線とする。
   - 数値はbit一致を目指しN0。NVFP4のtensor scaleはreduction契約を変えないことを先に確認する。
   - 2026-09-22の試行はconsumer側（gate/up projection pack kernel内）へ量子化を取り込むもので、段階7の対象ではなかった。
     WU2のC2と同種で1%未達と判定済みの設計であり、実装もexact gfx1030の最初のsmokeで`execution resource is busy`となって計測へ到達していない。
     productionへ残っていたdraftは破棄し、対象familiesは一つも評価していない。[破棄の記録](../../../../../history/2026/09/21-30/phase87-stage7.md)
   - **consumer側（matmul kernel内）への量子化取り込みは段階7の候補にしない。** producerへの融合だけを対象とする。
     実装は[AGENTS.mdのkernel融合方針](../../../../../../AGENTS.md)に従い、ビット一致版を先に作って分解版を対照にする。
   - 段階6のfork（V620のNVFP4 M1/M3、FP8 M1）を対照に含める。両者は別種の削減なので加算しない。
2. **段階9: MTP draft専用の縮小語彙lm_head（完了、2026-09-24）**
   - MTPありだけに効き、見込みは正味約+3.8%（V620）／+3.2%（R9700）。段階7（MTPなし中心のnode削減）とは
     対象の費用も触るfileも別なので、並行して進めてよい。
   - 段階3より先に置く。companion形式の速度比較を、draft lm_headが軽くなった後の費用構成で行うためである。
3. **段階4: W×A16の廃止と契約の整理（WU-4Rを含め完了、2026-09-24）**
   - 呼び出し元のない旧NVFP4 W4A16 kernelも削除し、数値8／9／10は監査用tombstoneとして保持した。
   - 両GPUでlowp／HIP runtimeとQwen3.8のW4A4生成tokenを確認した。
4. **WU-P1: Paged Attentionの試作と軽い検証（2026-09-24完了）**
   - 両GPUのkernel単体と数値が受入を通過。本番sourceは変更せず、本移行は段階10としてPhase 87へ統合した。
5. **段階3: MTP companionの形式（WU-3Sまで完了、2026-09-24）**
   - NVFP4とMXFP6のM1差は最大約1.01ポイントで5ポイント規則は不成立。MXFP6 companion sidecarは退役した。
   - NVFP4はM4で両GPU1%以上改善した一方、通常AB/BAのTPOTは両GPUで悪化した。共通ルールを今回の判断へ適用せず、ユーザー決定でNVFP4を既定化して段階3を閉じた。
   - vllm-mxfp4はdrafterをMXFP4にすると受理が2.5→2.21へ落ち、FP8 per-channelなら2.60〜2.80を維持したと記録している。
     「低bit化で受理率が落ちると全体が遅くなる」という論点の先行事例として、評価時に参照する。
6. **段階2（V620のFP8 W8A8は新規採用なしで完了、2026-09-25）と段階1（NVFP4 W4A4、次）**
   - WU0再計測の余地はV620 FP8約1.05、V620 NVFP4約0.83、R9700 NVFP4約1.41 ms/token。
   - 着手前に、vllm-mxfp4側の技法（weightのWMMA fragment順配置＋non-temporal load、LDS padding、
     SGPRへのwave-uniform base address、小M用のTM=ceil(M/16)と深いsplit-K）とsLLM現行providerの差分を
     読み取りだけで確認し、未実装のものだけを候補にする。相手の数値は倍率として転用しない。
   - V620のNVFP4には[D系統](../../../../../history/2026/09/21-30/phase87-stage6.md)で打ち切った近傍依存（約1.9 ms/token）が残る。
     再開条件は満たさないが、このstageの対照は実モデルの先行条件で取る。
7. **段階12: MTP幅3・4への対応と最適化（段階1の後、2026-09-25追加）**
   - 幅を2・3・4で選べるようにする（機能として必須、既定にしなくても残す）。a₃・a₄とM3からM4を見積もり、
     M=4・5で重い処理を最適化し、既定の幅を変えるかを判断する。
   - 見込みは1回あたりの確定token数で幅3が約+16%、幅4が約+25%。attentionの行間KV共有（段階11のC1）は前倒ししてよい。
8. **段階10: Paged KV／Attentionへの本移行（2026-09-24、Phase 87へ統合）**
   - 2026-09-25完了。vAttentionを廃止し、本番のdecode・prefillをblock単位のKV poolへ接続した。計算の構造は変えていない。
   - 段階1（行列積）とは触るfileがほぼ重ならないため、並行してよい。graph／runtimeの共有fileを変えるときだけ順序を調整する。
9. **段階11: attention kernelの最適化（段階10の後、2026-09-24追加）**
   - 2026-09-25採否完了。MTP verifyのM=3行間KV共有を両GPUで限定採用。R9700のGQA共有とV620のM1内側loop、prefillのdot2／FP8 WMMA試作は不採用。
   - 分析時点の上限の目安はMTPありで確定tokenあたり約10〜11%、prefillでTTFTの2〜3割で、Phase 87の残り候補の中で最も大きい。
10. **段階8: gate/upとGDN qkv/zのdual-output bundle（2026-09-25にPhase 87から外し、[backlog](../../../../backlog.md)のP13へ移した）**
   - 段階7の後に判断する予定だったが、判断されないまま残っていた。段階7後は、NVFP4 gate/upの活性値がproducerで量子化済みになり、
     GDN qkv/zの量子化も共有済みのため、減るのは各組の起動1回分だけである。上限の目安は104組×node gapで
     V620約0.54、R9700約0.46 ms/token（TPOTの約1%前後）。新しいkernel変種を伴う「重い変更」でend-to-end 1%が必要なため、
     V620は並列枝の分だけ届かない見込みで、R9700がぎりぎりである。
11. **[Phase 88（llama.cpp比のkernel効率）](../../../../active/2026/09/21-30/phase88-qwen38-llama-kernel-parity.md)**（2026-09-26新設）
   - decode attentionのtile化、NVFP4 matvecのK方向分割、gate/up融合（旧段階8を含む）、GDN、prefill（A-tiled producer-consumerを含む）を扱う。
12. **Phase 89（リクエストバッチ処理、旧Phase 88）**
   - KV容量設計（group size選択）は、ここで扱う。
   - 段階10（Paged KV本移行）はPhase 87の中で完了し、バッチ用のattention kernelとKV管理はpagedを前提に書く。

### 段階7の着手時上限と候補（2026-09-22）

計画の「着手時に対象familiesごとに上限を算出し、半分を打切り線とする」に従い、実装前に上限を固定する。
式は `削減できるnode数 × nodeあたりgap（V620 5.2µs／R9700 4.4µs）＋ 量子化kernel時間`。
node数と量子化時間は[段階0結果JSON](../../../../../history/2026/09/11-20/phase87-stage0-results.json)の
MTPなし・127 transition実測（`sllm_matmul_bf16_to_fp8_outer_v2` 185回、
`sllm_matmul_bf16_to_nvfp4_block16_wave8_v1` 112回、
`sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1` 16回）を使う。
当時の通常計測はV620 16.8134／R9700 21.4513 tok/s（MTPなし）である。
以下は2026-09-22時点の探索余地と打切り線であり、現在の採否ルールを定める表ではない。

| family | node/token | 上限 V620 | 打切り線 V620 | 上限 R9700 | 打切り線 R9700 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP8 per-row | 185 | 2.9215 | 1.4607 | 2.4514 | 1.2257 |
| NVFP4 block16＋tensor scale | 112 | 0.9869 | 0.4935 | 0.8217 | 0.4109 |
| MXFP8 KV前処理 | 16 | 0.1590 | 0.0795 | 0.1192 | 0.0596 |

単位はms/token。MXFP8 KV前処理の上限0.1590／0.1192は、当時の全体TPOTの約0.27%／0.26%にとどまる。
新kernelを伴う変更では[共通ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)の全体改善1%にも届かない見込みであり、
候補C3としては実装せず[backlog](../../../../backlog.md)へ移した。

**量子化185回／112回の producer 対応表（MTPなし）**

| producer | 消費者 | FP8 node | NVFP4 node |
| --- | --- | ---: | ---: |
| `input_rmsnorm`（FullAttention層のみ） | full q/k/v | 48 | 0 |
| `input_rmsnorm`（LinearAttention層） | GDN qkv/z（FP8）＋b/a（**BF16**） | 48 | 0 |
| `linear_attention_state` | GDN out | 48 | 0 |
| `post_attention_rmsnorm` | mlp gate/up | 16 | 56 |
| `mlp_silu_mul` | mlp down | 8 | 56 |
| `full.sigmoid_mul` | full o | 16 | 0 |
| `final_rmsnorm` | lm_head | 1 | 0 |
| 合計 | | **185** | **112** |

**候補（作業単位あたり3つまで）**

1. **C1: FP8 per-rowのproducer融合**。対象は上表のうち LinearAttention層 `input_rmsnorm` を除く
   **137 node**（`input_rmsnorm` FullAttention層48＋`linear_attention_state` 48＋`post_attention_rmsnorm` 16
   ＋`mlp_silu_mul` 8＋`sigmoid_mul` 16＋`final_rmsnorm` 1）。到達上限 V620 2.161／R9700 1.810 ms/token。
   FP8の行スケールは `amax/448` を当該kernel内で完結でき、**producerへ渡す外部スケールが不要**。
2. **C2: NVFP4 block16＋tensor scaleのproducer融合**。対象は `post_attention_rmsnorm` 56＋`mlp_silu_mul` 56
   ＝ **112 node**（family全量）。上限0.9869／0.8217、採用に必要な0.5948／0.4662は上限の60%／57%。
   producerには `input_global_scale`（`lowp`の`input_tensor_scale[0]`）が必須で、**現行producer descriptorは
   このスケールを持たない**。ABI拡張の要否は着手時に決める。
3. **C3: MXFP8 KV前処理** — 上表のとおり到達不能のため対象外。

**LinearAttention層の`input_rmsnorm`（48 node）をC1から外す理由**: 同一出力 `normed` を
BF16の`linear.b_matmul`／`linear.a_matmul`とFP8のqkv/zが同時に消費し、出力をencodedへ置き換えられない。
2本目の出力bindingが要るため、段階7では残件とする。C1到達上限はこの除外を織り込む。

段階7の実装方針は下記のままとし、**consumer側（matmul kernel内）への量子化取り込みは候補にしない**。

**段階7の採否（2026-09-24）**: C1のうちproducer内でFP8行scaleを完結できる88 nodeを両GPUで採用した。
同一processのHIP graph AB/BAでは88 node構成の短縮がV620 0.842541、R9700 0.704588 ms/replayで、
各7 roundすべて短縮率が1%を超えた。通常8192/128のMTPなし／ありも両GPUで段階6より速く、
生成token列と固定128位置の全語彙logitsは段階6相当対照と一致した。
`linear_attention_state`→GDN outの48 nodeはheadごとに分かれたkernelから全headの行amaxを求められず、
`final_rmsnorm`→lm_headの1 nodeは最終行aliasからscale planeの最終行を参照できないため、今回の融合範囲から除外した。
既に除外したLinearAttention `input_rmsnorm` 48 nodeと合わせ、未融合97 nodeを採用数に含めない。
C2（NVFP4 producer融合、112 node）は、同一process単体でV620 0.263、R9700 0.325 ms（TPOT比0.45%／0.71%）の短縮だった。
それでも2026-09-24のユーザー決定で採用した。理由は、N0で全roundが改善し、実装が完了済みで、無効のまま経路を残さないため。
C1+C2の通常計測（MTPなし）はV620 17.2036、R9700 21.9810 tok/sで、生成token列は段階6と一致した。MTPありはユーザー指示で再計測していない。
C3は着手時の上限が小さく対象外とした。[検証記録](../../../../../history/2026/09/21-30/phase87-stage7-c1.md)へ数値と残件を記録する。

### 当面着手しないもの

- MTP draft headの低bit化＋厳密rerank、MTP幅3の再比較、GDNのconv＋recurrent追加融合など、計画のない候補は
  [未計画の課題一覧](../../../../backlog.md)へ移した（2026-09-24）。
- vllm-mxfp4由来のうち[分析](../../../../../history/2026/09/21-30/vllm-mxfp4-optimization-analysis.md)で
  「適用しない」と整理した項目（MXFP4 W4A8形式、R4D attention、DFlash2、int2 target verify head、
  lazy GDN snapshot、KV pin、ParoQuant、TP関連、sourceの直接流用）。

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
- llama.cpp比のkernel効率（Phase 88）とリクエストバッチ処理（Phase 89）。

## 実行時の注意

- Codexは既定のOpenAI profileで動かす（2026-09-19、週間制限の回復によりdeepseek-flashは当面不要）。
  subagentの選択はAGENTS.mdのprofile別規則に従う。
- V620を使う間はローカルQwenサービスを止め、R9700はservice lease手順に従う。

段階0履歴: [計測・棚卸しと優先順位](../../../../../history/2026/09/11-20/phase87-stage0.md)

WU0履歴: [読み出し専用帯域と打ち切り線](../../../../../history/2026/09/11-20/phase87-wu0-read-bandwidth.md)

WU1履歴: [MXFP8 E4 decode attentionのGQA共有](../../../../../history/2026/09/11-20/phase87-wu1-attention.md)

WU1.1履歴: [split増の組合せと新基準での採否](../../../../../history/2026/09/11-20/phase87-wu1-1-attention.md)

WU-C1履歴: [不要な切替と実験経路の削除](../../../../../history/2026/09/11-20/phase87-wu-c1-cleanup.md)

段階6履歴: [graph内の並列枝と再計測](../../../../../history/2026/09/21-30/phase87-stage6.md)

R9700 FP8単独追加評価: [不採用の測定根拠](../../../../../history/2026/09/21-30/phase87-r9700-fp8-fork.md)（出力一致、M1速度不安定・M3退行）。

段階7履歴: [FP8 producer融合とC2/C3採否](../../../../../history/2026/09/21-30/phase87-stage7-c1.md)。

段階4履歴: [W×A16廃止と契約整理](../../../../../history/2026/09/21-30/phase87-stage4.md)。

段階9履歴: [MTP draft縮小語彙head](../../../../../history/2026/09/21-30/phase87-stage9.md)。

WU-P1履歴: [Paged Attention試作](../../../../../history/2026/09/21-30/phase87-wu-p1-paged-attention.md)。

段階3履歴: [MTP companion NVFP4／MXFP6比較](../../../../../history/2026/09/21-30/phase87-stage3.md)。

WU-2V履歴: [V620 FP8 decode残件](../../../../../history/2026/09/21-30/phase87-wu-2v-v620-fp8.md)。

段階1履歴: [NVFP4 W4A4 decode](../../../../../history/2026/09/21-30/phase87-stage1.md)。

段階12履歴: [MTP幅3・4](../../../../../history/2026/09/21-30/phase87-stage12.md)。

段階10履歴: [Paged KV本移行](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)。

段階11履歴: [Paged attention kernel最適化](../../../../../history/2026/09/21-30/phase87-stage11.md)。
