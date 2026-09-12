# Phase85: MXFP8／MXFP6共通カーネルの改善と推論効果検証

> 状態: 実装・検証完了（公開結果は当該commitのGitHub Checksで確認）
> 作成日: 2026-09-13
> 起点: Phase84.5完了 `c2667ad885512329977d5883f3ea7b55dcfe2838`

## 目的と順序変更

2026-09-13のユーザー指示により、既存Phase85・86を後回しにして本Phaseを挿入する。
MXFP8／MXFP6の共通カーネルを広範な行列形状で最適化した後、共通処理を利用する推論が改善したかを、
モデル本体の量子化、KVキャッシュ、MTPの三用途で確認する。旧Phase85はPhase86（他精度の残差）、
旧Phase86はPhase87（NVFP4リクエストバッチ処理）へ繰り下げる。

2026-09-13の追加指示「Phase85を完了する」により実装・検証・公開まで開始する。以下のshape集合・測定回数・実行段階は、この目的を実行可能にするAI提案であり、
独立した承認段階や新しい速度下限を追加しない。実装開始時に実在shapeと既存検査の棚卸しから有限の比較manifestを確定し、
数値判定・比較条件を候補実装前に記録する。未達を隠すための途中の条件変更は行わず、必要な変更は理由を残して同じ作業を再計画する。

## 対象と共有境界

GPUはcanonical V620 exact `gfx1030`とR9700 exact `gfx1201`、各single GPUとする。
ROCm 7.14.0、target別Code Object V6／wave32を基準に、実行時のdriver・kernel・compiler・GPU UUIDを記録する。
既存の[GPU](../../../../../compatibility/gpu.md)、[AMD GPU](../../../../../compatibility/amd-gpu.md)、
[software](../../../../../compatibility/software.md)の契約を維持し、今回の結果を他targetの実機証拠へ拡張しない。

| 共有部分・利用先 | 対象 | 比較で分離する内容 |
| --- | --- | --- |
| block codec・packed I/O | E4M3FN／E3M2、block32、E8M0 scale、量子化・展開 | codec単体、scale読出し、matmul／KVでの利用 |
| activation量子化・matmul | MXFP8 W8A8／MXFP6 W6A6、M=1・small-M・prefill | activation pack費用、packed weight ingress、tile／reduction、出力 |
| モデル本体 | 既存MXFP8／MXFP6 model lockと通常provider | prefillとdecode、実際の選択kernel、全推論への寄与 |
| MTP companion | Phase84のMXFP8／MXFP6専用8行列、共有matmul | prefix準備・draft・verify・replay・採用率と実効decode |
| KVキャッシュ | 既存standard OCP MXFP8 E4 append／attention | append、K/V decode、attention schedule、長さ・配置・state寿命 |

モデル本体とMTPのmatmulは同じ演算契約で改善する。KVは主にcodec・scale・packed I/Oを共有するが、
attentionを通常の重み行列積へ読み替えない。共有部分の改善がKVへ届かない場合も、経路と実測値を明記する。
MXFP6 KVは現行対象ではなく新設しない。既存MXFP8 E5は共有変更が届く場合に限り、対応targetの回帰確認へ含める。

共通化はformat policyとGPUごとの実行scheduleを分ける。gfx1030のsoftware／half2経路とgfx1201のWMMA経路を
同一tileへ無理に統一せず、scale・layout・ingress・epilogueを共有し、能力とshapeでspecializationを選ぶ。
selectorはtarget、format、M/N/K、layout、alignment、資源条件に基づき、model名やbenchmark case IDに依存させない。

### 現行sourceと改善が届く範囲

| 入口 | 現行の責務 |
| --- | --- |
| [low_precision_block_codec.hpp](../../../../../../native/hip/src/low_precision_block_codec.hpp) | format codec、E3M2 packed access、E8M0、block view、量子化の共有境界 |
| [low_precision_matmul_provider.hpp](../../../../../../native/hip/src/low_precision_matmul_provider.hpp) | format／layout契約、shape selector、prepared provider |
| [matmul_kernel.hip.cpp](../../../../../../native/hip/src/matmul_kernel.hip.cpp) | activation quantizer、MMQ、half2／WMMA body、format別ingress、launch |
| [mtp_quantized_sidecar.rs](../../../../../../crates/sllm-core/src/mtp_quantized_sidecar.rs) | MTP専用8行列のsidecar。graphから通常MXFP8／MXFP6 providerへ接続 |
| [kv_state_kernel.hip.cpp](../../../../../../native/hip/src/kv_state_kernel.hip.cpp) | KV append／quantization。matmulとは別scheduleでcodecを利用 |
| [causal_attention_kernel.hip.cpp](../../../../../../native/hip/src/causal_attention_kernel.hip.cpp) | MXFP8 KV読出しとattention。shape選択は対応runtimeと合わせて確認 |

現行gfx1030の共通half2 ID55／57は主にM>=128向けであり、MTPのM=1〜3はdecode／row8／tiled16経路へ進む。
gfx1201でもMXFP8 WMMAは主にM>=128、MXFP6はM>=17が境界となる。したがってlarge-M改善の延長だけにせず、
**低M向けの共有load／reduction／tileとselectorを明示的な候補に含める**。既存Phase84のM=1棄却候補との違いを先に整理する。
MTP prefix準備だけが速くなった場合はdraft decode改善と区別し、実際のconsumerとMを記録する。

## 広範なshape比較

表記はactivation `[M,K]`、weight `[N,K]`、output `[M,N]`とする。
既存model／MTP graphの実在shapeを抽出し、次の合成shapeを加える。全直積は作らず、各軸・境界・矩形分類を
両format・両targetで覆う有限集合にする。実在shapeの出現回数とwall寄与は、合成shapeの性能分布と別に集計する。

| 軸 | 初期候補 | 狙い |
| --- | --- | --- |
| M: decode／MTP | 1, 2, 3, 4, 7, 8, 15, 16, 17 | M=1、提案／検証のsmall-M、tile切替 |
| M: prefill | 31/32/33, 63/64/65, 127/128/129, 255/256/257, 511/512/513, 1024, 2048 | tail、tile境界、chunk-sized large-M |
| N | 17, 31/32/33, 63/64/65, 127/128/129, 1023/1024/1025, 2559/2560/2561, 4096, 5120, 6143/6144/6145, 16383/16384/16385, 17408, 32767/32768/32769 | 小出力、wide-N、現行selector内外 |
| K（有効値） | 32, 96, 992/1024/1056, 2016/2048/2080, 2560, 4096, 5120, 12288, 17408 | K32 block、非2冪、長reduction、資源境界 |
| N/Kの組 | 正方形、N≫K、K≫N、実在q/k/v/o・gate/up/down・fusion・head | 一つの都合のよい形状への偏りを防ぐ |

大きいN/KとMは代表組合せに絞り、allocation見積りで収容を確認する。実在する更に大きいhead等は、そのformatを
実際に使う場合だけ追加する。M/Nのtailは現行契約が受理する範囲を数値検証し、未対応layoutを黙って追加しない。
K非32倍数（例31/33、2047/2049）、zero dimension、未対応stride／targetは拒否契約として別に検査する。
Kを切り上げて元の非対応入力を成功扱いにしない。

実在shapeの初期候補は、4BのK/N=`2560/9216`・`9216/2560`、9Bの`4096/12288`・`12288/4096`、
27B／MTPの`5120/17408`・`17408/5120`・`10240/5120`・`6144/5120`・`5120/1024`とする。
これらはoperator集合への追加であり、全サイズのfull-model測定を要求するものではない。

KVは別の形状集合とする。query countは1, 3, 7, 17, 31/32/33、127/128/129、1023/1024/1025、KV長は31/32/33、127/128/129、
1023/1024/1025、4095/4096/4097、8191/8192/8193を出発点とする。head dimension・Q/KV head比は
既存対応modelから選び、causal tail、chunk境界、非zero append位置、partial block、capacity grow前後を含める。
実際のselectorに追加境界があれば置き換え・追加の理由をmanifestへ残す。

## 実行手順

以下は一つのPhase内の実装順であり、段階ごとの新しい公開・独立reviewを要求しない。

1. **共有経路と基準の棚卸し。** Phase62〜75、83.5、84の採用済み実装・棄却理由を読み、model／MTP／KVから
   provider・kernel・codecへの対応表を作る。現行通常経路でshape一覧、数値基準、時間・資源を取得する。
   起点commit以後に意味のある変更があれば実際のsource/build identityを基準にし、古い測定を同一条件の対照にしない。
2. **広範shapeでの最適化。** 両GPU・両形式の基準を見て、寄与の大きい共有処理から改善する。
   activation quantization、E8M0 scale／packed load、tileとK staging、reduction、tail、出力書込みを候補とする。
   MXFP8でscheduleを評価後、同じ骨格のMXFP6とformat固有ingressを比較し、共通改善とE3M2固有効果を分ける。
   small-Mとlarge-Mを両方扱い、register／LDS／spill／occupancyとlaunch費用を確認する。
3. **KVの共有処理への適用。** codec・packed I/Oの候補をappend／attentionで評価する。K/Vのvalueとscale、
   vector load、decode再利用を優先し、必要なattention schedule変更は独立候補として効果と数値差を記録する。
   matmul、codec/KV、両方の候補を必要な代表行で切り替え、改善源を分離する。
4. **通常selectorへの接続と三用途の推論比較。** 下表のモデル本体・KV・MTPを両GPUで実行する。
   強制kernelだけの勝利にせず、通常CLI/APIから採用経路へ到達したことをdispatch/profileで確認する。
   未改善・退行があれば費用内訳を確認し、共有候補または適用scopeを修正する。全model最適化へ際限なく広げない。
5. **採否と引継ぎ。** 最終shape分布、三用途の前後比較、数値結果、資源、採用selector、不採用と未測定の理由をまとめる。
   Phase86へ残差だけを引き継ぎ、既存のPhase完了手順で関連検証・一回の統合review・commit／push・当該CI確認を行う。

既存llama.cpp checkoutのMMV/MMQ、codec、attentionから直接再利用できる構造を検討する。
既に棄却した候補は同じ条件で再試行せず、新しいshapeや費用内訳による根拠を記録する。
外部コードをcopy／adapt／portした場合はscratchを含め、同じ作業中に取込み一覧と詳細noticeを更新する。
参照だけの調査では新しい取込み記録を作らない。[来歴管理方針](../../../../../provenance/README.md)に従う。

## 推論比較の構成

全行で同一formatの最適化前後を主比較とし、異なるformatやBF16との比較は別列にする。
tokenizer／model revision、input token、sampler、seed、KV、MTP幅、prefix cache、chunk、Graph設定を揃える。
三用途を混ぜた一行だけで改善を認定せず、以下の対照で寄与を分ける。

| 用途 | 代表構成・対照 | 入出力と確認内容 |
| --- | --- | --- |
| モデル本体 | Qwen3.5-4Bの既存MXFP8／MXFP6 lock、MTP off、FP16 KV固定 | 512/128、2048/128でprefill/decode。513/17で未整列入力の公開経路 |
| KV単独の影響 | 同4B BF16 weight、MTP off、MXFP8 E4 KV。FP16 KVを同条件の対照にする | 513/17、8192/128。append／attention時間、長文decode、grow・要求再利用 |
| 本体＋KV | 同4BのMXFP8／MXFP6 weight＋MXFP8 E4 KV、MTP off | 2048/128。組合せ時の速度・数値・workspaceを確認 |
| MTP | Qwen3.8-27B NVFP4 target＋MXFP8 E4 KV固定。MTP off／BF16／MXFP8／MXFP6 companion | 8192/128。Phase84と同じprefix準備・draft・verify・replay・sampling内訳を取得 |
| 通常API | 上記各用途の代表構成を通常serverから実行 | 短文non-stream／SSE、cancel後recovery、別要求、unload／cleanup、dispatch到達 |

4Bは[BF16 source lock](../../../../../models/locks/qwen3.5-4b-bf16.json)と、既存MXFP8／MXFP6 GGUFの
derived lock・変換recipe・artifact digestを組にして固定する。Qwen3.8は専用runtimeのsource/artifact identityと
MTP sidecar identityを記録し、Qwen3.5-27B lockを代用しない。MTP主比較の幅は既存の2、
固定samplingはtemperature=1.0／top_p=0.95／top_k=20とする。chunk／capacityは既存8192/128条件を照合して固定する。

KVには現行gfx1201のresident選択を使い、旧VMM不具合を再導入しない。prefix cacheは速度主比較で無効または空にし、
要求再利用の正しさは別ケースで確認する。MTPの採用0/1/2後の有効KV value/scale、GDN／companion state、次の計算も照合する。
Phase84.5のBF16限定診断をそのまま量子化MTPの証拠にせず、量子化companionを通る既存検査を使い、不足分だけ拡張する。

MTP採用率はPhase84の12言語／タスク条件×3 seedを最終候補で再利用し、同一GPUのBF16対照とMXFP8／MXFP6を比較する。
候補ごとの全suite反復は行わず、探索中は短い固定履歴と代表入力を使う。採用率だけで採否を決めず、
確定token/s、target検証回数、棄却・replay費用と合わせる。同じseedでも生成履歴が異なるため、GPU間の率差をkernel誤差へ直結させない。

大型の追加full-model測定は、4B／27B MTPとoperator集合で覆えない共有経路が見つかった場合に、
既存lockから最小の代表を選ぶ。新規model対応、全model・全形式の直積検証は追加しない。

### 再利用する検証入口

| 既存入口 | 再利用・必要な拡張 |
| --- | --- |
| [sllm-mxfp-wa-evidence.rs](../../../../../../crates/sllm-hip/src/bin/sllm-mxfp-wa-evidence.rs) | Phase62〜75／84のoperator、oracle、dispatch、repeat／resource出力を再利用し、有限shape集合と候補比較を追加 |
| [low_precision_block_codec_gpu_test.hip.cpp](../../../../../../native/hip/tests/low_precision_block_codec_gpu_test.hip.cpp) | 全code・packed／scale・provider境界。既存host selector検査と合わせる |
| [sllm-qwen35-mx-weight-quality.rs](../../../../../../crates/sllm-hip/src/bin/sllm-qwen35-mx-weight-quality.rs) | 固定10ケースのlogits／top-1／KLD／perplexity。現状MXFP8中心なのでMXFP6比較と最適化前後の同形式対照は不足分を実装 |
| [sllm-kv-mxfp8-e4-evidence.rs](../../../../../../crates/sllm-hip/src/bin/sllm-kv-mxfp8-e4-evidence.rs) | head dimension 31/32/33/255/256/257のbyte／tail検査と、対応head dimensionのattention oracle |
| [sllm-qwen35-kv-quality-probe.rs](../../../../../../crates/sllm-hip/src/bin/sllm-qwen35-kv-quality-probe.rs) | 固定10ケースのFP16 KV／MXFP8 E4品質比較。速度行と混同しない |
| [sllm-phase78-qwen38-benchmark.rs](../../../../../../crates/sllm-hip/src/bin/sllm-phase78-qwen38-benchmark.rs) | Phase84のcompanion指定・8192/128・MTP内訳を再利用。Phase84.5のBF16専用診断とは区別 |

GPU benchmark・quality runnerの既存長さやsampling制約は最初に確認し、表の速度行に不足する計時・引数だけを拡張する。
runnerが存在することだけを全条件実行可能の証拠にしない。過去Phase73のwide-N MXFP8 selector拡張はhost検査中心なので、
今回のN=32768内外を含む実GPU比較で測定範囲を明確にする。

## 数値・性能の判定

- codecは全256 E4M3FN code／全64 E3M2 code、E8M0境界、zero、subnormal、NaN scale、Inf saturation、
  block/lane/tailを既存の独立NumPy oracleへ照合する。量子化recipe、scale選択、丸め、NaN伝播を維持する。
- matmulは独立のdequantized FP32 oracleと既存providerの両方を対照とし、absolute／relative error、
  nonfinite位置、BF16出力を記録する。bit維持変更と加算順変更を分け、加算順変更では固定履歴のhidden／logits差と
  既存品質指標も調べる。既存検査の許容差と対象範囲を採用し、未定義なら基準測定で提案を明示してから候補を比較する。
- KVはvalue/scale byte、非更新prefix、独立attention oracle、grow／rollback後の状態を確認する。
  kernel oracle成功だけでBF16比のfull-model品質同等性を認定しない。旧KV defaultのtop-1閾値をW/A最適化へ流用しない。
- GPU成功にはexact target、HIP-only、fallback未使用、実行case数、数値oracle、cleanupを記録する。
  CPU代替・compile成功・未選択・timeout・crashをGPU PASSにしない。CPU CIは小さい契約・oracleとcompile-onlyに限定する。
- operatorは同一buffer条件のwarmup後反復で比較する。初期案の3 warmup＋10 measuredから、実行時に確定した
  3 warmup＋13 measuredへ揃える（訂正経緯は履歴に記録）。kernel時間に加え、
  activation packを含むoperator全体の時間も測る。前後を交互に測り、必要ならcontrol/candidate/controlでdriftを確認する。
- 推論は1 warmup＋3 measuredの中央値・MAD／範囲を基準に、TTFT、prefill tok/s、TPOT／decode tok/s、
  E2E、peak／resident VRAMを記録する。model loadと初回実行は別に報告する。profile付きrunを速度代表値にしない。
  prefillはMTP prefix準備を含み、decodeは最初の確定token後の実出力をwall時間で割る。draft・棄却tokenを出力数へ足さない。
  早期EOSは実出力数を記録して別扱いにし、固定samplingや終了条件を変えて比較を成立させない。
- shapeごとの比率、分類別中央値／幾何平均、最悪退行とそのscopeを示す。未実行・拒否・OOMは速度集計から区別する。
  推論も三用途ごとに「改善」「ばらつき内」「退行」「未測定」を明記し、単体kernelの勝利をE2E改善へ読み替えない。

採用は数値契約と資源寿命を満たし、測定ばらつきを超える効果を確認したtarget／shapeに限る。
退行する条件は既存providerへ事前にdispatchし、実行失敗後のsilent fallbackを追加しない。
量子化MTPの既定変更は、同じ最終実装のBF16対照との採用率・数値・実効速度から別に判断する。
共通kernel改善の成功を理由に量子化MTPを自動昇格させない。

Phaseの目的は広範shapeの改善と三用途への効果確認である。全shape・全用途が必ず速くなる保証や必達倍率は置かない。
ただし、探索だけを行って三用途の比較を省略した状態は完了としない。候補が不利なら既定を維持し、
改善が届かなかった理由、測定値、適用scopeとPhase86への残差を報告する。

## 実行環境・証拠と作業範囲

GPU測定時はUUIDを再照合して単一GPU可視化し、同一GPUの常駐推論とbenchmarkを重ねない。
V620を使う際はローカルQwenの利用状況を確認し、idle serviceを停止してpairを確保する。Qwenのsingle-V620縮退は行わない。
既存R9700常駐serviceの設定・binaryを記録してから必要な停止／測定／復帰を行い、起動・healthを確認する。
進行中の他要求を強制終了しない。調査・実装の並行作業はnative Lunaを使い、編集範囲を分ける。

既存sLLM runnerとHIP profilerを基本に、profile→source対応→候補比較→通常推論再測定の流れを使う。
Magpieを補助利用する場合はlocal CLIと対応targetを先に確認する。sLLMをMagpieの既存inference backend対応済みとは扱わず、
PyTorch／Triton／torch traceを必要とする計測を持ち込まない。ツールの導入自体を完了条件にしない。

draftはdirty treeを許容し、source差分・build入力・binary・model lock・比較manifestの対応を記録する。
統合／公開時は最終実装に対応する関連検査と証拠を揃える。raw trace、model、binary、巨大shape出力は
`.local-artifacts/phase85/`等へ置き、追跡文書にはcompact結果と参照・digestを記録する。
一律のGPU全面再実行や段階ごとのimmutable identityを追加しない。停止・再計画はAGENTS.mdの既存条件に従う。

対象外は、新規MXFP6 KV、MXFP4 W4A8／NVFP4 W4A16固有最適化、汎用FP8 artifact対応、quantization recipe変更、
新規model／ABI、リクエストバッチ処理、TP／multi-GPU／RDMA、MI300X実機再検証、公開APIの機能拡張とする。
共通source変更が既存の別targetへ届く場合は影響するcompile／contractだけ確認する。

## 記録と引継ぎ

### 完了結果

- primary87、small-M54、selector／M3／N上限の境界を両GPUで比較した。採用A/B/C8と、consumerを限定したC6を通常selectorへ接続した。
- モデル本体・KV併用の各12条件、MTP off／BF16／MXFP8／MXFP6、同形式品質8構成、通常APIと216言語／task／seed条件を確認した。
- 通常MTPのMX companionはM1とlarge-M prefixを使うため、Rows4のconsumer確認を通常CLIの515入力／chunk512末尾M3へ追加した。
  prefillはMXFP8／MXFP6がV620で約10.0%／3.5%、R9700で約24.3%／11.3%改善した。
- R9700の通常MX本体decodeは約4〜6%改善した。V620の通常長の本体とMTP全体は概ね横ばい。
  KV向け候補は退行したため撤去し、量子化MTPはBF16の速度を上回らずBF16既定を維持する。
- 状態照合はV620 normal、R9700の同一履歴attention-only対照で成功した。R9700 normalの既存演算順差はFAILとして保持し、main attentionは変更しない。
- 詳細な採否、基準／候補identity、ばらつき、失敗と再検証は[集約結果](../../../../../../ci/matrix/phase85-mxfp-common-kernel-results-v1.json)と履歴に固定する。
  Phase86へ残差を渡し、全shape・全model・全用途の高速化やBF16比品質同等性へ拡張しない。


成果は共有経路対応表、有限shape manifestと基準値、候補別数値・性能、最終selector範囲、三用途の前後比較、
不採用理由・未測定範囲として履歴へ集約する。Phase86は採用済み改善を引き継いで残差を再棚卸しする。
実装完了または中止時に本計画をarchiveへ移し、履歴との相互リンクを更新する。

[メイン計画](../../../../main-plan.md) /
[Phase76〜87ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md) /
[履歴](../../../../../history/2026/09/11-20/phase85-mxfp8-mxfp6-common-kernels.md)
