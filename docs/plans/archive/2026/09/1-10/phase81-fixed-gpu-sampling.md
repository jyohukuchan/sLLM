# Phase 81: 固定サンプリングの共通GPU経路とAPI性能

> 状態: 完了（2026-09-08）。実装・CI修正commitの公開host／HIP compile-only CI成功を確認。
> 作成日: 2026-09-07
> 根拠: ユーザーによる固定設定の全面採用と、prefill／decode等への速度影響をほぼなくすPhaseの優先実施指示。

## 目的と順序

主要モデルのコーディングエージェント用途に絞った固定sampling profileを、公開APIから共通GPU経路で
実行できるようにする。既存の高速greedy経路に対するprefill／decode・TTFT／TPOTの追加負担をほぼなくし、
CPU samplingに起因する速度低下を解消する。方針の正本は[main-plan](../../../../main-plan.md)とする。

Phase 80のCI修復は完了済み。本Phaseを次に実施し、従来のPhase 81（static FP8 KV・MTP・文章生成）を82、
従来82（他精度）を83、従来83（NVFP4 batching）を84へ繰り下げる。後続の作業内容は保持する。
本計画の作成を実装完了・GPU性能確認済みとは扱わない。

## 固定設定とAPI契約

2026-09-08着手時に以下の設定、対象runtime、完了条件を実装範囲として確認した。
性能の5%目安は非拘束のままとし、新規モデルbring-upや全形式の直積検証を追加しない。

| 設定 | 採用値 |
| --- | --- |
| `temperature`／`top_p` | `1.0`／`0.95` |
| `presence_penalty`／`frequency_penalty` | `0`／`0` |
| `repeat_penalty`／`repeat_last_n` | `1.0`／`0` |
| `min_p`／`typical_p` | `0`／`1` |
| DRY・XTC・Mirostat・dynamic temperature | 無効 |
| ユーザー指定`logit_bias` | なし |
| `logprobs`／`top_logprobs` | 無効／`0` |
| `ignore_eos` | `false` |
| `top_k` | モデル読込時に固定。Qwen3.5 coding／Qwen3.8 thinkingは20、Gemma4は64、Ministral 3は0（無効） |

- 省略時は採用profileを適用し、同じ値の明示指定も受け付ける。非対応値・非対応stageは実行前に
  未対応エラーを返す。現在のAPI既定top_p=1.0から0.95への変更を仕様・テスト・クライアントへ反映する。
  requestを黙って書き換えたり、遅いCPU samplingへ送ったりしない。
- `seed`、出力token上限、stopは変更可能なままにする。tool calling、JSON制約、reasoningの制御と
  内部token maskを維持する。任意logit_biasの廃止を内部maskの廃止として実装しない。
- top_kはモデル・モードの採用profileとして記録する。他モデルの値は対応着手時に公式推奨、元の
  generation_config、GGUF metadataを確認して決め、未指定を一律20／64へ読み替えない。
  `general.sampling.top_k`はGGUFの任意metadataであり、存在を前提にしない。確定profileを
  読込時に解決し、metadataやファイル形式の違いで採用値が暗黙に変わらないようにする。
- top-k→top-pの順序、候補内の正規化、境界tokenの包含、同率logitの順序を明示する。
  top-k追加は生成分布の変更であり、top-p単独の高速な近似として扱わない。
- GPU RNGのseed／counterと再現範囲を定義する。同じseedでも既存CPU RNGとのtoken列一致を
  無根拠に約束せず、変更があれば版とAPI仕様へ記録する。

## 対象と既知の接続課題

最初の実機対象はV620 `gfx1030`とR9700 `gfx1201`。新しいGPU、モデルアーキテクチャ、重み形式の
bring-upは含めない。共通samplerは語彙数、logitsのdtype/layout、mask、top_k、GPU能力で選択し、
NVFP4やQwen3.8のモデル名をkernel適用条件にしない。

| 対象 | 本Phaseで扱う内容 |
| --- | --- |
| Qwen共通経路 | Qwen3.5の小さい既存モデルで基礎確認し、Qwen3.8 NVFP4のAPIまで同じsamplerを接続 |
| Gemma4 Dense | 既存TokenSelect接続を共通samplerへ移し、top_k=64を確認 |
| Gemma4 MoE等の既存Argmax-only終端 | GPU常駐logitsを共通samplerへ渡す接続を追加。greedy限定をsampling対応と誤認しない |
| KV・重み形式 | 既存FP16 KVとOCP MXFP8 KV、BF16とNVFP4等の既存経路で終端接続を共有。全形式の直積再検証は課さない |
| tool／JSON／reasoning | 現在対応済みのAPI経路で制約を保持。モデルごとの未対応機能の新規bring-upは含めない |
| MTP | 通常target-only samplingの契約を整える。確率的accept/rejectとMTP実用化はPhase 82へ引き継ぐ |

着手時に既存runtime・APIの対応表を作り、接続済み、今回修正、元から未対応を区別する。
Qwen3.8だけの成功を共通化完了とせず、既存の対象runtimeで接続漏れを解消する。
未実機のtargetや未接続の将来モデルをPASS扱いにしない。
既存のgreedy-only MTPを固定profile対応済みと扱わず、非対応の組合せは要求実行前に明示する。
確率的要求をgreedyへ変更して既存MTPを動かす回避は行わない。

着手時の主な障害は以下である（解消状況は実装・実機記録を参照）。

- `sampling.rs`のdevice selector適格条件はtop_p=1.0であり、0.95では全語彙host samplingへ進む。
- Qwen3.8 serverはdevice selector seedの設定を除外する。`qwen_execution.rs`のHIP Graphと
  KV append/attention chainも`device_selector.is_none()`を条件にしている。
- 既存selectorは全語彙additive logits・maskをCPUで用意してH2Dし、tokenごとに作業bufferを確保する。
- 一部の終端providerはArgmaxだけを返すため、APIの設定変更だけではGPU samplingへ接続できない。

## 作業順と成果物

1. **比較条件と採用profileを固定する。** 既存のmodel lock、実行可能なAPI/runtime、KV・重み形式、
   Graph／chain／kernel selectorの状態を整理する。実装前のgreedyと同じ固定sampling設定のhost基準を保存する。
   代表条件をここで選び、無関係なモデル・形式を途中から完了条件へ追加しない。
2. **共通GPU samplerを実装する。** 既存TokenSelect契約の再利用を検討し、GPU上でmask適用、
   top-k、top-p、乱数抽選を行う。temperature=1.0と無効stageの処理を省く。K=20／64を共通骨格で扱い、
   全語彙ソートを避ける候補選択を評価する。K0ではtop-kを省略した正確なtop-pをGPU上で処理する。
   作業bufferとRNG状態を再利用し、hostへ最小結果を返す。
3. **実行経路へ接続する。** Qwen・Gemma等の終端logitsとsamplerを同じqueueの正しい順序で実行する。
   Graph capture/replay中のアドレス・seed/counter更新、KV commit、completion、cancelとbuffer寿命を扱う。
   条件を単に削除する修正で済ませず、sampling時も既存Graph／chainが選ばれることを確認する。
   prefillは必要な最終行だけをsamplingし、全prompt行や中間chunkへ不要な処理を追加しない。
4. **APIとエージェント経路を統合する。** request解析、モデル別profile解決、generation executor、
   Chat／Completions／Responses等の既存入口で同じ契約を使う。SSE、stop、seed、reasoning、tool引数制約、
   cancellation・再要求を確認する。既存クライアント・WebUI・対象サービス設定も固定profileへ合わせる。
   metadata保存とruntimeでの採用値適用を別々に確認する。
5. **残る負担を測り、除去する。** CPU使用時間、D2H/H2D量、allocation、fence、kernel launch、
   sampling時間とGraph選択を観測し、実測で支配的な箇所を直す。内部grammar maskのCPU生成等が残る場合は
   寄与を分離し、無制約textの速度だけでtool経路も高速と結論しない。
6. **対象検証・文書・公開を完了する。** 影響するhost／HIP compile-only／GPU検証とAPI計測を行い、
   成果・未達・対応範囲を記録する。Phase完了時にcommit・pushし、公開HEADのCIを監視、必要な修正・再pushを行う。

実装入口: `crates/sllm-core/src/sampling.rs`、`qwen_execution.rs`、`gemma4_execution.rs`と既存MoE終端、
`crates/sllm-frontend/src/generation.rs`、`reasoning.rs`、`crates/sllm-server/src/api.rs`、`production.rs`、
`phase43_service.rs`、`native/hip`のTokenSelect実装とRust binding。変更箇所に応じてCIのABI／symbol／source
manifestも同期し、Phase 80で修復した検査を無効化しない。

## 性能の比較方法と目標

ユーザー目標は「prefill／decode等の速度にほぼ影響しない」。samplingの演算自体は残るので、ゼロ時間や
過去のgreedy token/sを事前に保証しない。次の三条件を同じtarget・model revision・dtype・KV・context・
出力長・高速化設定で比較する。

| 条件 | 用途 |
| --- | --- |
| A: 既存の最速の正しいgreedy経路 | sampling追加負担の基準。公開APIからgreedyを廃止しても内部比較用に保持 |
| B: 同じ固定profileの既存host sampler | CPU→GPU移行の改善量を確認。QwenとGemmaで各採用top_kを明示 |
| C: 固定profileの共通GPU sampler | 採用候補。Aとの差とBからの改善を別に報告 |

- sampler単体は同じlogits・maskを使う。model実行への影響は同じprefix／位置／token履歴を再生する
  内部benchmarkで分離し、実際の自由生成API計測も併記する。異なる生成token・EOS到達・出力長による差を
  sampling時間と混同しない。Bを実行できない旧providerでは欠測理由を示し、架空の基準値を作らない。
- prefill本体、最終sampling、TTFT、steady decodeのTPOT／token/s、要求全体のwall timeを分ける。
  CPU使用時間、転送量、peak VRAM、workspace、Graph／chainの適用状態も記録する。
- 短いpromptと既存の長い実入力（Qwen3.8では9,435-token条件）を使い、cold captureとwarm replay、
  無制約textと対応済みtool／JSON・reasoning要求を分ける。既存最適化を無効にしたAと比較して達成扱いにしない。
- warmup後の複数回測定で中央値・ばらつき・decode遅延の裾を確認する。初期目安は3回とし、
  ノイズや差の帰属が不明な条件だけを追加測定する。profile有効時の速度を通常性能として用いない。
- **非拘束の数値目安（AI提案）:** Aに対するprefill時間、TTFT、TPOTの増加を各5%以内、
  可能なら測定ばらつきの範囲内へ抑える。起源は本計画作成時の「ほぼ影響しない」の具体化案であり、
  ユーザー承認済みの数値gateではない。対象は選んだ代表条件、費用は上記比較の集計のみとし、
  初回baseline確認時に妥当性を再評価して更新・失効させる。平均で条件ごとの退行を隠さない。
- 明確な速度低下が残る場合は、原因・追加ms/token・相対差・残作業を示し、本来の目標を達成したとしない。
  条件の除外や品質・性能目標の妥協が必要なら、ユーザー決定として記録する。

## 正しさ・統合確認と完了条件

- [x] 固定profileと省略／明示同値／非対応指定のAPI契約が実装され、他設定への暗黙変換・CPU sampling fallbackがない。
- [x] 共通samplerがtop-k／top-pの候補・正規化・抽選を小さい独立NumPy oracleと実GPUで確認できる。
  Kより小さい語彙、K前後、非整列語彙、同率、極端なlogits、0.95境界、単一有効token・全maskを含め、
  非有限値の扱いも契約化する。seed/counterの再現性と複数seedの分布を確認し、greedy一致だけで正しさを証明しない。
- [x] 既存の対象runtime・APIが共通経路へ接続され、Qwen3.8 NVFP4に限定されない。既存KV形式の接続を
  保ち、対象ごとの実装・実機確認・未対応を区別できる。
- [x] samplingに伴う全語彙logits D2H・全語彙CPU候補処理・毎tokenの不要なallocation/H2Dを除去し、
  Graph／chainを維持する。残る制約処理の転送や同期は用途と寄与を説明できる。
- [x] 選んだcoding・tool／JSON・reasoning経路が正常に生成・停止・cancel/recoveryできる。
- [x] 両local targetの影響する代表条件でA/B/Cと実APIの結果を示し、速度への影響がほぼないという
  ユーザー目標への達成判断を根拠とともに記録する。性能と数値correctnessの証拠は分ける。
- [x] 対象host・compile-only検査を通し、GPU PASSはexact target・数値oracle・fallbackなし・cleanup成功を伴う。
  CPU CIでfull-modelやGPU規模の正しさを代替しない。
- [x] main-plan、API仕様、計画・履歴を更新し、完了計画をarchiveへ移す。commit・push後に最終HEADの
  対象CI成功とremote同期を確認し、必要な修正・再検証・再pushまで完了する。

検証入口は`sllm-validation`スキルと既存CI戦略に従い、影響範囲に絞る。実機作業時は互換性文書と
local QwenのGPU占有規則を守る。参照実装の流用は既存provenance方針に従う。
既存の停止・再計画条件を適用し、反復失敗を検査追加で解決しない。

## 2026-09-08の着手記録

- 開始HEADは`54d01a77bef488246264b81c12d245102a7f3f39`。計画文書の未commit変更を保持し、
  基準計測は変更前source／binaryを隔離して採取する。作業用データは`.local-artifacts/phase81/`へ保存する。
- `sllm-validation`と`magpie-kernel-evaluator`を確認した。Magpie CLI・MCPはこの環境では見つからず、
  sLLMを対象とする既存HIP testcase・direct／HTTP計測入口を使う。TraceLens/PyTorchを追加するためだけの
  framework導入は行わない。GPU oracleと通常性能を別々に採取する方針は維持する。
- 固定selector requestの語彙数を明示し、補正なし・制約なしを空配列で表す実装を開始した。
  従来の全語彙配列を毎token作る処理を省き、Qwen／Gemmaの終端へ共通request-owned bufferを接続する。
- Qwen、Gemma Dense／MoE、Ministralの既存終端へ共通bufferとselectorを接続した。
  Qwen selectorのhost検証6件、Gemma MoE 16件、Ministral 5件が成功した。これらはfake backendによる
  接続・readback・buffer再利用の確認であり、実GPUの数値・性能達成を意味しない。
- 既存`sllm-phase78-qwen38-benchmark`へ内部比較用の
  `SLLM_PHASE81_SAMPLING=greedy|host-fixed|gpu-fixed`と`SLLM_PHASE81_REPLAY=1`を追加した。
  replayでは次token入力を生成結果から独立させ、A/B/Cで同じ履歴を使う。設定・seed・replayの有無を
  JSONへ記録し、公開APIにはgreedy／host比較用の設定を追加しない。計測器のhost検証5件が成功した。
  変更前binaryでのV620基準計測は進行中であり、GPU samplerとAPIの達成判定は未完了。
- profile解決でQwen3.5／Ministralを全面拒否する中間実装を検出した。Qwen3.5は公式coding推奨のK20、
  Ministralはlocked metadata未指定を確認した上で追加top-k制限なし（K0）をモデル別に明示採用する。
  根拠と位置付けはmain-planへ記録した。K0の共通GPU top-p対応を含め、既存runtimeの拒否で共通化完了を代替しない。
- core全体のhost unit testは565成功・20既存ignore（`.local-artifacts/phase81/core-host-r1.log`）。
  native並列sortのWG256／tile1024比較漏れをレビューで検出し修正した。実GPU検証r1はROCmとAMD-SMIの
  index不一致によるV62043の基準計測との重複を検出して中断し、PASS扱いにしない。
  r2はV62003のHSA UUID `GPU-76a08c022586fed6`を固定して開始し、AMD-SMIで実際の配置を確認した。
  重複した基準計測の性能値は採用前に取り直す。結果は`.local-artifacts/phase81/correctness/`へ保存する。
- V62003のr2はexit 0で成功した。K20／K64とlegacyのC++数値oracle、16-byte record、fallbackなしを確認した。
  この結果はK0追加前binaryのdraft証拠であり、NumPy fixture接続・fixed flags/status・K0・実モデル/API性能を
  完了扱いにしない。K0／K20／K64のrequest-buffer再利用host検証は追加後に成功した。
  変更前A計測はfastpath opt-inなしで開始されていたため、default参考値として区別する。正式なA/B/Cは
  Graph／chain等を同じ採用設定で有効化し、auditで選択を確認して比較する。

## 2026-09-08の実機比較（途中結果）

- V62043（`GPU-08b2ddcbd6e6b36c`）のQwen3.8 NVFP4／FP16 KVで、17入力・17出力の
  matched replayを1回warmup・3回測定した。同じ高速化設定を使い、greedy／host固定／GPU固定の
  全runでfallbackなし・cleanup成功、greedyとGPU固定ではGraph replay 1,920回・span 128回・KV chain 256回を確認した。
- prefill中央値はgreedy `208.826 ms`、host固定 `227.718 ms`、GPU固定 `209.223 ms`。
  TPOTはそれぞれ`60.726 / 86.089 / 60.650 ms`で、GPU固定のgreedy比はprefill `+0.19%`、TPOT `-0.13%`だった。
  短文では追加負担がほぼない結果だが、長文・実API・他モデルの達成を代替しない。
  証拠は`.local-artifacts/phase81/candidate-gfx1030-r1/short-comparison.json`と同directoryのbinary identity・A/B/C原票。
  集計器はtarget・seed・Graph・出力budgetの不一致を拒否する負例4件も確認した。
- R9700の既存APIではcoding入力44 token・出力64 tokenに揃えたgreedy／host固定の基準を保存した。
  先行の25出力対32出力のwall timeをsampling overhead比較には用いない。GPU固定のAPI比較は未完了。
- K0のGPU top-pは並列BF16 bin処理へ変更し、V62003のV248,320単体で中央値`2.433→1.002 ms`へ短縮した。
  最終native testcaseはK20／64の非整列語彙境界、K0同率境界、複数seed・再生、異常statusをPASSした。
  `.local-artifacts/phase81/correctness/v620-selector-k0-final-identity.json`へbinary SHAと条件を保存した。
  この単体結果からMinistral等のモデル全体の速度達成は推定しない。
- Qwen3.8の9,435入力・128出力でもgreedy／GPU固定の比較が成功した。prefill中央値は
  `30,213.255 / 30,232.248 ms`（`+0.063%`）、TPOTは`64.999 / 64.809 ms`（`-0.293%`）。
  host固定のBも成功し、TPOT `92.213 ms`に対してGPU固定は`29.72%`短縮した。
  完全な比較は同directoryの`long-comparison.json`へ保存した。Gemma Dense NVFP4の17入力・17出力は3条件とも成功し、
  TPOTはgreedy／host固定／GPU固定で`67.436 / 87.563 / 67.858 ms`、GPU固定のgreedy比は
  prefill `+0.440%`、TPOT `+0.625%`だった。各GPU結果は同じ入力履歴のdirect実行であり、API性能と区別する。
- V62003ではQwen3.5のFP16／MXFP8 KVとGemma Dense NVFP4の接続・seed再生・単一token mask・cleanupが成功した。
  R9700のnative selector数値検証も成功した。一方、Ministralの実モデルK0 prefillはrecord status `299`で失敗し、
  修正中である。単体test成功を実モデル接続成功に読み替えない。
- Qwen3.8の公開serverは既存仕様でgfx1201専用のため、gfx1030での起動は実行前に拒否された。
  V620のdirect結果とは区別し、Qwen3.8 APIの比較はR9700で行う。
- R9700 APIのcoding 44入力・64出力は全run成功した。1回warmup・3回測定のwall中央値は
  greedy `3.734 s`、host固定 `5.350 s`、GPU固定 `3.520 s`。GPU固定はhost固定比`34.21%`短縮した。
  自由生成のtoken列は異なるためsampling演算だけの差とは扱わない。
  `.local-artifacts/phase81/candidate-gfx1201-r2/api-coding-comparison.json`へ保存した。
  別のAPI probeではJSON schemaがHTTP 500、Responses toolがHTTP 400となり、制約付き生成は未完了として診断中。
  診断ではJSONの非限定string schemaがgrammar active-state上限65,536を超え、toolはQwen3.8専用profileの
  既存`tool_protocol_v1_available=false`で拒否されていた。前者は有限enumで再確認し、後者は未対応を明示して
  既存のtool対応モデルで接続を検証する。未対応機能を本Phaseの新規bring-upへ拡大しない。
  有限enumの再確認はHTTP 200で`{"answer":"ok"}`、stop併用もHTTP 200・`finish_reason=stop`・停止文字の非含有を確認した。
  `.local-artifacts/phase81/candidate-gfx1201-r2/api-probe-r2-bounded.json`はこの2件だけのPASSであり、
  初回suiteの失敗やtool未実施を上書きしない。候補API終了後は既存R9700 serviceへ復帰させる。
- 統合レビューでQwen画像executorのselector転送漏れとMinistralの明示`logprobs:false`拒否を修正した。
  前者は画像embeddingとmRoPEを保った共通prefill seamを追加し、後者はfocused host testが成功した。
  Gemma MoEの取消時無処理とfull-logits未対応要求の実行後失敗も修正した。
  取消済みownerはpoison化し、公開済みKVを巻き戻さずfresh requestを要求する。frontend全96件と関連core testが成功した。
- Gemma MoE NVFP4のV62043実機接続検証も成功した（62.21秒）。K64 prefill／decode、seed再生、単一token mask、
  HIP audit、fallbackなし、cleanupを確認した。これは取消レビュー修正前binaryでのsampling接続証拠と区別する。
  公開HIP H3 compile-onlyはgfx1030／gfx1201とも全5段階成功、公開ABI 117関数・未知stubなしを確認した。
  gfx1201の初回は`/proc`観測raceで停止し、再試行で成功した。GPU数値検証とは別の証拠として保持する。
- 統合レビュー修正後のhost lib検証はcore 566成功・20 ignore、frontend 96成功・1 ignore、server 127成功・1 ignore。
  `.local-artifacts/phase81/host-libs-review-r2.log`に保存した。未実施GPU testのignoreをGPU成功へ計上しない。
- Ministral K0のstatus 299は、Rust descriptorが`top_k=0`だけでlegacy ABIを選び、未使用の中立maskを
  legacy kernelが読んでいたことが原因だった。workspaceを持つ固定契約とlegacyを区別する修正後、
  V62003の実モデル接続は成功した。別途K0の並列nucleus境界で複数blockがheaderを書き換える競合も検出し、
  不変のtotal massと境界blockの所有判定、明示boundary countへ修正した。非同率248,320語彙、複数seed／counter、
  mask有無の回帰検証はgfx1030／gfx1201で成功した。Ministral K0の実モデルも両targetで成功した。
- Qwen3.5のAPIは既存Phase62 MXFP8 GGUF＋明示FP16 KVでV62043上の起動に成功した。
  Phase20の古いBF16 GGUFはrank 5の物理tensorで拒否されたが、現行converterのPhase46 BF16 bundleは
  rank 4へflattenした有効artifactであり、parser緩和やconverter追加は不要だった。
  API probeのtool／stopは16-token budgetで未完了となったため、制約の成立に十分なbudgetで再確認する。

## 最終候補の検証記録（2026-09-08、公開前）

- Qwen3.5 BF16 GGUF／FP16 KVの公開API経路で15項目が成功した。固定値の省略・明示同値のseed再生、
  非対応設定の実行前拒否、JSON制約、画像付きJSON制約、tool引数、reasoning、stop、生成tokenを受信した後の
  SSE切断と次要求の回復を含む。画像試験の初回は未対応の`image_url.detail`を指定して400となり、
  既存の対応形式へ修正した再試行が成功した。API仕様の緩和は行っていない。
  証拠は`.local-artifacts/phase81/candidate-gfx1030-r3/qwen35-api-probe-r2.json`。
  shutdown auditは完了17要求がHIP・fallbackなし、取消2要求、最終allocation／workspace／quarantineは0。
- 最終候補のQwen3.5／MXFP8 KV接続はgfx1030とgfx1201で各1件成功した。Gemma MoEのgfx1030は
  K64・mask・seed再生・cleanupを含む1件が69.39秒で成功した。
- 最終レビューでGemma MoEのproduction wrapperに残るArgmax-only拒否を検出し、固定K64のseed解決と
  共通selectorのprefill／decode転送を追加した。host focused testとgfx1030 server buildは成功した。
  API wrapperの固定K64生成・JSON制約・取消と回復も実機成功した。別モデル用K値の指定に対し
  HTTP500を返す不整合はモデル解決後・scheduler投入前の検証で修正し、最終APIで400を確認した。
  Completions／Responsesでsampler拡張を受けない既存契約も確認した。最終shutdownは残留0。
  Clippy、Rust/C++整形、source inventoryも成功した。公開HEADのCIは未完了であり、Phase完了とは扱わない。

- r3の最終Qwen3.8 A/B/Cでは、greedyに対するGPU固定のprefill差は短文−1.71%／長文＋0.59%、
  TPOT差は短文−0.13%／長文−0.39%だった。CPU固定比のTPOTは約29〜31%短縮した。
  Gemma Denseではprefill＋0.49%／TPOT＋0.68%、Ministral K0ではprefill＋1.20%／TPOT−4.40%。
  選んだ代表条件では「ほぼ影響しない」を支持する。詳細・ばらつき・制限は履歴と集約evidenceへ記録した。

## 後続への引継ぎ

Phase 82は本Phaseの固定target samplingを前提にstatic FP8 KVとMTPを接続する。Phase 83／84も同じprofile、
共通sampler、API契約を再利用し、別のhost sampling経路を新設しない。batching時のRNG状態と作業領域は
要求ごとに独立させる設計を維持するが、本PhaseでB>1や新規MTPの性能達成を要求しない。

[メイン計画](../../../../main-plan.md) ·
[Phase 76〜84ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)

[実装・検証履歴](../../../../../history/2026/09/1-10/phase81-fixed-gpu-sampling.md)へ結果を記録する。
実装commitの公開CI成功を確認してarchiveへ移した。完了記録のcommitも公開CIを確認し、最終報告へ記載する。

初回公開CIは基本H3・H1・H2が成功した。H0の新規Cargo target台帳未登録と、公開H3の短い`/proc` readを修正した。
Rust依存validator／MSRVとrunner43件が成功した。詳細は上記履歴へ記録し、修正HEADのCI成功を確認して完了処理を行った。

[メイン計画](../../../../main-plan.md) /
[履歴](../../../../../history/2026/09/1-10/phase81-fixed-gpu-sampling.md)

## 公開時の完了確認

実装commit `fe8bb12644da237da8c35e94b656852742371326`で以下がすべて成功した。

- [h3-compile-only (non-required)](https://github.com/jyohukuchan/sLLM/actions/runs/34160168408): PASS
- [h3-public-runtime-compile-only (non-required)](https://github.com/jyohukuchan/sLLM/actions/runs/34160168438): PASS
- [host-required](https://github.com/jyohukuchan/sLLM/actions/runs/34160168406): PASS

完了記録の変更は文書のみで、source／build inputs／toolchain／モデル／実機artifactを変えない。
記録commitの最終公開CIとremote同期は、公開後の完了報告で確認する。

[メイン計画](../../../../main-plan.md) / [履歴](../../../../../history/2026/09/1-10/phase81-fixed-gpu-sampling.md)
