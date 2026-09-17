# Phase85 follow-up: M=1経路のA16化によるMTP高速化

> 状態: 完了（Stage 0〜3・撤去後検証・区間別priming確認を完了。Stage 1 opt-inを保持し、BF16既定は維持）
> 起点: `e25bcc077e301f3157b9d7e984e7663f9f3a45c2` ＋ 未commitのColumns2/branchless作業ツリー
> 2026-09-14ユーザー指示: 「実験を含めてM=1経路のA16化によるMTP高速化の計画を作成して」

## 背景と仮説

Phase84以降、MXFP8 W8A8／MXFP6 W6A6のMTP companionはBF16 companionのdecodeを一度も上回っていない。
[bottleneck診断](../../../../../history/2026/09/11-20/phase85-mxfp-m1-bottleneck-diagnosis.md)、
[V620 branchless](../../../../../history/2026/09/11-20/phase85-v620-branchless-mtp.md)、
[R9700改善実験](../../../../../history/2026/09/11-20/phase85-r9700-mtp-improvement.md)の実測を
`decode wall = proposal block数 × (ターゲット検証 + draft w回)`、`生成token数 = block数 + 採用数`へ分解すると、
退行の内訳は次のとおりで、**主因はkernel時間ではなく採用率**である。

| 実験 | decode wall | block数 | 退行総量 | うちblock数増加の寄与 |
| --- | ---: | ---: | ---: | ---: |
| V620 BF16 → MXFP8候補 | 4998.1 → 5602.5 ms | 53 → 58 | +604.4 ms | +471.5 ms（約78%） |
| R9700 BF16 → MXFP6候補 | 3642.4 → 4120.5 ms | 50 → 56 | +478.1 ms | +437.1 ms（約91%） |

（BF16 draft wallはPhase84 r4、decodeは各追試の再測定値であり、run跨ぎの合成である点は結果記録でも明示する。
block数の寄与だけで退行の大半を説明できるという結論はrunの混在に依存しない。）

採用率低下の原因仮説は**活性化側の量子化**である。根拠は三つ。

- `low_precision_matmul_provider.hpp:415-436` のWMMA述語は `Gfx1201 && m>=128`（MXFP8）、
  `Gfx1201 && m>=17`（MXFP6）で、gfx1030にはMXFP8/MXFP6のWMMA経路が無い。
  M=1は `TilePolicy::DecodeRowReduction` ＋ `InnerProduct::DecodedBlockScaledFp32` ＝ FP32復号FMAであり、
  **W/Aを揃えて行列コアを使い切るという方針の前提条件がM=1では成立しない**。
- 帯域面でも利得が無い。K=5120の活性化はBF16 10,240 byte、MXFP8 5,280 byte。
  重み91,914,240 byteに対し差は4,960 byte（0.005%）。
- 本体NVFP4は既に `KernelVariant::Nvfp4DecodePackedDequant` → `Nvfp4W4A16Block16` /
  `InnerProduct::E2M1Bf16Fp32`（`matmul_runtime.inc:532-537`）で、
  **prefill/バッチはW4A4、M=1 decodeはW4A16**という分岐を実装済みである。
  MTP companionだけがM=1でA16の対応物を持たない。

したがって本作業は「MTPをA16にする」のではなく **「M=1経路だけA16にする」** とする。
prefill・高バッチのW8A8／W6A6とWMMA選択は一切変更しない。

## 非対象

本体NVFP4、KV encoding、sampling、MTP幅制御、batching（Phase87）、MXFP4 W4A8、
MTP側embedding／共有FP8 head、既存sidecarのbyte列とmanifest名、常駐serviceへの新binary適用、
commit／push、公開CI。BF16既定は各段階の実測がBF16を上回るまで維持する。
新しい速度下限や必達倍率は設定しない。外部コードの新規取込みは予定しない。

## 受入基準（実装前に凍結）

1. Stage 0は、いずれの結論でも数値付きの判断記録を残して完了とする。
   「効かなかった」も完了であり、Stage 1着手は成果条件ではない。
2. Stage 1を実施する場合、A16選択は `m == 1` に限定する。
   host testで `m >= 17`／`m >= 128` のWMMA述語と、`m > 1` の全selector出力が
   現行と同一であることを、境界の両側（m=1,2,16,17,127,128,129）で確認する。
3. A16はW8A8と数値が異なるため、**同形式の前後digest一致は要求しない**。
   代わりに両GPU・実6形状・境界で独立FP32 oracleのPASSと、
   Stage 0のCPU側往復可逆性検査（下記）を根拠とする。
4. 実MTP測定はexact target／UUID、HIP-only、非zero dispatch、fallbackなし、cleanup 0を満たす。
   CPU emulation、timeout、crash、zero selectionはPASSにしない。
5. 既定変更は、両GPUでdecode中央値がBF16を上回った場合にのみ提案する。上回らない場合は
   選択可能な経路として残し、BF16既定を維持して理由を記録する。
6. 検査は32の冪や整列値だけにせず、K 2016/2048/2080/17376、N 1023/1024/1025 の両側を含める。

これらは本計画の分岐条件であり、新しい承認gateや独立review要件を追加するものではない。

2026-09-14実行開始時に受入基準と分岐条件を凍結した。Stage 0の基準は作業ツリーのColumns2であり、
既存branchless候補は別snapshotの診断結果として扱う。列挙した比較対象は計5系列（「4系列」の誤記を訂正）。

## Stage 0: fake-quant ablation（kernel変更ゼロ・判断用）

**目的**: 採用率低下が活性化側の誤差に由来するかを、HIP変更なしで切り分ける。

MXFP8 E4M3は有効4bit、MXFP6 E3M2は有効3bit、E8M0 scaleは2の冪なので、
`dequant(quant(w))` はBF16（仮数8bit）で**厳密に表現できる**。よって
「重みだけMX量子化→BF16へ復号して書き戻したsidecar」は、既存BF16 kernelで走らせたとき
**W8A16／W6A16の数値と（加算順を除いて）等価**になる。これを利用する。

- 変換器に `--encoding bf16-roundtrip-mxfp8` / `bf16-roundtrip-mxfp6` を追加する。
  `MtpWeightEncoding::Bf16` payloadを出力し、recipe digestは既存2形式と区別する。
  変更は `crates/sllm-core/src/mtp_quantized_sidecar.rs` と
  `crates/sllm-cli/src/bin/sllm-convert-qwen38-mtp.rs` のRustを変更する。
  実装確認により `qwen_graph.rs` がBF16 sidecarを明示拒否していたため、同ファイルの
  BF16既存layoutを維持する受理分岐と `lib.rs` の変換API公開も最小限追加する。HIPは触らない。
- **往復可逆性検査（CPU、必須）**: 8行列すべてで `bf16(dequant(quant(w))) == dequant(quant(w))`
  をbit単位で確認する。overflow/underflow・非finiteが1要素でもあれば等価性の主張を取り下げ、
  その要素数と位置を記録したうえでStage 0の解釈を限定する。
- 比較する5系列: `bf16`（対照）／`bf16-roundtrip-mxfp8`（＝W8A16相当）／
  `bf16-roundtrip-mxfp6`（＝W6A16相当）／既存 `mxfp8`・`mxfp6`（＝W8A8・W6A6）。

**主指標（交絡なし）**: BF16 runの出力token列を固定prefixとして各系列のdraftへ与え、
BF16 draftのtop-1との一致率と、logitの差（相対L2とtop-5順位の変化）を測る。
生成履歴が分岐しないため、これまでの採用率比較にあった交絡が無い。
既存の12言語/タスク条件×3 seedを再利用する。旧API記録は本文のみでtoken ID列を
保存していなかったため、同じprompt/seedで現行BF16 runを取得し、その出力token ID列を
固定prefixとして保存する。旧本文の再tokenizeを元の生成token列とは扱わない。

**副指標（交絡あり、確認用）**: 代表2条件で通常のMTP loopを回し、
proposal block数・採用/提案数を取る。代表条件は既存suiteの `coding-en` と `creative-ja`、
seed123、128出力を各系列・GPUで1回ずつとする（counts確認、時間は参考値）。
履歴分岐による交絡を明示したうえで主指標と併記する。

**続行判断**（実装前に凍結、承認gateではない）:

| top-1一致率でW8A8→BF16の差を埋めた割合 | 判断 |
| --- | --- |
| 60%以上 | 活性化が主因。Stage 1へ進む |
| 30〜60% | 両方が寄与。Stage 1へ進むが、Stage 2でblock数の改善を確認するまでStage 3へ進まない |
| 30%未満 | 重みformatが主因。Stage 1を実施せず、scale精度・per-channel・companion非量子化の再検討へ切り替えて記録する |

GPUは両機で実施する。Stage 0はkernel変更が無いため、失敗しても撤去対象は変換器の追加encodingだけである。

### Stage 0実行時の修正

logit保存がFP32値ごとの4-byte `write` となり、3条件で約9,500万syscallを発生させていた。
`/proc/<pid>/io` とソースで原因を確認し、行単位の一括writeへ変更した。
生成済みの両GPU・36組のBF16 prefixは固定したまま再利用し、途中のlogit収集だけを
`primary-r2` で再実行する。旧部分出力は保持し、完了した行列ファイルのbyte一致も確認する。
これは測定I/Oの修正であり、GPU演算・sampling・受入基準は変更しない。

### GPU別の続行判断

MXFP8主指標の全36組・raw logits照合では、V620はA8 4501/4608 → 重みのみMX8
4536/4608で差の32.71%を回復し、30〜60%の続行範囲に入った。R9700は
4507/4608 → 4531/4608で23.76%となった。両GPU合算は59/208＝28.37%だが、
本計画には集約方法の指定がなかったため、当初はGPU別の判定で続行した。
2026-09-14のユーザー回答により、GPU別に判定し、V620の条件成立を根拠に
Stage 1へ進んで共通実装を両GPUで検証する方針を明示的なユーザー決定とした。
R9700は活性化の寄与が小さい結果として保持し、共通実装の比較検証に含める。
数値基準は変更せず、Stage 0のW6A6集計と副指標の確認も完了した。
BF16既定は維持し、Stage 3はStage 2のblock数条件を満たすまで実施しない。

## Stage 1: W8A16／W6A16の実装（Stage 0が続行判断のとき）

- `MatmulFormat::Mxfp8E4M3W8A16`、`Mxfp6E3M2W6A16` と
  `InnerProduct::E4M3Bf16Fp32`、`E3M2Bf16Fp32` を追加する。
  既存 `Nvfp4W4A16` が `provider.hpp:468` で活性化のblock-scaled layout検査を
  バイパスしている分岐をそのまま拡張する。
- kernel bodyは現行Columns2を複製し、活性化側の
  `BlockCodec<...>::load(activation_view, 0U, inner)` を
  `bf16_to_float(activation[inner])` へ置換するだけにする。
  laneのK割当、FP32積和、reduction順、重み側の復号とscale経路は変更しない。
- **sidecarのbyte列とmanifest名は変更しない。** 重みはMXFP8/MXFP6のまま共通で、
  runtime内のformat/provider選択だけを増やす。既存sidecarがそのまま使え、rollbackは選択を戻すだけになる。
- **統合上の未解決点**: 活性化のlayout（RowMajor BF16 か block-scaled か）は
  provider requestを組む前に決まるため、M依存の分岐はRustのgraph生成層で
  「MX quantizer＋W8A8 matmul」か「W8A16 matmulのみ」かを選ぶ必要がある。
  graph層でM依存の切替が高くつく場合は、**MTP draft opに限定して固定でA16を選ぶ**を代替とする
  （draft opは常にM=1のため挙動は同じで、影響範囲がMTPに閉じる）。
  どちらを採ったかを実装時に記録する。
  実コードの確認では、v6/v7へ既にBF16 activationが渡され、量子化はnative execute内の
  queue scratchで行われていた。したがってRust graph/public ABIは変更せず、
  `SLLM_MX_WA_M1_A16=1` をprepare時に読み、実M=1だけquantizerとworkspaceを省く。
  未指定・M>1・force-baseline指定時は既存経路。prepared planは後のenv変更に影響されない。
- 量子化dispatchが1つ減るため、operator eventの内訳（quantizer / matmul）を
  A8側と分けて記録し、A16の短縮を「kernelが速くなった」と読み替えない。

## Stage 2: 実MTPと非対象領域の非退行測定

- 実MTP: 8192入力／128出力、chunk2048、state8320、MTP幅2、MXFP8 E4 KV、
  固定sampling seed123、1 warmup＋3 measured、stock auto clock、両GPU。
  BF16／W8A8／W8A16／W6A6／W6A16の5系列。
- 報告は **`block数 × (draft/block, 非draft/block)` への分解を必須**とする。
  decode tok/sだけでなく、proposal block数・採用/提案数・draft wallを併記し、
  改善がどちらの要因から来たかを区別する。
- **非退行確認（A8領域を壊していないこと）**:
  - prefix priming（M=chunk2048）が現行A8の値から退行しないこと。
    参考値: V620 BF16 1582.11 / MXFP8 454.27 ms、R9700 118.57 / 166.63 ms。
  - gfx1201でM=128（MXFP8）とM=17（MXFP6）の演算子比較を行い、
    WMMA providerが選ばれ続けることをprofileのkernel IDで確認する。
  - 受入基準2のhost test。
- 演算子は実6形状（K/N = 10240/5120、17408/5120、5120/17408、5120/12288、5120/1024、6144/5120）
  ×2形式×A8/A16、3 warmup＋32 measured。境界は受入基準6の値を使う。
  形状manifestは候補実装前に `ci/matrix/phase85-a16-m1-v1.json` へ凍結する。

### priming上限の明示

既存benchmarkの本体chunkは2048だが、MTP graphのpriming上限は1024に固定されていた。
通常設定の5系列比較を保持し、計画のM=2048条件を満たすため、benchmarkだけに
`SLLM_PHASE85_MTP_PRIMING_CHUNK_CAPACITY`（既定1024）を追加した。
`stage2-priming2048`で同じ8192/128・1 warmup＋3 measured・5系列を両GPUで測定し、
reportへ実際にgraphへ渡したpriming上限を明記する。通常runtimeの既定値は変えない。

### 総priming非退行の区間別確認（完了）

ABBA追試の+0.584%を成功扱いにせず、先頭M=1と後続batchを分離した診断を実行した。
`SLLM_PHASE85_MTP_PRIMING_TIMING=1` の場合だけ各成功呼出しのrow_countとwall_nsを保存する。
8192入力では実区間を1／2048／2048／2048／2047行として検査する。
同じR9700 MXFP6・ABBA4job・各1 warmup＋3 measuredで比較し、従来の総時間も保持する。
診断の追加はRustの時間記録とbenchmark reportだけで、kernel・graph・samplingは変更しない。

区間別ABBAの全6 measured/armでは、総priming中央値がA6 152.711 ms、A16 152.590 ms
（-0.0793%）、先頭M=1が1.785→1.722 ms（-3.51%）だった。
後続batch合計は150.840→150.842 ms（+0.0010%）で測定上ほぼ同等。
以前の+0.584%は再現しなかった。全区間の実row数、生成token hash一致、HIP-only、解放、
サービス／clock復元を確認し、非退行確認を完了する。小差を有意な高速化とは主張しない。
以前の正の差を消さず、すべてのcampaignを結果に保持する。

## Stage 3: packed load kernel（条件付き）

Stage 2の通常priming1024ではA16の4組すべてがBF16よりblock数を増やした。
計画指定のpriming2048ではR9700のW6A16のみ54 block（同条件BF16は55）となり、
このtarget／formatでStage 3を実施した。decode中央値はBF16比+1.88%、W6A6比+4.95%。
prefix primingの中央値はW6A6比+0.62%の時間増加を観測したため、ばらつきと
同一binaryのABBA追試を完了した。A6中央値151.175 ms、A16 152.058 ms（+0.584%）で、
この初期追試だけでは時間非増加を実証できなかった。上記の区間別ABBAで確認を完了した。
総primingには先頭M=1処理が含まれ、M2048だけではない。
M>1のselector・出力・WMMA symbol維持は別途確認した。
Stage 3の3候補は全18形状で既存A16より遅く、不採用とした。
V620およびMXFP8へのStage 3採用は行わない。

以下は実装前の実験案と期待値であり、追加3候補の採用結果ではない。
Stage 2でblock数がBF16と同等まで戻った場合にのみ着手する。A16単独では
「BF16とほぼ同点」までしか見込めず（53 block想定で約5021 ms vs BF16 4998 ms）、
プラスを作るのはこちらである。

- 1要素1 byteのglobal loadを `uint4` 等のpacked loadへ置換する。
- scaleを要素ごとに再ロードしている `BlockCodec::load()`（`low_precision_block_codec.hpp:878`）を、
  ブロック単位でレジスタへ載せる経路へ寄せる。現状 `make_wave_block32` を使っているのは
  gfx1201のMXFP8のみで、gfx1030のMXFP8とMXFP6全targetはスカラー経路のままである。
- ブロック内をpair/整数dotで畳んでからE8M0 scaleを1回だけ適用する
  （2の冪なので指数加算で足りるかを含めて比較する）。
- 目標は実効帯域である。現状の到達度は gate `[17408,5120]` で
  BF16 486 GB/s（ピーク比95%）に対し MXFP8 252 GB/s（49%）／MXFP6 184 GB/s（36%）。
  ここを埋めた場合のdecode改善見込みは +2.5〜3%程度であり、これを超える期待値を置かない。
- llama.cppのmmvq経路の直接reuseを先に検討する（`docs/provenance/README.md`）。
  実際に複製・移植した場合は同じ作業中に
  [import log](../../../../../../THIRD_PARTY_NOTICES.md#import-log) へ追記する。
  `.local-artifacts/` 下のscratch copyやprobeも対象とする。

## 最終判断

凍結した受入基準1〜6の数値・経路・既定方針を確認した。Stage 2の初期ABBAでは総priming時間に
+0.584%を観測した。先頭M=1を分離した追加ABBAでは総時間-0.0793%、後続batch合計
+0.0010%となり、非退行確認を完了した。両campaignと測定限界を保持する。両GPU共通のBF16超えも未達で、既定変更は行わない。
Stage 3は全18形状で遅かったため不採用とし、追加の未採用候補検証は終了した。
撤去後のfresh両target build、host検査、両GPU72ケースのStage1出力・ID一致を確認した。
サービスのunit／binary／run.shとhealth／ready、両GPUのauto復元も確認した。

## 実行と記録

- laneはDraft。dirty treeを許容し、関係するテストに絞って実行する。
  Stage 1完了時に統合reviewを1回、findingの再確認のみ行う。checkpointごとの再reviewは行わない。
- 進め方の並列性: Stage 0のRust変換器追加とStage 2の測定harness整備は独立なので同時に進める。
  Stage 1のHIP実装はStage 0の判断待ちであり、先行実装しない。
- 再計画の条件（AGENTS.md準拠）: 同一work unitが2回却下、review時間が実装時間を超過、
  機能的進捗が1時間以上停止、検証/文書が作業の30%超、見積り1.5倍超、
  またはgate・受入基準の変更。いずれかで新規の検証を止めて同じwork unitを再計画する。
- GPU運用: 使用前にexact UUID（V620 `GPU-76a08c022586fed6`、R9700 `GPU-a8e9ddefa2d60f55`）を確認する。
  V620はローカルQwen subagentがx2で占有するため、V620を使う間はQwen serviceを停止して
  利用不可として扱う。R9700の既存serviceは測定時だけ停止し、終了時に元のunit／binary／run.sh hash一致と
  healthz／readyz 200へ復帰する。両GPUのperformance levelは元の`auto`へ戻す。
- raw測定・build・profile・生成sidecarは `.local-artifacts/phase85-a16-mtp/` に置き、Gitへ追加しない。
  `prep/` に正確なcommandと環境、`build-gfx1030|gfx1201/identity.json` にbuild入力とbinary hash、
  `stage0/`・`operators/`・`mtp/`・`profiles/`・`analysis/` を分ける。
  Gitへ入れるのは集約値・hash・文書のみとし、model／binary／raw traceは追跡しない。
- 集約結果は `ci/matrix/phase85-a16-mtp-results-v1.json`、形状manifestは
  `ci/matrix/phase85-a16-m1-v1.json`。完了時に本計画を
  `docs/plans/archive/2026/09/11-20/` へ移し、対応する
  `docs/history/2026/09/11-20/phase85-m1-a16-mtp.md` を作って相互リンクする。

[メイン計画](../../../../main-plan.md) /
[実測履歴](../../../../../history/2026/09/11-20/phase85-m1-a16-mtp.md) /
[bottleneck診断](../../../../../history/2026/09/11-20/phase85-mxfp-m1-bottleneck-diagnosis.md) /
[M=1高速化実験](../../../../archive/2026/09/11-20/phase85-mxfp-m1-mtp-followup.md) /
[MTP companion量子化の利用手順](../../../../../development/mtp-companion-quantization.md)
