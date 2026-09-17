# Phase86: MTP catch-upの条件付けを外部エンジン準拠で検証する

> 状態: 完了（有意な改善を確認できず、既定採用せず。2026-09-17）
> 起点: `e25bcc07` ＋ 未commitの作業ツリー
> 2026-09-17ユーザー指示: 「次のPhaseとして計画を作って」
> 番号: 2026-09-17ユーザー決定によりPhase86とし、旧Phase86（他精度残差）をPhase87、旧Phase87（NVFP4リクエストバッチ処理）をPhase88へ繰り下げた。

## 背景（コードで確認済みの事実）

### 外部エンジンは検証済み位置をtargetのhiddenで再計算する

| エンジン | catch-upの実装 | 条件付け |
| --- | --- | --- |
| vLLM | `vllm/v1/spec_decode/llm_base_proposer.py:502` の最初のforwardが検証済みの全token（`target_token_ids: [num_tokens]`）を処理し、最初の提案と**融合** | `target_hidden_states` |
| SGLang | `eagle_worker_v2.py` の `draft_extend`。コメント「*we get 1 token from draft prefill and (#spec steps - 1) tokens here*」 | target hidden |
| llama.cpp | `common/speculative.cpp:1427` の `process()`。`verify_h` は「*Hidden rows from the most recent target verification batch*」。「*kv is shared with target (e.g Gemma4) なら catch-up decode を省略できる*」とあり、**Qwenのように共有しないモデルでは実行する** | target hidden |

### sLLMのQwen MTP経路はこれをしていない

`crates/sllm-frontend/src/generation.rs` の `QwenMtpGenerationExecutorV1` では、
`propose_mtp_draft`（2118行）と `propose_mtp_draft_with_device_selector`（2066行、これまでの全測定が使ったgpu-fixed経路）がいずれも

- chain 1歩目: `last_target_hidden_bf16`（targetのhidden）で遷移
- chain 2歩目以降: **draft自身が出力したhidden**で遷移

し、検証後は棄却分を `rewind_mtp_rows`／`rewind_last_decode_transition` で巻き戻すだけである。
全受理時だけ `decode_mtp_state_only_batch` で1行を足す。
したがって**2歩目以降のdraftが受理された位置のMTP KVは、targetの真のhiddenではなくdraftの推測したhiddenから作られたまま残る。**

同じ関数の**非MTP provider分岐（外部／n-gram）では、targetのhiddenでcatch-upしている**（`generation.rs:2243` の `hidden_before`）。
必要な部品 `decode_mtp_state_only_batch(tokens, target_hidden_rows)`（`qwen_execution.rs:3154`、1 tokenにつきhidden 1行）は既にある。

Phase84.5は「rewindと状態復元が設計どおり一貫している」ことを確認したが、
**外部エンジンと同じ条件付けかは検査対象ではなかった。** 一貫性と条件付けの妥当性は別問題である。

### これまでのteacher-forced測定（M1）は本番と違う条件を測っていた

`sllm-phase78-qwen38-benchmark.rs` の `stage0_fixed_entries` は、強制行ごとに
`draft.decode_mtp(token, &last_hidden)` を呼び、`last_hidden` を**毎行targetのdecode出力で更新する**。
つまりStage 0は

1. **全位置をtargetのhiddenで条件付け**している（外部エンジン型であって本番型ではない）
2. **chain 1歩目型の提案しか測っていない**（本番の2歩目は「draftのhiddenを入力にした提案」で、Stage 0では一度も測られない）

したがって[ベンチマーク](../../../../../development/mtp-acceptance-benchmark.md)のM1値
（MXFP8 0.9731／0.9824 など）は、**本番がまだ到達していない条件付けでのdraft忠実度**である。
この限界は本Phaseの結果に関わらず記録する（受入基準5）。

## 仮説

本番では受理されたchain位置のMTP KVがdraft条件付けのまま蓄積し、
以降の提案を劣化させて採用率を下げている。targetのhiddenでcatch-upすれば採用率が上がる。

**これは仮説であり、効果が無い・小さい可能性も十分ある。** 否定的結論も本Phaseの完了とする。
効果は既知の性質から、near-tie位置（top1−top2 logit差の小さい位置）に集中すると予想する。

## 非対象

batching（Phase88）、tree drafting（topk>1）、並列drafting（DFlash型）、greedy受理規則の実装、
MXFP kernelの変更、MXFP8のGPU差調査（終了済み）、FORCE_BASELINE（完了済み）、
MTP幅の変更、commit／push、公開CI。tree／並列draftingは本Phaseの結果を踏まえて将来検討する。
外部エンジンのコードは設計の参照にとどめ、取り込まない。取り込む場合は同じ作業中に
[import log](../../../../../../THIRD_PARTY_NOTICES.md#import-log)へ記録する。

## 受入基準（実装前に凍結）

1. Stage 0の差分特定は、結論に関わらず数値付きで記録して完了とする。
   「catch-upに効果は無い」も有効な完了である。
2. catch-up候補はopt-inの環境変数で有効化し、**採否が決まるまで既定経路の出力・選択・性能を変えない。**
3. 有意性は**prompt単位のcluster**で判定する（位置を独立標本として扱わない）。
   MXFP8 GPU差調査で、位置単位のMcNemar z=3.91がprompt cluster検定では p=0.0625 になった教訓による。
4. GPU証拠はexact target・UUID、HIP-only、非zero dispatch、fallbackなし、cleanup 0で判定する。
   CPU emulation、timeout、crash、zero selectionはPASSにしない。
5. **M1がtarget条件付け・1歩目型のみを測っていた限界を、ベンチマーク仕様書へ本Phaseの結論と独立に記録する。**
6. 比較は同一GPU内で行う。GPU間で `target_hidden` が異なることが判明しているため、GPU間の採用率の絶対値を比較しない。
7. 出力token列が変わる変更を既定採用する場合は、[数値出力変更台帳](../../../../../compatibility/numerical-output-changes.md)へ
   最初の分岐位置・対象scope・rollback commitを記録する。
8. **BF16 companion（本番既定）を最初に評価する。** 量子化companionはその後に扱う。

## 段階

### Stage 0: 差分の正確な特定（実装なし）

1. 幅2で受理数0／1／2の各場合について、両提案経路で**どのMTP位置のKVが保持され、それぞれ何のhiddenで作られたか**を表にする。
   `rewind_mtp_rows` と `committed_rows` の計算を読み、実装から確定させる。推測で埋めない。
2. 同じ表をvLLM／SGLang／llama.cppについても作り、差がどの位置に生じるかを明示する。
3. 既存の実測（[mtp-bench](../../../../../../ci/matrix/mtp-bench-v1.json) の基準run、Phase85の採用数）から、
   **本番で保持されるMTP位置のうちdraft条件付けのものが何割か**を見積もる。
   これが小さければ、効果の上限も小さい。
4. 見積りが無視できるほど小さい場合は、Stage 1以降の実施をユーザーと再確認する。

### Stage 1: 測定器の拡張（本番忠実な強制モード）

既存Stage 0 harness（強制prefix、全logits保存、HIP監査）を拡張し、同じ強制経路で2つの条件付けを比べられるようにする。

- **モードT（既存）**: 全位置をtargetのhiddenで条件付け。
- **モードP（新設、本番忠実）**: 強制経路をブロック単位で進め、1歩目はtargetのhidden、2歩目はdraftのhiddenで提案し、
  強制列に対する受理結果に応じて本番と同じ規則でKVを保持／rewindする。
  **2歩目の提案も記録する。**
- 既存の凍結prefix（`.local-artifacts/mtp-bench/claimb/gen2-bf16/prefixes.json`）を再利用し、**新規生成しない**。
- 1歩目・2歩目それぞれについて、モードT／Pの同一位置でのtop-1とmargin（top1−top2）を記録する。
- ホストでの契約検査（ブロック境界、受理数0／1／2、rewind後の長さ）を先に通す。

モードPとモードTの差が、「本番の条件付けがどれだけdraftを劣化させているか」の直接の測定になる。

### Stage 2: catch-up候補の実装（opt-in、既定不変）

2段階で進める。

- **2a: 分離catch-up（正しさと効果の測定用）**
  検証後にそのブロックのdraft遷移をすべてrewindし、確定tokenとtargetのhidden行で
  `decode_mtp_state_only_batch` を1回呼んで進め直す（M＝確定行数、最大で幅＋1）。
  1ブロックあたりMTP forwardが1回増える。
- **2b: 融合catch-up（vLLM型、速度用）** — 2aで採用率改善が確認できた場合のみ。
  catch-up行と次ブロック1歩目の提案を1回のforwardで行い、forward数を増やさない。
  M＞1になるので、Phase85の小M kernel（M=2〜4）の性能特性が効く。

いずれも環境変数で有効化し、既定では従来経路を通す（受入基準2）。
KV・hidden・次計算の一致は、既存のPhase84.5型の状態検査（受理数0／1／2、rewind）で確認する。

### Stage 3: 効果測定

**BF16 companionを先に**、両GPUで行う（受入基準8）。

1. **採用率（主）**: Stage 1のモードPで、2a有効時と無効時を同一強制経路で比較する。
   Tier A 26条件（凍結ベンチマーク）、prompt cluster単位の検定（受入基準3）。
   margin帯ごとの内訳も出し、near-tie集中の予想と照合する。
2. **自由生成での確認（副）**: 既存の8192／128 fixtureで、ブロック数・採用数・per-block時間を記録する。
   decode tok/sは `tokens / (blocks × (draft/block + 非draft/block))` で導出し、単一runの端点時間を主指標にしない。
3. 2aで改善した場合のみ2bを測り、forward数とper-block draft時間の変化を確認する。
4. BF16で効果が確認できた場合のみ、MXFP8／MXFP6 companionでも同じ比較を行う。

### Stage 4: 採否

[ベンチマークの採用判断ルール](../../../../../development/mtp-acceptance-benchmark.md)に従う。
両GPUで導出decode tok/sがBF16基準を上回り、採用率が劣化しない場合に既定採用を提案する。
片側GPUのみで成立した場合は既定を変えず、選択可能な経路として範囲を記録する。
既定を変える場合は受入基準7に従う。

## 実行と記録

- laneはDraft。Stage 2で本番のMTP遷移経路を変えるため、Stage 2完了時に統合reviewを1回行う。
- 再計画条件（AGENTS.md準拠）: 同一work unitが2回却下、review時間が実装時間を超過、
  機能的進捗が1時間以上停止、検証/文書が作業の30%超、見積り1.5倍超、または受入基準の変更。
- GPU運用: exact UUID（V620 `GPU-76a08c022586fed6`、R9700 `GPU-a8e9ddefa2d60f55`）を確認する。
  V620使用時はローカルQwen serviceを停止して利用不可として扱う。
  R9700は `sllm-qwen38-r9700.service`（user unit）を測定時だけ停止し、
  unit／run.sh／binaryのSHA256一致、healthz／readyz 200、performance level復元を確認する。
  既存の `ci/tools/run_mtp_teacher_forced_r9700.py` のlease処理を流用してよい。
- 状態のcapacityは、最長 rendered prompt 8,324 token＋出力を収める値（既存測定では10,240）を使う。
- raw出力・logits・trace・buildは `.local-artifacts/phase86-mtp-catch-up/` へ置き、Gitへ追加しない。
  集約は `ci/matrix/phase86-mtp-catch-up-v1.json`。完了時に本計画を
  `docs/plans/archive/2026/09/11-20/` へ移し、対応する
  `docs/history/2026/09/11-20/phase86-mtp-catch-up-conditioning.md` を作って相互リンクする。

## 参照

- [MTP採用率ベンチマーク](../../../../../development/mtp-acceptance-benchmark.md)
- [Phase84.5 MTP経路の限定診断](../../../../../history/2026/09/11-20/phase84-5-mtp-path-correctness.md)
- [MXFP8 GPU差の調査記録](../../../../../history/2026/09/11-20/mxfp8-gpu-divergence.md)
- [メイン計画](../../../../main-plan.md)

## 実行時の測定契約の具体化（2026-09-17）

- Stage 0で、draft hidden由来の保持位置はTier A 26条件の約29.13〜41.48%と
  見積もられたためStage 1以降へ進む。受理1でも2歩目の入力状態が保持される。
- 指定済みgen2 prefixは8条件である。その8列はそのまま再利用し、Tier Aの不足18条件は
  `mtp-bench-v1`の凍結BF16列とruntimeのchat templateから補う。新規生成は行わず、
  由来・token列・promptをhashで固定する。
- 強制列では本番のp/qが選ぶreplacement／bonusを出力できないため、モードPは
  draft argmaxと次の強制tokenの一致によって受理数を決める診断とする。
  hidden連鎖・保持・rewindは本番規則に従うが、本番p/q受理率そのものとは呼ばない。
  target top-1との一致も別に記録し、自由生成の固定GPU p/qで副確認する。
  Pの比較はprompt単位で行い、Tとの位置対比較は対応位置と比較件数を記録する。
- R9700常駐サービスは開始時点でinactive。測定後も元のinactive状態と
  unit／run.sh／binary hashを保ち、稼働中だったサービスをリースした場合だけ
  再起動・healthz／readyzを検査する。

### 入力準備の実行方法の見直し

初回full campaignは両GPUで入力準備時に停止した。manifestのbare hex digestと
helperの`sha256:`付きdigestをそのまま比較した測定器の不備であり、実データの
変更ではなかった。またtoken digestの正本はカンマ区切り十進文字列で、runtimeの
LE i32 digestとは別契約である。全26件のprompt/file/token/countをhostで照合した。
受入条件と26条件の範囲を維持し、manifest補完をGPU接続前のprepare-onlyへ分離する。
以後は1回だけ補完したprefixを各mode/GPUへ再利用し、同じ準備失敗のために
GPU model loadを繰り返さない。追加の広範検証は行わず、既定の主評価へ戻る。

## 完了判定

Stage 0の設計・保持割合、Stage 1のT/P測定器、Stage 2aのopt-in実装・状態検査、
Stage 3の両GPU Tier A 26条件と自由生成、Stage 4の採否を完了した。
M1の差はgfx1030 +0.5355ポイント／gfx1201 +0.2333ポイントだが、prompt clusterの95%区間は
両方0を跨いだ。M4の相対差も−0.374%／+0.305%で両方0を跨ぎ、既定採用しない。
条件未成立によりStage 2bと量子化companion比較は未実施とし、BF16既定を維持する。
受入番号1〜8の証拠対応は集約JSONに記録した。既定変更を条件とする番号7は適用外。

履歴: [Phase86検証履歴](../../../../../history/2026/09/11-20/phase86-mtp-catch-up-conditioning.md)。
集約: [Phase86結果](../../../../../../ci/matrix/phase86-mtp-catch-up-v1.json)。
