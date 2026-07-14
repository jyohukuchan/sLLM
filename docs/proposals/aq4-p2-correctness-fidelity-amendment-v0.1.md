# AQ4 P2 correctness/fidelity 契約改定案 v0.1

## 前回の要点

既存のP2契約は、独立BF16 source oracleに対しても `greedy` の完全一致と
policyで要求する `top-k` 一致を要求している。一方、同一AQ4 artifactの
`same_artifact_all_m1` path oracleは、候補と同じ量子化実装を使うための実装・挙動
oracleであり、production candidateの置換を許す比較ではない。数値5指標のpolicyは
まだ未承認で、現在のthreshold auditも `blocked` である。

今回のattempt2は、3行のbounded intermediate traceを生成できたが、wrapperの
`sha256sum -c` が出力ディレクトリを作業ディレクトリにせずrc1になった。内容検証、
SHA-256、identity、service復旧は独立に確認できるものの、この観測を閾値決定へ使っては
ならない。

## 今回の変更点

この文書と同梱JSONは、既存binding specをまだ変更しない非拘束の改定案である。改定の
中心は、BF16 source比較とactive AQ4 path比較の責務を分離することである。

### 1. 現行契約の監査

- `docs/specs/prefill-validation-v0.1.md` §8.3 はsource oracleにfinal hidden/logitsと
  greedy tokenを要求し、§9 はexact greedy列とpolicy-required top-k agreementを
  correctness gateにしている。
- `docs/specs/qwen35-aq4-p2-oracle-v0.1.md` はsource oracleの全語彙tie規則に基づく
  exact greedy/top-kを記録し、`same_artifact_all_m1` をsource oracleの代替にしない。
- `docs/specs/qwen35-aq4-p2-full-source-calibration-v0.1.md` はsource gateとpath gateを
  分離し、hidden/logitsのrelative-L2、max-abs、top-k overlapを観測値として計算するが、
  threshold policyを自動生成しない。
- `docs/specs/aq4-p2-calibration-evidence-binding-v0.1.md` はgreedy mismatch rowsを0、
  5指標を事前bound policyと比較する。
- `benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json` は
  `greedy_token_exact_required=true` とsource/path oracle必須を宣言するが、5つの数値は
  nullのplanning templateである。`correctness-threshold-audit.json` はこの状態を
  正しくblockedと報告している。P0/P1の現行成果物にもAQ4 correctness数値policyはない。

### 2. 改定案: BF16 sourceは独立holdoutのfidelity契約へ分離

現行のexact greedy/top-kは、互換性canary（toolchain/identity検査、promotion不可）では
維持する。本番candidateのsource gateについては、独立したBF16 calibration holdout
（independent holdout）を
先に固定し、次の統計を事前bound policyで判定する。

- token agreement率
- top-k overlap
- logits cosine similarity と relative-L2
- hidden cosine similarity、relative-L2、max-abs drift
- 固定品質タスクのスコアとnon-regression

holdoutは量子化探索・候補選択に使ったtuning casesと重複させない。source、既存active
AQ4、candidateの全identityとcase splitをhashで固定し、集計方法（row平均、worst-row、
quantile、margin）もcandidate実行前にpolicyへ書く。proposal自体は閾値数値を持たず、
今回3行やattempt2の観測分布から数値を推測しない。

### 3. 改定案: active AQ4 pathはexact behavioral oracleとして維持

candidateと同じartifact/packageの `same_artifact_all_m1` を、candidateの挙動oracleと
して必須化する。candidateはprefill implementation、resolved chunk plan、declared
workspaceだけを変更でき、次の値は同一case・同一stepで完全一致しなければならない。

- context token hash、greedy token、ordered top-k IDs
- KV/cache length、absolute position、block table、scheduler ownership
- generated-token counter、terminal outcome、reset/lifecycle

このpath比較の数値vector driftは診断値として保存するが、attempt2の観測値から閾値を
生成しない。P3候補はall-M=1のexact behavioral mismatch、unexpected fallback、OOM、
状態commit/reset不一致を即時に拒否する。

### 4. 閾値を事後調整しない改定方法

閾値は次の順序で固定する。

1. source checkpoint、tokenizer、cases、tuning/evaluation holdout splitをhash-bindする。
2. 独立BF16 sourceと既存active AQ4のbaselineをcandidate実行前にcaptureする。
3. metric定義、集計、quantile、safety margin、品質タスク、非回帰規則をレビューする。
4. policyをself-hashしてからcandidateとactive-pathを実行する。
5. candidate結果は凍結済みpolicyに対してだけ判定する。

許可できる根拠は、同一identityで承認済みのP0/P1 policy、独立holdoutの事前baseline、
量子化設計の事前誤差上限、またはpredeclared active-AQ4-vs-BF16 envelopeである。異なる
identityの履歴値、candidateが選んだ行、attempt2の3行/VRAM/power、producer summary、
mismatch後の閾値緩和は根拠にできない。

## 最小P2 amendmentの証拠・validator・拒否条件

必要な証拠は、disjoint holdout、独立BF16 full-vector calibration、既存active AQ4
all-M=1、candidate all-M=1、optimized candidate、source/path comparison、凍結policy、
固定品質タスクである。validatorはpolicyの未束縛・後変更・holdout重複、active pathの
token/top-k/context/state mismatch、identity/case swap、missing/nonfinite/partial
vector、attempt2-only threshold justificationをfail-closedで拒否する。

## P3候補のGo/No-go

P3は、active pathでexact behavioral no-regressionを満たし、source holdoutで凍結済みの
token agreement/top-k/cosine/relative-L2/hidden drift/品質タスク基準を満たす場合だけGoと
する。性能は既存の2 warmup + 10 measured scheduleと同一identity・power conditionで
比較する。改定案と独立validatorがレビュー・bindされるまでpromotionはNo-goである。

## 次の行動

1. 本JSONをレビュー用proposalとして承認または差し戻す。
2. source-vs-existing-active-AQ4の事前baselineとdisjoint holdoutをcaptureする。
3. 数値policyをcandidate実行前にbindし、validator testを追加してからP3を実行する。
4. 承認後にのみ既存binding specとthreshold schemaを改定する。
