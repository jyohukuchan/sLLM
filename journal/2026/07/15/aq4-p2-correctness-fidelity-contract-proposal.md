# AQ4 P2 correctness/fidelity contract proposal audit

## 前回の要点

既存P2仕様は、source oracle、同一AQ4 artifactのall-M=1 path oracle、full-vector
comparison、事前bound thresholdを別の責務として定義している。しかし、AQ4の数値
thresholdは未承認で、今回のattempt2はtrace生成とwrapper検証失敗が同一runに残った。

## 今回の変更点

- 現行仕様のBF16 source gateがexact greedy/top-kを要求する箇所を監査した。
- candidateとactive AQ4 pathは、同一artifact all-M=1をexact behavioral oracleとして
  維持する案を整理した。
- BF16 sourceは、exact compatibility canaryと、本番promotion用の独立holdout fidelity
  統計（token agreement、top-k overlap、logit cosine/relative-L2、hidden drift、品質
  タスク）へ分離する案を作成した。
- thresholdはattempt2の3行、VRAM/power、producer summaryから導出せず、事前baseline、
  量子化設計上限、同一identityの承認済みpolicy、またはactive-AQ4-vs-BF16の凍結済み
  envelopeだけを根拠にする順序を定義した。
- 非拘束proposal JSON、レビュー文書、契約テストを追加した。既存binding spec、
  threshold template、attempt2 raw evidenceは変更していない。

## 検証

- `python3 -m unittest -v tests.test_aq4_p2_correctness_fidelity_proposal` を実行する。
- proposalの非拘束・threshold-free、source exact現行契約、active path exact契約、
  evidence/validator/rejection条件、既存auditのblocked状態を検証する。

## 残課題

- 独立BF16 full-vectorと既存active AQ4のdisjoint holdout baselineは未captureである。
- 数値policyは未承認のため、P3 candidateのpromotionは引き続きNo-goである。
- proposal承認後にのみ既存binding specとthreshold schemaを改定する。

## 次の行動

1. proposalを独立レビューし、source exact gateをcompatibility canaryへ限定するか決める。
2. holdout splitとactive-AQ4 baselineをcandidate実行前にhash-bindする。
3. 凍結policyとvalidator testsを追加してからP3候補を実行する。
