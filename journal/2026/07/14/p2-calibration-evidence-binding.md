# P2 calibration evidence binding

## 前回の要点

P2 result builderとcomplete-matrix validatorはsampled oracleとraw-v2 evidenceを検証していたが、full-vector calibration comparisonをcase単位で束縛していなかった。

## 今回の変更点

- P2専用`ullm.aq4_p2_calibration_evidence.v1`を追加し、case/case-set、model/source/package/worker/device/prompt/step/policyとcomparison hashをexact bindした。
- 全caseのsource gateと、optimized Mだけのsame-artifact all-M1 path gateを必須化した。all-M1/decodeのpath、source/path swap、case swap、hash/identity drift、comparison reuse、partialを拒否する。
- full comparisonの5指標を事前bound policyへ照合し、null/nonfinite、greedy mismatch、blocked、unknownを拒否する。calibration timingはperformanceへ混ぜず、raw-v2の2+10 scheduleを維持した。
- 両toolsのJSON/hash読取をfd固定、`O_NOFOLLOW`、single-link、ctimeを含む安定性確認へ変更し、全path componentのsymlinkを拒否した。publishはatomic no-replaceにした。
- synthetic正例とswap/hash/identity/threshold/nonfinite/unknown/hardlink/missing/reuse/optimized path負例を追加した。
- 独立QAの指摘に対応し、source gateを`independent_source_full` manifestからsampled-v2親のpath/hash/schema/model/revision/checkpoint/tokenizer identityまで直接再構築するようにした。target manifestのsource path/hashも再hashする。
- optimized path gateへ`path_oracle_case_id`、`path_oracle_result_sha256`、`path_oracle_calibration_manifest_sha256`を必須化した。all-M1 resultのsource gateから`aq4_target` calibration root/path/hashを再構築し、`comparison.reference`へexact bindした。
- 別caseの正当comparison、bound source reference、all-M1 result/result hash/calibration manifest hash/root pathのswapと、最終validator内link改変の負例を追加した。matrix reuse検査だけには依存しない。

## 次の行動

source/target calibration schemaの最終変更は隔離validatorだけで吸収する。GPU/live/model loadはこの作業では実行していない。
