# AQ4 P2 frozen holdout runner v0.1

## 前回の要点

P2 split は8 strata（prompt_tokens 4種 × baseline_mode 2種）を各6行に固定し、各stratumの3行をcalibration、残る3行をholdoutにした。calibration 24行から一度だけfreeze receiptを作り、holdoutはその後に一度だけ評価する。

## 今回の変更点

`tools/prepare-aq4-p2-fidelity-holdout-cases.py` はholdout 24行だけをRustのsource-cases schemaへ変換する。`tools/run-aq4-p2-fidelity-holdout.py` は `preflight` と `execute` を分離する。preflightはGPUを起動せず、次を検証して `ready_for_execute` のcreate-new planを発行する。

- split/policy/calibration-cases/holdout-cases のSHA、8 strata各3行、全行のstep=0・row_count=1を固定する。
- freeze receiptのpath/SHA、`status=frozen_calibration_envelope`、`holdout_status=not_started`、`holdout_evaluations_remaining=1`だけを受理する。policyを再導出せず、holdout値をcalibrationへ戻さない。
- `actual_verified` receiptを通常ファイル、nlink=1として読み、絶対pathとSHAをplanへ記録する。
- source full-vector artifactがholdout 24行だけを持つこと、source casesのSHA/IDをholdout casesへexact bindする。source/activeのmodel ID、upstream revision、tokenizer identityを一致させる。
- served model、package、worker、capture binary/build、device architecture/ID、quantized revisionをexact bindする。
- Rust captureへ明示`--subset holdout`、`--cases-file`、`--expected-holdout-cases-sha256`を渡す。既定のcalibration CLIとcalibration SHA検証は変更しない。capture manifestはsubset、split/cali/holdout SHA、one_process=1、one_model_load=1、gpu_parallelism=1を記録する。

executeはplanを再解釈せず、attempt markerをcreate-newで先に発行し、固定commandを1回だけ起動する。ROCR/HIP/CUDAの可視GPUを指定index 1枚に限定する。timeout、OOM（SIGKILL/137）、nonzero、欠落・部分artifact、非有限値、shape/greedy/top-k/relative-L2>1/state違反はimmutable failure receiptにする。同一attempt markerが残る限り再試行を拒否する。

成功したcaptureは凍結boundsと比較し、metricsのmean/diagnostic maxだけを結果へ記録する。成功またはNo-Goのholdout resultは `holdout_evaluation_count=1`、`holdout_evaluations_remaining=0` としてimmutableに発行する。failureはremaining=1を保つが、attemptは消費済みであり再試行できない。

## 次の行動

1. CPU fixtureでpreflightのsplit/receipt/identity拒否とcreate-new topologyを検証する。
2. GPU環境でRust captureを`--subset holdout`として一度だけ実行し、active artifactをvalidateする。
3. result receiptを別の昇格判断へ渡す。holdout値をfreeze policyへ書き戻さない。
