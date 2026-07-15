# AQ4 P2 frozen holdout runner v0.1

## 前回の要点

P2 split は8 strata（prompt_tokens 4種 × baseline_mode 2種）を各6行に固定し、各stratumの3行をcalibration、残る3行をholdoutにした。calibration 24行から一度だけfreeze receiptを作り、holdoutはその後に一度だけ評価する。

## 今回の変更点

`tools/prepare-aq4-p2-fidelity-holdout-cases.py` はholdout 24行だけをRustのsource-cases transport schemaへ変換し、`subset=holdout`、専用observation、split/policy/calibration/holdout SHA、casesのpath/SHA/size/mode/nlinkを持つformal source-holdout receiptも発行する。`tools/run-aq4-p2-fidelity-holdout.py` は `preflight` と `execute` を分離する。preflightはGPUを起動せず、次を検証して `ready_for_execute` のcreate-new planを発行する。

- split/policy/calibration-cases/holdout-cases のSHA、8 strata各3行、全行のstep=0・row_count=1を固定する。
- freeze receiptのpath/SHA、`status=frozen_calibration_envelope`、`holdout_status=not_started`、`holdout_evaluations_remaining=1`だけを受理する。policyを再導出せず、holdout値をcalibrationへ戻さない。
- `actual_verified` receiptは共有のSQ8昇格契約と同じexact top-level schemaで検証する。prepared receipt、maintenance evidence、executor recordを共有検証器で再構成し、profile、served manifest、package、worker、overlay inventoryまでlive identityへ結び付ける。
- source full-vector artifactがholdout 24行だけを持つこと、source casesのSHA/IDをholdout casesへexact bindする。source/activeのmodel ID、upstream revision、tokenizer identityを一致させる。
- build receiptのsource commit/tree、clean worktree、Cargo.lock、build log、capture binaryをpath/SHA/size/mode/nlinkでexact bindする。capture `build_sha256` は実行binary SHAそのものであり、任意文字列は受理しない。package contentは全regular memberを1 MiBずつstreaming hashし、Rustと同じrelative-path/NUL/raw-SHA/newline集約でpreflightとexecute直前の双方で照合する。
- served model、package manifest/content、worker、guard receipt、capture binary、device index/backend/name/architecture/ID、quantized revisionをexact bindする。device IDは必須である。
- Rust captureへ明示`--subset holdout`、`--cases-file`、`--expected-holdout-cases-sha256`を渡す。既定のcalibration CLIとcalibration SHA検証は変更しない。capture manifestはsubset、split/cali/holdout SHA、one_process=1、one_model_load=1、gpu_parallelism=1を記録する。

executeはcommand SHAだけでなく、planの全固定値からcommandを再構成して完全一致を要求する。attempt markerをcreate-newで先に発行した後、split/freeze/source/actual/package/worker/guard/capture/build receiptと全参照をpath/SHA/size/mode/nlink/symlink条件で再検証し、別の`--device-index`は受け取らず固定commandを1回だけ起動する。ROCR/HIP/CUDAの可視GPUとguard環境もplanだけから設定する。

runnerはLinux `/proc` からcapture child PID、process group、session、実行ファイル、command hashを取得し、process group内PID、package root配下のopen fd/memory map、`/dev/kfd`/`/dev/dri/renderD*`を開いたGPU processを実行中に反復censusする。成功にはprocess groupでchild 1個だけ、外部観測されたpackage file、child PIDを含むGPU censusが必要である。Rust manifest側では24行の開始・完了、各行前のclean generation state、generation state観測、同期reset、reset後clean state、direct captureでscheduler未使用・pending 0をexact evidenceとして持つ。self-reportだけでは成功にしない。

Rust artifactは全regular memberを0444/nlink=1、`vectors/`とartifact rootを0555にし、各file、directory、rename後のparent directoryをfsyncする。manifestとSHA256SUMSはcreate-newで発行し、Linux `renameat2(RENAME_NOREPLACE)`以外で公開しない。runnerもこの封印topologyを再検証する。

attempt後のpre-spawn、stdout/stderr open、spawn、process census、timeout、OOM、nonzero、partial artifact、比較、result publicationの失敗は、可能な限りstage/errno/external evidence付きimmutable failureへ封印する。failureは`attempt_consumed=true`、`retry_permitted=false`、`partial_artifact_adopted=false`である。同一attempt markerが残る限り再試行を拒否する。

成功したcaptureは凍結boundsと比較し、metricsのmean/diagnostic maxだけを結果へ記録する。成功またはNo-Goのholdout resultは `holdout_evaluation_count=1`、`holdout_evaluations_remaining=0` としてimmutableに発行する。failureはremaining=1を保つが、attemptは消費済みであり再試行できない。

## 次の行動

1. build/guard/source-holdout/actual receiptを含むCPU preflightを独立監査する。
2. 監査済みplanだけをGPU環境へ移し、`execute`を一度だけ実行する。
3. result receiptを別の昇格判断へ渡す。holdout値をfreeze policyへ書き戻さない。
