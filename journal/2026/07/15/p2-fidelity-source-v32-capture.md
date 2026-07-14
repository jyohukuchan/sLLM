# P2 source fidelity v32 capture

## 前回の要点

v32の固定入力とCPU専用preflightを完了し、親エージェントのGoを待っていた。旧source-oracle-v2のfixture行はproduction splitの`attempt2_exclusions`に属するため、legacy確認はdisjoint-by-policyとして扱う契約を固定していた。

## 今回の変更点

固定済みwrapperを編集せず、一度だけCPU専用source captureを実行した。

- 実行: `--threads 32`、OMP/MKL 32、`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`空、timeout 2h。
- 所要時間: 42:47.64（`time-v.txt`）。User 69,944.73 s、System 1,189.81 s。
- 最大RSS: 16,870,968 KB。swapはpreflightおよび実行中に使用なし。
- 成果物は`source-full`に公開され、24/24行、unique case ID 24件。`runtime.model_loads=1`、`runtime.run.nonfinite_rows=0`。
- legacy status: `not_applicable_disjoint_by_policy`、legacy overlap 0、hidden/logit差分 0。
- sidecarはhidden 393,216 bytes、logits 23,838,720 bytes。ストリーミング検査でf32値の非有限値は0件。
- `validate-qwen35-aq4-p2-full-calibration.py`は`status=valid`、`row_count=24`、`nonfinite_rows=0`を返した。

成果物SHA256:

- manifest: `78a6de7d2cae4c2ff31952cfe345fefbce55dfd67db7a4904ba10f4e5f7438bc`
- rows: `ac2919e3ffa4d7c790ea942bd0a56c3b9e3a00b9f666b726648bb644b77b1124`
- hidden: `1eb708665fb3a2c4b3c5e020a164240ecdff03261e2b16286117f5cd1883ed7d`
- logits: `0309e08dbdef916d59ddc409bbdabc9e35c7bfd0469e97cd413f77a42b9ac268`

raw evidenceの`source-full/SHA256SUMS`検証は全4件OK。attempt側のpreflight、stdout/stderr、time、vmstatも保持した。CPU専用で実施し、サービス操作とGPU操作は行っていない。

## 次の行動

親エージェントは限定stage済みraw evidenceと本journalをcommitし、source artifactのpromotion可否をactive gateへ報告する。再実行は行わない。
