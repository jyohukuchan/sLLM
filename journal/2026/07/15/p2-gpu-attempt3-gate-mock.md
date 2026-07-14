# p2 GPU attempt3 clean-gate mock

## 前回の要点

- attempt2 の production run は trace artifact を生成したが、gate の終端 `sha256sum -c "$OUTPUT/SHA256SUMS"` が output 外の cwd で相対ファイル名を検証したため rc=1 になった。
- attempt2 の raw evidence と layerwise root-cause analysis は別 commit に保存済みで、freeze script は変更していない。

## 今回の変更点

- `run-gpu-gate-attempt3.sh` を attempt2 の機械的コピーから作り、attempt3 固有の output/log/marker/candidate paths と、終端 verifier の `(cd "$OUTPUT" && sha256sum -c SHA256SUMS)` だけを変更した。
- `tests/test_qwen35_aq4_p2_attempt3_gate_template.py` を追加した。test は GPU/service を起動せず、normalized diff、終端行、relative SHA256SUMS の 3/3 success、payload tamper の nonzero、attempt1/2 との path 分離を検査する。

## 次の行動

- `bash -n`、専用 unittest、normalized diff を実行した結果を commit に保存する。
- attempt3 の production/GPU run は独立 review と明示許可があるまで実施しない。

検証結果: `bash -n` pass、専用 unittest 4 tests pass、temporary output verifier は正常時 rc=0（manifest/payload/runtime 3/3 OK）、payload 改ざん時 nonzero、normalized diff pass。GPU/service 操作と attempt3 production run は未実施。
