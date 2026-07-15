# SQ8 Rust lineage v2 parser修正

## 前回の要点

one-shot actualは新しいv2 lineage referenceを含むserved-modelをworkerへ渡したが、Rust側がv1の5-key referenceとv1 manifestだけを型付きdecodeしていたため、`invalid_manifest`で終了した。Pythonの中央validatorとgatewayはv2 migrationをすでに検証できていた。

## 今回の変更点

- Rustのlineage referenceとmanifestをschema discriminatorによるstrictなv1/v2 unionへ変更した。v1互換を維持し、v2ではentry count、current implementation audit、predecessor、typed entriesを検証する。
- external/runtime copyと全live receiptについてcanonical absolute path、0444、single-link、live SHAを検証する。
- first-v2のv1 migration prefix、actual failureとcurrent GO、およびsubsequent-v2のappend-only prefixを検証する。unknown、missing、type、count、digest、sequence、duplicate、predecessor drift、second migrationなどのtamperを拒否する。
- 新served-model SHA `31ba7f6483a5baf7d84f8b45a5d86d02c2c22d72d229ca74cfe593192e98ccdd`と旧v1 SHA `a4d541a8c44edd73e505f223b15cf92933b4e0bf2a257e8e9d08dbad94192542`をCPU loaderで受理する回帰試験を追加した。
- dedicated targetと`CARGO_BUILD_JOBS=1`でserved-model試験、worker試験、package checkを通した。full lib suiteも同じ条件で実行した。

## 次の行動

このRust修正を統合した新commitから、既存の認可手順に従ってfresh workerとruntime chainを再構築する。今回の作業ではGPU、service、sudo、actual executionは行わない。
