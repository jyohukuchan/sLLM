# SQ8 Rust lineage v2 parser修正

## 前回の要点

Rustのv2 lineage parserは新旧served-modelとportable migrationを処理できたが、独立監査でv1 predecessorの移行時に捨てる意味項目を十分に検証していないことと、対象ファイルのrustfmt未適用がblockingとして見つかった。

## 今回の変更点

- Rustのlineage referenceとmanifestをschema discriminatorによるstrictなv1/v2 unionへ変更した。v1互換を維持し、v2ではentry count、current implementation audit、predecessor、typed entriesを検証する。
- external/runtime copyと全live receiptについてcanonical absolute path、0444、single-link、live SHAを検証する。
- first-v2のv1 migration prefix、actual failureとcurrent GO、およびsubsequent-v2のappend-only prefixを検証する。unknown、missing、type、count、digest、sequence、duplicate、predecessor drift、second migrationなどのtamperを拒否する。
- v1のseq0、seq1-2、seq3-4、seq5について、schema、固定verdict/status、actual、reason code(s)、authorization eligibility、live receiptとの一致を移行前に検証する。entryとlive receiptを同じ不正値へ変更する6系列の負試験を追加し、移行後に捨てる項目もfail-closedにした。
- 新served-model SHA `31ba7f6483a5baf7d84f8b45a5d86d02c2c22d72d229ca74cfe593192e98ccdd`と旧v1 SHA `a4d541a8c44edd73e505f223b15cf92933b4e0bf2a257e8e9d08dbad94192542`をCPU loaderで受理する回帰試験を追加した。
- 対象ファイルへrustfmtを適用した。`CARGO_BUILD_JOBS=1`でserved-model試験11件、worker試験18件、package checkを通した。full lib suiteはテストスレッドも1に固定し、744件pass、0件fail、isolated HIP試験1件ignoreで完走した。

## 次の行動

このRust修正のfresh independent auditを実施する。GOの場合だけ、既存の認可手順に従ってfresh workerとruntime chainを再構築する。今回の作業ではGPU、service、sudo、actual executionは行わない。
