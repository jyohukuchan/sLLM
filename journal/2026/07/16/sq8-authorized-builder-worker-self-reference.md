# SQ8 authorized builder worker自己参照修正

## 前回の要点

authorized materializationは独立監査済みunauthorized runtimeのworkerをcreate-new runtimeへコピーしていたが、`build-receipt.json`の`worker.source_path`だけは旧runtimeを保持していた。このためlatest authorized pre-exec監査は、authorized 8メンバー内に旧runtime参照が1件残るとしてNo-Goだった。

## 今回の変更点

- authorized/unauthorizedの両build receiptでworker source/immutable pathを、新runtime自身の`ullm-aq4-worker`へ正規化した。unauthorized→authorizedの認可連鎖はbuild/receipt/Gate/manifestに伝播する外部independent audit receiptのpath/SHAで保持し、旧runtime pathはauthorized runtimeへ複製しない。
- worker copyは`O_NOFOLLOW`の同一fdでhash/copyし、前後fdと最終pathのdev/ino/mode/uid/gid/nlink/size/mtime/ctimeを比較する。sourceはsingle-link 0555/0755、authorized copyはsingle-link 0555として再検証する。
- authorized output pathはaudit receipt SHAだけから導出する専用関数へ分離し、promotion request derivationに依存させない。
- authorized 8メンバーを再帰走査し、監査済み旧runtimeのexact bytes、nested JSON path、`..` alias、`file://` aliasを拒否する。新output自身への正規pathは許可する。
- runnerはbuild receipt workerのexact shape、自己参照canonical path、live SHA/bytes/mode/nlinkをdry-run/executeの双方で検証する。candidate snapshot開始/終了のfull runtime fingerprintも比較し、検証中TOCTOUを拒否する。

## 次の行動

新しいunauthorized runtimeをfresh materializeし、worker自己参照と全historical runtime参照ゼロを確認する。独立監査後に新audit SHA由来のauthorized runtimeをfresh materializeする。既存authorized runtimeはstaleとして再利用・手編集しない。GPU/service actual executionは、新runtimeの独立pre-exec監査がGoになるまで行わない。

検証はbuilder/runner専用100件とpromotion receipt 7件が成功し、py_compile、`uvx ruff check`、`uvx ruff format --check`、`git diff --check`も成功した。generatorを含む追加33件は32件成功し、1件だけisolated worktree内のfixtureが本repo絶対pathを保持する既知のpath差で失敗した。GPU、service、sudoは使用していない。
