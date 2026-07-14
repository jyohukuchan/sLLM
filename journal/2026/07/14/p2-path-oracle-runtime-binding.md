# P2 path oracle runtime binding

## 前回の要点

作業開始時のworktreeには、path exporterへGPU device metadataとsource replay hashを追加する差分、active served-modelへpackage-only identityを結ぶ差分、bounded comparisonのshape/relative-L2/cosine/top-k指標を追加する差分が未コミットで存在した。これらは別caseへのreplay接続と、誤ってCPUと記録されたGPU path evidenceを修正する意図を持つため、破棄せずgroup Bの入力として引き継いだ。途中でartifact identity、comparison metrics、既存path test、detached attestationが`939e6e9`へcommitされたため、本作業はそのcommit後のbaselineへ残りのexporter/validator差分を統合した。

## 今回の変更点

- artifact identityの検査対象をpackage/artifactのSHA fieldだけへ限定し、binding kindやpathをSHAとして扱う回帰を閉じた。
- production exporterをGPU index 1、visible mapping 1へ固定し、`HIP_VISIBLE_DEVICES`と`ULLM_HIP_VISIBLE_DEVICES`を子processへ実際に設定した。CPUはfixture-onlyに限定した。
- runtimeへserved v2 manifest、product package、worker binary、gfx1201 profile、path binary、source root、cases入力、source replay sequence/contextをexact bindした。
- validatorはpath runtimeとSHA256SUMSを必須化し、runtimeの全fieldとroot file coverage、served/package/worker/device/env/replay linkを元artifactから再構築する。
- manifest、payload、runtime、外部binding fileをsingle-link、`O_NOFOLLOW`、fd/path identity固定で読み、unknown/duplicate、stale SHA、symlink、hardlink、read中identity変化を拒否する負例を追加した。
- 独立QAで見つかったcross-open TOCTOUを閉じるため、manifest/payload/runtime/SHA256SUMSの最初のpinned bytes、identity、digestをvalidation contextへ保持した。source replayとlink comparisonを含むsemantic/hash検証は同snapshotを再利用し、完了時に全path identityを再照合する。
- 決定的test hookにより、checksum後のrename、same-size rewrite+mtime復元、semantic後のruntime差替え+SHA256SUMS更新を発生させ、いずれもsnapshot identity/digest不一致として拒否した。
- 外部binary/package/artifact/workerをJSON bytes上限から分離し、single-link pinned fdのstreaming digestとidentityをvalidation contextへ登録した。product/package directoryは全entryの相対path、種別、size、identity、same-fd file digestを固定し、完了時にtreeとfile identityを再照合する。
- path semantics後のbinary同一size rewrite/rename、worker/package/artifact置換を決定的に発生させる負例と、4 MiBを超えるbinary/workerを受理する正例を追加した。

## 次の行動

このcommitではGPU/live/model loadと既存evidence rootの書換えを行わない。既存のdetached GPU attestationは履歴として保持し、次のsanctioned runがある場合だけ新しいruntime schemaをexporterから直接発行する。
