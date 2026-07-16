# SQ8 operator audit reconciliation hardening

## 前回の要点

認可済みcandidate `44e1b3ea77b3adbe`のone-shot actualは、128-token prefillと2-token generation、SQ8 batch/pair telemetry、zero host staging、worker terminal auditまで完了した。その後、capture validatorが`backend_operation.load.v1`の192件をrequest invocationとして重複計上し、M128 terminal implementationをkind表記差で欠落させたため、`audit_missing`で失敗した。

消費済みfailure evidenceは次のidentityで固定し、今回の実装では変更、削除、再試行を行っていない。

- actual failure receipt: `a7ab2cd590e192fcf05cc544babd6278bd0970ea887c931b4170e5fd6f602223`
- maintenance evidence: `089575934da699bef8428bf24807fa8a12014ff43e16a246fbb64f4f64a9f127`
- actual `SHA256SUMS`: `14fffa383fd42b6ed7f86e3735af47adfe89b154fab8d7b56878fa345e1531e8`
- consumed source commit: `db84e98fb8e96a848d70d65520bd7dddbd8f3f93`
- consumed lineage: exact18、SHA-256 `e1ca0a5c3c89d75104acc8ae03a09d08bbda8082807a5dd438519a361e6a1727`

## 今回の変更点

- terminal `request_execution_audit.operation_audit`をrequest invocation countの唯一のsourceとした。外側のterminal copyとはexact equalityを必須にした。
- SQ8 128+2契約を、positive implementation exact-8、physical invocation `128`、`total_steps=129`、`decode_steps=1`、token-equivalent coverage `8256`として固定した。runtimeの14-slot audit全体もcanonical ID/kind/countで検証する。
- `backend_operation.load.v1`はload/resolution evidenceだけとして扱い、invocation countへ加算しない。24 linear-attention layerと8 self-attention layer、3 phases、合計192 recordsのtopology、Primary resolution、重複、layer partitionを検証する。
- root `operator_resolutions`にはpositive terminal implementation 8件だけを出力する。全192 load recordsは`ullm.qwen35_aq4.sq8_load_resolutions.v1`へcompact recordとcanonical SHAで分離して保存する。
- terminal auditは`ullm.qwen35_aq4.sq8_operator_audit.v1`へsource audit SHA、deterministic digest、式、exact countsを保存する。runnerとreceipt writerも両evidenceをexact schema、hash、topology、countで再検証する。
- CamelCase load kindとsnake_case audit kindをcanonical化する。M128 paged-KV chunkはload-time fused KV-write resolutionへ明示的にbindingする。未知、欠落、重複、bool/float/safe-int overflow、count sum、steps、decode、coverage、再hashしたload重複をfail closedにした。
- production trace operator schemaに合わせ、load count `0`をroot operator listへ混在させず、terminal operatorだけをpositive invocationとして残した。

## 検証

GPU、service、sudo、authorization materializationを使わず、jobs 1相当の逐次pytestだけを実行した。

- capture、runner、receipt writer: `268 passed`
- production execution trace: `7 passed`
- generator、prepare、product promotionを含む拡張集合: `334 passed, 5 subtests passed`。別worktreeの絶対pathをfixtureに保持する既存のreasoning candidate test 1件だけがroot path差で失敗し、今回の変更とは無関係だった。
- terminal operator 8件をproduction trace validatorのoperator schemaへ直接入力し、count合計`128`で通過した。
- `git diff --check`通過。

## 次の行動

この変更はcapture、maintenance wrapper、receipt writerのtrusted component SHAとsource commit/tree/archiveを変更する。消費済みcandidate、request ID、attempt marker、exact18 lineage、independent auditは再利用できない。

次のauthorizationには、今回のactual-failure receiptをsuccessor entryとして追加し、その後にこの修正sourceのfresh current implementation GOを追加する。現在のexact18から2 entriesを追記するため、次はexact20 lineageが必要である。新しいsource identityからworker、unauthorized runtime、independent audit、authorized runtime、固定request/output/markerをすべてcreate-newで構築し、明示的なone-shot認可後にだけactual executionへ進む。

## 最終監査とpre-actual materialization

### 前回の要点

commit `b88ce2165d6aba3310a9e6a06572ba52217b7cda`をread-onlyで監査し、192件のload resolutionがtopology専用であり、terminal `operation_audit`だけがinvocation countのsourceであることを確認した。消費済みactual、failure receipt、markerには接触していない。

### 今回の変更点

- source identityはtree `6d375047bea11afb4d3a6795fe4ce1c986e1ade5`、archive `0627b19e947c5f76fca9a61d0d9cac9a5ba67b13b9b37c89abd0e0772041b447`で固定した。
- gateway locked suiteは`269 passed`、Rust full libraryは`749 passed, 1 isolated HIP test ignored`を`CARGO_BUILD_JOBS=1`、`--test-threads=1`で通した。Rust CurrentV2 loader、timeout、error、EOS、trusted ingressの試験を含む。
- root全件走査は、対象外のactive bundle絶対pathとmodeがisolated worktreeに一致しないため、`853 passed, 5 skipped, 110 failed, 23 errors, 57 subtests passed`時点で診断終了した。対象証明集合の`268 + 7 + 341`とgateway/Rust結果を判定根拠とした。
- fresh workerは専用targetへjobs 1でbuildし、SHA-256 `e1f8fa90dcac5c493645f3de887b86cfc22e8fa18b1fd1f2185ff7cf62954ef9`、5,293,016 bytes、CPU `--help` exit 0を確認した。
- exact18を完全prefixとして、seq18にactual failure `a7ab2cd590e192fcf05cc544babd6278bd0970ea887c931b4170e5fd6f602223`、seq19にcurrent GO `a4485a584a188b6187bff667575208160cc671e9881028e047640a1eba919995`を追記した。exact20 manifestはSHA-256 `82d5696cada2abe9d116f223ddbf9ab72c012a019ef03972e02d0ed8c01ead33`、entries SHA-256 `1c47ea621d7c4c90888986f58a257f421f9cd291896eed96f9df45da2d79db15`で、Python append-only validatorを通した。
- fresh unauthorized runtimeはrequest `sq8-promotion-e57edab532ed2e9833fc83fdbc566bf8880ddfbeaa5a46c5505805c5a0b311aa`、prompt IDs 1..128、completion 2、EOS `[248044,248046]`を固定した。served-model SHA-256は`f94e7ff789605f05e27de20e4f84cf924b0d790eb0ad18b6776328f69f5258ee`である。
- strict CurrentV2 audit receipt候補はSHA-256 `17581136c57bb90edb0ac6ba78f03c45d94118b11791aaf38918b479a0654739`でPython validatorを通した。そのreceiptを埋め込んだfresh authorized pre-actual served-model `c76a86f985d0dccf522ab2bc782a155f7407a3c6463f75fd79b17f0df5d5df33`は、fresh Rust typed loaderとPython production loaderの双方を通した。
- exact actual output `/tmp/ullm-sq8-overlay-gpu-promotion-actual-17581136c57bb90e`を指定したwrapper dry-runは`dry-run-ready`、candidate SHA-256 `7938074e8722601aa8a6f4a29f9eec78ad1199850824e55b64d88a8c2abbc442`だった。outputは実行前後とも存在しない。GPU、service、sudo、actual one-shotは実行していない。

### 次の行動

pre-actual判定はGOである。actualを行う場合は、authorized candidate `/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-17581136c57bb90e`とcreate-new output `/tmp/ullm-sq8-overlay-gpu-promotion-actual-17581136c57bb90e`だけを使い、明示的なone-shot指示後に1回だけ実行する。
