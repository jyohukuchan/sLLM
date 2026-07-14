# Qwen3.5 AQ4 P2 target calibration v0.1

## 前回の要点

`ullm.qwen35_aq4_source_calibration.v1` は、独立BF16 sourceのpost-final-RMSNorm hiddenとraw logitsをf32 little-endian sidecarへ保存する。engine coreは、通常生成のprepared token直後だけ同じhidden/logitsをchunk observerへ公開し、source greedy列をhash-bound teacher forcingとしてcommitできる。

## 今回の変更点

`ullm-aq4-p2-calibration` は1 processで1 caseだけを実行し、`ullm.qwen35_aq4_target_calibration.v1` rootをatomicかつ非上書きで発行する。M=1は`oracle_kind=aq4_target`、M>1は`oracle_kind=aq4_optimized`とし、別々の新規rootを必須とする。前者はsource gateとpath gateのall-M=1 referenceを兼ねる。

入力は既存full-model driverと同じcase v2、identity v2、served-model、package tree、worker、runtime device、preflightへexact bindする。source manifest/cases/rows/SHA256SUMS/sidecarを含む全入力ファイルは、symlinkではないregular fileかつ`st_nlink=1`でなければならない。`O_NOFOLLOW`で開いたfile descriptorへ読み取りを固定し、read前後のfd/pathについてdevice、inode、size、mode、mtime、ctime、nlinkが一致することを検証する。SHA256SUMS検証時の各file identityとexpected digestを保持し、manifest、rows、hidden、logitsの実利用fdを開いた直後に両方を再照合してから、その同じfdをparse/scanへ使う。これによりsum検証後から実利用open前までのrename replacementと同一size rewriteも拒否する。package treeもdirectory列挙時identityとhash fd identityを照合し、aggregate byte countには同じhash fdのsizeを使う。JSONとrowはbyte上限、sidecarはmanifestから導いたexact sizeとchunk scanを使い、shape、offset、row SHA-256、nonfinite count、stable top-10、greedy、input-token hash、case/step順を全rowについて再検証する。hardlink、rename replacement、append、同一size rewriteとmtime復元は拒否する。`ULLM_ENABLE_AQ4_LM_HEAD_DIRECT_TOP1`が有効なら、full logitsが生成されないためmodel load前に拒否する。

source caseのgreedy列は`ullm.qwen35_aq4.calibration_replay.v1` domain、u64 little-endian token count、u64 little-endian token IDsのSHA-256へ正規化する。各stepではAQ4 predicted tokenを観測した後、source tokenだけをcommitする。したがってdivergence後も次stepのKV/historyはsource列へ固定される。

出力は次の2種類のrow indexを持つ。

- `rows.jsonl`: 98d9433のcompare toolと互換なexact vector row。case/step/input hash、hidden/logits offset・bytes・elements・dtype・endianness・SHA-256・nonfinite count、AQ4 greedy、stable top-10、finiteを持つ。
- `execution-rows.jsonl`: vector rowとcase/stepで1対1に対応し、source sequence/row SHA-256、predicted/committed token、divergence、generation epoch、observation完了、publication commit、row lifecycleを持つ。manifestがrecord countとfile SHA-256をbindする。

manifestはrequested/resolved/actual token M、actual request width、case、source、policy、served/package/content/worker/capture binary/device identity、terminal lifecycle/reset、operation auditを記録する。calibration observerのdevice-to-host転送を含むため、`performance.timing_eligible=false`かつ`raw_v2_schema_emitted=false`であり、既存raw-v2 schemaは変更しない。

出力は`.`、`..`、重複separatorを含まないnormalized pathに限定し、leafは非存在でなければならない。既存の非symlink parentをcanonicalizeしてcandidateを作り、source artifact、独立source checkpoint、tokenizerの各canonical rootと同一・内包・被内包にならないことを、成功rootとblocked rootの両方で確認する。

成功rootは全row、両sidecar、execution rows、terminal reset、operation coverageが完全な場合だけ`available`となる。nonfiniteを含む完全captureは`blocked`、OOM・identity不一致・cancel・short/extra row・hash不一致・step swap・partial writeはpartial payloadを破棄してmanifest-onlyのimmutable blocked rootとしてatomic発行する。既存rootは成功・失敗のどちらでも上書きせず、2 processが競合してもLinux `renameat2(RENAME_NOREPLACE)`で一方だけをpublishする。

## 次の行動

GPU/live captureはこの実装作業では行わない。R9700で実行する際は、source gate用M=1 rootを先に固定し、optimized Mごとに別rootを作成してから、既存compare toolでsource gateとpath gateを別々に計算する。
