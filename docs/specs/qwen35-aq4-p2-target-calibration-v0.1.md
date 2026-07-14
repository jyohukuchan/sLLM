# Qwen3.5 AQ4 P2 target calibration v0.1

## 前回の要点

`ullm.qwen35_aq4_source_calibration.v1` は、独立BF16 sourceのpost-final-RMSNorm hiddenとraw logitsをf32 little-endian sidecarへ保存する。engine coreは、通常生成のprepared token直後だけ同じhidden/logitsをchunk observerへ公開し、source greedy列をhash-bound teacher forcingとしてcommitできる。

## 今回の変更点

`ullm-aq4-p2-calibration` は1 processで1 caseだけを実行し、`ullm.qwen35_aq4_target_calibration.v1` rootをatomicかつ非上書きで発行する。M=1は`oracle_kind=aq4_target`、M>1は`oracle_kind=aq4_optimized`とし、別々の新規rootを必須とする。前者はsource gateとpath gateのall-M=1 referenceを兼ねる。

入力は既存full-model driverと同じcase v2、identity v2、served-model、package tree、worker、runtime device、preflightへexact bindする。source manifest/cases/rows/SHA256SUMS/sidecarはregular fileかつnon-symlinkでなければならず、shape、offset、size、row SHA-256、nonfinite count、stable top-10、greedy、input-token hash、case/step順を全rowについてchunk scanで再検証する。`ULLM_ENABLE_AQ4_LM_HEAD_DIRECT_TOP1`が有効なら、full logitsが生成されないためmodel load前に拒否する。

source caseのgreedy列は`ullm.qwen35_aq4.calibration_replay.v1` domain、u64 little-endian token count、u64 little-endian token IDsのSHA-256へ正規化する。各stepではAQ4 predicted tokenを観測した後、source tokenだけをcommitする。したがってdivergence後も次stepのKV/historyはsource列へ固定される。

出力は次の2種類のrow indexを持つ。

- `rows.jsonl`: 98d9433のcompare toolと互換なexact vector row。case/step/input hash、hidden/logits offset・bytes・elements・dtype・endianness・SHA-256・nonfinite count、AQ4 greedy、stable top-10、finiteを持つ。
- `execution-rows.jsonl`: vector rowとcase/stepで1対1に対応し、source sequence/row SHA-256、predicted/committed token、divergence、generation epoch、observation完了、publication commit、row lifecycleを持つ。manifestがrecord countとfile SHA-256をbindする。

manifestはrequested/resolved/actual token M、actual request width、case、source、policy、served/package/content/worker/capture binary/device identity、terminal lifecycle/reset、operation auditを記録する。calibration observerのdevice-to-host転送を含むため、`performance.timing_eligible=false`かつ`raw_v2_schema_emitted=false`であり、既存raw-v2 schemaは変更しない。

成功rootは全row、両sidecar、execution rows、terminal reset、operation coverageが完全な場合だけ`available`となる。nonfiniteを含む完全captureは`blocked`、OOM・identity不一致・cancel・short/extra row・hash不一致・step swap・partial writeはpartial payloadを破棄してmanifest-onlyのimmutable blocked rootとしてatomic発行する。既存rootは成功・失敗のどちらでも上書きしない。

## 次の行動

GPU/live captureはこの実装作業では行わない。R9700で実行する際は、source gate用M=1 rootを先に固定し、optimized Mごとに別rootを作成してから、既存compare toolでsource gateとpath gateを別々に計算する。
