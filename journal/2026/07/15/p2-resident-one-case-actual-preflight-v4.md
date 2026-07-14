# P2 resident one-case actual preflight v4

## 前回の要点

- v3 actualはservice stop後にsystemd `RuntimeDirectory=ullm`がlock parentを削除するため、stable 2前のlock probeで停止した。
- v4はlock substrateを作成してstable 2からrunnerまで同一inodeを維持する契約だった。

## 今回の変更点

- commit `e566f18b7ad02a3684890a27ca8bd0a66b1a1ab0`、manifest SHA-256 `08b292d39412e31e8623fe22fb90b326905de1e88fba35d91eb8ac75599bd933`のHEAD/input/SHA256SUMS/permissions、3 fresh output ABSENTを確認した。
- production preflightのKFD owner snapshotで`reason_code=gpuid_schema_differs`、`stage=gpuid_parse`となったためNO-GOとした。sudo prime、canonical command、service stop、substrate作成、launcher、model load、warmup/measured、profileはすべて0回で、v4 outputは3つともABSENTのまま。
- O_NOFOLLOW、regular-file確認、open前/fstat identity確認付きでPID `18897`のqueue `0/1/2`にある`gpuid`を取得した。全rawは5 bytes、ASCII repr `'51545'`、SHA-256 `53bfb177258a9c4448495c24270e0712ac0a01039c23a51bc28a74a1a268c6ab`、strip後は正のdecimal `51545`、末尾改行なし。
- v4 scannerは`text.endswith("\n")`かつ`text[:-1].isdigit()`を要求するため、kernel sysfsの正常な改行なし値との差分は`missing_trailing_newline`だけである。`strip()`後の全体がASCII decimalで正の値、raw length bound内なら受理する必要がある。空、符号、空白混入、非ASCII、非数字、0以下、oversizeはfailを維持する。
- topology node 2の`gpu_id=51545`と一致し、propertiesの`domain=0`、`location_id=18176`からBDF `0000:47:00.0`を導出してready bindingと一致した。KFD ID/node/BDF mappingはPASS。
- immutable diagnostic `resident-one-case-kfd-gpuid-schema-diagnostic-v1/`は`0555`/`0444`、SHA256SUMS PASS。diagnostic SHAは`e5245ab189b3eb3e33781484161404f29bb9f6ff299a4623431c9f639b1cf1e8`、SUMS SHAは`c4c4c38fc1cd4486dd850b964b3059ba0c7bb170596079a7c0ee1224cf109fbe`。
- production serviceはmain PID `18825`、worker PID `18897`、active/running、`NRestarts=0`で不変。service/GPU/HTTP/actualには変更を加えていない。

## 次の行動

- scannerを実kernel sysfs形式に合わせ、末尾改行の有無を意味差として扱わず、bounded ASCII positive decimalを受理する。
- KFD parser修正と回帰QA、新しいready/operator artifactが揃うまでactualを実行しない。v4のfresh output pathは未使用だが、今回のmanifestでの実行は行わない。
