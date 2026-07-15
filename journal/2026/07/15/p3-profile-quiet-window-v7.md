# P3 profile quiet-window v7 rolling

## 前回の要点

- v6は181 samples・907.386秒を監視したが、全pts process set変化83件をblockingに含めたため最大clean streakが1 sampleとなり、NO-GOだった。
- v6 evidenceをread-onlyで再分析すると、pts inventory/setだけをdiagnostic-onlyにした場合はsamples 0〜43の44連続clean・`214.970613735` 秒が成立していた。

## 今回の変更点

- fresh `resident-one-case-smoke-profile-quiet-window-v7` で、PTS inventory/set変化をdiagnostic-onlyへ変更した。PTS上でもsystemctl、maintenance、capture、rocprof、GPU diagnostic、probeに一致するtargeted commandはblockingのままである。
- blocking条件はforeign AMD/KFD owner、targeted external command、service/worker/lock/formal health identity、HEAD/tree、関連30ファイルのbytes/identity、profile v3 fresh outputs 5件である。blockingごとにstreakをresetし、cleanな状態からnew baselineを作る。
- 最初の起動はforeign AMD PID `3077792`（`/tmp/ullm-p2-depth-accumulation/target/debug/ullm-engine`）のため開始formal healthでfail-closedし、fresh outputを作らず終了した。foreign解消後にrolling collectorを再起動した。
- 再起動後は52 samplesを取得し、監視経過は `346.315937148` 秒だった。samples 17〜23でblocking reset 7件を記録した。PID `3098307` のbashから起動されたGPU diagnostic `target/debug/ullm-engine package-linear-attn-mlp-block-smoke` PID `3098649`をtargeted external commandとして検出し、samples 18〜23ではforeign AMD ownerとしても検出した。KFD foreign ownerは0だった。
- blocking解消後のsamples 24〜50で27連続clean samples・`192.447962750` 秒を満たした。候補後のconfirmation sample 51を含む最終streakは28 samples・`200.537321143` 秒である。PTS diagnostic changeは50件あったが、targeted commandを伴わない変化は判定へ使用していない。
- HEAD/treeは開始・終了・最終baselineで `3023148fd63ef93805837077b574be28478e7f54` / `d057eb82ef0b1b3a21e99c385453edf9c2161e12` に一致した。関連30ファイルのbyte aggregate `5f8b91af3bfb90d39ba830b3242aa46e00d30dbc9bfd2d4c10f5aa6dcc349cce` とidentity aggregate `a6051a932743cf29f6d997bb5bc57d9a07d698098ba011601b0142bbb6587759` は不変で、fresh outputs 5件は全samplesで不在だった。
- GO候補後にeb484 canonical artifacts 11件、c3/A4/B4/C pins、QA source commit:path:blob strict provenance `12/12`をreadbackした。execute binding、base ready、profile readyの全`SHA256SUMS`が成功した。
- canonical targeted testsはstrict QA provenance、base canonical readback、base dry-run zero-process、profile canonical readback/dry-run zero-processの4件がPASSした。confirmation sampleとGO時formal healthもPASSし、開始・終了formal identityは一致した。
- 判定は **GO** である。actual、GPU workload、service操作は行っていない。GPU diagnosticは並行プロセスを検出しただけで、collector自身は実行していない。
- `quiet-window.json` のSHA-256は `5d210245c52248aec489b9ec6820c4d208ad8346cdf09c4a2f3babe4909eb7b6`、`SHA256SUMS` のSHA-256は `57fbbd5697f8aae52bbbe2c001cb5123e77798996ec4db161d8306e7bd1bc2c7` である。成果物は0444、ディレクトリは0555で、`sha256sum -c SHA256SUMS` は成功した。

## 次の行動

- v7 GOをprofile operatorのquiet-window根拠として使用できる。実行直前にも固定HEAD/tree、関連bytes/identity、strict trust pins、fresh outputs、formal healthを再確認し、差異があれば実行しない。
