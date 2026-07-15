# P3 profile quiet-window v6 rolling

## 前回の要点

- v5 は固定27 samplesの監視中にforeign AMD owner PID `2929769`を4 samplesで検出し、NO-GOとした。

## 今回の変更点

- fresh `resident-one-case-smoke-profile-quiet-window-v6` で、2026-07-15 17:24:40 JSTから17:39:40 JSTまで、5秒間隔のrolling監視を最大15分実施した。181 samples、監視経過は `907.386449985` 秒である。
- GO条件は27連続clean samplesかつfirst-to-lastが130秒以上である。foreign AMD/KFD owner、対象external process、service/worker/lock/health、全pts process set、HEAD/tree、関連30ファイルのbytes/identity、profile v3 fresh outputsの変化でstreakを即resetした。
- reset eventは99件だった。理由の内訳は全pts process set変化83件、foreignまたはmissing AMD owner 17件、GPU/KFD owner set変化2件、最大15分到達1件である。理由は同一eventで重複し得る。clean streakの最大長は1 sampleで、終了時は0 samplesだった。
- foreign AMD ownerはPID `2993067`（samples 44〜47）、`2999623`（56）、`3006777`（78〜81）、`3011743`（92）、`3012105`（93）、`3019377`（115）、`3019767`（116）、`3025205`（133〜136）である。KFD ownerは全181 samplesで通常worker PID `2635236`だけだった。
- 対象external processは0件だった。service epoch、worker PID `2635236`、lock device/inode `26:772895`とholder PID `2634680`、profile v3 fresh outputs 5件の不在は保持された。開始・終了formal healthはともに取得でき、gateway/OpenWebUI endpointsはHTTP 200、formal process countsは同一だった。
- HEAD/treeは開始時 `380866065d7da7a52e1de5cd51ccc344cb6f54d3` / `b8b1bbfb424d8c05edd3030bd6d2997481ed4973` から、sample 167（17:38:35 JST）で `5658f8db7a493b19a4bc7fa7a5e245cd7cc6ad9e` / `c78e82891fab9a713b4b722292659652f6fe9c5c` に変化した。commitは `Add layer0 QKV Z interaction diagnostic` である。
- 関連30ファイルのbyte aggregateは `5f8b91af3bfb90d39ba830b3242aa46e00d30dbc9bfd2d4c10f5aa6dcc349cce`、identity aggregateは `a6051a932743cf29f6d997bb5bc57d9a07d698098ba011601b0142bbb6587759` で開始・終了時に一致した。
- 27連続clean samplesへ到達しなかったため、GO直前にだけ行うeb484/c3/A4/B4/C strict readback、3組のSUMS、canonical targeted testsは実行段階へ進んでいない。pin不整合を示す結果ではなく、quiet streak未達による **NO-GO** である。
- actual、GPU workload、service操作は実行していない。AMD-SMI/KFD監視、formal health、Git/readbackなどread-only操作だけを行った。
- `quiet-window.json` のSHA-256は `e21ebdf2c2a5301ac5de845e741fa8bb75545b46f735d4a24288b9596a78ea15`、`SHA256SUMS` のSHA-256は `c566a7f81277654b01c313f04c893329f2f2048272615fac36d73675ed203b45` である。成果物は0444、ディレクトリは0555で、`sha256sum -c SHA256SUMS` は成功した。

## 次の行動

- このv6 NO-GOを保持する。全pts process setの変化とforeign AMD ownerを発生させる並行診断を止められる時間帯を確保した場合だけ、別のfresh pathでrolling quiet-windowを再実施する。
