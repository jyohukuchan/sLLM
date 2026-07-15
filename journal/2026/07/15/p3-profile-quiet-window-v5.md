# P3 profile quiet-window v5

## 前回の要点

- v4 は 27 samples と 130 秒以上の sample spanを満たし、runtime側も安定していたが、監視中に HEAD/tree と関連 trust artifacts が変化したため NO-GO とした。

## 今回の変更点

- fresh `resident-one-case-smoke-profile-quiet-window-v5` に、5秒間隔で27点を取得した。sample 0からsample 26までのmonotonic spanは `130.847051378` 秒である。
- 開始・終了HEADは `3f6a8073bbd9abe7de4901452a9720bda0b8285e`、treeは `5437bc3ed76e374e7ed610be87e10c7eb6a25516` で不変だった。
- `eb4840f2f3ddcfb27c0e6e5de259f1b6002f7c07` のcanonical artifact 11件、`c3a676a962e542b997c14a695328d5cdbfa6c120` のmaintenance source、A4/B4/Cの固定commit:path Git objectsを開始前にreadbackした。
- 関連30ファイルのbyte aggregateは `5f8b91af3bfb90d39ba830b3242aa46e00d30dbc9bfd2d4c10f5aa6dcc349cce`、identity aggregateは `a6051a932743cf29f6d997bb5bc57d9a07d698098ba011601b0142bbb6587759` で、開始・終了および全標本で不変だった。profile v3 fresh output 5件は全標本で不在だった。
- formal health、service epoch、worker PID `2635236`、lock device/inode `26:772895` とholder PID `2634680`、KFD owner、全pts process setは安定し、対象external processは0件だった。
- sample 13〜16の約20秒間だけ、AMD-SMI ownerにworker PID `2635236` とforeign PID `2929769` が併存した。KFD ownerはworker単独のままだった。foreign PIDは監視終了時点で終了済みで、ttyを持たず、保存済み証跡からargvは確定できない。
- 変化を1件でも許容しない契約により、sample identity change 4件とGPU/KFD owner change 4件、計8件を記録し、判定を **NO-GO** とした。
- actual、GPU workload、service操作は実行していない。AMD-SMI/KFDのread-only監視だけを行った。
- `quiet-window.json` のSHA-256は `88f912ca69e3e767404f08a65b6a78983c17426fe933b8be31a0b6ef015cf6a3`、`SHA256SUMS` のSHA-256は `e9d0bfb8d64035f1f3f5e55fd5c9b5902a60d39671e6147fc1fb3128176c4c86` である。成果物は0444、ディレクトリは0555で、`sha256sum -c SHA256SUMS` は成功した。

## 次の行動

- このNO-GOを保持する。foreign AMD ownerを発生させる並行処理がない時間帯を確保できた場合だけ、別のfresh pathでquiet-windowを再実施する。
