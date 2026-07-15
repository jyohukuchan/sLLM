# AQ4 P3 device lock and live preflight parser contract

## 前回の要点

- device lock契約はcommit `431c69b40d99e4ea8bbb74b9e26b731ad785b3ad`で修正し、rawとsummaryのlock ownerを完全一致で照合するようにした。
- その時点のactual-v8 full validateには、次の境界として`links.live_preflight`の現行embedded evidenceを旧ref shapeとして扱う不一致が残っていた。actual-v8はimmutable failure evidenceであり、retryは禁止されている。

## 今回の変更点

- `ullm.aq4_p2_device_lock_owner.v1` の現行契約に `device` と `inode` を必須フィールドとして追加した。
- `device`、`inode`、`pid`、`acquired_unix_ns` と driver 側の同種フィールドは、boolを許可しない正整数として検証する。
- summaryにも同じdevice lock検証を適用し、rawとsummaryのlock ownerを完全一致で照合する。run IDとresident driver SHAのidentity bindingもfail-closeのまま維持した。
- fixtureを現行契約へ更新し、missing、zero、negative、string、bool、unknown、raw/summary mismatch、driver SHA mismatch、run ID mismatchのnegative matrixを追加した。
- diagnostic current contractでは、raw `links.live_preflight`とsummary `validation.live_preflight`にembedded 8-field evidenceを必須化した。promotion contractは従来どおりlive preflightを含まないため、mode境界を明示している。
- linked documentのexternal byte SHA、absolute traversal-free path、stat device/inode、mode `0444`、nlink 1を検証する。document自身にはself-hash fieldがないため、external byte hashと混同しない。
- `ullm.aq4_p2_resident_live_preflight.v1`の13 root fieldsと、runtime mapping、lock、VRAM、commands、compute owners、environment、prepared preflight、services、worker PIDsのnested shapeをexact検証する。
- runtime mappingをresident identityへ、preflight lockをdevice lockへbindingし、rawとsummaryのembedded evidenceを完全一致で照合する。VRAMはused 0かつfree/headroomがtotalと一致する証明を要求する。
- missing、legacy ref-only、extra、type、bool、zero、negative、path、hash、device/inode、mode、nlink、document/nested mismatch、raw/summary mismatchのnegative matrixを追加した。
- 封印済みactual-v8のraw SHA-256 `397f02a2cd87e5d30eb9eb569b5d022351b1f994358e71535f2ce697af5df25c` とsummary SHA-256 `b82409bf997e207df5576ba7e38ebefddff363440c256250ffc8f7b521dcb3f5` を固定し、full resident pair、12 runs、device lock、live preflightの正受理を回帰テストにした。
- producer全103件、selector/family関連53件はpassed。`py_compile`と`git diff --check`もpassedした。GPU、service、actualは実行していない。
- producer SHA-256は`d0360a494f30c2bbac7ca1d043385dd6de9384fa2d81ab99881e54afeaaed934`へ変わった。capture側のpinは最終producer commit後に再連鎖する。

## 次の行動

- producer修正をcommitした後、capture側のproducer SHA pinを最終SHAへ再連鎖する。
- 再連鎖後にcapture CPU testsを再実行する。launcher/maintenance/artifactsには触れず、actual-v8は再実行しない。
