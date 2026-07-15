# AQ4 P3 device lock parser contract

## 前回の要点

- profile actual-v8 は resident driver、12 runs、ROCTx、rocprof trace の生成まで成功したが、P3 producer が raw の `device_lock.device` と `device_lock.inode` を未知フィールドとして拒否した。
- actual-v8 は immutable failure evidence であり、retryは禁止されている。

## 今回の変更点

- `ullm.aq4_p2_device_lock_owner.v1` の現行契約に `device` と `inode` を必須フィールドとして追加した。
- `device`、`inode`、`pid`、`acquired_unix_ns` と driver 側の同種フィールドは、boolを許可しない正整数として検証する。
- summaryにも同じdevice lock検証を適用し、rawとsummaryのlock ownerを完全一致で照合する。run IDとresident driver SHAのidentity bindingもfail-closeのまま維持した。
- fixtureを現行契約へ更新し、missing、zero、negative、string、bool、unknown、raw/summary mismatch、driver SHA mismatch、run ID mismatchのnegative matrixを追加した。
- 封印済みactual-v8のraw SHA-256 `397f02a2cd87e5d30eb9eb569b5d022351b1f994358e71535f2ce697af5df25c` とsummary SHA-256 `b82409bf997e207df5576ba7e38ebefddff363440c256250ffc8f7b521dcb3f5` を固定し、device lock境界の正受理と完全一致を回帰テストにした。
- producer全53件、selector/family関連53件はpassed。`py_compile`と`git diff --check`もpassedした。GPU、service、actualは実行していない。
- producer SHA-256は`57471a9145e3448b128d4838c87099b7360bd40c127d476293c3a875ecb12b3a`へ変わった。capture側の旧pinは担当外なので変更しておらず、capture testは意図どおりhelper SHA mismatchでcollectionを停止する。
- actual-v8全体には今回のdevice lockとは別に、producerの`links.live_preflight`表現が現行の埋め込み証跡と一致しない既存差分がある。今回のactual回帰は担当境界であるdevice lockに限定した。

## 次の行動

- capture/launcher/maintenance側でproducer SHA pinを再連鎖し、担当外の`links.live_preflight`契約差を別作業としてfail-closeに解消する。
- 再連鎖後にcapture CPU testsを再実行する。actual-v8は再実行せず、既存のimmutable evidenceを入力として使う。
