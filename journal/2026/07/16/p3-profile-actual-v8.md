# P3 profile actual v8

## 前回の要点

- quiet-window-v13とoperator-command-v8はcommit `46219af1ce52c6af3bd29d6e84a0297ab6301823`で確定した。
- quietは27/27 clean、span `311.965172889`秒、reset 0、confirmation passed。operatorはready-v10をpinし、exact 10 argv、`shell=false`、maximum invocation 1、retry forbiddenを固定した。

## 今回の変更点

- double independent GO後の直前監査でservice main/worker `790940/791055`、active/running、NRestarts 0、exclusive AMD/KFD ownership、fresh outputs 9/9 absent、targeted process 0を確認した。
- 同一PTYでsudo cacheを確立し、sealed manifestのargvを`os.execv`でshellを介さず1回だけ実行した。invocationは1/1、return codeは`1`、retryは0。
- failure reasonはcapture parserの`resident raw device_lock fields differ: missing=[], unknown=['device', 'inode']`。ready candidate reason codeは`ready_candidate_marker_absent`で、今回のprofileはfailure evidence onlyであり、measurement/promotionには使用しない。
- operator resultはstatus `failed`、actual auditはstatus `failed_immutable_evidence_preserved_restore_passed`。operator result JSON SHA-256は`83ed58b27eeaba5d5feaf28ffec4b0f9fe08521b3050b62e784f25138313e32a`、result SUMS SHA-256は`9f03e26cdd484aa98f8f89a91717afa10390b1fa5d3a30034bb61834b67af4d8`。
- actual audit JSON SHA-256は`066b9f016e7f54c0ae047b965dd6cbb7e2b45449654b9dba3b4912b9d96ecd2e`、audit SUMS SHA-256は`9e6d1514b030c5b98ef491ce139d3bda02dca6331b91f6cdec5d3e0868759ba9`。capture failure JSON SHA-256は`78ea515b3bba273e5c8179ffce1452d994a8e50fa4d0aebe9c101ffd57a432b5`。
- package full content hashはexactly 1回でpassedし、7,700,872,459 bytes、1,045 files、SHA-256 `a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1`を確認した。
- outer-finally restoreは`14.926430339`秒で120秒のabsolute deadline内に成功した。新service epochはmain/worker `872658/873053`、active/running、NRestarts 0、AMD/KFD ownerはworkerのみで、formal healthとworker/package/served hashesはpreflightと一致した。
- capture children、launcher children、lock holders、targeted residual processはいずれも0。secret scanの唯一の文字列一致はrocprof agent infoにあるCPU製品名`AMD Ryzen Threadripper PRO 3995WX`であり、credential、Authorization、Bearer、API keyは記録されていない。
- maintenance-v7、execute-evidence-v7、runtime-v7、capture-v7、operator-result-v8、actual-audit-v8を封印した。各SUMS SHA-256は順に`8af6e055aca65de5d4b0def3e3776d2bc678b69d7ba6bd2ea1b4893dcc9c88f4`、`aeba908fc66d7160f533b07c8cf5fbf5a89c2e24a7262320f7fac21bb39aef63`、`e4697d4c08469e9b6bd22b0b5df5fe3c0ce68c14430ec524295c227e7a35b87c`、`f66be352a652e02b09f785682e24f3f267ad5bb940d7c5c9127f43011219e44d`、`9f03e26cdd484aa98f8f89a91717afa10390b1fa5d3a30034bb61834b67af4d8`、`9e6d1514b030c5b98ef491ce139d3bda02dca6331b91f6cdec5d3e0868759ba9`。
- 6 rootはすべて`0555`、全filesは`0444`かつnlink 1。formal actual validatorと全`SHA256SUMS`がpassedし、旧sealed成果物は不変。

## 次の行動

- retryは禁止されたまま維持し、同じv8/v7 outputを再利用しない。
- `device_lock`の`device`/`inode` fieldsとcapture parser contractの不一致を、今回のimmutable failure evidenceを入力にした別作業として修正する。
