# P3 profile actual v9

## 前回の要点

- quiet-window-v14とoperator-command-v9はcommit `2df19a16723df952c0be58a5cff4a1d86bb80d99`、tree `c60eb183096efffd7fcea1b4993f68aa26e37c18`で確定した。
- quietは27/27 clean、span `312.301672447`秒、reset 0、confirmation passed。operatorはready-v11をpinし、exact 10 argv、`shell=false`、maximum invocation 1、retry forbiddenを固定した。

## 今回の変更点

- double independent GO後の直前監査でservice main/worker `872658/873053`、active/running、NRestarts 0、exclusive AMD/KFD ownership、fresh outputs 9/9 absent、targeted process 0を確認した。
- 同一PTYでsudo cacheを確立し、sealed manifestのargvを`os.execv`でshellを介さず1回だけ実行した。invocationは1/1、return codeは`1`、retryは0。実行時間は`82.541920446`秒だった。
- failure reasonはcapture parserの`unknown kernel family in source trace: __amd_rocclr_fillBufferAligned`。ready candidate reason codeは`ready_candidate_marker_absent`で、今回のprofileはfailure evidence onlyであり、measurement/promotionには使用しない。
- operator resultはstatus `failed`、actual auditはstatus `failed_immutable_evidence_preserved_restore_passed`。operator result JSON SHA-256は`81f3e354dfea5a86fa52b668788ca02a5ec4afdfd2988a66db14debb2d84353c`、result SUMS SHA-256は`0ed11a72b9c1cf86ea29b8631db3c3ba4a7124063502ab2b2992e436ff30c93b`。
- actual audit JSON SHA-256は`bb51e8609744efc9b30d9a997129f6ced13f623f335c21575d0116046b0da5bf`、audit SUMS SHA-256は`30a40b6c29fafb143f21c2b294653542059ebb0f54a1b86c64f65780d5ee5860`。capture failure JSON SHA-256は`bbcd13d768e650c5f099cdbf17128bc003fba196065d9f08d6c71d413dc07c8b`。
- package full content hashはexactly 1回でpassedし、7,700,872,459 bytes、1,045 files、SHA-256 `a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1`を確認した。
- outer-finally restoreは`14.998021894`秒で120秒のabsolute deadline内に成功した。新service epochはmain/worker `1205475/1205902`、active/running、NRestarts 0、AMD/KFD ownerはworkerのみで、lock busy、formal healthとworker/package/served hashesはpreflightと一致した。
- seal後のcommit直前readbackではserviceがさらに新しいmain/worker `1212941/1213021`へ遷移していたが、active/running、NRestarts 0、exclusive AMD/KFD ownership、lock busy、trusted hashes一致、targeted execute process 0を確認した。artifact内のrestore判定は実行直後のepoch `1205475/1205902`に対するimmutable記録である。
- capture children、launcher children、lock holders、targeted residual processはいずれも0。credential値パターンは0件。sudo認証文字列との2件の文字列衝突はagent-info CSVのCPU製品名`AMD Ryzen Threadripper PRO 3995WX`だけだった。
- maintenance-v8、execute-evidence-v8、runtime-v8、capture-v8、operator-result-v9、actual-audit-v9を封印した。各SUMS SHA-256は順に`458be04877a644e918e45369ad4acab589516b574f3bf5b0791bf0a8e6bcd614`、`87507f693f2d537a7eae158d206bf1e1d3e4f755bb41ce758afe554d24519d2c`、`91170d9480041e4c16a8a0778821622d106bd948db579fec805dc35c0f228d37`、`03cc0adecd43aeecb799bb574294e6e5a108077fc392bbc58103a67f3ec21c15`、`0ed11a72b9c1cf86ea29b8631db3c3ba4a7124063502ab2b2992e436ff30c93b`、`30a40b6c29fafb143f21c2b294653542059ebb0f54a1b86c64f65780d5ee5860`。
- 6 rootはすべて`0555`、全filesは`0444`かつnlink 1。formal actual validatorと全`SHA256SUMS`がpassedし、旧sealed成果物は不変。
- operator source testsはpost-actual適用可能な10件がpassedした。残る1件はpre-actual専用のfresh 9/9 absence assertionであり、actual evidence生成後はexpected failとなるため、source/testをこのartifact commitでは変更せず、次のoperator source作業でfinal-state independentに修正する。

## 次の行動

- retryは禁止されたまま維持し、同じv9/v8 outputを再利用しない。
- `__amd_rocclr_fillBufferAligned`をkernel familyへ分類するcapture parser修正は、今回のimmutable failure evidenceを入力にした別作業として扱う。
