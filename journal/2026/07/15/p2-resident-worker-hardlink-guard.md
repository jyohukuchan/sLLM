# P2 resident worker hardlink guard

## 前回の要点

- production workerはCargo release由来のprimary/`deps` aliasというexact 2 hardlinkで、汎用single-link検査がresident driverのready前に終了していた。
- 過去のactual v2 failure evidenceは、この終了を記録した既存証拠として保持する。

## 今回の変更点

- served-model manifest schemaは変更せず、worker専用のexact-two hardlink-set guardをresident driverへ追加した。
- release root/`deps`を有界・no-symlinkで走査し、primaryと唯一のaliasだけが同一inodeを指すことを確認する。2個の`O_NOFOLLOW` FDでdev/ino/uid/gid/mode/size/mtime/ctime/nlink、SHA-256、byte count、pre/open/post安定性を検証する。
- package/device/`RealExecutor`確立後、ready直前にもworkerを再hashしてguardを再検証する。汎用file helperのsingle-link規則は維持した。
- resident binary、prepared bundle、strict B sidecar、launcher、execute binding、maintenance harness、base/profile readyとdry-run artifactを順番に再固定した。
- 過去v2 output/evidenceは削除せず、次のexplicit actual用output/evidence/run-idはv3へ進めた。
- QA attestationは重複し得るlegacy集計を廃止し、11個のdistinct test fileについてpath、source commit、Git blob、exact pytest argv、collect/pass countを記録するv2へ更新した。

## 検証

- resident driver Rust tests: 12 passed。
- trust-chain 6 files: 255 passed。
- ROCTx ranges: 5 passed。
- diagnostic capture: 11 passed。
- selection raw producer: 21 passed。
- profile family exclusion: 27 passed。
- candidate selector: 26 passed。
- base/profile canonical dry-runはpassedで、actual processを起動していない。
- actual v3はworker hardlink guardを通過し、HIP gfx1201/device 1でmodel load 1のreadyまで成功した。case_beginでrunnerがprepared preflight linkをlive preflight linkで上書きしたためdriverが`preflight fields differ`で終了し、そのfailure output/evidenceは保持した。

## v3 failure後の変更点

- `PreparedPreflightLink`と`LivePreflightLink`を別型・別validator・別変数へ分離した。case_begin builderはexact 2-field prepared linkだけを受理し、その参照先がexact 7-field prepared documentであることを検証する。
- live preflight linkはrunner gate、lock identity、raw/evidenceだけで使用し、driver protocolへ渡さない。
- fake resident driverはcase_begin top-level、execution、sampling/control、5 linkのexact fields/hash、prepared 7-field documentをdriver同等に検証する。
- live/prepared swapとfield collisionを負例として追加した。
- runner、validator、B、launcher、execute binding、maintenance harness、base/profile ready/dryを再固定し、次のexplicit actual用output/evidence/run-idをfreshなv4へ進めた。
- exact QA集計は345 collected / 345 passed / 0 failed / 0 deselectedである。

## 次の行動

- explicit actualを行う場合だけ、ready artifactに固定されたv4 outputを一度使用する。
- actualが失敗しても証拠を保持し、再試行時は新しいversionへ進める。

## v8 device vocabulary failure後の変更点

- source manifestのdevice family語彙は`RDNA4`、prepared caseとdriver runtime identityのarchitecture語彙は`gfx1201`であることをschema監査した。
- resident driverがruntime `gfx1201`を`RDNA4`へ変換していた処理を削除し、case/runtimeに共通するdevice 5 fieldを変換なしでexact比較するよう修正した。`P2Device` schemaにはfieldを追加していない。
- active production fixtureにsource `RDNA4`とbound/runtime `gfx1201`を別field集合として固定し、正常系、gfx/RDNA語彙交換、case identity不一致、runtime identity不一致をRust testへ追加した。resident driver testは16/16 passedである。
- release driver、prepared bundle、strict B、launcher、execute bindingを再固定し、fresh output/evidence/run-idをv5へ進めた。offline launcher dry-runはpassedで、actual/GPU/service操作は行っていない。
- exact QAのうちdevice修正に直接対応するRust 16件と、独立した既存pytest 90件はpassedである。

## 現在の阻害要因

- 並行するfidelity作業がproduction workerを旧exact-two hardlink inodeからsingle-linkの別inodeへdetachした。resident側の既存exact-two fixture/guardは意図どおりこれを拒否する。
- このためprimary trust-chain pytestはdevice bindingへ到達する前にworker trust rootで拒否され、最終exact QAとmaintenance ready artifact再生成は未完了である。
- 「無関係guard維持」に従い、resident guardをsingle-linkへ変更せず、production workerも変更していない。exact-twoを復元してfixtureごと再固定するか、resident guardをsingle-linkへ変更するかの明示判断が必要である。

## single-link authoritative決定後の解決

- production workerとfidelity証拠は変更せず、worker fixtureを`roots`、順序付き`paths`、`primary_path`、期待metadata/SHAを持つv2へ更新した。
- guardは`paths`数と`nlink`の一致を必須とし、宣言された全pathを`O_NOFOLLOW`で開いてmetadata/SHAを検査する。各bounded rootのsame-inode path集合が宣言集合とexact一致することをpre/postで確認するため、single-linkとexact-twoの両方をfixture-drivenで扱い、unknown/extra/missing/swapを拒否する。
- current active fixtureはsingle-link identityへ固定し、synthetic testsではsingle-linkとexact-twoの正常系および初期・遅延変異の負例を維持した。一般file helperのsingle-link規則は変更していない。
- driver source/binary、validator、prepared、B、launcher、execute binding、maintenance harness、base/profile readyとdry-runを再固定し、次の明示actual用run/output/evidenceをfreshなv6へ進めた。
- exact QAは12 distinct files、362 collected / 362 passed / 0 failed / 0 deselectedである。内訳はprimary trust-chain 256、resident Rust 16、その他5 suites 90である。
- base/profile canonical dry-runはpassedで全actual process countが0、service/GPU/model loadは未実行である。v6 run/evidence、profile run/evidence、rocprof capture outputはすべて不在である。

## 次の行動（更新）

- explicit actualを行う場合だけ、ready artifactに固定されたv6 outputを一度使用する。
- actualが失敗しても証拠を保持し、再試行時は新しいversionへ進める。
