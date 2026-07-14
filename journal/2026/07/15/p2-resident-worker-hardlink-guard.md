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
