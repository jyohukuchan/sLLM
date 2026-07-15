# P3 profile actual v14

## 前回の要点

- quiet-v19はcommit `1a45447b1eaa76a645fff6cca31cc007f034b4ff`でGO、27/27 clean、span `358.729450382`秒、reset 0として封印された。
- command-v14はcommit `ba7ab7d41c6de84a9165aa8e3592a9b18fcb0e6d`、root tree `6b0ad082ad999eb7e9269686949beb4868dca1a8`のexact-one pending authorityだった。
- 実行直前にはexternal SQ8 0、AMD-SMI/KFD ownerはproduction workerのみ、service epoch不変、fresh9 absent、operator/quiet/current audit cleanを確認した。

## 今回の変更点

### exact-one実行

- command-v14の10引数argvを`cwd`固定、`stdin=DEVNULL`、`shell=False`で正確に1回だけ実行した。
- start Unix nsは`1784155500308479749`、end Unix nsは`1784155588135817749`、elapsedは`87827338000` nsだった。
- return codeは1、operator stdoutは255 bytes、stderrは0 bytesだった。
- invocationは1/1、retryはfalseであり、失敗後の再実行は行っていない。

### 失敗分類と証拠

- actual statusは`failed_immutable_evidence_preserved_restore_passed`として固定された。
- workload runtime summary自体は`complete`、resident model load 1、warmup 2、measured 10、transaction 12、case 1/1 completeだった。
- capture validatorが`kernel trace row 83 interval/order is invalid`を検出したため、capture/runner/operatorは失敗扱いになった。
- ready candidate markerはabsent、measurement/promotion eligibleはいずれもfalseである。
- capture process group cleanup、launcher/driver/lock substrate cleanupはpassedであり、残留children、残留targeted process、lock holderはいずれも空だった。
- capture failure SHA-256は`a6774da72c214bf1ec7967e819e8d13a4ff4a62559cc60f0446880f54b4bdcb4`である。

### 復旧と封印

- 既定finalizerを1回だけ実行し、restore attempted/passed、poll 6回、errorなしを確認した。
- 復旧後はservice MainPID `2822256`、worker PID `2822676`の新epochでactive/running、NRestarts 0だった。
- AMD-SMI/KFD ownerはともに`[2822676]`、targeted processは0であり、formal health SHA-256は`b032d38fcdb8e17f2452daa47ce07f2335875451df2ac47f73d117a4331b3722`だった。
- `validate-actual`、`validate-operator`、`validate-quiet`はすべて通過した。
- 6 root、35 filesをroot mode 0555、全file mode 0444、各`SHA256SUMS`整合としてcommit `a2fe1ebac5d631919ca9082e17fda2126759a385`へ封印した。
- evidence commit treeは`ce8b024ff3bf2a516eac07275a93c171184fa279`である。
- operator result JSON SHA-256は`15982bf0ba01eeac23720f19b35a4c4cddad6ec949fa9e2397c7027126097d27`、actual audit JSON SHA-256は`cb68c59909b10fca0b3230d1550fdd06f22550e8a1209b930c46e12753568347`である。
- 全35 fileについてarchive bytesとGit object bytesのSHA-256一致を確認した。

| root | Git tree | files | `SHA256SUMS` SHA-256 |
| --- | --- | ---: | --- |
| maintenance-evidence-v11 | `3aaec547b8b77018b2209c69e9874603b325c83d` | 5 | `66963d122d201d7e9eba3a09bb57b28f615742eacb6b18c058f45df49e3d322f` |
| execute-v10 | `d922290115d14a641d2d072145f16c767cc03c6c` | 7 | `9e3e4f429223d5d4f5567a6a37fd104027a3142da7ba6afbde683dd43e727f5c` |
| execute-evidence-v10 | `efb84aa13feb998c3e3d4d738a43cb4d638bd2cc` | 8 | `08c21a39197bd150c7aff02a4b7a3749cc5cfd795f626ba87cf0402107d6959e` |
| operator-result-v14 | `2a74b0886243f0aa8d7d57f0d492a469c1679f35` | 4 | `0b78274b890f8804eb29b78118d0845cc3ad5d34778da5cf1e4a50bbdef6acf8` |
| actual-audit-v14 | `a9df3fdb625636c2c333c8c4e1ad3b8cc3aab4c2` | 2 | `558519d5e7623899db6b559a4a003d6fb866c35214f406b9364e6aeb9eec42f3` |
| diagnostic-capture-v10 | `5b4c5e1fdd514676d0703fd6a94e7add4bf0a2fc` | 9 | `7c275a461699947b5075f5e491be88cf1e64c8c1a0a05c523a55f8ed0856a09d` |

## 次の行動

- v14 actualはimmutable failureとして保持し、同じcommand-v14を再実行しない。
- 次の調査では、封印済みkernel trace row 83のbegin/end intervalと周辺行をread-onlyで解析し、validatorの入力順序仮定とrocprofv3の出力仕様を照合する。
- 新たな実行が必要になっても、原因修正、source/test/ready/quiet/commandの新version化、独立監査、別の明示的認可が揃うまではGPU actualを行わない。
