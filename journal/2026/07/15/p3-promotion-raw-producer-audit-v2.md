# P3 promotion raw producer canonical integration

## 前回の要点

- 専用branchではP3 raw producerのidentity境界、sample identity、atomic no-replace publicationを修正した。
- 旧branch後半の固定SHAは現mainのA4履歴より古いため、6 commitの一括cherry-pickは不可と判定した。

## 今回の変更点

- canonical main上でproducer/testの実質差分だけを順にreplayした。
  - `f99102db`: identity root、resident driver、runtime device、hash bindingの厳密化。
  - `c743007f`: `(run_id, case_id, run_index)` sample identity、device index 0、integer device ID、atomic no-replace、file/parent fsync。
- producer file SHA-256 `ce31daba6737a64efd2db3b897bcbef56289052978e7b3be544f89d82b91da52`をcapture/launcherへ固定した。
  - `86fb3df3`: producer consumer pin。
  - `2e005c49`: current launcher/captureのcommit、tree、blob、file SHAとraw 26件のQA manifestをmaintenanceへ固定。
- canonical generatorでbase/profile ready artifactを再生成した。
  - `2b211a3a`: base/profile ready trust artifact。
  - profile-ready rootには旧固定`target-command-manifest.json`を置かず、live preflight後にfresh生成する現契約を維持した。
- canonical launcher generatorでexecute-binding trustを再固定した。
  - `446e5042`: execute-binding launcher trust。
- 既存の未commit input/evidenceとAQ4 fidelity capture系ソース変更には触れていない。

## 検証

- raw producer: 26 passed。
- raw producer + selector: 52 passed。
- capture helper pin relevant: 5 passed。
- capture full: 27 passed。
- launcher relevant/full: 76 passed。
- maintenance full: 135 passed。
- ready artifact readback: 4 passed。
- base/profile/execute-bindingの`SHA256SUMS`: passed。
- Python `py_compile`、`git diff --check`: passed。
- GPU、profile実測、rocprof capture、service操作は実行していない。
- 旧branchで残っていたinput-root、bundle、execute-binding driftはcanonical main上では再生成後に再現せず、上記回帰はすべてgreenになった。

## 次の行動

1. 実GPU/profile前に、7 case × 10 measuredの各runへ固有のkernel/HIP API trace、capture capability、resident identity、full-model pairを割り当てる。
2. producer rawをselectorへ渡し、`selected`以外ではP3 runtime候補を昇格させない。
3. 実測時もone-case diagnostic rawをpromotion evidenceへ流用しない。
