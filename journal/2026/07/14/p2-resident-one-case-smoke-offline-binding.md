# P2 resident one-case smoke offline binding

## 目的

`0fd7993843d0d7f1096d89079ce06922871d9f1a` の resident driver を、R9700 の実モデル load 前に検査できる one-case smoke 入力 bundle として固定する。

## 実施内容

- detached clean worktree の commit `0fd7993` で `CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-p2-resident-driver` を実行した。
- release binary を byte-copy し、`nlink=1`、mode `0555` の detached copy とした。
- active served manifest、worker、package manifest、package tree 1045 files、resident binary、guard set、protocol v2、build commit を SHA-256 で固定した。
- representative/full_model/cold_prefill/cold_batched、prompt 128、requested M=128 の 1 case を R9700 `gfx1201`、device index 1、visible device 1 に束縛した。
- case、fixture、identity、synthetic preflight、policy、fake-ready、runner dry-run を exact bundle と `SHA256SUMS` に収録した。
- bundle validator は exact schema、hash coverage、single-link regular file、mode、relative member path、symlink、hardlink、二重 stat/hash による TOCTOU 検査を行う。

## 検証

- offline bundle validator: passed
- `SHA256SUMS`: bundle root 内で全 12 entries passed
- `python3 -m pytest -q tests/test_prepare_aq4_p2_resident_smoke_bundle.py tests/test_run_aq4_p2_resident_batch.py`: 17 passed
- resident runner の synthetic fake-ready validation: passed
- resident runner の one-case dry-run: 1 case、12 transactions、1 resident model-load plan

## 非実施事項

- GPU command は実行していない。
- worker/resident driver の process は起動していない。
- model load は実行していない。
- live service は変更していない。
- actual runtime identity、power、VRAM は取得していない。
- status は `prepared_not_executed`、promotion は `false` のままとした。

## v2 trust-root修正

初版validatorはbundle内のhashを再束縛するとsemantic driftを独立検出できなかったため、schema v2へ更新した。

- `0fd7993`のcommit/tree/source blobとclean build binary SHAを固定した。
- active served model、worker、package manifest、1045-file package tree、guard setを外部pathから再読・再hashする。
- trusted Git blobのofficial expanderからofficial caseを再生成し、R9700 host bindingを別objectとして明示した。
- 全JSON payloadをtrust rootsから再構築し、全階層とexact bytesを比較する。
- semantic valueとtransport hashを同時に再束縛する8種類のnegative testを追加した。
- v2 CLI validator、SHA256SUMS、17 testsが通過した。

## v3 normative launch-boundary修正

- trust rootをlaunch-boundary hardening commit `319d618`のclean treeとdriver/runner blobsへ更新した。
- detached clean worktreeで`CARGO_BUILD_JOBS=1` release buildを行い、resident binary SHAを固定した。
- absolute served manifestと全protocol link契約をtrusted driver blobから再検証する。
- driver executable、served manifest、device index、build commitをexact launch argvとpath/SHA bindingへ固定した。
- 旧`0fd7993` provenanceをhistorical superseded、execution不可として明示した。
- validator終了時のdirectory再列挙とdirectory identity比較を追加し、late unknown/missing/replaceを拒否した。

## 3dc4aa6 trusted runner更新

- current source/runner trust rootを`3dc4aa612b6cfd87675d0bd9fe506426f43e64f9`、tree `bd46e713c658878e66fcab6d49ef863e43a06bd8`、runner blob SHA-256 `e7dae31c64b3844a09fbba7ef36bbae7834e21d5d217bad679dd50bdf314ff02`へ更新した。
- resident driver source blobはnormative `319d618`とcurrent `3dc4aa6`でbyte同一、SHA-256 `d42e283d231dc177b929bcffb0f51acb0c13900be7bd040f6e24bd51aede95b7`であることをGit blobから検証した。したがって、normative clean binary SHA-256 `62f720835de60a61bad0a9aab5b80d778624d4d97ef5c8998e179418dab730f1`を継続した。
- prepare時にbundle同梱のtrusted runnerを`--one-case-smoke --dry-run`でsubprocessとして1回実行し、validate-only/fake-ready handshakeを通過したrunner生成planを`dry-run.json`へ収録した。planは1 case、12 transactions、warmup 2、measured 10、`smoke_only=true`、`promotion_eligible=false`である。
- exact argv、exit code 0、stdout/stderrのexact bytesとSHA-256、plan SHA-256を`runner-dry-run-evidence.json`へ固定した。stdoutとstderrはいずれも空である。
- 通常profile 84 casesとone-case smokeを分離し、one-case成果物のpromotionはfalseのままとした。
- launch bindingはtrusted runner validate-only argvとresident driver direct argvを分離した。runnerの`argparse`がoption風のdriver引数を`nargs=+`で透過できないため、両者を単一argvとして偽装していない。

## 検証更新

- offline bundle validator: passed
- `SHA256SUMS`: `bundle.json`と全required membersを含むexact coverage passed
- `python3 -m pytest -q tests/test_prepare_aq4_p2_resident_smoke_bundle.py tests/test_run_aq4_p2_resident_batch.py`: 43 passed
- GPU command、resident driver、worker、model load、live service operationは実行していない。
