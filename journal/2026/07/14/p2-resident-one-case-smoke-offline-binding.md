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
