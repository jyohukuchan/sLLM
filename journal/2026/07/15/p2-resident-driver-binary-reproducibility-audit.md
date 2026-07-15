# P2 resident driver binary reproducibility audit

## 前回の要点

- canonical resident driverはsource commit `81ceebb13518f590b5dbf439cd00b35e508c1c3f`、tree `5e98c3812f9eebdaed3e6085ab2e13521e249521`からclean buildされた。
- canonical identityはSHA-256 `458b8603d6823a1c20ea93e7c0d757c8910f3c36c9a2a34ab536853c0c9e7d34`、3,506,904 bytes、Build ID `e7313ba6f51feac74f14b5ffd100333265362e1e`である。
- recorded toolchainはcargo 1.96.0、rustc 1.96.0、LLVM 22.1.2、host `x86_64-unknown-linux-gnu`だった。

## 今回の変更点

- exact commitをdetached new worktree `/tmp/ullm-resident-repro-81ceebb1-wt`へ展開した。開始前とbuild/test後のworktreeはcleanだった。
- target `/tmp/ullm-resident-repro-81ceebb1-target`が開始前に存在しないことを確認した。
- Cargo.lockはGit blob `fb12cb0388ea1c6fc6368e7ea5d5100c11a20666`、SHA-256 `10df8371ae3a33ed792dc4e8c15dd6196a8a7e176e377ef275e75b3219aa157b`でexact commitと一致した。
- `CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=... cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver`を実行し、1分32秒で成功した。
- rebuildはSHA-256、size、Build IDがcanonicalと一致し、`cmp` return code 0でbyte-for-byte identicalだった。
- ELF section headers、program headers、notesのread-only digestも双方で一致した。byte差異がないためdiffoscope相当の原因分類は不要で、分類は`none`である。
- 同じtarget/toolchain/逐次条件でresident driver unit testsをCPU実行し、22 passed、0 failedだった。
- canonical packageのmode 0555とCargo build outputのmode 0775には包装上の差があるが、ELF bytesは同一である。canonical packagingがbuild outputを0555へ封印する契約と整合する。
- 監査後に一時worktreeとtargetを削除した。canonical binary、source、GPU、service、actualには触れていない。

## 次の行動

- canonical resident driverは現在の記録済みtoolchainとexact source/Cargo.lockから再現可能と判定する。
- 今回は差異がないため、ELF差分調査やcanonical binaryの再生成は不要である。
