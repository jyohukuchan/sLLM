# AQ4 layer-0 QKV runtime probe

## 前回の要点

Candidate1 の Python CPU oracle は、f32 reference を再現するだけで実際の
`ullm_runtime_aq4_matvec_f32` を呼び出していなかった。CPU context 0 の
runtime API を直接実行し、将来の GPU tensor 出力へ安全に接続できる診断
経路が必要だった。

## 今回の変更点

- `crates/ullm-engine/src/bin/ullm-aq4-layer0-qkv-runtime-probe.rs` を追加した。
  外部 `input_normed` f32 JSONL sidecar を逐次処理し、固定した layer-0 QKV
  tensor を `PackageAq4ResidentMatvec::load` と単独 `matvec` で実行する。
- package manifest SHA、QKV index/scale/codebook の実体 SHA、tensor identity と
  geometry、device info/backend、operation、fused=false、guard/effective env、
  入力 sidecar SHA と context/input SHA を report に固定した。
- package の相対パス、symlink、regular file、nlink=1、有限値、shape、重複 case、
  出力先上書きを fail closed にした。出力 f32le/report は temporary file を同期して
  no-overwrite hard link で公開する。
- 入力 sidecar は canonical path を解決して regular file/`nlink=1` を pre/post で再確認し、
  device/inode/size/mtime_ns と probe が逐次消費した SHA256 を report に記録する。
  metadata または consumed SHA の不一致は fail closed とし、置換を検出する unit test を追加した。
- `docs/aq4-layer0-qkv-runtime-probe.md` に sidecar 仕様と CPU-only 実行手順を記録した。

## 検証

- `cargo check -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe` 成功。
- `cargo test -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe` 成功（5 tests）。
- device 0 の CPU context で active package と synthetic-zero 4096 要素 input を実行。
  `output.f32le` は 8192 要素、`report.json` は `status=valid`、
  `classification=unclassified`、`fused=false`、`promotion_eligible=false` になった。
- v7 の 3-case input sidecar を device 0 で再実行し、`output.f32le` SHA256 は
  `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473`、
  `report.json` SHA256 は `1683b8ee03ba7d58f61f3f60764b5b6ef4d8e5fb624f0dd07743423ba412f019`。
  report の `input.identity` は canonical regular path、pre/post 同一 stat、`nlink=1`、
  consumed SHA256 `c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17` を含み、
  sidecar の `sha256sum` と一致した。
- GPU device、service、holdout は実行していない。

## 残課題

- GPU tolerance/holdout の承認前なので、GPU report の classification は意図的に
  `unclassified` のままにする。GPU の実行証拠はこの作業では生成しない。
- 入力 sidecar の `context_token_ids_sha256` は外部契約として固定し、probe 内で
  token 列を再構成しない。token 列まで検証する段階では別の sidecar schema が必要になる。

## 次の行動

親エージェントへ限定 commit を渡し、既存 candidate1 の Python oracle と統合する。
runtime probe の CPU report を kernel promotion の根拠には使わず、GPU tensor 出力と
承認済み tolerance gate が揃ってから比較する。
