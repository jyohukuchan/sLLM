# SQ8 P1 source-correct canonical artifact

## 前回の要点

旧`SQ8_0` v0.1 artifactは、Qwen3-14B-FP8のraw F8 weightをsourceの
`weight_scale_inv`なしで再量子化していた。このartifactはsource checkpointと数学的に
異なるため、P0で既存結果を隔離し、P1ではsource payloadと2D block scaleをbyte-exactに
保持することを受入条件にした。

## 今回の変更点

- `sq-fp8-artifact-v0.2`をsource-correct canonical schemaとして追加した。
- safetensorsの`F8_E4M3`領域とBF16 `weight_scale_inv`領域を、F32変換や再量子化をせず
  bounded chunkでコピーするPython importer/verifierを追加した。
- v0.2のblock shapeをQwen3-14B-FP8 source contractに合わせて`[128, 128]`へ固定した。
- canonical artifactとGPU固有のtranspose、padding、swizzle、prepackを分離した。
- Rust readerにtyped manifest validation、全payload checksum/有限値検査、row/block復元を
  追加した。read後の同サイズ改変も復元前の再検証で拒否する。
- 旧v0.1 builderは選択された`F8_E4M3` sourceをfail-closedで拒否する。BF16/F16からの
  legacy再量子化経路は維持した。
- source/outputの同一・親子関係、symlink escape、不正な既存出力先を拒否し、検証済み
  v0.2だけをLinux `renameat2(RENAME_EXCHANGE)`でatomic overwriteできるようにした。
- 既存artifactは検証前後・exchange直前・exchange後に`st_dev/st_ino`を照合する。途中で
  出力先が差し替えられた場合は無関係なdirectoryを削除せずrollbackする。初回promotionは
  `renameat2(RENAME_NOREPLACE)`を使い、途中で現れた出力先を置換しない。
- exchange後の競合entryが通常ファイルやsymlinkでも型に依存せずrollbackし、両側のinodeを
  確認する。cleanup対象は新規0700 quarantineへatomicに移し、所有inode一致時だけ削除する。
- safetensors headerのduplicate JSON key、offset overlap/gap/trailing bytes、indexとshardの
  不一致、JSON booleanを数値として扱う入力を拒否した。
- Python生成artifactをRust readerで読む相互運用testと、実checkpoint goldenを追加した。
- Rust readerは宣言byte数の読込前後に同じopen fileをstatし、宣言領域後のEOFも確認する。
  artifact読込後にpayloadが追記された場合も、checksumと復元の両方で拒否する。

## Full artifact結果

source:

- model: `Qwen3-14B-FP8`
- config SHA-256: `c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793`
- index SHA-256: `6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151`

artifact:

- path: `/tmp/ullm-qwen3-14b-fp8-sq8-canonical-full-v0.2`
- source tensors: `723`
- FP8 weight/scale pairs: `280/280`
- unpaired tensors: `0`
- passthrough tensors: `163`
- weight payload: `13,212,057,600 bytes`
- scale payload: `1,612,800 bytes`
- manifest SHA-256: `23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2`
- canonical content SHA-256: `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`

layer0 q projection golden:

- raw weight SHA-256: `55b69346ebbd88c2946655d91617ee2abcaddd0942ca4835349df648063894ca`
- raw scale SHA-256: `297ad4da1210a22ac5114cc6a4811e8141014f7ebf48e613e7705285e669e948`
- sampled 5-block reconstruction SHA-256: `a33d5e3d995a32597855384a191d110030b7a995d54e35029872952e1a083ac3`

## 検証

- Python compile: 成功。
- Python unit/policy: `33 passed`。
- 実checkpoint goldenとPythonからRustへの相互運用: `4 passed`。
- `cargo test -p ullm-engine -- --test-threads=1`: lib `151 passed`、main `36 passed`。
- `cargo fmt --all --check`: 成功。
- `cargo check -p ullm-engine --example verify_sq8_canonical`: 成功。
- Python full artifact verify: `11.81 s`、最大RSS `28,580 KiB`、全280 pair成功。
- Rust release full artifact verify: `32.71 s`、全280 pair成功。
- Python/Rustのfull artifact content SHA-256は一致した。
- debug buildでの全量Rust検証は純Rust SHA-256が遅いため中断した。focused testと26 MBの
  実artifact検証はdebugでも成功しており、全量はrelease readerで完走させた。

P1の受入条件であるbyte-exact payload/scale、source reconstruction一致、完全pair accounting、
bounded memory、atomic output、Python/Rust reader一致は満たした。GPU実行や性能の主張は
含まない。commitは`dd35c01 Add source-correct SQ8 canonical artifacts`。

## 次の行動

P2ではcanonical block-2D scaleを使う小さなCPU linear oracleを実装し、同じ入力を使う
source-correct GPU reference pathとtyped correctness reportを追加する。一つのq/o projectionが
固定済み許容誤差を通るまで、dynamic W8A8最適化とfull-layer統合へ進まない。
