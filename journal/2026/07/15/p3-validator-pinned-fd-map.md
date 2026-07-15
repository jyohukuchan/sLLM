# P3 validator pinned FD map

## 目的

A4 profile 実行で論理 argv/path を維持したまま、trusted bundle validator が capture 時に保持された FD だけから bundle control bytes を読むようにした。

## 実装

- `ULLM_AQ4_PINNED_FD_MAP` の sealed memfd を最大 1 MiB で `pread` し、共有 file offset を変更しない。
- map root と9-key binding schema（`resolved_path` を含む）、canonical bytes、self-hash、4 seal、role/path/FD 一意性、closure/method、identity、content SHA-256を fail-closed で検証する。
- bundle member bytes は logical path で binding を引き、保持 FDを `pread` する。元の member path は content read のために再 open しない。
- `bundle_root` は保持 dirfd に対する `listdir` と `stat(..., dir_fd=..., follow_symlinks=False)` で検証し、logical root path と保持 FD の7項目 identityを開始時と終了時に照合する。
- FD map がない通常の `validate` は従来の path-based validation を維持する。

## runner provenance

prepared-v1 の `trusted-runner.py` は prepared dry-run/evidence を生成した履歴固定 control memberであり、profile の actual code execution targetではない。この bootstrap pinは `3dc4aa...` / `e7dae31c...` のまま維持した。

binding-v4 の `trusted-runner.py` は actual generic runnerであり、A4基盤 `ede2b872...`、blob `7adc7f...`、raw SHA-256 `0b3e55f...` へ更新した。binding manifest の `runner_roles` が両者を明示的に分離し、同一 runner とは主張しない。

## 再生成と検証

- official `prepare` で prepared-v1 を fresh 再生成し、既存 canonical bytesと同一であることを確認した。
- authoritative validator commitはcombined integration commit `ec257544...`、tree `ce304dde...`、source blob `13de5f3d...`、source SHA-256 `c14bff3c...` である。このcommitはvalidator source修正に加え、独立したP2 fused GPU probe attempt4 closeout 17 pathsを含む。両者のprovenanceと目的は分離されており、attempt4はA4/P3の実行証拠やpromotion判断へ流用しない。
- 上記validator commit/SHA-256を使い、official `prepare-binding` で binding-v4 を再生成した。
- `validate` と `validate-binding` の canonical readbackは成功した。
- prepared-v1 と binding-v4 の `sha256sum -c SHA256SUMS` は全 memberで成功した。
- binding evidenceは runner subprocess 1、trusted validator subprocess 1、fake driver subprocess 1を記録する。
- `tests/test_prepare_aq4_p2_resident_smoke_bundle.py`: 63 passed。

GPU command、モデルロード、service操作、actual runは実行していない。
