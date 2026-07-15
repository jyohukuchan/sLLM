# Canonical QA source provenance repair

## Scope

- normal/profile readyのQA attestationに記録されていた不存在commitを修正した。
- 誤: `c743007f9748d2baf6d699744f7dad4fd3b1cd21`
- 正: `c743007f97486e7c7e070f4258ce4e98f0665aad`
- 対象blob: `8167859108c68fa27c67fe21c3d772e4899e384a`
- actual実行、GPU実行、サービス操作は行っていない。

## Source hardening

- strict provenance test: commit `b89a0ff683884e4d0b1014512259bce5596dd05c`, blob `2ae3b109a95770b70523a1711e57cd7969619e43`
- maintenance source: commit `c3a676a962e542b997c14a695328d5cdbfa6c120`, tree `582f2c4fee68e7251afea43cb33fea5ef4cd39e6`, blob `907eaf5590d98d2abf493127f4aa45961c4736ca`, SHA-256 `1771bc16d75a946180e38a9ebac54974b97781186ea64ca733a85fb9c3c91aa1`
- QA manifestの12ファイルすべてで、`source_commit:path`の実在と記録Git blobの一致を生成時・readback時に検証する。
- 不存在commitとblob不一致の負例を追加した。

## Official regeneration

- normal/profile readyディレクトリを削除し、存在しないことを確認してからmaintenance公式CLIで再生成した。
- normal `SHA256SUMS`: `20e2c676b7b6e6b71ee476d790b9681641d5ca9822eb03ac4991e9bb29e418c1`
- profile `SHA256SUMS`: `98e7a10aded874feddee3f937446fb73f5a987c4a3449cb7a3d2bcb10beef76e`
- 共通QA attestation SHA-256: `bd13865b433c9ec723534824e8bfe3bd8f69c8394436f1de0f0d58371e029239`
- normal ready-binding SHA-256: `a15fadb54781b748469bf8ec101a0493c18ddeb69d25c49a05858e1dcdab9ca2`
- profile ready-binding SHA-256: `6550cc2e740dfbc153de10c977b7ea807c3c3434e8939e09c08d74497eccc21c`

## Execute binding

- execute-bindingはready artifactのcommit/treeを参照しない独立launcher trustであるため、再生成していない。
- 前後で3ファイルのSHA-256が不変だった。
  - `SHA256SUMS`: `c2dbe6c9a27b7a243d08388d4b8cd8b687f20172ae33e8d5a2f7b8a79f6618d2`
  - `execute-binding.json`: `2c613a6950f0a4631ae47f7dffadc7d6c0abc443bcb3bcce3eea147585a87b66`
  - `launcher-trust.json`: `d897a6b64c418a9ceebfc6f4fcbd6ee8ccf494a9e850bee960ba329ff55d3e22`

## Verification

- QA attestation記載の全suite: `468 passed`（Python 452、Rust 16）
- normal/profileで12/12の`source_commit:path`とGit blobが一致した。
- QA attestation SHA、ready-binding内のattestation pin、harness-trust内のready-binding pinとsource pinが一致した。
- normal/profileの`SHA256SUMS`は全メンバーで成功した。
- profile v3のrun、evidence、maintenance evidence、dry-run evidence、rocprof captureの5出力はすべて不在である。
