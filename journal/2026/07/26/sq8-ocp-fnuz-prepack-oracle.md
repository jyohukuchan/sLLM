# SQ8_0 OCP E4M3FN → FNUZ prepack oracle

Date: 2026-07-26

## 前回の要点

- canonical `SQ8_0` は OCP E4M3FN raw payload と BF16 `[128,128]`
  block dequant multiplier を保持する。gfx942 FP8 MFMA に raw byte をそのまま
  渡せないことが Phase 0 で判明していた。
- `0x80` は OCP では negative zero、FNUZ では NaN であり、変換 operand ごとの
  scale は二倍、両 operand を変換した積は四倍に補正する必要がある。
- A′（CK gfx942 XDL reuse）は、byte/scale format gate を通るまで isolated
  prototype にも進めない。physical gfx942 の数値・occupancy・性能検証も未実施である。

## 今回の変更点

- CPU-only Rust oracle `crates/ullm-engine/src/sq8_fnuz_prepack.rs` を追加した。
  `0x7f`/`0xff` は拒否、`0x80` は `0x00` へ正規化、他の有限 OCP byte は raw
  bit を維持する。256 byte 全件で `OCP = 2 * FNUZ(mapped)` を検証する。
- BF16 scale transform を fail-closed にした。one converted operand は x2、two
  converted operands の aggregate scale は x4 とし、non-positive/non-finite、
  overflow、underflow、BF16 非exact をすべて拒否する。x2 は `0x7f00` から、x4
  は `0x7e80` から overflow とした。正の有限 BF16 を x2/x4 する限り underflow
  範囲は空であることを全 bit pattern で確認した。
- `crates/ullm-engine/tests/sq8_fnuz_prepack.rs` に外部 integration test を追加し、
  256 byte、NaN/negative-zero、数値 x2/x4、BF16 境界、fail-closed payload、
  hash-checked fixture scan を実行した。`cargo test -p ullm-engine --test
  sq8_fnuz_prepack` は 6 passed / 0 failed だった。
- CPU-only scanner example `sq8_fnuz_prepack_scan` を追加した。canonical manifest
  を検証した後、全 weight/scale を再度 SHA-256 照合しながら 256-bin frequency と
  scale gate を集計する。
- 実 artifact
  `/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact`
  を 64 MiB chunk で全走査した。identity は
  `SQ8_0` / `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`、
  280 tensor、13,212,057,600 payload byte、806,400 scale だった。`0x80` は
  207,515件、`0x7f`/`0xff` は 0件、invalid scale と x2/x4
  overflow/underflow/non-exact はすべて 0件だった。scale の最小/最大は
  `0.00012493134` / `0.005645752`。
- `/etc/ullm/served-models/active.json` は read-only で確認し、現在 `AQ4_0` /
  artifact null であるため SQ8 artifact source ではないと記録した。repository の
  SQ8 profile から上記 product root を辿った。active manifest、サービス、
  `/opt/ullm`、既存 build/release tree は変更していない。GPU/R9700 は使用していない。

## 次の行動

1. format gate は通過したため、A′の FNUZ-only derived buffer を使う isolated
   gfx942 prototype を作る。ただし既存 gfx1201 body/ABI/dispatch は変更しない。
2. physical gfx942 でのみ fragment/lane map、kernel/end-to-end differential、
   occupancy/residency、partition 別の性能を検証する。今回の結果を physical
   correctness/performance の代替証拠にはしない。
3. final activation、`active.json` の byte replacement、service lifecycle は人間の
   明示承認がある別作業まで行わない。
