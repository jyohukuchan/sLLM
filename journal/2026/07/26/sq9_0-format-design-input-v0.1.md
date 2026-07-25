# SQ9_0 Format Design Input v0.1

## 前回の要点

- `SQ8_0` is the existing E4M3 path, while V620/gfx1030 has no native FP8 matrix path.
- The existing AQ4 decode source establishes 16-byte `uint4` wide-load discipline, but its promoted
  width-8 shuffle reduction is gfx1201/RPB=32-specific.

## 今回の変更点

- Added the [SQ9_0 design input](../../../../docs/plans/sq9-format-design-input-v0.1.md).
- Fixed a no-scale E5M3 format with a two-plane 128-value packing layout and IEEE special-value
  behavior.
- Added and passed the exhaustive 512-pattern CPU conversion proof:
  `python3 -m unittest tests/test_sq9_e5m3_bit_conversion.py -v`.

## 次の行動

1. `SQ9_0` は保留中の future option としてこの design record を保存し、reader、quantizer、
   validator、kernel、runtime selector を実装しない。
2. V620/gfx1030 の direct decoder profiling を次の action にしない。V100 または exact RDNA1
   target の実 requirement と target 固有の capability confirmation が揃った場合だけ、別途 review
   した plan で quality と matched current-format comparison を定義する。
3. `AQ4_0` / `SQ8_0` / `SQ8_1` を当面の current scope とし、`SQ9_0` の artifact、campaign、
   release、authorization、activation を行わない。
