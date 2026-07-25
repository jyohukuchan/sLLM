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

1. Evaluate no-scale quality and explicit scale-bearing alternatives before implementing a reader.
2. Profile a direct V620/gfx1030 decoder and compare it with matched `SQ8_0` fallback traffic.
3. Keep artifacts, campaigns, releases, and activation outside this design-input change.
