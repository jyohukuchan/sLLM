# AQ4 Phase 3d: layer 0--31, final norm, and LM-head boundary analysis

## Valid CPU-only measurement

- Status: `valid`; `layer 0:31` was measured as one contiguous chain.  The
  diagnostic rejects start > 0 because such a split would not have the true
  preceding decoder hidden state.
- All eight self-attention layers were included: `3, 7, 11, 15, 19, 23, 27,
  31`.
- Final RMSNorm is a full-hidden comparison.  LM head uses 34 fixed AQ4 rows
  (`0..31, 220, 41330`) and is explicitly not a full-vocabulary L2.
- `time -v`: wall `408.41 s` (6:48.41), max RSS `330744 KiB`, process swap
  operations `0`.  This is below the preflight 20-minute / 512 MiB envelope.
- Inputs and report bindings are in `compare/comparison.json` and
  `aq4-chain/aq4-report.json`; their SHA-256 values are respectively
  `daf9f44edaf5e77315bc107244cd29679f1e2900919c7fbcf20250cc7ee842df` and
  `abeb41d71e74bc50f6e231a664330500f0e8d685819dd124eac5c8b9c6e078f5`.

## Growth curve

The plot-ready CSV is `compare/growth-curve.csv`; the same data are rendered
in `compare/growth-curve.md`.

| stage | family | relative L2 | cosine | max abs |
| --- | --- | ---: | ---: | ---: |
| layer 0 | linear attention | 0.042451 | 0.999107 | 0.069627 |
| layer 1 | linear attention | 0.075076 | 0.997375 | 0.174330 |
| layer 2 | linear attention | 0.092594 | 0.995869 | 0.253928 |
| layer 3 | self attention | 0.106254 | 0.994378 | 0.202241 |
| layer 4 | linear attention | 0.119419 | 0.992886 | 0.466560 |
| layer 5 | linear attention | 0.125536 | 0.992172 | 0.557333 |
| layer 6 | linear attention | 0.077143 | 0.997134 | 1.431293 |
| layer 7 | self attention | 0.094488 | 0.995626 | 1.429813 |
| layer 8 | linear attention | 0.094775 | 0.995630 | 1.403173 |
| layer 9 | linear attention | 0.092623 | 0.995813 | 1.345047 |
| layer 10 | linear attention | 0.074961 | 0.997391 | 2.475082 |
| layer 11 | self attention | 0.080827 | 0.996919 | 2.402580 |
| layer 12 | linear attention | 0.082044 | 0.996822 | 2.437775 |
| layer 13 | linear attention | 0.080715 | 0.996935 | 2.471966 |
| layer 14 | linear attention | 0.077336 | 0.997172 | 2.444794 |
| layer 15 | self attention | 0.086443 | 0.996394 | 2.267082 |
| layer 16 | linear attention | 0.090945 | 0.995938 | 2.238621 |
| layer 17 | linear attention | 0.096662 | 0.995381 | 2.109135 |
| layer 18 | linear attention | 0.086750 | 0.996378 | 2.433884 |
| layer 19 | self attention | 0.141563 | 0.989972 | 2.836046 |
| layer 20 | linear attention | 0.148977 | 0.988851 | 3.186935 |
| layer 21 | linear attention | 0.150603 | 0.988610 | 3.164642 |
| layer 22 | linear attention | 0.121062 | 0.992646 | 14.237411 |
| layer 23 | self attention | 0.147672 | 0.989042 | 7.387459 |
| layer 24 | linear attention | 0.144614 | 0.989491 | 8.158051 |
| layer 25 | linear attention | 0.131568 | 0.991308 | 7.411285 |
| layer 26 | linear attention | 0.082310 | 0.996785 | 25.694946 |
| layer 27 | self attention | 0.158523 | 0.987408 | 6.456627 |
| layer 28 | linear attention | 0.156894 | 0.987690 | 7.032410 |
| layer 29 | linear attention | 0.170875 | 0.985320 | 6.312180 |
| layer 30 | linear attention | 0.151532 | 0.988456 | 5.233986 |
| layer 31 | self attention | 0.127881 | 0.991806 | 15.799225 |
| final norm | RMSNorm, full hidden | 0.501033 | 0.965215 | 24.646279 |
| LM head | AQ4 fixed 34 rows | 0.586050 | 0.969689 | 8.347778 |

## Boundary conclusion

- Decoder growth is nonmonotonic, not a depth-accumulation curve: its maximum
  is `0.170875` at layer 29 and it returns to `0.127881` by layer 31.  The
  complete 32-layer observation is only 20.8% of the prior known production
  final relative L2 `0.6151289249`.
- The clear boundary is **decoder layer 31 output -> final RMSNorm**.  Full
  hidden relative L2 jumps by `+0.373152`, from `0.127881` to `0.501033`
  (`3.917953x`).  Final norm alone reaches 81.45% of the known production
  final value.
- The 34-row AQ4 LM-head sample then reaches `0.586050` (95.27% of the known
  production scalar), but this number is a fixed-row sample and must not be
  represented as a full-vocabulary L2.

## Tensor-family assessment

- Self-attention layers have local, nonpersistent jumps (18->19:
  `+0.054813`; 26->27: `+0.076212`), but both are followed by declines and do
  not produce monotonic growth toward `0.615`.
- The only dominant, full-hidden boundary is norm-specific.  The source
  Qwen3.5 final RMSNorm formula is `normalized * (1 + raw_weight)`; the AQ4
  chain applies `normalized * raw_weight`, matching the AQ4 runtime helper's
  suffix policy.
- `final-norm-weight-semantics.json` proves that source and package raw BF16
  payload bits are identical (SHA-256
  `44f7283137ae75c262c152f7e529b70c708ea13afc1bfaa565c8ea74b61ecf88`).
  The observed discrepancy is therefore a final-norm weight interpretation
  issue, not a package-payload mismatch.  This Phase 3d evidence does not
  implement a production fix.

## Integrity checks

- Contract validation asserted 32 ordered decoder summaries, all eight
  self-attention indices, `final_norm` full-hidden records, and `lm_head`
  fixed-row records (nine timesteps each).
- `sha256sum -c compare/SHA256SUMS` passed for `comparison.json`,
  `growth-curve.csv`, and `growth-curve.md`.
- Earlier invalid attempt directories are preserved: attempt 1 stopped before
  terminal output due to a passthrough/AQ4 LM-head reader mismatch; attempt 2
  exposed and then fixed comparator terminal-frame ordering.  Neither is used
  for the values above.
