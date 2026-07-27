# R9700 CO hardware measurement

This directory is the R9700 (`gfx1201`) half of the RDNA4/CDNA3 comparison.
The binary was rebuilt here from `tools/hw-microbench-rdna4-cdna3.hip.cpp`, and
the accompanying ISA audit passed before timing. `HIP_VISIBLE_DEVICES=1` maps
to the physical R9700; the benchmark identifies the selected runtime target as
`gfx1201`, never `gfx1030`.

`bandwidth.jsonl` and `gemm.jsonl` are the source of the parent comparison
table. They use 256 MiB per STREAM vector, dense `2MNK` FLOPs, five warmups,
11 median samples, 10 launches per sample, and HIP event timing only.
`window-wall-clock.txt` records 39 seconds including build and ISA audit;
`runtime.txt` records validate/bandwidth/GEMM group wall times.

The validation executable completed before bandwidth and GEMM, and would have
aborted the wrapper on either BF16 CPU-reference or FP8 OCP-E4M3FN mismatch.
The source revision used in this window wrote its numeric validation line to
stdout rather than `validate.jsonl`; the later successful bandwidth/GEMM files
therefore prove pass-by-control-flow but do not preserve the two max-abs values.
The source has since been corrected to write that line to the requested output
stream for the MI300X fill-in run.

The six `telemetry-*.json` files contain an `amd-smi metric -j` option error.
They are retained as honest evidence, not treated as telemetry. Window-level
R9700 metrics immediately before and after the hardware phase are stored in
`../../sq8-grouped-tile-sweep/co-window/telemetry/` and summarized in the
parent comparison: before edge 45 C, GFX 234 MHz, 13 W; after edge 48 C,
GFX 430 MHz, 13 W. The wrapper now uses `amd-smi metric --json` for valid group
telemetry on the future MI300X run.
