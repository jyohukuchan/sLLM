# SQ8 P2 reference correctness

## 前回の要点

P1ではQwen3-14B-FP8のraw F8 weightと128x128 `weight_scale_inv`をbyte-exactに保持する
canonical artifactを実装した。P2では、このartifactを旧row-block scaleとして誤読せず、
1 projectionでCPU oracleとGPU referenceが一致することを受入条件にした。

## 今回の変更点

- C ABIとRust wrapperに明示的なblock-2D scalar/batch APIを追加した。
- scale indexを`(row / block_rows) * ceil(cols / block_cols) + col / block_cols`に固定した。
- 旧tensor/row/row-block APIは内部で`block_rows=1`を渡し、ABIと挙動を維持した。
- 新HIP APIはnative kernel失敗時にruntime errorを返し、host stagingへfallbackしない。
- runtimeは`CpuReference`または`HipKernel`を型で返す。
- canonical weightを1 MiB bufferで読み、compact BF16 scaleとF64 accumulatorを使う独立CPU
  oracleを追加した。F32 weight matrixは生成しない。
- artifact readerに対象pair/path/checksum/compact scaleの安全な公開APIを追加した。
- resident loaderはweightを16 MiB以下のchunkで再hashしながら転送し、scaleは行展開せず
  compact F32として転送する。
- typed correctness reportにartifact/input hash、profile、execution path、fallback、nonfinite、
  MSE、最大絶対誤差、relative L2、cosine、固定閾値、判定を保持した。

## 固定入力とgate

- input: `x[k] = (((73*k + 19) mod 257) - 128) / 256`
- 5120要素F32-LE SHA-256:
  `93f05449d07327c1237992938233030f1058dbe965504e343c8ae656dbe2e781`
- max absolute error: `2e-5`以下
- relative L2: `1e-5`以下
- cosine similarity: `0.999999`以上
- nonfinite: `0`
- HIP fallback: 不可

閾値とCPU oracleはGPUの正式結果を取る前に`b4b4e9b`で固定した。block-2D runtimeは
`aaef5a3`で先に固定した。

## 実Qwen projection結果

- artifact content SHA-256:
  `29857be65d162ca1150f91e5c159186a0086d0f2eb463b2b113d92db4acd5c6a`
- tensor: `model.layers.0.self_attn.q_proj.weight`, `5120 x 5120`
- scale: `40 x 40`, block `128 x 128`
- oracle output SHA-256:
  `24661953d983d9532ef4c3413a420e73f139604762cc07e68eebd09da5b53469`

CPU runtime:

- execution: `cpu_reference`
- max abs: `4.291534423828125e-6`
- relative L2: `7.800539426079679e-7`
- cosine: `0.9999999999996875`
- result: pass

R9700 runtime enumeration index 2 / HIP device ID 1:

- execution: `hip_kernel`
- fallback: `not_used`
- max abs: `2.384185791015625e-7`
- relative L2: `8.499422347560135e-8`
- cosine: `0.9999999999999909`
- result: pass

正式JSONは
`uLLM-project/benchmarks/results/2026-07-10/sq8-reference-correctness-v0.1/`に保存した。
この結果はscalar W8A16 referenceの正しさだけを示し、matrix-core FP8性能は主張しない。
resident loader、実行CLI、正式reportは
`5cda189 Validate SQ8 block-2D reference execution`で固定した。

## 次の行動

P3でR9700/gfx1201のhipBLASLt、Composable Kernel、rocWMMAを順にcapability確認し、実projection
shape、fallback無し、profiler上のmatrix instructionを満たす最初の経路を選ぶ。対応経路が無い
場合だけ、boundedなdirect HIP kernel作業へ進む。
