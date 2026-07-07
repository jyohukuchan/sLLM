# Runtime causal attention batch cold prefill v1

Date: 2026-07-07

Command shape:

```bash
ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1 \
  target/release/ullm-engine runtime-causal-attn-batch-smoke \
  2 B N 3 16 4 256 256
```

Device:

- index: `2`
- backend: `hip`
- name: `AMD Radeon Graphics`
- scope: R9700/RDNA4

The smoke runs a real batched runtime primitive with q/k/v/output layout
`[batch, sequence, head, dim]`. It is not a full model prefill benchmark.
It isolates cold causal self-attention and reports total input token/s plus
attention pair/s.

## Results

| B | N | repeats | mean ms | min ms | max ms | total input tok/s | attention pair/s | sample diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 3 | 1.384234 | 1.373984 | 1.398394 | 92469.914769 | 5964309.502584 | 0 |
| 1 | 512 | 3 | 18.603645 | 18.093167 | 19.221686 | 27521.487903 | 7059261.647216 | 0 |
| 2 | 512 | 3 | 35.809583 | 35.532571 | 36.227503 | 28595.697680 | 7334796.455042 | 0 |
| 4 | 512 | 3 | 68.139151 | 66.070095 | 70.937661 | 30056.141879 | 7709400.392004 | 0 |
| 8 | 512 | 3 | 135.607344 | 134.453550 | 136.727181 | 30204.853802 | 7747545.000218 | 0 |
| 1 | 2048 | 3 | 274.000056 | 273.283873 | 275.220157 | 7474.451027 | 7657575.077284 | 0 |
| 4 | 2048 | 3 | 1095.987421 | 1092.576287 | 1099.089092 | 7474.538339 | 7657664.528476 | 0 |
| 8 | 2048 | 3 | 2208.166702 | 2203.859284 | 2213.431606 | 7419.729673 | 7601513.050319 | 0 |
| 1 | 4096 | 3 | 1127.648016 | 1124.018902 | 1133.346221 | 3632.339117 | 7440846.681293 | 0 |
| 4 | 4096 | 3 | 4649.452562 | 4618.483638 | 4668.964355 | 3523.855719 | 7218618.439491 | 0 |

## Interpretation

- The new runtime primitive verifies a real batch input shape for cold
  causal attention. This fills the Phase C4 batch-width component gap for
  `N=512/2048`, with an additional `N=4096` point.
- At `N=512`, total input token/s rises only from `27.5k` to `30.2k` when
  increasing `B=1` to `B=8`.
- At `N=2048`, `B=1/4/8` stays around `7.4k` input token/s and about
  `7.6M` attention pairs/s. Runtime grows almost linearly with batch count.
- This means the current implementation can measure real batched causal
  attention, but it does not yet provide meaningful batch-width efficiency
  gains. The next optimization is still the attention kernel itself:
  score reuse, tiled/block causal prefill, and better K/V read scheduling.

## Verification

- `cargo fmt --all --check`
- `cargo check -p ullm-runtime-sys`
- `cargo test -p ullm-runtime-sys causal_attn_batch -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 smokes with `ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1`
