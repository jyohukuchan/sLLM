# Qwen3.5 9B AQ4 tiled prefill candidate rejection

## 前回の要点

BM8/BN32/BK128 の gfx1201 AQ4 tiled GEMM 候補は、当時の実験で選択された既定候補として resident smoke を実行した。比較対象は full-native AQ4 prefill resident evidence である。

## 今回の変更点

候補の raw `resident-evidence.json` は変更せず、7ケースを baseline と機械比較して候補を棄却した。prompt token IDs、generated token 列、prompt progress、`length` 終了、`reset_complete`、`clean_shutdown`、子プロセス状態は全ケースで一致した。

| prompt | candidate prompt tok/s | full-native baseline tok/s | candidate / baseline |
|---:|---:|---:|---:|
| 1 | 26.035045358 | 22.011351034 | 1.182800879338864 |
| 8 | 27.877384447 | 47.474999428 | 0.587201364564516 |
| 127 | 56.289696364 | 116.608569477 | 0.482723496362350 |
| 128 | 56.877337304 | 116.561201985 | 0.487961142604688 |
| 129 | 56.944708375 | 115.971695955 | 0.491022468080299 |
| 255 | 56.606947984 | 115.847450692 | 0.488633523191032 |
| 256 | 56.605322839 | 115.590148179 | 0.489707156970322 |

比率は raw の `timings.prompt_per_second` を丸めずに candidate / baseline として計算した。定常境界 p127, p128, p129, p255, p256 の候補比率は 0.4827234963623502555、0.4879611426046877325、0.4910224680802992618、0.4886335231910315437、0.4897071569703218524 で、候補は全境界で baseline を下回るため rejected / no promotion とした。

現在の gating/default は commit `4ab1181c58b765d0f084ea0edddfafa79c66c364`（`4ab1181`）であり、既定 dispatch は Legacy である。`ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM=1` を明示した場合だけ tiled 候補を選択する。別途行われた post-gating の p128=111.17 tok/s manual check は非 raw 注記であり、本 evidence のケース・検証には含めていない。

## 検証

- candidate raw source commit: `9a1506b71ed0b7d01e058b16015812fb188845b2`、worker SHA256: `7fcce92d12d3dea635aae14bea0e04c13622e4c299ab6a8cf30565ac3290f472`
- baseline raw source commit: `ca0e11fbddedfab3f5e65b8daf575d9b8c2f4197`、worker SHA256: `fef7f95522018ece2e5412f2444aba38ce522ff137c70aa18a3f4855ca7bc5ee`
- raw SHA256: candidate `f4b77e149c9907bbbfbaf4d06d66259c40728dc5a220b209b353112d9ef62d33`、baseline `c339c0930d7b7eacbb46a41686ede76efd451c1019cad7f205c3e77b75fa0e56`
- JSON parse と独自 assert（7件、ID、prompt IDs、generated tokens、progress、outcome、reset、clean shutdown、子プロセス）を実行した。
- 比較元 raw と `.rocprofv3/` は変更していない。`git diff --check` は summary/journal の whitespace を確認する。

## 次の行動

既定 Legacy を維持し、tiled 候補の性能改善または再評価は別の実験として gated opt-in で行う。
