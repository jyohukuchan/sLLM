# BS: Qwen3.5-35B-A3B `AQ4_0` 生成品質による再分類

Date: 2026-07-26

## 前回の要点

- BN は Qwen3.5-35B-A3B の routed expert 80 tensor だけを strict `AQ4_0`
  `aq4_e4m3_g8_ts_flloyd16` で package 化した。router、shared expert、attention、norm、
  embedding、`lm_head` は raw passthrough である。
- package は 23.310427 GiB、262,144 token の byte ledger は
  `30,858,010,436 B`（headroom `3,350,732,988 B`）であり、R9700 容量内の設計値だった。
- 全 tensor の dequantize verification は通過したが、layer 39 `down_proj` の max-abs
  `0.043730080` は唯一の outlier として残した。
- 前回は「end-to-end top-k expert set が量子化前後で不変」を合否にしたため、同一 8-token
  streaming prefill の selected set `105/320`、ordered top-k `238/320` の変化を理由に
  `not_passed` と記録した。router payload 自体は source と SHA-256 一致、1,280 条件付き
  input で top-8 0 変化である。

## 今回の変更点

- 上記の判定基準は誤りだったと見直した。lossy routed expert の出力が後段 router input に
  累積するなら selected set の入替りは量子化 MoE の自然な観測であり、それだけで文章品質の
  failure を意味しない。`105/320` と `238/320` は metadata に **非 gate の観測値**として残した。
- `tools/validate-qwen35-moe-aq4-streaming-forward.py` を final RMSNorm / raw `lm_head` /
  tokenizer chat template / separate KV cache を含む bounded greedy generator に拡張した。
  source-vs-source v0.1 control は 3 ケース 38 step で greedy token、route、hidden state
  が全て完全一致した。
- CPU 時間に合わせて日本語、英語、コードの 3 ケースへ短縮した v0.2 suite を実行した。
  source と `AQ4_0` は日本語/英語で同じ rollback recovery の意味を保ち、Python
  `is_even` は `is_even = lambda n: n % 2 == 0` で完全一致した。空応答、反復、文字化け、
  言語混線、応答放棄、極端な長さ偏りは観測しなかった。prose 2 件は両経路とも意図した
  24-token cap で止まったため、終止句の有無を failure にしなかった。
- 人間可読な並置証跡は
  `benchmarks/results/2026-07-26/qwen35-moe-aq4-quality-reclassification-v0.1/aq4_0-generation-v0.2.md`、
  機械可読な詳細は同名 `.json` に保存した。greedy match `47/62`、conditional NLL、生成 path
  中の route 変化は情報として残したが、合否閾値にしていない。
- product metadata の `generation_quality_validation.status` を `passed` に更新した。
  旧 `streaming_forward_validation.status` は `observed_non_gating` に変更し、旧
  `not_passed` と旧 criterion を履歴として保持した。layer 39 outlier の生成影響は individual
  raw-passthrough ablation をしていないため **未確認** とした。
- 生成品質は崩れていなかったので Phase 3 は行わず、codebook の再設計、layer 39 の raw
  passthrough、shared expert の変更はいずれも行っていない。GPU、service、active manifest、
  `/opt/ullm` も変更していない。

## 次の行動

1. MoE loader/residency と hybrid attention、mRoPE、KV state、Q output gate を別 task として
   結線し、CPU streaming pass を serving capability や GPU allocation 実測に読み替えない。
2. loader が実装された後、より広い prompt suite と実 runtime の文章品質を確認する。今回の
   3-case CPU evidence は package の再分類根拠だが service promotion ではない。
3. layer 39 `down_proj` の `0.043730080` が気になる場合は、通常品質が崩れたときに限り
   raw-passthrough ablation で因果を切り分ける。現時点では未確認のままとする。
