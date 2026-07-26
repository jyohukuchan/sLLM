# BF: config 駆動ローダーと追加アーキテクチャの入口

Date: 2026-07-26

## 前回の要点

- Qwen3 loader は `config.json` と `architectures` を読まず、Qwen3型の attention、SiLU
  MLP、RoPE、norm、embedding/head、paged KV を固定していた。
- Gemma4 E2B、Qwen3.5 dense、Qwen3.5 MoE の実 config は揃っていたが、unknown
  architecture を黙って Qwen3 として読む fail-open を防ぐ入口がなかった。
- HF trace harness は CPU reference と comparison schema を持っていたが、Qwen3-14B
  SQ8_0 の実 candidate は未採取だった。

## 今回の変更点

- `model_config` を追加し、package manifest の `source_model_dir/config.json` を
  SHA-256付きで読み、single `architectures` contract を fail-closed に解決した。
  Qwen3、Gemma4 text、Qwen3.5 dense text、Qwen3.5 MoE text の実 config field を型付き
  descriptor に組み立てる。unknown/multiple architecture、欠落 source config、未対応
  Qwen3 rope scaling は拒否する。
- Qwen3 generic loader、Qwen3-14B `SQ8_0` serving/generation loader、Qwen3.5-9B
  `AQ4_0` runtime に load 前の config/geometry contract を入れた。既存の Qwen3 rotary
  dim、MLP epsilon、数学演算、sampling は変更していない。
- Gemma4 は `Gemma4TextExecutor`、Qwen3.5 MoE は `Qwen35MoeExecutor` が未実装として
  設定組立て後に明示停止する。dense Qwen3.5 は既存 `AQ4_0` executor を許可する。
- Qwen3-14B `SQ8_0` 用 diagnostic-only trace writer を追加した。campaign、FP32 corpus、
  numerical gate、service は使わず、isolated R9700 の 1-step serving forward から
  embedding / 40 layer / final norm / logits の 43 tensor を `architecture_hf_trace.py`
  schema で保存する。GPU isolation を artifact read より先に検査するようにし、可視化の
  誤設定を重量 payload 展開前に fail-closed にした。
- 実行回帰: Qwen3.5-9B `AQ4_0` は R9700 の M=2 prefill + M=1 decode を成功し、
  deterministic input `2 produce` の次 token 491（` new`）を返した。Qwen3-14B
  `SQ8_0` trace も token 198 から greedy token 262 を返した。
- HF Qwen3-14B-FP8 CPU BF16 trace と SQ8_0 candidate は input/config/shape/top-1 262 が
  一致した。一方 strict comparison は 42/43 tensor fail であり、embedding は exact、
  因果順の最初の乖離は layer 0（relative L2 0.008560965）だった。これは strict numeric
  equality の達成ではない。SQ8_0 と FP8/BF16 reference の差を量子化と既存 executor
  差へ分解する unquantized uLLM path は未確認である。
- 検証: `model_config` 7件、`qwen3_loader` 9件、trace writer 2件が pass。実 trace と
  report は `benchmarks/results/2026-07-26/config-driven-loader-v0.1/` に保存した。

## 次の行動

- Gemma4 を着手する場合は text-only とし、mixed local/full attention、extra norm、PLE、
  tied head、soft-cap の layer trace contract を先に固定する。
- Qwen3.5 MoE は routing/gather-scatter/grouped GEMM/weighted reduction/shared expert の
  runtime primitive が必要であり、config descriptor だけで executor に進めない。
- Gemma4 48--72 h、Qwen3.5 MoE 72--120 h の見積りは変更しない。まず unquantized
  Qwen3 diagnostic path を用意できれば、今回局在化した SQ8_0-vs-reference 差を
  quantization と executor semantics に分けて確認できる。
