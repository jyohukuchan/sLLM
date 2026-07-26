# Gemma4 E2B resident BF16

実装 commit: `0c6ae998` (`feat(gemma4): add resident BF16 text executor`)

## 前回の要点

BL の `Gemma4TextExecutor` は Transformers 5.12.1 を根拠に、Gemma4 E2B text decoder の
local/full attention、PLE、4 residual norm、shared K/V、tied head、final soft-cap を実装し、
HF と同じ greedy continuation まで確認していた。一方で source matrix を projection ごとに
stream する diagnostic path であり、weight residency と K/V cache は未実装だった。

## 今回の変更点

- 全 BF16 checkpoint payload 2,011 tensor / 10,246,357,958 B を R9700 に resident upload する
  path を `Gemma4TextExecutor::load_resident` に追加した。text-only forward でも complete
  payload を載せ、PLE と multimodal payload を含めて VRAM を会計した。
- device F32 K/V source cache を追加した。非共有 0--14 の local 12 source は 512-token
  ring、full 3 source は config maximum まで確保し、15--34 は layer kind に応じて source
  13/14 を reuse する。HF shared-state timing に合わせ local window の縮小は次 append の前に
  行う。
- M=N prefill と M=1 decode の entry point、resident weight direct matvec/row read、memory
  plan、logical traffic accounting、trace/cache/boundary driver を追加した。既存 generic HIP
  primitive のみを使い、BH/BK の編集中ファイルと `AQ4_0` / `SQ8_0` production code は変更していない。
- BL の二 prompt・各4 greedy token、cache/full-reprefill equivalence、window 512 を越える
  513-token boundary、20 shared K/V layer source mapping を R9700 で確認した。誤った physical
  K/V reproject diagnostic が別列になることも確認した。
- R9700 resident BF16 の wall-clock throughput は prefill 18.296336 tok/s、decode 15.613216
  tok/s。llama.cpp `68a5592` は Gemma4 BF16 GGUF を実行でき、同条件（F32 K/V、FA off）で
  218.955938 / 69.959983 tok/s だった。詳細は結果 artifact に保存した。

## 次の行動

- 本 task の serving 統合は行わない。必要になれば package/tokenizer/worker contract と
  quality prompt suite を別 scope として設計する。
- 現在の prefill は resident weight である一方、generic matvec を token 順に発行する。
  llama.cpp 差を縮める次の性能課題は、Gemma 固有の意味論を保った batch prefill / activation
  residency であり、BH の attention work と衝突しない別の設計判断が必要である。
- `benchmark.json`、`validation-with-vram-telemetry.json`、`sliding-boundary.json`、
  `llama-cpp-benchmark.json` と `summary.md` をこの実装の evidence として保持する。
