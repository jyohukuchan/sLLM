# Qwen3.5 AQ4差分トレース v2

## 前回の要点

v1のsource差分トレースは3 rowsと固定座標を保持していたが、step 1のcontext hashが直前replay tokenだけを対象にしていた。

## 今回の変更点

- source captureとAQ4専用binを、`prompt_token_ids + replay_token_ids[..step]` のfull-context規約へ統一した。
- rowのcontextは、step 0が`[11,12,13]`（長さ3、`42ea52c7…`）、step 1が`[11,12,13,220]`（長さ4、`6af1601b…`）、case 1が`[21,22]`（長さ2、`3bca9e21…`）となった。
- v2 CPU source traceは3 rows、greedy `[220,16,15]`、最大RSS約15.3GiB、単一プロセス・1スレッドで再生成した。旧v1は`invalid_superseded`としてv2 manifestと`SUPERSEDED.json`へ記録した。stage値は旧v1と一致し、context bindingだけが修正された。

## 次の行動

AQ4 GPU中間トレースは、measurement laneの固定HEAD再reviewと排他/service復旧証跡の確認後に実行する。旧v1のcontext hashを比較対象へ再利用しない。
