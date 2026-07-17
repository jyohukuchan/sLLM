# SQ8 P8-C KFD v2 rerun

日付: 2026-07-11

## 前回の要点

P8-C正式run 1回目はcancel latencyとresource request 1..86の全5 samples、request 87のsample 0まで合格した。request 87のsample 1で、短命KFD PID directoryが列挙後に消え、v1 producerが`ENOENT`でfail-closed停止した。

## 今回の変更点

- v1 incomplete evidence: `/home/homelab1/datapool/ullm/evidence/sq8-worker-acceptance-acbc60e-2026-07-10`
- v1 incomplete raw SHA-256: `cce782ebda91132a0816267c8a43be29b989604a7b85024e60f3548646ff29a5`
- v1 incomplete stderr SHA-256: `a67cd901880abdba07ab2c56d50de17d21966ea8391a4a47f75a8e61838bbc00`
- v1の`raw.jsonl` は公開されず、worker/process group/R9700 VRAMは回収済み。
- v2 commit: `4e627bc537ce493cbe6a7387144229331d943b03`
- v2 release worker SHA-256: `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950`
- schema: `ullm.sq8.worker_acceptance.raw.v2` / `ullm.sq8.worker_acceptance.validation.v2`
- KFD観測を全attempt raw付きの1秒stable double-collect intervalへ変更した。PID directory identity、partial read、required worker、他processの正VRAMをretryで隠せない。
- producer 56 + validator 42 = 98 tests、`py_compile`、`git diff --check`、独立監査が合格。
- 実機で`amd-smi process` -> KFD snapshotを600回繰り返し、600/600合格。

## 次の行動

1. P8-Dで専用Python packageとoffline tokenizerを実装する。
2. 常駐workerを1個だけ管理するfail-closed supervisorと、待ち行列を持たない非stream OpenAI APIを実装する。
3. HTTP経由の日本語応答を確認してからP8-EのSSEへ進む。

## 正式run結果

- 出力先: `/home/homelab1/datapool/ullm/evidence/sq8-worker-acceptance-v2-4e627bc-2026-07-11`
- 独立validation: 合格、gate errorなし
- raw SHA-256: `fb52f5e172196d8ccc4b41caa7c0ff6aede9f41e32f09d0526a874b93722a8ab`
- worker stderr SHA-256: `305eb3d3e1009f7ce05f0e956c3f735c4f3c4da4e34d87a2c854d3d56f51ea3c`
- validation SHA-256: `1f5674ee9f8c67386769bf8132cbc9863e6c1a76b71b4bdac377a00c317a8a4c`
- measured cancel p95 upper bound: `145,297,962.95 ns`、全34 cancelの最大値: `298,216,883 ns`
- resource: 100 requests / 505 samples、最終VRAM差分0、最終RSS差分0、両Theil-Sen slope 0
- KFD: 641 snapshots、retry 0
- worker: exit code 0、shutdown `62,034,509 ns`
- repo内証跡: `benchmarks/results/2026-07-11/sq8-worker-acceptance-v0.2/`

P8-Cは完了した。最初のv1停止は製品側の失敗ではなく、fail-closedな観測器が短命PIDを扱えなかったことが原因だった。v2では観測の透明性と禁止条件を維持したまま解消し、再実行で製品側のcancel・回復・資源安定性を確認した。
