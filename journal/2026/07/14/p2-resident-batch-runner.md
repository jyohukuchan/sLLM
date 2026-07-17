# P2 resident batch runner

## 前回の要点

- P2 の representative expanded matrix には、R9700/RDNA4 の `full_model`、`cold_prefill`、`aq4_0_target` が 84 ケースある。
- 既存の単ケース経路では model load をケースごとに繰り返すため、同一 resident session の証跡として不足していた。
- active production worker の identity（source commit `ae8b2bb…`、served manifest SHA `feb3190d…`）と、P3 current-head diagnostic は同一 baseline として扱わない必要がある。

## 今回の変更点

- `tools/run-aq4-p2-resident-batch.py` を追加した。expanded manifest と fixture index の自己ハッシュを検証し、対象 84 ケースを case ID 順に選択する。
- resident driver の標準入出力プロトコルを定義した。driver は最初に `model_loads=1` を宣言し、各ケースについて 2 warmup + 10 measured を直列実行する。各 run の reset 完了、terminal audit digest、resource samples/peak、actual batch width を検証する。
- ケースごとに raw JSON を atomic write し、fixture・identity・policy の hash link、resident session、reset 集計、OOM の immutable status を記録する。OOM はそのケースを保存した後、残りのケースを実行せず中断する。
- `tests/test_run_aq4_p2_resident_batch.py` に fake driver を追加し、84 ケース選択、one model load、12 run/reset、atomic raw output、OOM 中断を CPU のみで検証する。
- dry-run の実測計画値は 84 ケース、1,008 transactions、prompt tokens 1,389,024、resident model load 1 だった。`production_server` 48 ケースはこの direct full-model driver と混ぜず、active gateway を 1 ケースずつ通す別 runner とする。gateway 側の trace/resource sidecar と requested/resolved/actual M を同じ粒度で検証し、M を固定できない場合は unsupported と記録する。

## 所要時間の見積もり

- active product の既存目安 117–129 tok/s を仮置きした場合、1,389,024 prompt tokens の full-model target は約 2.99–3.30 時間（1,389,024 / 129 ～ 1,389,024 / 117）で、reset・resource sampling・atomic write の overhead を別途見込む。これは現在の P1 trace にこの resident matrix の有効な op wall-time がないため、R9700 の保証値ではない。今回の dry-run は時間を測定せず、構造上の下限（model load 1 回、case transaction 1,008 回、各 transaction 後の reset）だけを固定した。
- gateway 別 48 ケースは 576 transactions、prompt tokens 780,768。full-model と合わせると 1,584 transactions、2,169,792 prompt tokens になる。

## 検証

- `pytest -q tests/test_run_aq4_p2_resident_batch.py` — 3 passed。
- `python3 -m py_compile tools/run-aq4-p2-resident-batch.py tests/test_run_aq4_p2_resident_batch.py` — 成功。
- `git diff --check` — 成功。
- real GPU、常駐 worker、gateway は起動・停止・変更していない。

## 次の行動

- 実 GPU 実行時は、active production identity 用 run ID と P3 current-head diagnostic 用 run ID を分離して渡す。resident driver の実装をこのプロトコルへ接続し、resource observer の sidecar schema を追加検証する。
- `production_server` 48 ケースは active gateway runner を別実装し、direct full-model raw と混在させない。

## active gateway runner と policy 判定

- tools/run-aq4-p2-active-gateway-batch.py は direct resident runner と分離した HTTP 境界を担当する。readyz と identity endpoint を初回および case ごとに再取得し、served manifest SHA、worker SHA、guard SHA、PID/starttime、runtime device、M capability を束縛する。stream/nonstream、fixture SHA、trace/resource evidence、release/reset、429/HTTP failure、固定 M の unsupported を raw に保存する。
- source-vs-AQ4 bounded comparison の exact greedy/top1 は、same-artifact path 回帰だけの要件ではない。source/path oracle link の validator は exact greedy と exact top-k を要求し、candidate source gate も greedy mismatch 0 / greedy_tokens_exact を要求する。candidate top-k は bound policy の minimum overlap で判定する。
- AQ4 P2 threshold template は unbound_template / planning_only で、正しさの L2・max-abs・top-k threshold、power 値、hash binding が null である。したがって観測値から閾値を逆算せず、bound policy がない現時点の resident 84 / gateway 48 live execution は blocked とする。
