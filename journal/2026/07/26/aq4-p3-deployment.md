# AQ4_0 P3 deployment

## 前回の要点

- 本番の `AQ4_0` worker は source `0cd76056…` / manifest `c57a2b6…fca4` に留まり、P3 の prefill/
  decode kernel 改善は未デプロイだった。
- policy は実生成品質を軽量判定に使う方針へ変更済みであり、top-1/logits/FP32 reference/campaign を
  昇格 gate に使わない。昇格と復帰には generic lightweight tools を使う。
- 既存 delta analysis は P3 endpoint `c4c9a9b…` を、動く shared `HEAD` ではなく安定 source として
  推奨していた。

## 今回の変更点

- detached `c4c9a9b…` worktree から P3-only worker と timing binaries を build し、fresh release
  `/opt/ullm/aq4-p3-deployment-v0.1/` に stage した。worker SHA-256 は
  `ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265`、候補 manifest SHA-256 は
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` である。active と同一 product
  package manifest `a790a033…e5ad` を維持し、P3 36-guard contract を検証した。
- R9700 (`gfx1201`) の isolated direct timing は prefill **970.6107 tok/s**（historic 982.3835 の
  -1.198%）、decode **73.4568 tok/s**（historic 74.29 の -1.122%）だった。熱条件/同居プロセスが
  異なるため厳密同値とは主張しない。historic 56.6% efficiency の raw denominator は未確認であり
  gate に用いなかった。
- Qwen3.5-9B config を独立確認し、32 layer と `linear_attention×3 → full_attention` の8回反復を
  確認した。`b21b2723` / `b3d78b42` の config-driven loader は AQ4 path に届くが、P3-only binary
  には混ぜていない。SQ8 v2 shared runtime は AQ4 に届き得るため「完全に無関係」ではないが、選択
  source より後であり候補から除外した。
- 初回 generic promotion は baseline third request 中の他 session `systemctl stop` に中断された。
  tool は `baseline_failed_before_mutation` で fail-closed となり candidate は active にならなかった。
  BH の R9700 lock と別 session の StartLimit 競合も検出したため、lock 解放と safety window を待った。
- 再試行は active old text 10件と candidate text 10件を保存して成功した。比較は blocking 0、各 category
  の nonempty text を確認。exact output match 1.000 は診断値であり判定基準にはしていない。generic tool
  の service operation は成功した restart 1回、StartLimit recovery 0回だった。
- 新 active manifest は `a98910dc…fe49`、gateway は active/running / `Result=success` /
  `NRestarts=0`。rollback no-`--yes` preflight は `ready: true` で、旧 `c57a2b6…fca4` へ戻せる状態を
  確認したが、rollback は不要なので実行していない。
- activation 後に BJ の SQ8_0 projection measurement が gateway を一時停止した。candidate manifest
  は不変で、BJ restore 後に `/readyz` HTTP 200、新 P3 worker path / SHA `ba8c46…e265`、
  active/running / `Result=success` / `NRestarts=0` を最終確認した。
- その後 22:35:50 JST に BJ が second `--speed-first` SQ8_0 window を開始し、22:35:52 から R9700
  lock と gateway stop を保持している。22:41 JST handoff 時の manifest は `a98910dc…fe49` のまま、
  gateway は意図的に inactive/dead だった。AQ4 deployment は競合 start を行わず、この window の
  restore trap に委ねた。これは P3 activation の失敗や rollback ではない。

## 次の行動

1. P3 candidate を再昇格しない。問題が出た場合だけ
   `tools/rollback-promoted-served-model.py` を activation outcome に対して実行する。
2. 速度の厳密 A/B を必要とするなら、同じ温度・clock・resident load を明示的に揃えた別の R9700
   measurement window として行う。今回の 970.6107 / 73.4568 は実測として保持する。
3. `AQ4_0` P3 後の config-loader、SQ8 shared runtime、AQ5、Gemma 作業はこの release へ混ぜず、別の
   candidate として評価する。
