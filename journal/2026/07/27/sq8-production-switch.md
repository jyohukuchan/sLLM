# 依頼CF SQ8_0 本番切替: quality hold

## 前回の要点

- `SQ8_0` の decode grouped tile-20 と adaptive prefill は、BQ 時点で品質 hold のまま
  だった。BQ の 96/128 token 上限では direct 側にも途中終了があり、tile-20 起因かは
  未判定だった。
- 現本番は `AQ4_0` manifest
  `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`。

## 今回の変更点

- shared worktree を動かさず、開始時 main `c5e7dc16` の別 worktree から single SQ8 worker
  build を作成した。BH `3d914439`、BR `b46cc8ac`、BK `17a531a2`、CC `1c660223`、BX
  `8412e170` の祖先を確認した。F32 KV default を維持し、FP16 / S1E4M3 は有効化していない。
- code と日本語 multi-turn を含む 8 prompt を direct と grouped tile-20 に、最大 512 token
  で実行した。両方が 8/8 自動 blocking なしでも、grouped の JavaScript 説明だけが
  `Infinity`/`NaN` の truthiness を誤った。direct は正しかった。上限を上げても grouped
  単独の内容破綻が残ったため、交絡ではなく tile-20 の quality failure と判定した。
- decode は 27.394198 tok/s。prefill N=4095 は 126.761 tok/s、N=128 は一発同期 sample で
  426.744 tok/s だった。N=128 は CC の five-sample median protocol を再現できなかったので、
  887 前後との合否比較には使わない。
- SQ8 release を `/opt/ullm` に置かず、manifest promotion も行わなかった。AQ4 active
  manifest は不変で、復旧済み。
- 実測隔離の cleanup が lock 解放より先に AQ4 start を試み、`WorkerBusy` から
  `NRestarts=3` / start-limit になった。lock 解放後 `reset-failed` と 1 start を実施し、
  最終 `ActiveState=active`, `NRestarts=0` を確認した。

## 次の行動

- tile-20 の grouped generation を、直接経路と token-level / kernel-level に突き合わせて
  JavaScript 説明の差が発生する理由を調査する。品質修正なしに tile-20 を昇格しない。
- 再試験時は隔離 gateway の終了・lock 解放を確認してから AQ4 を start し、cleanup で
  service start を先行させない。N=128 は CC と同じ one warmup + five synchronized samples の
  protocol で取り直す。
