# AQ4_0 本番 artifact 保護配置是正

## 前回の要点

- BQ は `AQ4_0` grouped decode（`aq4_gqa_grouped_split`、`split_tile: 128`）を
  固定 10 prompt の lightweight promotion suite で通し、active manifest
  `69a5e1eb2e7713a1d017332539a587b9a13cf925cbfb28d7c89719ba6709ec2e` を
  稼働させた。worker SHA-256 は
  `7e34eed1c3aa2bef80e248b3446ff6668300d0aa4e99e03dc3ff9c7a8d238fa3`、
  decode は P3 の 74.110977 tok/s に対し 74.509830 tok/s（1.005382x）だった。
- ただし active worker は repository 作業木内の uid 1000 / `0775` staging path を
  指しており、削除・書換え可能だった。served source commit `c8074928` も main に
  は存在しない。共有 runtime 2 ファイルに他セッションの未コミット変更があったため、
  それを main に無理に統合しない判断自体は正しかった。

## 今回の変更点

- 案 A を選択した。既に quality gate を通過した worker を byte-for-byte で
  `/opt/ullm/aq4-gqa-grouped-deployment-v0.1/releases/aq4-gqa-grouped-c8074928-7e34eed1/`
  へコピーし、`root:root` / `0555` に固定した。元の staging worker は削除・移動して
  いない。
- 新しい root-owned manifest と provenance receipt を同じ release root に `0444` で
  配置した。候補 manifest SHA-256 は
  `3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7` で、
  変更は worker path と protected provenance receipt だけである。`paged_decode_attention`
  の kernel / tile は `aq4_gqa_grouped_split` / `128` のままである。
- provenance には `c8074928` が main に無い事実と、同一 runtime patch-id
  `f69740849b7bd56b065f7b8f1842ba95ed72b042` を持つ current-main-base integration
  branch `bq-aq4-grouped-integration` の `9d864350` を記録した。後者の release build は
  成功しているが worker SHA-256 は異なるため、served binary の byte-identical rebuild は
  未確認として明記した。
- service 操作前は BX/BY の R9700 測定 lock を待った。測定終了後に gateway が `active` に
  復旧したため、追加の start は行わず `tools/promote-served-model.py --yes` の一回の
  `restart` だけで切り替えた。start-limit recovery は不要で、`NRestarts=0` のままである。
- active manifest は
  `3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7` に更新された。live
  確認時の worker PID 642085 の `/proc/.../exe` は protected worker path を指した。worker は
  `root:root` / `0555`、manifest と provenance receipt は `root:root` / `0444` である。
- promotion outcome は `activated`。固定 10 prompt は candidate / baseline とも全件 HTTP 200、
  blocking finding なし、出力完全一致率 1.0 だった。post-restart readiness は 7 回目の
  probe で healthz / readyz / models のすべて 200 になり、`structured_reasoning` の実応答は
  HTTP 200・372 文字だった。root-owned evidence は
  `/opt/ullm/aq4-gqa-grouped-deployment-v0.1/lightweight-promotion-evidence-bz-20260726T193310Z/`
  にある。
- 証跡取得後の 04:34:49 JST に、別の systemd 操作で service が停止されたことを journal から
  観測した（起点のセッションは未確認）。本タスクは追加の stop/start を発行していない。最終観測は
  `inactive/dead` / `Result=success` / `NRestarts=0` であり、R9700 測定との競合を避けて
  inactive 状態を尊重する。

## 次の行動

1. mutable staging worker は削除・移動せず残す。active worker は既に protected release を
   指すため、今後の作業木清掃で本番起動可能性を失わない。
2. runtime 2 ファイルの owners が main 統合を完了した後、`9d864350` 相当を clean main から
   再buildし、served worker との byte identity を再確認する。現時点では source-level patch
   の同一性と integration-branch の release-build 成功までを確認済みで、byte identity は未確認である。
3. service は測定 window の都合で inactive の可能性があるため、他セッションが使用中なら
   起動しない。再昇格、rollback、または明示的な運用復旧が必要になった場合だけ、R9700 lock と
   start-limit 状態を先に確認して lightweight route を使用する。
