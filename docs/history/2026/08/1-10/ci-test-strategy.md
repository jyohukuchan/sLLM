# CI・テスト方針策定履歴

## 2026-08-02

- 前回の開発で発生した、GPU処理をCPUで実行して長時間化した問題と、テスト導入が遅れて細かな不具合が蓄積した問題を設計上の入力とした。
- ローカル`reference/`の配置状態を確認した。各参照directoryは空であり、固定source配置後の再調査が必要と判断した。
- workspace外の読み取り専用mirrorからvLLMとTensorRT-LLMのtest tier、hardware metadata、代表model、quarantine、性能回帰の分離方針を調査した。
- GitHub ActionsとROCmの公式資料から、公開fork PRとself-hosted runnerの安全境界、ephemeral runner、timeout、matrix、GPU isolationを調査した。
- CPU CIで許可する処理と禁止するGPU-scale処理を分離した。
- H0〜H3、G0〜G4、P0〜P1のテスト階層、初期時間予算、result state、shape方針、数値比較方針を草案化した。
- PR、信頼済みdispatch、main push、daily、weekly、releaseの実行matrixを草案化した。
- repository skeletonからGPU runner、kernel、model slice、end-to-end、performanceへ進むPhase 0〜6を定義した。
- file変更を伴わない4つの調査用Codex sessionは全て終了コード0で完了した。
- 読み取り専用reviewで、GPU runnerのcommit/cache信頼境界、CPU testの累積resource制限、main-planのCI導入順序、result集約、AMD binary key、KV dtypeの不足を検出して草案へ反映した。
- 再出発レビューを受け、`255/256/257`を含むperformance-cliff sanity、同一reviewed SHAのGPU merge gate、`always()`を含むfail-closed集約条件を追加した。
- repository hygieneとcredential方針をPhase 0/1のgovernance baselineへ追加した。
- push前reviewを受け、H3 required昇格条件、P1 weekly/release予算、build・ROCm・target・codegen変更のH3/G4 gate表現を統一した。
- source-lockの7件を固定完全SHAの一次sourceとして4 reader sessionで再調査し、全sessionが終了コード0で完了した。外部codeのcopy、adapt、portは行っていない。
- 段階化、明示登録、決定的sharding、per-test timeout、GPU preflight、immutable artifact再利用、isolated test、warmup/metricを採用し、暗黙skip、0件成功、soft-fail、可変外部data、root/privileged runnerを不採用とした。
- H0/H1/H2の並列required row、`host-required`集約、8/10/8分のhard timeout、workflow p95 10分・hard上限15分を確定した。
- tier/属性marker、versioned suite registry、path-to-suite manifest、test-result/compatibility-tuple schemaの正本pathと必須概念を確定した。
- H3のPR compile rowを`gfx1030`/`gfx1201`とし、20回以上・7日以上に加えて全期待rowのPASS、他state/cancel/schema errorなし、artifact hash一致、時間・infra条件をrequired昇格条件にした。`gfx1200`はnightly/release compile-onlyに残した。
- 初期GPU evidenceを専用local hostの`gfx1030` 1台と`gfx1201` 1台の直列実行とし、2台目の`gfx1030`をspare/nightly用とした。暫定local経路はtrusted project commitだけに限定し、public fork PRからはGPU runnerを直接使わない。

[対応する計画](../../../../plans/active/2026/08/1-10/ci-test-strategy.md)
