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

[対応する計画](../../../../plans/active/2026/08/1-10/ci-test-strategy.md)
