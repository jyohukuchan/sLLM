# AQ4 P2 証跡ブリッジ

## 前回の要点

P1 の実行トレースと独立検証のスキーマ監査を完了し、CPU の再現可能な証跡を保持した。P1 の親ゲートとエンジン側の実行ブリッジは別担当であり、ここでは変更していない。

## 今回の変更点

- `expand-aq4-production-p2.py` を追加し、ケースを canonical JSON として展開した。smoke 84、representative 1,705、full 3,075、合計 4,864 件で、all-M1 の経路オラクル参照を付与する。
- `bind-aq4-production-p2-identity.py` を追加し、モデル、トークナイザー集合、worker、package、graph、state、source oracle、power、baseline、Git の SHA-256 を束ね、bound policy を生成する。
- `run-aq4-production-p2.py` を追加し、shell 無しの argv 実行、bounded streaming、OOM/失敗/未対応/skip の不変状態、R9700 排他ロック、事前メモリ余裕、atomic 出力を実装した。production の実体不足は fail-closed とする。
- `build-aq4-prefill-validation-result.py` と `validate-aq4-production-p2-evidence.py` を追加し、raw/result/trace/source-oracle/identity/policy のリンクと経路オラクルを独立検証する。CPU/component/full-model は promotion 不可で、production-server は独立 trace が揃うまで不可とする。
- `tests/test_aq4_production_p2_evidence.py` と専用 fixture を追加した。CPU synthetic の正常系、hash 改ざん検出、production 実体不足の fail-closed を確認した。

## 次の行動

専用テストは 3 件すべて成功した。GPU/live の実測、R9700 の電力・VRAM capture、実 package/worker の production trace、P1 親ゲートの承認が残課題である。これらが揃うまで P2 は測定準備と CPU 証跡に限定する。
