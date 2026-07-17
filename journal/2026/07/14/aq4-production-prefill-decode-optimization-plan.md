# AQ4 production prefill/decode最適化計画の作成

## 前回の要点

- 現行製品はQwen3.5-9B AQ4_0 reasoning v2で、release gateと100-chat soakまで完了している。
- production prefillは約117–129 tok/sで、Qwen3.5 AQ4/R9700向けのprefill最低条件へ未達である。
- paged decode splitはproduction配備済みで、p1339/g64は約66.53 tok/sである。
- production execution trace生成、独立validator、完全なproduction graph identity、generic M>1 prefillのresident接続は未完である。
- release bundleのrollback environment hashと現在のenvironment file hashに差がある。

## 今回の変更点

- `uLLM-project/docs/plans/aq4-production-prefill-decode-optimization-plan-v0.1.md`を作成した。
- P0 identity/rollback、P1 trace/validator/runner、P2 baseline/profile、P3 prefill、P4 freeze、P5 decode、P6 promotion、P7 closeoutの8 Phaseに分けた。
- P2完了後はprefill実装とdecodeのread-only解析を並列化できるようにした。
- CPU oracle、runtime kernel、registry/engine integration、evidence toolingを別laneとし、共有ABI、session、R9700、activationは直列化した。
- R9700の一件実行、実package GPU smokeとworkspace testの逐次実行、OOM evidence保持を明記した。
- component改善だけではpromotionせず、full model、direct worker、production server、OpenWebUIまで同じidentityで検証するGateを置いた。
- 対象実装はAQ4に限定しつつ、validation仕様に従ってSQ8_0をcross-format controlとして測定matrixへ残した。
- 独立レビューを反映し、executor-record sidecar、SQ8/Qwen3 full-model control、cached-prefix必須coverage、100/20-chat soak、429契約、traceのtoken ID禁止を明記した。

## 判断

- kernel最適化の前に必要なのは、完全なgeneric executor移行ではなく、rollback binding、production trace、独立validator、固定baselineである。
- full generic migrationは上位計画として継続するが、この計画のcritical pathを必要以上に広げない。
- decode splitの既存production pathを保持し、prefill変更とdecode変更を別candidateとして評価してから統合する。
- 過去のBM8等は探索資料には使うが、現active identityと異なるevidenceをpromotion根拠へ流用しない。
- p1339/g64の約129.33/66.53 tok/sも旧manifest/source identityの履歴値であり、現active baselineはP2で取り直す。

## 次の行動

P0でrollback environment hash差分、service topology、active identityを読み取り確認し、active serviceを変更せずにrollback bindingとthreshold policyを固定する。その後、P1-A trace producerとP1-B independent validatorを並列に開始する。
