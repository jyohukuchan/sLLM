# P3 AQ4 prefill candidate shortlist

## 前回の要点

- P3 prefill candidate auditは、M=128 cold prefillの全体wall timeは見えるが、operation別wall timeは未計測であり、候補順位は仮説だと整理していた。
- family-exclusive profiler commit `28ec343` はrocprofv3 interval unionを診断用に保存し、profileを通常性能証拠へ流用しない。

## 今回の変更点

- AQ4 native prefillのlayer sequence loop、self-attention Q/K/V projection、paged KV writer/readerを読み取り監査した。
- 候補を3つ（A: sequence出力D2D copy削減、B: QKV grouped dispatch、C: paged KV metadata/sync削減）へ整理した。
- 最有力はAとした。ただしP2 R9700 profileでD2D bytes・GPU interval・p50/p95が支配的であることを確認するまで、実装やactivationは行わない。
- Aのfallback、CPU oracle、component/full-model/direct-worker/productionの昇格順と、OOM/state/reset/identityの停止条件をproposalへ記録した。

## 次の行動

- capture fixed commitの独立tamper reviewを先に完了する。
- GPU/serviceを変更せず、承認済みR9700 profile runのraw evidenceから候補Aの支配性を再判定する。
