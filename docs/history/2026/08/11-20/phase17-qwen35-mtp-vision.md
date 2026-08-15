# Phase 17 Qwen3.5 MTP・vision履歴

## 2026-08-16: 詳細計画作成

- fixed `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`のknown-unconsumed
  MTP 15 tensorとvision 297 tensorを正式消費する計画を作成した。
- debugging範囲を分けるため、text-only MTPのreader/oracle/graph/speculative transaction/serviceを先に完了し、
  image processor/vision graph/multimodal prompt/APIを後から実装する順序にした。
- MTPはgreedy token完全一致、stochastic rejection/residual sampling、accepted prefixだけのopaque KV publication、
  stop/EOS/cancel、target別内部provider選択を受入条件にした。Phase 16のFP8 KVを代表caseで回帰する。
- vision processorは同じrevisionのpixel area `65,536..=16,777,216`、patch 16、temporal patch 2、merge 2、
  mean/std 0.5とspecial tokenを固定した。NumPy/Pillow oracleを使い、PyTorchは使用しない。
- OpenAI公式OpenAPI 2.3.0とImages/vision guideを2026-08-16に確認した。Chat Completionsのtext/image content arrayを
  versioned profileへ追加するが、初期server sourceはBase64 data URLだけとし、HTTP(S) fetch/Files APIを実装しない。
- MTPとvisionを個別にPASSした後だけcombined image+MTP smokeを行う。本時点ではsource、model lock status、API、fixtureを変更していない。

[対応する計画](../../../../plans/active/2026/08/11-20/phase17-qwen35-mtp-vision.md)
