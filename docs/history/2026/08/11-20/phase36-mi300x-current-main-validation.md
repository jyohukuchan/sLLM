# Phase 36 MI300X latest-main実機再検証履歴

## 2026-08-20: 計画作成

- ユーザー指示により、Phase 35後のlatest mainを単一MI300Xで再検証し、問題があれば修正するPhase 36を計画した。
- 課金GPU sessionをA〜Eへ分割した。Session Aはidentity、ROCm/artifact、Phase 12相当99 operator、
  Qwen3.5-4B BF16/FNUZ FP8短生成までを2〜3時間、上限4時間で実行する詳細計画とした。
- Session Bはlow-bit KV/chunked prefill/10k+、CはMTP/vision/OpenAI service、Dはperformance/llama.cpp/profile、
  EはGemma/MoE/安定性のconditional extensionとした。
- この時点では計画のみであり、VM作成、credential作成、GPU実行、production source修正は開始していない。

[対応する計画](../../../../plans/active/2026/08/11-20/phase36-mi300x-current-main-validation.md)
