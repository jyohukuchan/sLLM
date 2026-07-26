# SQ8_0 resident descriptor boundary

## 前回の要点

BF の config loader は source `config.json` を Qwen3 / Gemma4Text / Qwen3.5 dense /
Qwen3.5 MoE の typed contract に解決できるようになった。一方、resident `SQ8_0` の
実行側は Qwen3-14B の hidden/head/KV/head-dim/intermediate、40層、two-residual norm、
single full RoPE、全 layer own K/V、独立 head を固定しており、Gemma4 E2B の local/full、
PLE、shared K/V、tied head、soft-cap を表せなかった。

## 今回の変更点

- `ResidentModelDescriptor` を config SHA とともに追加した。decoder、embedding/output、
  per-layer attention/RoPE/KV state/norm/MLP/PLE を closed typed representation にした。
  Qwen3.5 MoE は linear state、mRoPE、MoE expert/top-k/shared-expert metadata を記述するが、
  executor が未実装であることは変えない。
- Qwen3-14B `SQ8_0` は exact legacy descriptor を `require_qwen3_14b_sq8_0` で受理する。
  generation、serving、architecture trace、stack load が descriptor と artifact
  `source.config_sha256` の一致を allocation 前に確認する。seven-projection kernel と
  40-layer array を汎用化せず、既存の出力経路を変えない境界とした。
- Gemma4 resident BF16 executor は descriptor から checkpoint validation、memory/KV plan、
  local/full attention、KV source sharing、window、RoPE、four residual norms、layer-dependent
  MLP、PLE、tied embedding/head と final soft-cap を読むようにした。新 kernel は追加せず、
  BH が管理中の runtime source には触れていない。
- `docs/plans/multi-architecture-support-plan-v0.1.md` に resident path の 15 fixed contract、
  generalization boundary、Gemma4 を MoE より先にする理由を追記した。
- CPU では `model_config` 10、`gemma4_text_executor` 5、`sq8_stack_runtime` 14、
  `sq8_generation_runtime` 10、`sq8_serving_runtime` 41 test が通過した。実 Qwen3-14B、
  Gemma4 E2B、Qwen3.5-35B-A3B config の再解決も行った。

## 次の行動

- R9700 lock が空いたことを指定の `fuser` / `pgrep` / `systemctl show` で再確認してから、
  既存 regular lock file を `O_CREAT` なしで nonblocking `flock` する。一回でも busy なら
  実行せず待つ。取得できた時だけ `HIP_VISIBLE_DEVICES=1` で Qwen3-14B `SQ8_0`
  architecture trace と Gemma4 resident validation を行い、終了時に解放する。service の
  start/stop、lock の作成・奪取はしない。
- 判定は既存 BL/BO greedy token 列と生成文で行う。FP32 reference corpus、bitwise gate、
  campaign、厳密な数値閾値は使用しない。
- Gemma4 run の evidence と Qwen3 regression を追記し、artifact schema の未対応範囲
  （BF16 Gemma4 と rank-3 MoE の `SQ8_0` conversion）は未完として明記したまま commit する。
