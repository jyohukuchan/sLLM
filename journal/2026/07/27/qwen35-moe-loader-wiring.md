## 前回の要点

- BI は Qwen3.5-35B-A3B の top-k routing、gather/scatter、grouped/decode GEMM、
  weighted reduction、shared-expert gate を runtime C ABI として実装済みだったが、
  package descriptor から resident executor への loader 結線は未実装だった。
- BN の text-only `AQ4_0` package は 23.310427 GiB、262,144 token ledger は
  30,858,010,436 B で、R9700 34,208,743,424 B に収まる見込みだった。ただし実 allocation
  は未確認だった。
- BS の生成品質 pass と、source/`AQ4_0` 間の expert 選択変化が non-gating observation
  である判断は前提として保持する。

## 今回の変更点

- `Qwen35MoeAq4Runtime` を追加し、descriptor は `resident_descriptor()` で読む一方、
  `model_config.rs` の意図的な `Qwen35MoeExecutor` 未実装停止を変更せずに通過する。
- raw BF16 passthrough の attention/router/shared expert と、rank-3 `AQ4_0` routed expert
  を同居させた resident loader を実装した。decode は raw router → top-8 → selected slab
  stage/dequant → decode GEMM → gated SiLU → scatter weighted → raw shared expert gate の順で
  実行する。
- Qwen3.5-9B `AQ4_0` の full attention（mRoPE、Q output gate、paged KV、Q/K norm）、
  linear attention（conv/recurrent state）、1+weight RMSNorm を bridge 経由で流用した。
  9B の dense `run_device_step` は変更していない。
- 35B linear attention の hidden 2048 と value stream 4096 を descriptor geometry として
  渡し、9B default geometry を保った。
- MoE decode scratch は 40 層に複製せず、111.20 MiB の shared workspace とした。初版の
  4.344 GiB の重複確保を避け、BN ledger の一層分 workspace 前提と整合させた。
- BN の 262,144-token ledger の full-attention KV は 2 B/value（合計 5 GiB）である一方、
  BW 初版 binary は F32 KV（10 GiB）だった。初版の静的 262k 合計は
  `36,226,719,556 B` となり、R9700 より `2,017,976,132 B` 超過する。この値は未実行の
  byte 計算であり、allocation failure を示すものではない。BR の typed KV cache が確定したら
  F16 2 B/value 設定で ledger 条件を実測する（数値的 BF16 同一性は主張しない）。
- CPU の source streaming control は 5 token × 40 層で final hidden/ordered route とも
  0 差。`architecture_hf_trace.py self-test` は corruption を検出した。完全 HF capture は
  66.965 GiB checkpoint に対する host RAM 不足のため未実行である。
- 確認済み: `cargo check`（default と `rocm-moe-gfx1201`）、MoE runtime unit tests、
  35B/9B linear geometry test。実機 R9700 実行は未完了。
- 実装コミット: `286ddc6d`（loader wiring）、`8146c7c3`（shared decode workspace）。
- 01:48 JST の read-only telemetry では、既存本番 Qwen3.5-9B `AQ4_0` worker が
  R9700 に `7,119,884,000 B` を保持し、free VRAM は `26,782,728,192 B` だった。
  262,144-token ledger `30,858,010,436 B` より `4,075,282,244 B` 少なく、同 worker が
  `/run/ullm/r9700.lock` も保持している。競合実行・service 停止は行わない。

## 次の行動

- `/run/ullm/r9700.lock` が解放され、外部 `ullm-openai.service` が GPU を保持していない
  ことを確認してから、`HIP_VISIBLE_DEVICES=2` / runtime device 0 で 262,144 token の
  resident load、短い greedy generation、AMD SMI telemetry を取る。typed KV cache が
  確定していれば `ULLM_KV_CACHE_DTYPE=f16` を明示し、確定前なら初版 F32 の 262k overflow
  見積りと 131,072-token fallback を区別して記録する。ロックの奪取、サービス停止・起動、
  `active.json` 変更はしない。
- 同じ条件で既存 Qwen3.5-9B `AQ4_0` baseline probe を走らせ、既知 top-1 token 220 と
  一致することを確認する。
- 生成 token を official tokenizer で文字列化し、最終 token の全 40 層で runtime route と
  raw BF16 router の独立再計算を比較する。tie-free layer は厳密一致、boundary tie は
  非断定として記録する。
