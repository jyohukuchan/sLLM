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
- HF `Qwen3_5MoeRMSNorm` と source `text_config.rms_norm_eps=1e-6` を再照合し、
  MoE bridge だけは input/post/QK/linear gated norm 全てに descriptor の epsilon を
  渡すよう補正した。historical dense bridge の post/QK `1e-5` default は public dense
  entrypoint に残し、9B `AQ4_0` の数値経路を変えていない。full-attention F32 KV
  fallback の fused writer operation plan にも descriptor epsilon を binding し、実行値と
  plan の mismatch を残さなかった。
- 35B linear attention の hidden 2048 と value stream 4096 を descriptor geometry として
  渡し、9B default geometry を保った。
- MoE decode scratch は 40 層に複製せず、111.20 MiB の shared workspace とした。初版の
  4.344 GiB の重複確保を避け、BN ledger の一層分 workspace 前提と整合させた。
- BN の 262,144-token ledger の full-attention KV は 2 B/value（合計 5 GiB）である一方、
  BW 初版 binary は F32 KV（10 GiB）だった。初版の静的 262k 合計は
  `36,226,719,556 B` となり、R9700 より `2,017,976,132 B` 超過する。この値は未実行の
  byte 計算であり、allocation failure を示すものではない。`d8389e59` の typed KV cache は
  R9700 9B `AQ4_0` で F16 8,192-token load と 3,968-token prompt/64-token exact generation
  を確認済みだが、35B の証拠には転用しない。35B は F16 2 B/value 設定で別途 ledger 条件を
  実測する（数値的 BF16 同一性は主張しない）。
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
- 空き window の probe で runtime index 0 が CPU fallback であることを確認した。HIP を
  一台に絞っても uLLM の first HIP device は runtime index 1 である。
- AMD SMI の physical R9700 は GPU 2 だが、ROCm/HIP ordinal は 1 だった。
  `HIP_VISIBLE_DEVICES=2` は V620 (`gfx1030`) を選ぶため、architecture guard が重み読込前に
  fail-closed した。R9700 の再試行には `HIP_VISIBLE_DEVICES=1` と runtime index 1 を使う。
  V620 には context 選択以外の allocation / kernel dispatch を行っていない。

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
- R9700 の AMD SMI GPU 2 と HIP ordinal 1 を混同しない。実行時は
  `HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1` / uLLM runtime index 1 を固定し、
  `gfx1201` admission を成功条件に含める。
