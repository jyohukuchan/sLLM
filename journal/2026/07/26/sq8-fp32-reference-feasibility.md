# SQ8 artifact-FP32 full-model reference の実現可能性 preflight

## 前回の要点

- `sq8-numerical-gate-v0.2-relative-to-fp32-reference.json` は SHA-256
  `64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf` で凍結済みだが、
  `artifact_fp32_strict_v1` の full-model CPU executor と capture は未実装だった。
- 既存の `sq_reference.rs` / `sq_optimized_reference.rs` は projection-only / F64、
  `cpu_reference_executor.rs` は small graph 向けであり、そのまま主参照には使えない。

## 今回の変更点

- GPU を使わず、`/etc/ullm/served-models/active.json`、systemd、`/opt/ullm`、
  activation、campaign、既存測定値を変更せずに、ローカルの SQ artifact と model binding を
  preflight した。
- frozen JSON の `scope.model_family` は `Qwen3-14B-FP8` だった。対応する canonical artifact
  と同 product の 163 raw passthrough payload package は `sq-fp8-artifact-v0.2` / `SQ8_0` /
  `full_model` / 280 pair / `[128,128]` BF16 block scale / 40 layer を満たす。本文の 40 layer
  と vocab 151,936 も Qwen3-14B config と一致する。
- 9B として見つかった Qwen3.5 artifact は `sq-fp8-artifact-v0.1` の部分 overlay で、48 FP8
  tensor、F32 row-block-256 scale だった。Qwen3.5 text config は 32 layer、vocab 248,320
  であり、frozen v0.2 の model binding と canonical artifact semantics を満たさない。
- 既存 `verify_sq8_canonical` を GPU 不可視条件で実行し、14B canonical artifact の 280
  weight/scale payload を checksum 検証した（15.96 s、max RSS 96,268 KiB）。同時に
  `model.layers.0.mlp.down_proj.weight` の block `[0,0]` を F32 復元し、16,384値の
  little-endian SHA-256 `7f48464a20b4ca17092c193a914a344be9b495fba09f9c5a572670136621b391`
  を得た。この結果は canonical decoder の再利用根拠であり、9B full-model timing の proxy
  にはしていない。
- `sq_canonical::tests::fast_fp8_finite_scan_matches_e4m3fn_decoder_for_every_byte` も GPU
  不可視条件で PASS した。canonical fast finite scan と E4M3FN decoder の 256 payload byte
  全値一致を確認する unit test である。
- したがって、9B の strict-FP32 1-token forward、8-step deterministic pilot、peak RSS、
  4,096 positions × 7 stream の時間外挿は実行していない。source model、F64 projection、
  partial overlay を代用して値を作らなかった。結果は
  `benchmarks/results/2026-07-26/sq8-fp32-reference/feasibility.json` に保存した。
- host snapshot の `MemAvailable` は 83,132,428,288 bytes だった。Qwen3.5 package の宣言
  element countからの F32 logical size は text model 31,746,738,176 bytes、全 tensor
  38,612,417,472 bytes だが、これは実 allocation/RSS ではない。

## 次の行動

1. 9B を対象にするなら、canonical full-model SQ8_0 artifact を先に固定し、32-layer /
   vocab-248320 semantics に合わせた新 gate version を review・freeze する。v0.2 JSON を
   後から変更しない。
2. Qwen3-14B-FP8 を対象に変更する明示承認が得られた場合だけ、既存 canonical artifact を
   入力に CPU strict-FP32 full-model runner と 8-step pilot を実装する。
3. 9B partial overlay を使う engineering-only pilot は可能でも、v0.2 合格・reference capture・
   candidate 判定には使わない。

## 14B canonical artifact の strict-FP32 実測追記

対象はその後明示的に `Qwen3-14B-FP8` の既存 canonical `SQ8_0` artifact へ確定したため、
上記の 9B 不採用判断を覆すのではなく、正しい v0.2 binding に対して別途実行した。artifact は
40 layer / 280 quantized pair / 163 BF16 passthrough / `[128,128]` scale で、content SHA-256 は
`2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147` だった。

### 実装と検証

- `sq8_fp32_reference.rs` を追加した。canonical OCP E4M3FN + BF16 block scale を F32 に復元し、
  matrix は K 昇順 F32 `mul_add`、F32 KV / RoPE / GQA / RMSNorm / MLP / LM-head まで実行する。
  GPU/HIP/runtime context/BLAS/activation quantization を使わず、weight は layer stream にした。
- `raw-p0001` token 1 の full 40-layer forward は 8.742120321 s、external peak RSS は
  528,100 KiB、runner `VmHWM` は 560,384 KiB。初期の full artifact/package validation は
  45.190505324 s だった。
- 8 feedback step pilot を二回 capture し、9 position × logits/final hidden/40 layer hidden の
  hash が完全一致した。final binary でも二回同じ summary hash を得た。
- canonical decoder の全 256 byte test、解析的 F32/scale/BF16/GQA tests は PASS。real artifact の
  layer 0 Q projection は既存 CPU F64 projection reference と max-abs `2.7418e-6`、relative-L2
  `1.0365e-6` で整合した。F64 は cross-check だけで、主参照には使っていない。

### 可否

- 全 weight を F32 化した論理 size は 59,073,228,800 B（55.016 GiB）。preflight
  `MemAvailable` 83,205,074,944 B に算術上は収まるが、all-resident allocation/RSS は未実測である。
  安全な結論は layer streaming 採用である。
- 1-token 実測を直接外挿すると、文字通りの 7 × 4,096 forward は 69.626 h（2.901 日）。
  frozen corpus の 7 primary stream は合計 4,096 position で 9.947 h だが、prompt、boundary、
  M=128 case を現在の逐次 runner で展開すると 28,853 token-forward 相当、70.066 h（2.919 日）に
  なる。いずれも長 context 増加を含まない楽観値である。
- よって full reference capture は開始しなかった。生成済みなのは `raw-p0001` の 9 position を
  二回だけであり、v0.2 full coverage は未達、status は `blocked_reference_or_capture` のままである。
  M=128 checkpoint capture scheduler も未実装である。

Receipt は
[`14b-full-model-feasibility.json`](../../../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-full-model-feasibility.json)
と
[`14b-full-model-SHA256SUMS`](../../../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-full-model-SHA256SUMS)
に保存した。frozen JSON、activation、service、GPU、既存結果には触れていない。
