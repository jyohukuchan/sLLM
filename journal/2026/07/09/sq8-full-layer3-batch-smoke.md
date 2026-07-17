# SQ8_0 full layer3 batch smoke

## 前回の要点

- `sq-fp8-package-self-attn-layer-batch-smoke` は、Qwen3.5 layer3 のpartial q/k/v SQ8_0 artifactで `real_batch=true`、`sq_fp8_batch_matvec_count=6`、`sq_fp8_expected_all_batch_matvec_count=14` まで確認済みだった。
- M10のvLLM + FP8比較は計画後半に置き、selected-layer診断行をfull serving比較として扱わない方針にしている。

## 今回の変更点

- Qwen3.5 layer3 の `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` を含むSQ8_0 artifactを作成した。
- artifact build:

```text
python3 tools/build-sq-fp8-w8a16-artifact.py --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B --base-package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d --output-artifact /tmp/ullm-sq8-layer3-full-projections-artifact --candidate-id sq-fp8-w8a16-r9700-v0-layer3-full-projections --include-regex 'model\.language_model\.layers\.3\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$' --scale-granularity row_block --scale-block-cols 16 --row-chunk 128 --summary-json /tmp/ullm-sq8-layer3-full-projections-summary.json --overwrite
```

- artifact summary:
  - `fp8_tensor_count=7`
  - `passthrough_tensor_count=768`
  - `compact_resident_bytes_estimate=19148930016`
- batch smoke:

```text
target/debug/ullm-engine sq-fp8-package-self-attn-layer-batch-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8-layer3-full-projections-artifact 2 1048576 3 len:2 1 256 32 10000000 0
```

- result:
  - `verified=true`
  - `real_batch=true`
  - `sq_execution_mode=direct_fp8_dequant_matvec_batch`
  - `sq_projection_boundary=batch`
  - `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`
  - `sq_fp8_batch_matvec_count=14`
  - `sq_fp8_expected_all_batch_matvec_count=14`

## 次の行動

- full-package real-batchまたはserver-style uLLM行に、同じdirect batch projection boundaryを接続する。
- M10では、uLLM側のfull-package real-batch/server-style行が揃ってからvLLM + Qwen3-14B-FP8行と比較する。
- vLLM FP8がR9700でunsupportedまたはfallbackになった場合は、failure/unsupported rowとして残し、BF16/FP16参考baselineと分けて扱う。
