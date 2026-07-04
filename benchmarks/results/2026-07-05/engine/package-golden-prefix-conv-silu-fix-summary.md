# Package Golden Prefix Conv+SiLU Fix Summary

## Finding

Qwen3.5 linear attention applies SiLU after depthwise Conv1d and before q/k/v split for recurrent input. The uLLM runtime depthwise Conv1d kernel is correct as a pure convolution kernel, but the package linear-attention smoke/workflow path was splitting pre-SiLU conv output.

## Change

- Keep `depthwise_conv1d_f32` as pure Conv1d.
- Apply `runtime_host_silu_f32(&conv_output)` before q/k/v split in package linear-attention recurrent/workflow/block/sequence paths.
- Emit both `attention_conv_pre_silu` and post-SiLU `attention_conv` in package/full-reference traces.

## Full 0..8 Golden-Before Results

| package | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity | dominant remaining layer |
| --- | ---: | ---: | ---: | ---: | --- |
| p4p6 | 0.012856537339 | 0.065956056 | 6.994463444 | 0.903654392 | layer 3 self-attention |
| p4p46-inproj | 0.012861041333 | 0.065597419 | 6.988537312 | 0.903646913 | layer 3 self-attention |
| p4p65-inproj | 0.012856537339 | 0.065956056 | 6.994463444 | 0.903654392 | layer 3 self-attention |

## Layer 6 After Fix

| package | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity |
| --- | ---: | ---: | ---: | ---: |
| p4p6 | 0.000511560667 | 0.016369533 | 0.645427704 | 0.998966216 |
| p4p46-inproj | 0.000487887468 | 0.016121546 | 0.486274719 | 0.999005169 |

## Interpretation

The layer 6 `21..22` max-abs outlier was an implementation mismatch, not an inherent gated RMSNorm sensitivity result. After the fix, the remaining dominant error moves to self-attention layer 3, hidden `3994`, token `3`, and is shared across p4p6/p4p46/p4p65. The next diagnostic target should be self-attention internals for layer 3 rather than another linear-attention in-projection policy swap.
