# T2 SQ FP8 full mixed qk pair prompt bundle v1

## 前回の要点

- layer3 `q/k/v` SQ FP8 triple候補は、短いB=1/4/8 smokeではAQ4 final top1と一致した。
- しかしfull mixed `manifest-all` のprompt bundleでは `case_a` のtop1がAQ4 `4105` からSQ `5582` に入れ替わり、strict top1は `2 / 3` だった。
- 次の切り分けでは、`v_proj` をAQ4のまま残し、layer3 `q/k` pairだけをSQ FP8にする必要があった。

## 今回の変更点

- layer3 `q/k` pair候補 `sq-fp8-w8a16-r9700-v0-qk-layer3-q32-k16` を、同じfull mixed prompt bundleで再測定した。
- `ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1` を付け、pair境界が実際に踏まれることをtelemetryで確認した。
- AQ4 baselineも同じrun directoryへ保存し、top-k overlap、AQ4 top1 rank、logit marginを `comparison.json` へ保存した。

## R9700 result

| row | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | true | true | 67.210503 | 81.070892 | 35.003929 | 5080682496 | `24218,4105,329` |
| SQ `qk-layer3-q32-k16` | true | true | 61.363454 | 78.083528 | 33.570044 | 4464140288 | `24218,4105,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `pair` |
| `sq_fp8_single_matvec_count` | 0 |
| `sq_fp8_batch_matvec_count` | 0 |
| `sq_fp8_pair_matvec_count` | 23 |
| `sq_fp8_triple_matvec_count` | 0 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | top1 stable |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | SQ top1 margin over rank2 is `0.000080586` |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | top1 stable |

## 判断

- layer3 `q/k` pair候補は、full mixed prompt bundleでstrict top1を維持した。
- `case_a` はtop1 marginが非常に小さいため、`q/k` pairは「通るが余裕が薄い」境界として扱う。
- 直前の `q/k/v` triple failureは、`q/k` だけではなく、`v_proj` 追加または `q/k/v` のscale組み合わせで発生した可能性が高い。
- このrunではAQ4 QKV triple fused dispatchを無効化してpair境界を強制しているため、throughputは最終速度比較ではなく、quality境界とpair direct path確認の診断値として扱う。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `aq4/raw.json`
- `sq-qk-layer3/raw.json`
- `aq4/stdout.log`
- `sq-qk-layer3/stdout.log`

## 次の行動

1. `q/k` layer3 pairを現在のfull mixed prompt-bundle pass境界として保存する。
2. `v_proj` 単体または `q/k + v` の別row-block/scale粒度を試し、`case_a` のtop1 swapを避けられるか確認する。
3. speed評価へ使う場合は、pair強制のためのAQ4 triple disableを外せる候補、または通常dispatchと比較可能な条件へ戻して測る。
