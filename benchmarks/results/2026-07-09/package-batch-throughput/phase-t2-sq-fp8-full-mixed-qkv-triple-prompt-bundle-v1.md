# T2 SQ FP8 full mixed qkv triple prompt bundle v1

## 前回の要点

- layer3 `q/k/v` SQ FP8候補は、短いB=1/4/8 smokeでAQ4 final top1と一致した。
- telemetry追加により、SQ FP8 direct `triple` projection境界をstdout/JSONLで確認できるようになった。
- ただし短い `len:2xB` smokeだけでは、SQ候補のquality guardとして弱かった。

## 今回の変更点

- `q/k/v` layer3 triple候補をfull mixed `manifest-all` request-state pathへ流した。
- prompt bundleは `len4`、`case_a`、`case_b` の3 requestにした。
- AQ4 baselineとSQ FP8候補を同じbatch=3、prefill/decode real batch pathで比較した。
- `final_topk_tokens_csv` / `final_topk_logits_csv` のrequest内区切りが `:` の場合もJSONLへ保存できるようにparserを修正した。

## R9700 result

| row | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | true | true | 67.314582 | 81.344364 | 35.092672 | 4317257728 | `24218,4105,329` |
| SQ `qkv-layer3-q32-k16-v32` | true | true | 60.207097 | 79.538849 | 31.731900 | 4466266112 | `24218,5582,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_single_matvec_count` | 0 |
| `sq_fp8_batch_matvec_count` | 0 |
| `sq_fp8_pair_matvec_count` | 0 |
| `sq_fp8_triple_matvec_count` | 23 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | top1 stable |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | top2 swap, SQ margin over AQ4 top1 is `0.001122474` |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | top1 stable |

## 判断

- `q/k/v` layer3 triple候補は、full mixed prompt bundleではstrict top1を維持できなかった。
- `case_a` はtop8集合自体は同じだが、AQ4 top1とSQ top1が入れ替わった。差は小さいが、現T2 guardはstrict top1なのでfailureとして扱う。
- direct triple pathそのものは `sq_fp8_triple_matvec_count=23` で確認できたため、この候補は「triple境界の回帰サンプル」として保存する。
- この候補はSQ policyへ昇格しない。次はlayer3より広げる前に、row-block幅、scale粒度、または対象familyを変えてstrict top1を守る候補を探す。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `aq4/raw.json`
- `sq-qkv-layer3/raw.json`
- `aq4/stdout.log`
- `sq-qkv-layer3/stdout.log`

## 次の行動

1. layer3 `q/k/v` tripleはregression boundaryとして残し、昇格候補から外す。
2. 次のT2探索では、`q/k` pair、`k`単体、または別row-block/scale粒度の `q/k/v` をprompt bundleで先に通す。
3. parser修正によりtop-k overlap、AQ4 top1 rank、logit marginをJSONLから直接集計できるので、以後の候補比較では同じ比較JSONを保存する。
