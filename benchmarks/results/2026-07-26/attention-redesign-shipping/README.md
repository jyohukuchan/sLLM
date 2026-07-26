# Attention redesign shipping audit

## `AQ4_0` production decision

BH の `SQ8_0` grouped tile-20 body 自体は `AQ4_0` に直接適用できない。
`AQ4_0` Qwen3.5-9B は Q/KV=16/4（GQA 4:1）、head/value dim=256 であり、
BH body の 5:1 / 128 と異なるためである。C=1339 の current-P3 ROCprof trace は
`ullm_paged_decode_attn_f32_kernel` が decode marker 内で 0 launch、split
partial/merge が 37.378910 ms / 411.411732 ms = **9.08552%** を占めることを示す。
32 層は `linear_attention` 24 層と `full_attention` 8 層で、対象は後者だけである。

GQA 協調の原理は 4:1×256 専用 body として実装・検証した。C=1339 の alternating
full-model control は direct **74.110977 tok/s**、grouped **74.509830 tok/s**、
**1.005382×**（+0.398854 tok/s）だった。これは profile driver の measured decode
interval であり、ROCprof range を throughput として扱っていない。

`aq4_gqa_grouped_split` / tile 128 を typed execution contract として manifest に
記録し、candidate manifest
`69a5e1eb2e7713a1d017332539a587b9a13cf925cbfb28d7c89719ba6709ec2e` を
`tools/promote-served-model.py` で昇格した。active model は引き続き `AQ4_0` であり、
`SQ8_0` は昇格していない。

## Evidence map

- [Phase 1 current-P3 trace](phase1/current-p3-compatible-c1339-20260726T160603Z/)
- [AQ4 specialized full-model control](aq4_0-grouped-final-c8074928-window-20260727T015800Z/)
- [AQ4 promotion suite and receipt](aq4_0-grouped-promotion-c8074928-20260727T020500Z/)
- [Candidate and P3 manifest records](manifest/)
- [`SQ8_0` direct versus grouped text-quality review](sq8_0-quality/quality-review-20260726T160603Z.md)
- [AQ4 prototype / ROCprof interpretation](aq4_0-grouped-prototype-analysis.md)

The same-model `SQ8_0` ten-prompt capture had no automated blocking failure,
but human review holds approval: the grouped output omitted requested Python
code, contained a JavaScript explanation error, and ended the Japanese
multiturn response incompletely. Exact-match observations are diagnostic only;
the hold is based on generated text quality.

## Source provenance

The served worker is from `c8074928e22b27801df78d65508fdd619d13a748`, retained
on local branch `bq-aq4-grouped-c807`. Its current-main integration equivalent
is `9d8643506a36659ecec3fc2d931deba26d29f574` on
`bq-aq4-grouped-integration`; its release build succeeded. At this record
point the shared main worktree has concurrent uncommitted edits to both runtime
source files, so this task intentionally did not overwrite the other owner's
work. The production candidate remains fully identified by its worker hash and
manifest above.
