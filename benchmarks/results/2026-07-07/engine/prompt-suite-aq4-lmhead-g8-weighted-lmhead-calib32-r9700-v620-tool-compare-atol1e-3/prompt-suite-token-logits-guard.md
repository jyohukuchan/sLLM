# Package Prompt Suite Generated-Token Guard

- JSON: `benchmarks/results/2026-07-07/engine/prompt-suite-aq4-lmhead-g8-weighted-lmhead-calib32-r9700-v620-tool-compare-atol1e-3/prompt-suite-token-logits-guard.json`
- Reference: `r9700_g8_weighted`
- Candidate: `v620_g8_weighted_tool`
- Passed: `true`
- Compared cases: `7`

| case | category | prompt match | generated match | logits match | stop match | both verified | output status match | generated tokens | sha256 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | --- |
| japanese_direct_answer | japanese | true | true | true | true | true | true | 29 | ef944af803b76204 |
| long_prefill_warmup_timing | timing | true | true | true | true | true | true | 192 | 9129dca8e4b8f03b |
| memory_vs_compute_direct | technical | true | true | true | true | true | true | 73 | 7a8de8b8a6bb49af |
| python_stop_helper | code | true | true | true | true | true | true | 51 | b9a99b54368752ee |
| short_qa_bandwidth | short_qa | true | true | true | true | true | true | 96 | 234c6d43c15b8b8b |
| throughput_checklist_direct | checklist | true | true | true | true | true | true | 84 | 3d94fb44537587b5 |
| warmup_direct_answer | technical | true | true | true | true | true | true | 65 | 2ecdd5d812c5145e |
