# Package Prompt Suite Generated-Text Guard

- JSON: `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-text-level-prompt-guard-v0.2/aq4-self-compare.json`
- Reference: `aq4-r9700-v0.3`
- Candidate: `aq4-r9700-v0.3-self`
- Passed: `true`
- Compared cases: `7`

| case | category | prompt match | token match | text match | no-stop text match | logits match | stop match | both verified | output status match | generated tokens | token sha256 | text sha256 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | --- | --- |
| japanese_direct_answer | japanese | true | true | true | true | true | true | true | true | 48 | 0f414b9c6f129ae3 | a9988c9d744d6994 |
| long_prefill_warmup_timing | timing | true | true | true | true | true | true | true | true | 192 | 9129dca8e4b8f03b | 80826b4157ab7ec1 |
| memory_vs_compute_direct | technical | true | true | true | true | true | true | true | true | 73 | 7a8de8b8a6bb49af | 06ea0ca212780b8b |
| python_stop_helper | code | true | true | true | true | true | true | true | true | 68 | 972e1e02f78dcf20 | aeaf3f98b1a37145 |
| short_qa_bandwidth | short_qa | true | true | true | true | true | true | true | true | 38 | 0865be503d10002c | e895f9b2eb42d258 |
| throughput_checklist_direct | checklist | true | true | true | true | true | true | true | true | 82 | 37ca6ef0680a56d3 | 99b619fce114723e |
| warmup_direct_answer | technical | true | true | true | true | true | true | true | true | 53 | 894b22f423f7431a | 7a37166e537dd63b |
