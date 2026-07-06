# Package Prompt Suite Generated-Token Guard

- JSON: `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/prompt-suite-token-logits-guard.json`
- Reference: `R9700-RDNA4`
- Candidate: `V620-RDNA2`
- Passed: `true`
- Compared cases: `7`

| case | category | prompt match | generated match | logits match | stop match | both verified | output status match | generated tokens | sha256 |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | --- |
| japanese_direct_answer | japanese | true | true | true | true | true | true | 49 | f9ba6ba327e9e382 |
| long_prefill_warmup_timing | timing | true | true | true | true | true | true | 192 | 6be141962dcb081a |
| memory_vs_compute_direct | technical | true | true | true | true | true | true | 62 | 3876cd8b6a716ab0 |
| python_stop_helper | code | true | true | true | true | true | true | 84 | 37c03e076f19178d |
| short_qa_bandwidth | short_qa | true | true | true | true | true | true | 35 | d30b2709cb879de4 |
| throughput_checklist_direct | checklist | true | true | true | true | true | true | 53 | c855ce236defcb73 |
| warmup_direct_answer | technical | true | true | true | true | true | true | 67 | 1775ce6734cb10c0 |
