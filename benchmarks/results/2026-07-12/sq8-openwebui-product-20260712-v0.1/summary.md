# SQ8 OpenWebUI full campaign

Run ID: `sq8-openwebui-product-20260712-v0.1`

Schedule: `{"cancel_phases":["after_started_before_progress","prefill_after_128","prefill_after_2048","decode_after_first_content","openwebui_stop_after_visible_content"],"decode_measured":10,"decode_warmups":2,"idle_settle_ms":5000,"latency_measured_per_case":10,"latency_warmups_per_case":2,"normal_requests":100,"normal_warmups":10,"openwebui_chats":20,"restart_requests":20,"restart_warmups":10,"sample_interval_ms":1000,"sampled_normal_indices":[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],"samples_per_point":5,"ttft_fixture_ids":["exact-p0032","exact-p0128","exact-p0512","exact-p2048","exact-p3584"]}`

Artifacts:
- `SHA256SUMS`
- `amd-smi-metric-normal-after.json`
- `amd-smi-metric-normal-before.json`
- `amd-smi-metric-restart-after.json`
- `amd-smi-metric-restart-before.json`
- `api-contract-results.json`
- `browser/openwebui-stop-before.png`
- `browser/post-header-failure.png`
- `cancel-results.json`
- `environment.json`
- `model-identity.json`
- `openwebui-smoke.json`
- `prefill-latency-results.json`
- `raw-session-results.jsonl`
- `release-matrix.json`
- `sampling-results.json`
- `service-journal.raw.jsonl`
- `soak-resources.raw.jsonl`
- `soak-results.json`
- `summary.md`
