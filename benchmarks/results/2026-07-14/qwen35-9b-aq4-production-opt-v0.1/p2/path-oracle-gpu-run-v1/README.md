# Qwen3.5 AQ4 P2 GPU path-oracle run evidence

This directory is an immutable copy of the bounded evidence captured around the
single sanctioned GPU path-oracle run on 2026-07-14. It contains no credentials.
The run was started with HIP physical GPU 2 exposed as runtime device index 1
(`HIP_VISIBLE_DEVICES=1`, `ULLM_HIP_VISIBLE_DEVICES=1`) and reported one `gfx1201`
device. The exporter completed model execution and wrote the path payload; the
first link attempt failed on a diagnostic sample-shape exception before the
comparison tolerance fix. The later link was generated from the same immutable
payload, without another model run.
