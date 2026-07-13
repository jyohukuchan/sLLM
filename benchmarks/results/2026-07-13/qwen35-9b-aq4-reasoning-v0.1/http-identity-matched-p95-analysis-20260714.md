# Identity-matched HTTP/SSE p95 comparison

Date: 2026-07-14

This report compares the previous v2 manifest with the current candidate under
the same hash-only HTTP/SSE fixture suite. It contains no prompt text,
response text, authorization header, or API key.

## Measurement identity

- Fixture: `tests/fixtures/generic-reasoning-release-v0.1/prompts.json`
- Cases per identity: 100 total; 20 per mode; 50 streamed and 50 non-streamed.
- Previous manifest SHA-256: `e6f749654e85a5f69f2d077bd55d4e27aff869d71803809386c5d36865183e72`
- Current manifest SHA-256: `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`
- Worker SHA-256 for both identities: `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`
- HTTP probe image: `sha256:5dce198cca467ce79994ed65e01d03882238f9efdd16a8c6f4bc55151c8a4a54`
- Percentile method: linear interpolation at rank `(n - 1) * p`.

Previous-manifest evidence is in `http-sse-campaign-old-e6f-01/` through
`http-sse-campaign-old-e6f-10/`. Current-candidate evidence is in
`http-sse-campaign-ae8b2bb/` and `http-sse-campaign-ae8b2bb-soak-01/` through
`-09/`. The previous manifest was activated only for the baseline collection;
the current candidate was restored with
`release-bundle-ae8b2bb-after-p95.json` and bundle-bound activation.

## Accounting and correctness

| Identity | Cases | Correct | Empty | Budget overshoot | Lifecycle reset | Stop outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous v2 | 100 | 100 | 0 | 0 | 100/100 | 100/100 |
| Current candidate | 100 | 100 | 0 | 0 | 100/100 | 100/100 |

## Latency and throughput percentiles

Values are aggregated by mode across 20 cases per identity. Latency is in
milliseconds and throughput is tokens/second.

| Mode | Previous latency p50/p95 | Current latency p50/p95 | p95 delta | Previous prefill p50/p95 | Current prefill p50/p95 | p95 delta | Previous decode p50/p95 | Current decode p50/p95 | p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 733.700 / 767.211 | 733.761 / 775.957 | +1.14% | 116.544 / 119.594 | 116.624 / 119.346 | -0.21% | 9.541 / 9.792 | 9.540 / 9.711 | -0.83% |
| budget-32 | 1298.988 / 1358.219 | 1301.705 / 1371.395 | +0.97% | 118.298 / 121.862 | 118.765 / 122.707 | +0.69% | 34.642 / 35.180 | 34.570 / 35.210 | +0.09% |
| budget-128 | 2753.821 / 2822.030 | 2752.970 / 2796.771 | -0.90% | 120.747 / 123.513 | 120.945 / 124.233 | +0.58% | 51.565 / 52.019 | 51.581 / 52.047 | +0.05% |
| budget-256 | 3227.460 / 3295.293 | 3217.936 / 3266.693 | -0.87% | 120.884 / 123.962 | 120.974 / 122.765 | -0.97% | 53.293 / 53.889 | 53.450 / 53.812 | -0.14% |
| unbounded | 3384.861 / 3449.831 | 3392.501 / 3439.705 | -0.29% | 116.341 / 118.090 | 116.518 / 117.743 | -0.29% | 54.064 / 54.635 | 53.943 / 54.383 | -0.46% |

## Gate interpretation

The identity-matched current-versus-previous p95 deltas are within the plan
thresholds of 3% for latency and 5% for throughput in every mode. The
disabled-mode regression check is therefore closed with the same 100-case
fixture population on both sides. Both populations also passed the correctness
and lifecycle accounting checks above.

