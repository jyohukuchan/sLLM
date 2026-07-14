# P1 schema repair

## 前回の要点

既存P1 trace/recordは旧flat schemaで、`memory.oom=false`、direct workerの
`production_server` claim、producer-only validation、循環し得る report SHAが
残っていた。

## 今回の変更点

- trace producer/validatorをspec v0.1のexact nested field setへ更新した。
- graph/state canonical digestとcompatibility digestを再計算し、manifest、worker、
  package、receipt、artifact identityを再hashするようにした。
- phase/operator/fallback/memory/state/aggregation/serverをfail-closeで再構成し、
  `memory.oom`をnullまたはobjectへ限定した。
- direct worker captureは`full_model`へ固定し、production serverはready/releaseを
  実boundaryで観測したrecordだけを許可した。
- detached validator reportを先にatomic publishし、そのSHAをverified traceへ記録する
  循環なしの独立検証を追加した。
- CPU fixture `tests/fixtures/production-execution-trace-p1/schema-r1/`、negative tests、
  matrix mechanics smoke label、P1-D→P2 handoffを追加した。

## 次の行動

P2でactive manifestと同一binary/packageの実production-server boundaryを再取得し、
このschema-r1 runnerをperformance evidenceへ昇格する。P1 mechanics smokeとread-only
bottleneck auditは性能証拠ではない。

P1中はactive R9700 workerと別workerを同時実行しない安全条件を優先したため、
production-server boundaryを新規取得していない。`p1-schema-r1/` は旧path-bound
artifactとして保持し、`benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/
p1-schema-r2/`を同一ディレクトリ内のrelative manifest pathで再生成した。どちらもCPU
`full_model` mechanics evidenceであり、production-server claimを含まない。これはP2
handoffとして明記する。

## 検証

- `python3 -m unittest tests/test_production_execution_trace.py -v`: 7 passed, including capture graph source, final-byte report attestation, phase-context tamper, and unreconstructed internal-step negatives
- 対象Python tools `py_compile`: passed
- `cargo fmt --all --check`: passed
- `cargo test -p ullm-engine execution_trace -- --test-threads=1`: 3 passed
