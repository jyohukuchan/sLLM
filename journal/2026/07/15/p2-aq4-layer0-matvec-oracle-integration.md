# P2 AQ4 layer-0 matvec integration evidence

## 前回の要点

既存の candidate1 oracle は GPU tensor-level output が無い attempt-3 traceに対して
fail-closedで停止し、Rust probeとの入力sidecar契約を固定していた。統合runではこの契約を
現mainのclean HEADへPython/Rustの順に取り込んで、同じ3固定ケースをCPU runtimeで照合した。

## 今回の変更点

- 現main HEAD `3407c0521012d101d45fcff0e03ef9ee0e00ec61` から作成した
  `integration-aq4-final` worktreeへ、Python oracle 5コミットとRust probe 3コミットを
  指定順にcherry-pickした。mainのdirty変更は触れていない。
- Python oracleを安定したsource/GPU traceの絶対pathで再生成した。reportは
  `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-matvec-oracle-integration-v0.1/python-report.json`
  （SHA-256 `e311995fea3f7f4f89f99094f01f4278224303ad2b369daa45be032fc94fb8a1`）、sidecarは
  `runtime-input.jsonl`（SHA-256 `c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17`、
  250573 bytes、schema `ullm.aq4_layer0_input_normed_jsonl.v1`）である。Python reportは
  `blocked_missing_gpu_tensor_output` / `inconclusive_missing_gpu_tensor_output`、
  `promotion_eligible=false` のままとした。
- Rust context-0 probeをdevice index 0で実行した。reportは
  `rust-context0/report.json`（SHA-256 `2e0e623d0cec8299944ee73db7586b0790e8a83eed9d037e0416fddba1e44145`）、
  outputは `rust-context0/output.f32le`（SHA-256
  `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473`）である。runtimeは
  CPU fallback、3 rows、finite、`promotion_eligible=false` を報告し、sidecar consumed SHAと
  pre/post statを固定した。GPU context/serviceは起動していない。
- `rust-vs-python-f32-comparison.json`（SHA-256
  `010d5dc7da4eaf84e750d6c8917efcb22cf87d3d12c3b62e4c1f3e55b1b8f33a`）は、Rust outputを
  sidecar順に8192要素ずつ読み、同じpackage/itemに対するPythonの
  `dequant_matvec(..., scalar_f32=True)`を比較した。3 rowsすべてで `max_abs=0`、
  `relative_l2=0`、`bit_mismatch_count=0`、`bit_exact=true`、非有限値なしとなった。
- Python 18 tests、py_compile、Rust probe 5 tests、低並列 `cargo check --bin
  ullm-aq4-layer0-qkv-runtime-probe` を通過した。GPU tensor outputが未取得のため、
  数値比較はCPU診断の確認であり、昇格判定やGPU threshold変更ではない。

## 次の行動

GPU tensor-level output（`ullm.aq4_layer0_matvec_tensor_output.v1`）を同じsidecar・package
identityで取得できるまで、candidateのpromotionは行わない。今回の統合artifactはCPU
runtimeとPython explicit-f32の一致を示す診断証跡として扱う。
