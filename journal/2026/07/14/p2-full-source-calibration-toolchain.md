# P2 full BF16 source calibration toolchain

## 前回の要点

既存のQwen3.5-9B AQ4 P2 source-oracle v2は、3 rowsのbounded hidden/logit sampleだけを保存しており、full-vector relative L2/max_absを計算できなかった。

## 今回の変更点

- `tools/export-qwen35-aq4-source-calibration.py` を追加した。旧v2とは別schema/rootで、hidden 4096とlogits 248320をf32le sidecarへ64K chunkで保存する。
- `tools/validate-qwen35-aq4-p2-full-calibration.py` を追加した。identity、legacy v2 3-row cross-check、offset/hash/top-k/nonfinite、duplicate/unknown/symlink/TOCTOUを検証する。
- `tools/compare-qwen35-aq4-p2-calibration.py` を追加した。source_gate/path_gateを分離し、chunk streamingとf64累積で相対L2/max_abs/top-k/greedy/nonfiniteを計算する。閾値は生成しない。
- `benchmarks/workloads/qwen35-aq4-p2-source-calibration-cases-v0.1.json` と仕様書を追加した。既存2 prompt/3 rowは互換性canary用で、promotion閾値根拠ではない。
- synthetic full-vector artifactのvalidator/comparator testsを追加した。

## 次の行動

synthetic testsとCLI検証を通した後、MemAvailable 2x checkpoint preflightを満たす場合だけ既存v2を上書きしないCPU canaryを検討する。AQ4 target側のfull-vector sidecarはRust/GPU/live担当であり、今回の範囲では変更しない。
