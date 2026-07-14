# p2 GPU differential trace attempt2 分析

## 前回の要点

- attempt1 は candidate に product root を渡したため、trace binary が `package_dir/manifest.json` を解決できず rc=78 になった。
- attempt1 の raw log/marker は固定 SHA で保存され、書き換えない条件だった。

## 今回の変更点

- attempt2 を `PACKAGE/package` 引数で一度だけ実行した。3 rows、35 stages、device index 1 の中間 trace を得た。
- attempt2 の candidate は全 3 行で `decoder_layer:0` が最初の差分、embedding は完全一致。lm_head の最大 max abs は 8.347782 だった。
- package manifest の量子化重み 256 件を監査した（g16=204、g8=52）。入力 cases/replay hash、固定 greedy top-k=1、30 kernel guard、active/package/build identity を照合した。
- gate rc=1 は trace failure ではなく、相対ファイル名の `SHA256SUMS` を output 外 cwd から検証した verifier cwd バグだった。output 内での read-only verifier は 3/3 OK。
- 根拠別の順位は、(1) 非可逆 AQ4 quantization、(2) AQ4 decode/matvec kernel 実装、(3) runtime/export/package mismatch、(4) input/sampling mismatch。top-k は intermediate schema に無いため、既存 source/path oracle の top-10 を補足参照した。

## 次の行動

- GPU/service を追加実行せず、package row の CPU dequantization と一段目 matvec の独立照合で、期待される lossiness と kernel/scale-index bug を分離する。
- attempt2 の raw output・logs・markers は保持し、analysis/evidence と script SHA を git に保存する。

検証: `/usr/bin/python3.12 tools/trace-qwen35-aq4-differential.py analyze` rc=0、`(cd .../differential-trace-gpu-v1-attempt2 && sha256sum -c SHA256SUMS)` 3/3 OK。未実施: GPU 再実行、top-k を intermediate schema に追加した再取得、全 256 row の独立 matvec oracle。
