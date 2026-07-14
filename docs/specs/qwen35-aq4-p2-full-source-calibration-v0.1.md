# Qwen3.5 AQ4 P2 full BF16 source calibration v0.1

## 前回の要点

P2の既存source-oracle v2は、Qwen3.5-9B BF16の3行についてhidden/logitのbounded sample、greedy、top-kだけを保存する。相対L2やfull-vector max_absを算出できないため、AQ4の正確性閾値を根拠付けるには別artifactが必要である。

## 今回の変更点

`tools/export-qwen35-aq4-source-calibration.py` は、既存v2を変更せず、同一revision・checkpoint・tokenizerのCPU BF16 forwardから、final RMSNorm後のhidden `[4096]` とraw pre-softmax logits `[248320]` をf32 little-endian sidecarへ逐次保存する。

- `vectors/hidden.f32le` と `vectors/logits.f32le` はrow順に連結し、`rows.jsonl` がbyte offset、shape、dtype、row SHA-256、nonfinite件数、greedy、top-kを持つ。
- 1 rowは `(4096 + 248320) * 4 = 1,009,664` bytesである。64K element chunkを使い、全row・全語彙行列をメモリに保持しない。
- 出力は新しいv1 schema/rootだけへ`renameat2(RENAME_NOREPLACE)`で排他的にpublishし、競合時は失敗する。既存 `source-oracle-v1/v2` を上書きしない。
- `--legacy-oracle` で既存v2のmodel revision、checkpoint aggregate、tokenizer aggregateを再検証し、旧3行のsample/top-k/greedyを必ずcross-checkする。不一致時は生成を失敗扱いとする。
- CPU preflightはcheckpoint bytesの2倍以上の`MemAvailable`と、予測sidecar容量の1.2倍以上の空きディスクを要求する。GPUが見える場合は実行しない。

`tools/validate-qwen35-aq4-p2-full-calibration.py` は全階層のunknown field、duplicate key、non-finite JSON、未登録file、全祖先/leaf symlink、hardlink、TOCTOU、sidecar offset/length/hash、stable top-k順序、legacy cross-checkを再計算する。読み取りは`O_NOFOLLOW`で開いたfdへ固定し、open前後のdevice/inode/size/mtime/ctime/nlinkを照合する。rowsはfile bytes、line bytes、record countを独立に制限する。nonfinite vector rowは順位計算を行わず、greedy/top-kをnullにした`blocked` artifactとして扱う。

`tools/compare-qwen35-aq4-p2-calibration.py` は次の2種類だけを受け付ける。

1. `source_gate`: independent BF16 sourceとAQ4 target sidecarを比較する。
2. `path_gate`: 同一AQ4 artifactのall-M=1とoptimized sidecarを比較する。

どちらもreference/candidateの全rowを要素列としてlockstepで読み出す。両artifactのchunk幅が異なる場合も境界を再分割し、短読と余剰要素を拒否して、f64累積で次を計算する。

```text
relative_l2 = sqrt(sum((candidate-reference)^2)) /
              max(sqrt(sum(reference^2)), 1e-30)
max_abs = max(abs(candidate-reference))
top_k_overlap = |set(top10_reference) ∩ set(top10_candidate)|
```

nonfiniteは別件数として保持し、threshold policyは生成しない。source/path identityは比較種別ごとに分離する。比較結果は観測分布artifactであり、閾値の自動bindやpromotion判定ではない。

## calibration case

`benchmarks/workloads/qwen35-aq4-p2-source-calibration-cases-v0.1.json` は、旧v2と同一の2 prompt/3 rowsを使うcompatibility canaryである。これはtoolchain検証用であり、AQ4閾値の十分な校正行列ではない。実運用のsubsetでは、AQ4 target fixtureと同一token ID列を使い、prompt/context境界、all-M=1、cold-batched、cached-prefix、M grid、decode first-stepを別caseとしてhash-bindする。

## CPU command

```bash
CAL=benchmarks/results/YYYY-MM-DD/qwen35-9b-aq4-production-opt-v0.1/p2/source-calibration-full-v1/run-id
CUDA_VISIBLE_DEVICES=-1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 tools/export-qwen35-aq4-source-calibration.py \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --cases benchmarks/workloads/qwen35-aq4-p2-source-calibration-cases-v0.1.json \
  --legacy-oracle benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2 \
  --output "$CAL/source-full" --chunk-elements 65536 --top-k 10 --threads 1
python3 tools/validate-qwen35-aq4-p2-full-calibration.py --artifact "$CAL/source-full"
```

既存v2のruntimeはCPU 1 thread、3 rowsで17.011秒（model load込み）だった。長いpromptのforward時間は未測定なので、canaryの実測を行列時間の根拠にし、未測定値から閾値や固定timeoutを推測しない。

## 次の行動

AQ4側はRust driverのresult JSONへfull vectorを埋め込まず、prepared token直後に同じrow contractのtarget sidecarを出す専用capture hookを追加する。その後、source gateとpath gateを別run rootで比較し、policy値は独立レビューと事前bindを経てから固定する。
