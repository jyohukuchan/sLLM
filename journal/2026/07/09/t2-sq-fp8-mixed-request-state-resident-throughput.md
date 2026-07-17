# T2 SQ FP8 mixed request-state resident throughput

## 前回の要点

- full mixed AQ4 `manifest-all` resident throughput baselineはB=1/4/8で取得済み。
- 次はSQ FP8 candidateを同じfull mixed resident pathへ接続し、AQ4/SQのqualityとthroughputを同じschemaで比較する段階だった。

## 今回の変更点

- `sq-fp8-token-ids-mixed-request-state-smoke` を追加した。
- `PackageAq4ResidentMatvec` にSQ/F32 materialized storageを追加し、artifactにあるtensorだけSQ FP8からF32 resident bufferへmaterializeする経路を作った。
- full mixed loaderからoptional `Qwen3PackageSqOverlay` をself-attention/linear-attention batch layerへ渡すようにした。
- stdoutに `sq_execution_mode=materialized_f32_fallback` を追加した。
- `run-external-benchmark.py` は `sq_execution_mode` をworkload metadataとして保持する。
- R9700で `kup6_gate5_down5` artifactをB=1/4/8のfull `manifest-all` で実行した。

## 結果

- B=1はAQ4/SQ final top1が一致した。
- B=4/B=8は2番目requestがAQ4 `5446` に対してSQ `1622` になり、quality guardは不合格。
- SQ resident end-to-end tok/sはB=1 `7.260317`、B=4 `14.585119`、B=8 `17.961835`。
- この速度はmaterialized F32 fallbackを含むため、native SQ kernel速度ではない。

## 次の行動

1. top1 driftが出ない保守的SQ candidateをfull mixed pathで再評価する。
2. SQ FP8 direct matvecまたは低遅延dequant matvecへ進む。
3. native SQ rowができたら、同じB=1/4/8 schemaでAQ4/SQ/vLLM比較へ戻る。
