# T1 mixed request-state AQ4 payload sharing

## 前回の要点

- `manifest-all` full mixed request-state smokeは32層全体で通った。
- ただしrequest slotごとにresident layerをロードしており、AQ4 payloadもslotごとに再ロードしていた。

## 今回の変更点

- `PackageAq4ResidentMatvec::load` で同じ `WeightRegistry` 内の同名tensorを再利用するようにした。
- linear-attn/self-attnのresident batch ownerで、request slot間に同じregistryを渡すようにした。
- R9700 `manifest-all` smokeは `verified=true` のまま通り、`slot_aq4_payload_registry_shared=true` を出力した。
- `layer_load_ms` は `18416.054962` から `17828.586114` に下がった。

## 次の行動

1. `scale_values_buffer` とpassthrough weight buffersをslot間共有する。
2. weight-only resident bundleとper-request state/cache/workspaceへ分ける。
3. real package throughput rowへ進む。
