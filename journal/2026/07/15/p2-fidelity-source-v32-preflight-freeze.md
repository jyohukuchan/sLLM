# P2 source fidelity v32 事前確認の固定

## 前回の要点

v16（`--threads 16`）のsource captureは24行を生成した後、旧source-oracle-v2のfixture行がproduction splitと重ならないため、旧互換確認で停止した。production splitと旧fixture行の関係をdisjoint-by-policyとして検証できるよう、exporterとvalidatorを更新した。

## 今回の変更点

v32用の新規入力・attemptパスを作成し、モデル読み込み前の確認だけを実施した。

- cases SHA256: `53f256bc8f5ed4036cfb1a9a98c0c9d9197bb980e1ef91d7ff01cf73001369a8`
- plan SHA256: `1b4f8c244e922ab73c0bb026216d8333a9cfe57c23e6695c4141554d117693c0`
- run script SHA256: `353f2287c703d2234e43ec6efce5d8b5bfe83bca0e4455f910109714640306c9`
- preflight SHA256: `bef52ca377a44596a18ef44e111a4a664f11123b354715ff90516cf21b393ea2`
- `--threads 32`、OMP/MKL 32を確認。
- `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES`は空。`torch.cuda.is_available()`はfalse、デバイス数は0。
- checkpointは4 shard、19,306,310,880 bytes。MemAvailableは79,938,424,832 bytes、必要headroomは38,612,621,760 bytes。空き容量は2,577,684,742,144 bytes。
- 旧fixture行 `fixture-prompt-0` と `fixture-prompt-1` はsplitの`attempt2_exclusions.case_ids`に完全包含。

## 次の行動

親エージェントのGo判断まで、source wrapperとモデル読み込みは実行しない。Go後にwrapperが生成するstdout/stderr、time、vmstat、source outputを追加し、attempt全体のSHA256SUMSを確定する。
