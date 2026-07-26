# AQ4_0 `matvec_add` 帯域効率の調査と候補実装

## 前回の要点

- `aq4-decode-walltime-accounting` は C=1339 で 292 module launch/token、
  `ullm_aq4_matvec_add_f32_kernel` が 64 launch/token・3.697842250 ms/token・
  payload 335.740 GB/s（640 GB/s rooflineの52.4594%）であることを確定した。
- 同じAQ4_0の `silu_mul` は3.402800344 ms/token・532.485 GB/s（83.2007%）。
  したがって本件は launch-gap やnorm融合ではなく、projection内の差分を調べる。

## 今回の変更点

- gfx1201向けの静的ISA、launch geometry、AQ4_0 payload構成、残差入出力を監査した。
  addもSiLU-mulもLDS treeではなくwave32 shuffle reductionであり、spillsもない。
  残差read + output writeの上限2,097,152 B/tokenはweight payloadの0.1689%で、
  payload帯域差の主要因ではない。
- addのgeneric g8/g16 traversalを、low-nibble-first、scale addressing、group scale、
  row scale、residual順序を保った固定g8/g16 traversalへ置換した。gfx1201/RPB=8では
  旧shuffle bodyを `ULLM_AQ4_MATVEC_ADD_USE_SHUFFLE_REFERENCE=1` でのみ選べるため、
  同一worker内A/Bができる。
- static candidateは1434→820 instruction、SALU 922→395、VALU 399→321、
  LDS/spillなし。VGPRは30→49なので静的値だけでは採用しない。
- current mainは本番の4:1 grouped-split実装を持たないため、main基準staging buildを
  R9700測定前に除外した。`9d864350`を隔離し、BZの`c8074928`と
  `ullm_runtime_hiprtc_sources.inc`／`part_01.inc` blobが同一であることを確認した。
  その上でcandidate worker、decode/prefill profiler、GPU differential testを再buildし、
  SHA-256を `candidate-grouped-build-provenance.json` に固定した。
- R9700の同一worker A/Bはdirect 73.895446→77.679674 tok/s（1.051211×）、
  production grouped 74.591159→78.284628 tok/s（1.049516×）だった。C=1339で
  両bodyとも292 module launch/token・add 64 launch/token、GPU/CPU差分とgreedy runtimeも
  一致した。prefill p=2048/M=128は974.984645→977.087601 tok/sで非回帰だった。
- matched traceでadd kernelは128 launchあたり7.406139→6.284984 ms、payload diagnosticは
  335.266→395.073 GB/sとなった。これはprofiler rangeをthroughputには用いていない。
  gfx1201 PMCのdynamic VALU/occupancy raw値は0だったため、achieved occupancyは未確認と
  記録した。
- `/opt/ullm/aq4-matvec-add-group-specialized-v0.1/` にroot:root/0555 workerを配置し、
  lightweight promotionは10実promptでblocking 0・10/10 exact match・`activated`を記録した。
  直後にactive manifestが外部操作により既存BZ SHAへ戻った（原因未確認）。指定どおり
  想定外のactive stateを上書きせず、既存workerをactiveへ復旧した。

## 次の行動

1. candidate artifactと品質証跡はimmutableに残す。current active SHAが想定外に変化したため、
   次のactivationはその時点のmanifest所有者とstateを再確認してから新しいtransactionとして
   行う（本記録のcandidateを無条件に再上書きしない）。
2. `matvec_triple` / `qkv_z_gate_beta` は本候補と同じ一stream traversal差分を持たないため、
   新しいhardware evidenceなしに変更しない。
