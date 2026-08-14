# Phase 11 CDNA3 port history

## 2026-08-14: 詳細計画の作成

- Phase 11をexact `gfx942`、wave64、BF16、FNUZ FP8の実装・compile/oracle phaseとして具体化した。
- model storageのOCP E4M3FNをload時にE4M3FNUZへ数値変換し、テスト専用FNUZ modelを作らない既存方針を維持した。
- AMDの公開MI300X例でVMMなしが示されるため、opaque KV契約を維持した`contiguous-resident` providerを
  Phase 11へ追加する。Paged Attentionへの選定変更や実行時silent fallbackではない。
- 実機evidenceはPhase 12へ分離し、未所有GPUをPhase 11実装完了のblockerにしない。
- 詳細は[archive](../../../../plans/archive/2026/08/11-20/phase11-cdna3-port.md)を正とする。

## 2026-08-14: FNUZとexact gfx942 provider

- scalar FNUZ codecを全256 byteと境界値で検査し、OCP E4M3FN sidecarをresident upload時に数値変換する経路を追加した。
- dynamic activationもFNUZへ量子化し、weight/activation FNUZ、outer FP32 scale、FP32 accumulation、BF16 outputを
  hipBLASLt planへ固定した。極小負数をunsigned zeroでなくFNUZ NaNへ丸める境界不具合を修正した。
- FNUZ GPU encoderを128候補総当たりから7段の二分探索へ変更した。
- `gfx942`はexact Code Object V6、wave64、`xnack=off`、`sramecc=on`、generic processor version 0でbuildする。
  `gfx9-4-generic`やOCP payloadのreinterpretは使用しない。

## 2026-08-14: wave64、KV、production統合

- M=1 BF16 MMVFとRMSNormにwave64専用kernel ID/symbolを追加し、gfx942 providerだけが選ぶようにした。
  native kernel監査で残った32定数はRoPE half dimension/head shapeであり、lane reductionではないことを確認した。
- opaque token-major FP16 KVへ`contiguous-resident` memory kindを追加した。VMM capability=falseならlogical capacityを
  通常のdevice allocationで確保し、trueなら既存virtual-contiguous providerを選ぶ。runtime failure後のfallbackではない。
- fake VMM capability fixtureでfalse時の1025-token allocation、metadata、cleanupと明示virtual拒否を検査した。
- CLI/server/model resident graphへgfx942 `native-fnuz`を接続し、selected KV memory kindをrequest auditへ追加した。

## 2026-08-14: handoffと検証

- MI300X x1候補manifestと、preflight、operator、slice、full-model、service、performanceの順序付きdry-run runner/schemaを追加した。
- host CTest 3/3、affected Rust tests、exact gfx942 native compile/link、R9700 native FP8、V620 FP8 emulationをPASSした。
- final local gfx942 candidateは全10 HIP bundleがexact `hipv4-amdgcn-amd-amdhsa--gfx942`であり、CLI SHA-256は
  `00c0d1cf3232dc984d4fb89178c23079cc980a1cd7dc2d13f8c5b095d0ab59b3`、server SHA-256は
  `d80e9361d58f5eb461335b4a580ff97b54f9da8ebdfe5b54b8319bda91072d52`だった。binary自体はrepository外の
  local build artifactであり追跡しない。
- local compileはMI300X実行、FNUZ hipBLASLt solution support、gfx942数値、性能を証明しない。これらは
  [Phase 12](../../../../plans/active/2026/08/11-20/phase12-mi300x-validation.md)でfail-closedに取得する。
