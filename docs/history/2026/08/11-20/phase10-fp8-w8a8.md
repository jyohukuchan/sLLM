# Phase 10 FP8 W8A8 history

## 2026-08-14: 詳細計画

- Phase 9のprepared execution/provider境界を再利用し、exact `gfx1201` native、exact `gfx1030` emulation、
  exact `gfx1030` load時BF16 conversionを別providerとして計画した。
- verified 4B BF16 lockからreproducible OCP E4M3FN sidecarを作り、第三者の小型FP8 checkpointへ依存しない。

## 2026-08-14: format、loader、graph

- logical dtype、FP8 encoding、scale granularity、resident representationを分離した。
- converterはQwen3.5-4Bのtext-linear 248 tensorをper-output-row scale付きE4M3FNへ変換する。
  block 128候補はhipBLASLt outer-vector contractと一致しないため、実装開始時にouter-rowへ固定した。
- source model lock、tool identity、完全artifactおよびtensor range hashをmanifestへ保存し、loaderはsource、
  manifest、artifact、shape、range、hashをfail-closedに検査する。
- production graph、resident provisioning、CLI、benchmark、OpenAI serverへFP8 sidecar/providerを接続した。

## 2026-08-14: target別実装

- R9700 exact `gfx1201`はdynamic per-row activation quantizationとhipBLASLt OCP E4M3FN W8A8を実装した。
  solution query、descriptor、scale pointer、workspaceをprepared planに所有し、request execution allocationを0にした。
- V620 exact `gfx1030`はbyte-decode W8A8 emulationをcorrectness providerとして実装した。
- 同じsidecarをload時に一度だけBF16へ変換する`converted-bf16` providerを追加した。これはnative FP8でも
  silent fallbackでもなく、resident encodingと追加VRAMを明示する。
- native/emulation operatorはM=1/M=3で独立oracleをPASSし、kernel id 5/6、fallbackなし、cleanup 0を確認した。

## 2026-08-14: model精度、service、性能

- R9700の実4B native generationをPASS。3入力のlogits比較はtop-1全一致、最大KLD 0.02394でgate 0.05を満たした。
- V620の実4B `converted-bf16` generationはBF16 output tokenと一致した。
- native FP8 serverで`/v1/models`、non-stream chat、SSE終端をPASSし、shutdown時のrequest/workspace、
  quarantineを0とした。
- R9700 32/32ではFP8/BF16のprefillが486.26/531.71 tok/s、decodeが31.58/37.04 tok/s、E2Eが
  1065.72/912.86 msだった。resident VRAMは4.847/8.412 GBで約42.4%削減した。
- 速度は改善しなかったためnative FP8をdefaultへ昇格しない。V620 emulationもcorrectness-onlyとする。
- workspace全target test、clippy、format、host CTest 3/3、両local GPUの最終operator evidenceを完了した。

詳細な契約と数値は[Phase 10 archive](../../../../plans/archive/2026/08/11-20/phase10-fp8-w8a8.md)を正とする。
