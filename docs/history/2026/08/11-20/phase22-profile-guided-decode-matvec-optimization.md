# Phase 22 profile-guided decode M=1 matvec optimization history

## 2026-08-18: Phase 21結果を受けた詳細計画とP22-A0開始

- Phase 21は17 ownerを1 terminal fence eventへ集約する構造削減をPASSしたが、Qwen3.5-4B BF16固定laneのE2E中央値は
  V620で0.14%、R9700で0.18%遅く、いずれもnoise内だった。production defaultはPROFILEDへ戻し、同じevent同期
  micro-optimizationをPhase 22へ継続しないことにした。
- 06:00 JST前にPhase 21を完了したため、ユーザー条件に従い、認識済み性能backlogからPhase 22を開始した。
  DeepSeek V4とTurboQuantは含めず、最初のwork unitをdense BF16 decodeのM=1 matvecへ限定した。
- current sourceを確認し、BF16 `M=1` provider selectionがRDNA2/RDNA4で単一decode v4、gfx942で単一wave64 variantを
  選び、関数へ渡される`K/N`を選択に使っていないことを確認した。decode v4は256 thread、1 output column/workgroup、
  paired BF16・non-temporal weight load、FP32 accumulation、BF16 RNE outputである。
- reviewed 4B specとgraphから、1-token dense decodeのM=1 matvecを249回と棚卸しした。内訳はMLP gate/up
  `K=2560,N=9216`が64、MLP down `9216,2560`が32、GDN/full q系`2560,8192`が32、GDN z
  `2560,4096`が24、GDN/full out `4096,2560`が32、full k/v `2560,1024`が16、GDN b/a
  `2560,32`が48、tied vocabulary projection `2560,248320`が1である。
- 直近のprofile履歴ではdecode matvecがGemma device timeの84.28%、Qwenの63.28%を占めた。Phase 9のQwen 1-token
  profileでもMMVFは約15.5/18.6 msだった。履歴値は候補選定だけに使い、Phase 22採用判断はcurrent sourceのfresh profileで行う。
- 反復回数とaggregate weight trafficが大きいMLP `2560→9216` / `9216→2560`を最初の比較familyへ固定した。
  1回のweight footprintが最大のvocabulary projectionは同じA0/A1 profileの独立controlとし、測定なしにMLP variantへ束ねない。
- 最初のproduction candidateは`target/M/K/N`で選ぶadditive static providerとし、最大3 microvariantから1つだけを実装する。
  gate/up fusion、RMSNorm/QKV fusion、H2D統合、event pool、graph replay、batching、量子化pathを救済策として混ぜない。
- primary final laneはPhase 21と同じQwen3.5-4B BF16 audit GGUF、prompt `Hello world`、greedy 3-token output、
  canonical V620/R9700単独可視化、PROFILED mode、各3 warmup + 10 measured以上、baseline/candidate交互順とした。
- 本時点のPhase 22進捗はP22-A0 source inventory、scope、contract、受入条件、最初のshape family選定までである。
  source kernel/providerの変更とfresh GPU profileはまだ行っていない。

## 2026-08-18: fresh profile、candidate評価、棄却、closeout

### Fresh baseline

- `sllm-phase22-matvec-evidence`を追加し、Qwen3.5-4B dense decodeの8 distinct shapeをdevice-resident prepared opとして
  各3 warmup + 10 measuredした。入力とweightはBF16 one、expected outputはexact BF16 `K`とし、transfer/preparationを
  kernel event時間から除外した。reportはtarget、kernel ID/symbol、grid/workgroup、effective bytes、全sample、fallback、
  allocation high-water、cleanupを記録する。
- current v4のcalls-per-token加重medianはV620 39.344 ms、R9700 28.172 msだった。V620のrole別medianはgate/up
  244.183 us、down 179.422 us、q 220.163 us、z 121.382 us、out 109.822 us、k/v 48.420 us、b/a 24.941 us、
  vocabulary 2.530 msだった。MLP 2 shapeは合計21.369 ms/tokenで最大familyだった。
- R9700はclock stateの段差を含んだため個別sampleとMADを保持し、candidateの採否はbaseline/candidate交互順と
  target別selectionで判断した。両targetともHIP-only、fallbackなし、output exact、cleanup 0だった。

### Candidateとoperator screening

- 256-thread blockを8 wave32へ分け、1 waveが1 output columnを担当する`wave32x8` candidateを新ID/symbolで実装した。
  paired BF16 load、non-temporal weight load、FP32 accumulation、BF16 RNEを維持し、最初はMLP 2 shapeだけをstatic selectionした。
- V620ではgate/up medianが244.503→165.163 us（約32%短縮）した一方、downは180.702→243.562 us
  （約35%悪化）した。2 shapeのcalls-per-token加重値は21.431→18.364 ms（約14%短縮）だったため、次の比較では
  candidate selectionをgate/upだけへ限定した。
- R9700は主要2 shapeの加重値が14.547→16.422 ms（約13%悪化）し、candidateをfull-model default候補へ進めなかった。
  current v4を維持した最終controlは3 warmup + 10 measuredで中央値161.104 ms、MAD 0.362 msだった。
- 独立f64 oracleへ`M=1,K=9216,N=2560`を追加し、両GPUで18/18 PASSした。監査中に既存evidenceのdispatch validatorが
  Phase 21で追加済みのM=2..8 serial-row ID 12/13を反映していないことを検出し、current provider identityへ同期した。

### Full-model採否

- modelは`phase20-audit-qwen35-bf16.gguf`、lock fingerprintは
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、request contentは`Hello world`、
  greedy 3 completion tokenに固定した。baseline/candidateは同じV620へ同時常駐させ、奇数組B→C、偶数組C→Bとして
  3 warmup + 10 measuredをcounterbalanceした。
- 全runでoutput `Hello! How`、prompt/completion token 14/3、submission/kernel 1404/1476、HIP-only、fallbackなし、
  request state/workspace cleanup 0が一致した。baseline中央値254.668 ms、MAD 0.319 msに対し、gate-only candidateは
  255.982 ms、MAD 0.531 msで0.52%遅かった。
- operator局所改善がproduction wall改善へ転化しなかったため、受入条件11/12に従いcandidateを棄却した。
  candidateのkernel ID/symbol/body/selectionは最終sourceから除去し、current decode v4/wave64を全target/shapeのdefaultに維持した。
  Phase 22は限定候補の否定結果と再利用可能なshape profile/oracle拡張を成果として完了し、高速化済みとは表記しない。

### Final verification

- `cargo fmt --check`、Phase 22 profileとmatmul evidenceのhost test 8件、`git diff --check`をPASSした。
- clippyは既存の`manual_contains`、`needless_borrow`、`collapsible_if`だけを明示allowし、Phase 22対象binを`-D warnings`でPASSした。
- exact gfx1030/gfx1201/gfx942 release compile、canonical V620/R9700の18-case f64 oracleと8-shape exact profileをPASSした。
- JSON/schema/manifest、H3 public-runtime contract validator、H3 runner/contract 65 testをPASSした。
- integration reviewではcandidate残存、shape selection、dispatch identity、fallback、cleanup、plan/history整合を確認し、
  correctness/security blockerは残らなかった。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase22-profile-guided-decode-matvec-optimization.md)
