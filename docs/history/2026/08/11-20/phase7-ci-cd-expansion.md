# Phase 7 CI/CD拡張履歴

## 2026-08-14: 受入条件固定

Phase 6完了時点のworkflow、CI matrix、Phase 5 performance runner、compatibility文書を監査した。
host H0〜H2、exact `gfx1030`/`gfx1201` H3、semantic G1、G2/G3、P0/P1の個別runnerは存在するが、
daily/weekly/releaseを選択するversioned profileと定期workflowは未実装だった。Phase 7では既存runnerを
再利用し、profile、互換性record、trigger、保持期間、compile target集合を新しい契約として追加する。

V620で長期実行される可能性があるGIMPS等のforeign workloadと性能測定を同時実行しない。daily代表tupleは
R9700 `gfx1201`とし、weekly/releaseのV620 rowはpreflightでforeign processまたはactivityを検出した場合に
性能PASSへ読み替えずfail-closedまたは明示的なinfrastructure dispositionとする。

## 2026-08-14: profile・workflow・compatibility compile実装

`phase7-ci-profiles-v1`でdaily、weekly、releaseのtrigger、host suite、GPU tuple、compile target、
retention、blockingの意味を固定した。scheduleはdaily `17 18 * * *`、weekly
`47 18 * * 6` UTCとし、releaseはmanualかpublished releaseだけが選択する。GitHub-hostedの
profile/host/compile jobとlabels `self-hosted`, `sllm-semantic-g1`, `rocm-7.14`のtrusted GPU jobを分離し、
public PR triggerは追加しなかった。official actionsは完全commit SHAへ固定し、workflow権限は
`contents: read`だけとした。

GPU controllerが実際に呼ぶPhase 5 direct full-model rowの範囲はG0 preflight、G3 end-to-end、
G4 exact tuple、registered P1 runnerである。profileの`gpu_tiers`はこの4 tierだけに限定し、
dedicated G1/G2 runnerを実行したとはclaimしない。dailyの`p0-observation`はこのP1 runnerの
値をhard thresholdに使わないlaneの意味であり、summaryは`executed_tiers`をexactに返す。

compatibility compile runnerは`gfx1030`〜`gfx1036`、`gfx1200`、`gfx1201`、`gfx942`の10 targetすべてで
local draft PASSとなった。link後ELFから`.hip_fatbin`を抽出し、bundle order、Code Object V6、
device metadataのexact targetを検査した。device imageは`gfx1030`〜`gfx1036` 4,432 bytes、
`gfx1200`/`gfx1201` 4,624 bytes、`gfx942` 5,264 bytesだった。これはcompile-onlyであり、
runtime、実機互換性、vendor supportのclaimを生成しない。

## 2026-08-14: R9700 daily path実測

dirty local candidate commit `423001990bc9daafb110a6292d5434e350f5b5f0`、base Git tree
`379705df86b8edbe35da37eb08b5bcd99167662a`でdaily profileを解決し、canonical R9700 `gfx1201`上の
Qwen3.5-4B `short-odd`を実行した。既存Phase 5 direct runnerのwarmup 3回・計測10回を再利用し、
exact tuple、fallbackなし、health、cleanupがPASSした。

| metric | median | p10–p90 |
| --- | ---: | ---: |
| TTFT | 2.862 s | 2.842–2.908 s |
| prefill | 5.951 token/s | 5.857–5.993 token/s |
| decode | 1.672 token/s | 1.670–1.675 token/s |
| TPOT | 597.963 ms | 595.261–600.450 ms |
| E2E | 12.440 s | 12.418–12.484 s |
| resident / peak VRAM | 8,411,592,192 / 8,540,569,292 bytes | single observation |

compact summary SHA-256は`3087f379a80ed30292c099253ade7b5669853f64e90f3211b4984cf4384f23ab`、
underlying report SHA-256は`9705fe6ff6dc1f0e181ecf7df0fecb389ac096b3e8ede00a6ddc056640c09d52`、
raw result SHA-256は`c56c6635e24fd1016e1ef8ae42fd8142f34b01eee20d14597097f60047e5cac0`である。
このrunはworkflow pathのdraft確認でありimmutable release evidenceではない。performance hard gateや
optimized/faster claimに使わない。

## 2026-08-14: verification

- Phase 7 lifecycle/schema検査: PASS。
- focused Python unit 7 tests: PASS。
- suite/host/path matrix検査: PASS（suite 32、host row 3）。
- JSON/schema/workflow manifest検査: PASS。
- exact target compile-only: 10/10 PASS。
- R9700 daily GPU observation: 1/1 PASS。
- H0 full row: 504/504 PASS（local-development、immutable=false）。

## 2026-08-14: GIMPS終了後のdaily運用更新

ユーザー指示によりGIMPS終了を確認し、V620をdaily観測へ復帰させた。profile revision 2ではdailyの
compile targetを`gfx1030` / `gfx1201`、GPU tupleをcanonical V620 / R9700の両方へ変更した。
foreign process、activity、health、fallback、cleanupのfail-closed契約は変更しない。以前のR9700限定runは
旧運用時点の有効な履歴として残す。

revision 2のdaily profileをlocal dirty candidate commit
`423001990bc9daafb110a6292d5434e350f5b5f0`、base Git tree
`379705df86b8edbe35da37eb08b5bcd99167662a`で実行し、canonical V620/R9700を2/2 PASSした。
両rowともfallbackなし、health/cleanup PASS、実行tierはG0/G3/G4/P1だった。実行前後に全GPUの
processがゼロであり、強制終了対象はなかった。

| target | TTFT median | prefill | decode | TPOT median | E2E median | resident / peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | 7.574 s | 2.246 token/s | 0.863 token/s | 1,154.459 ms | 26.110 s | 8,411,592,192 / 8,540,569,292 bytes |
| R9700 `gfx1201` | 2.849 s | 5.978 token/s | 1.671 token/s | 598.258 ms | 12.434 s | 8,411,592,192 / 8,540,569,292 bytes |

compact summary SHA-256は`d3780b3b38276623f0f4890ecc852d8d49538fa7d043789b410fa5d938fe8bb5`、
V620 report/raw SHA-256は`42b92fb51cff80d43c91983f2f18980c75f80f3e678ff30b13e7c16036e66f01` /
`186aa5b081a53c98b5219845c43613d5179fe40ff1d0ced0f3b558cade5189b6`である。これは新daily pathの
draft観測であり、immutable release evidence、性能優位性、hard threshold、長時間安定性をclaimしない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase7-ci-cd-expansion.md)
