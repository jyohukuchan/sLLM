# Phase 50 R9700実機移植・MI300X wave64引継ぎ履歴

## 2026-08-23: 詳細計画作成

- Phase 49完了後の既定次作業として、Phase 50をR9700 exact `gfx1201`実機移植とMI300X exact `gfx942` wave64引継ぎ準備へ詳細化した。
- 既存のPhase 50=R9700実機、Phase 51=MI300X実機という順序は維持した。Phase 50のMI300X範囲はexact feature compile/link、
  host selector非選択、ABI/workspace/数値契約の引継ぎまでであり、実機7行と性能PASSはPhase 51が所有する。
- Phase 49変更をtarget共通control-plane、gfx1201で再測定するwave32候補、gfx1030限定、不採用、gfx942 wave64再設計へ分類した。
  GQA P32や各bundleをgfx1201へ無条件に有効化せず、fresh 7行baseline/profileの残差順に個別採否する。
- R9700 tuple、target専用build、UUID/BDF/arch相互照合、ROCm loader closure、通常5行3+10と長時間2行1+3、
  selector境界、数値・資源、停止／再計画、V620 focused regressionを計画へ固定した。
- R9700の全7行llama.cpp同等は目標と残差報告であり、Phase 50完了やPhase 51開始のhard gateにはしない。
- この時点では計画文書だけを作成し、production source、GPU、VM、外部service、commit/pushを変更していない。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)

## 2026-08-24: 実機検証と限定採用による完了

### 判定

- Phase 50は`completed-limited-adoption`として完了した。R9700 exact `gfx1201`で7行を実行し、6行がPASS、
  `100000/2`はlayer 31のKV virtual commit OOMでFAILとなった。OOMをPASSや省略へ読み替えず、Phase 51へ未達として引き継ぐ。
- 全PASS行はHIP-only、CPU/別backend fallbackなし、生成反復一致、実行後のprocess終了とVRAM/GTT復帰を確認した。
  llama.cppとの差は目標報告として固定し、同等達成をPhase 50完了のhard gateにはしない。

### R9700 最終7行（sLLM対llama.cpp、E2E中央値）

| input/output | sLLM (ms) | llama.cpp (ms) | 比率 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 17/17 | 407.914958 | 332.726339 | 1.22598x | PASS |
| 32/32 | 759.729283 | 604.069210 | 1.25769x | PASS |
| 1,024/128 | 3,383.626517 | 2,509.156370 | 1.34851x | PASS |
| 32/256 | 5,959.859813 | 4,712.364031 | 1.26473x | PASS |
| 10,001/2 | 4,002.833893 | 2,072.475558 | 1.93143x | PASS |
| 100,000/2 | OOM（layer 31 KV、peak 26,414,592,000 B） | 69,783.055486 | — | FAIL（未達） |
| 32/20,000 | 532,486.026195 | 377,632.767706 | 1.41006x | PASS |

全行のrunner要約は`ci/matrix/phase50-r9700-summary-v1.json`に追跡し、raw producer要約は
`/home/homelab1/.local/share/sllm-evidence/phase50/r9700/final-current-20260824/sllm/phase50-r9700-sllm-v1.json`
（SHA-256 `39f95d1f0964624bcd3405108d794781cc44e1ab4e180d1d37d717927fda6e29`）へ保存した。固定peerはllama.cpp
tag `b10453`、commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`、wrapper SHA-256
`7e899df20101ee772e66f63b85d3c447c697a06bd74eef4758d2b1401e97faa1`である。
llama.cpp producer要約は
`/home/homelab1/.local/share/sllm-evidence/phase50/r9700/baseline-current-20260823/llama/phase50-r9700-llama-v1.json`
（SHA-256 `56777f25bf054335ba46df4ffaecbcbec688ac0bc95743608d9a58c0c6774bc8`）、外部aggregateは
`/home/homelab1/.local/share/sllm-evidence/phase50/r9700/final-current-20260824/phase50-r9700-summary-v1.json`
（SHA-256 `a8dce84097f760ec6da415e060e8264ca73ead34bd2c33ec001bcb9e4785a7f6`）である。追跡用に整形したaggregateの
SHA-256は`b14d0bdc5f532791abede149cc440fedd7ac3a2d7bfe6319de322c1ffc4f32e0`である。

開始前sLLM比の通常5行改善は、17/17 `9.43%`、32/32 `10.44%`、1,024/128 `9.26%`、32/256 `10.38%`、
10,001/2 `1.10%`、32/20,000は別baseline比 `50.49%`であった。

### 採用した候補と棄却・引継ぎ

- exact `gfx1201`で residual RMSNorm、GDN projection、MLP gate-up-SiLU bundle、GQA4 P32（KV `>=4096`）を採用した。
  A/Bは短行でcontrol `451.8648785 ms`から融合3 `410.794651 ms`、P32行でcontrol `10417.448939 ms`→P32
  `7671.955934 ms`→P32+融合3 `6969.896471 ms`となり、最終成果物へ統合した。
  raw A/Bは`/home/homelab1/.local/share/sllm-evidence/phase50/r9700/candidate-short-ab-20260824`と
  `/home/homelab1/.local/share/sllm-evidence/phase50/r9700/candidate-p32-ab-20260824`へ保存した。
- target共通のsemantic bundle、M=1 native／M>1分解、prepared completion、device property cache、DerivedContiguous、
  runner/schemaは採用した。gfx1030限定のattention preprocess、linear decode/short-column、short/mixed matmul、
  deferred completion、terminal last-row、scaled-prefillはgfx1201へ自動横展開せず`keep-gfx1030-only`または
  `decompose/baseline`とした。long-prefill v2とHIP Graphは`reject`を維持した。
- wave32固有のshuffle、GQA P32 partition、linear ownership、attention preprocess、matmul solutionはgfx942へ直接流用せず、
  wave64 lane ownership・block・LDS/register・barrierとTensile/hipBLAS solutionをPhase 51で再設計する。

### V620回帰とgfx942引継ぎ

- 共通source変更後のV620 exact `gfx1030`通常5行は5/5 PASS、E2E中央値は
  `428.8887465, 757.1793445, 4210.957128, 5781.122031, 13479.9155275 ms`で、Phase 49比は
  `+1.16%, +0.87%, -0.08%, +0.03%, -0.21%`。全行HIP-only、cleanup復帰を確認した。
  producer要約は`/home/homelab1/.local/share/sllm-evidence/phase50/v620/regression-current-20260824/phase49-v620-sllm-v1.json`
  （SHA-256 `f090592a946a136a2a4d6288311143932dea281f44e9fbf87a457feaa82eebd3`）へ保存した。
- MI300X実機実行は行わず、exact logical target `gfx942`のproduction Cargo build（binary SHA-256
  `9583b7d678897ce66d8ef8ce4ddcf4080a4ce4ede2bda6c0375281f2ac9af7a2`）、direct compile/link probe（SHA-256
  `1605d87928f719d60ba1e7f4bb51e5c20aba78f69d25d69ccf846268301f13f0`）、host selector 1/1 PASSを取得した。probeは
  `gfx942:sramecc+:xnack-` bundle、ELF ABI4、flags `0xE4C`、wave64、`.text` 1152 Bを確認し、gfx1201 provider非選択を示した。
  これはMI300X `project-verified`や性能PASSではなく、Phase 51の実機検証入力である。

### 数値・既知制約・固定identity

- residual RMSNormはBF16 RNEとFP32 accumulatorの順序を承認shapeで確認した。GDN projectionは有限値scopeで
  DecodeReduction 4要素の順序を確認したが、NaN payloadの完全一致はhelperのcanonicalization差により未検証である。
  MLPはfull-model反復と有限値scopeを確認し、専用operator oracleは未追加。GQA P32は既存のgamma8→gamma12近似とpartial
  merge順序の数値ポリシーを維持し、gfx942では再設計・再検証する。
- 最終sourceはbase HEAD `3ffefae3cf83cd4a0a9d560c01e277004d541e4d`上の未コミットcandidate。R9700 target binary SHA-256は
  `ef258419770e314fb2e6f0987426acbfee536a4021d0563ea1098851f0d5997d`、model SHA-256は
  `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、lock SHA-256は
  `425151d06832347a01b946b27336ceffac074eb7f6932af61e8c9821edc1e318`である。
- R9700 identityはBDF `0000:07:00.0`、UUID `GPU-a8e9ddefa2d60f55`、exact `gfx1201`。100k OOM、MI300X未検証、NaN payload未検証、
  MLP専用oracle未追加は既知制約として残し、Phase 51の課題に明示した。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)
