# Phase 4: Qwen3.5-2B・9B互換性確認履歴

## 2026-08-11: 計画作成

- Phase 3完了後のPhase 4として、2B/9Bを同じQwen3.5 dense text pathへ適用する計画を作成した。
- official Hubで2B revision候補`15852e8c16360a2fea060d615a32b45270f8a8fc`、9B revision候補
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`を観測した。
- 2Bは24 layer/hidden 2048、9Bはhidden 4096かつuntied LM headであり、単なる4B fixture差し替えでは
  ないことを計画へ反映した。
- 実装、model download、GPU testはまだ開始していない。

## 2026-08-11: M1 model lock完了

- 2B/9Bを各official Hub完全revisionへ固定し、全runtime/evidence file、license、LFS identity、
  safetensors catalogを含む完全lockを追加した。fingerprintは2B
  `sha256:304e19f8b8ef78bab1848a6cfb46ac619a8ca5c8fd052cac1c43fc3f4d6dcdb3`、9B
  `sha256:2d2bc642540e97d4681f8c66140e09f305f487476bb9fe238ca82a298febf893`である。
- checkout外cacheの全byte検証、load plan、1/3/17/255/256/257 token graphを同じtestで確認した。
  external cache testは678.50秒でPASSした。
- 2Bはtied embedding/output、9Bは独立`lm_head.weight`をrequired output projectionとして
  positive contractで確認し、既存のmissing/wrong shape/unexpected/tied矛盾等のnegative contractを維持した。

## 2026-08-11: M2 shape-driven共通実装完了

- 2B/4B/9Bをreview済みtyped specificationから選択し、model、weight plan、graph、execution、
  KV state、linear-attention state、frontendを単一のshape-driven pathへ一般化した。モデル別の
  graph/execution source複製は作成していない。
- native HIPのattention preprocess、KV append、causal attention、sigmoid output gate、linear attentionを
  reviewed head/hidden/state shapeへ一般化した。9Bは独立LM head buffer/upload/lifetime、2B/4Bは
  embedding aliasを同じ型付き分岐で扱う。
- `sllm_device_info_t`の予約領域をサイズ互換のまま空きVRAM fieldへ割り当て、`hipMemGetInfo`の値を
  sessionへ伝播した。全owned tensor/workspaceとrequest-local stateをchecked arithmeticで合算し、
  queue作成・allocation前に不足を拒否する。1 byte fixtureはbackend呼出し0件で拒否するtestをPASSした。
- host CTest 3/3、`cargo test --workspace --all-targets`、exact `gfx1030`/`gfx1201` native compile、
  release CLI buildをPASSした。release CLI SHA-256は`gfx1030`
  `1db0ef14415469e51be04e3b225972015170c89afb7b01e3e0f3c0ffa6637450`、`gfx1201`
  `63abc15894e2456f182b33f8b7539ac8cd797b40f61e43b0ba279e60f80385f5`である。

## 2026-08-11: M3 dual-target integration完了

- integration worktree tree `16282f9014186042580fc927e47750947216d694`（base commit
  `0e2526d8e8efa38deed88929977339d71ea03057`）に対し、canonical V620 `gfx1030`
  `GPU-76a08c022586fed6`とR9700 `gfx1201` `GPU-a8e9ddefa2d60f55`を直列実行した。
- 既存real-weight RMSNorm G2 binaryへ後方互換のreviewed `N=2048/4096` modeを追加し、2B/9Bの
  layer 0 input normをraw非保存memfdで渡した。rows 1/3/17、独立semantic RMSNorm oracle、
  `atol=0.0078125`、`rtol=0.015625`で各model×両targetをPASSした。全12 dispatchはHIP、
  fallbackなしである。cross-target output SHA-256は2B
  `756167897a5e27c028e02f0bd97748ecb3489718f1c9355dcf203d499b58e280`、9B
  `3dd542e0a612d6f590b3ba7902d2326793af7919fad8cf2d57643e96b4f6a777`で一致した。G2 binary
  SHA-256は`gfx1030=44c847ec73387dbace633b40b688b7c02d346603ff55b8b491dc8ede3fa62c25`、
  `gfx1201=0fbbeb5b5ec1246fb3006ca4af344c402a1e5ab654d4cb4e55f666c0e9106cd6`である。
- 2B/9Bともfixed `Hello`、Unicode chat、255/256/257 token、max-token、実stop-token
  `248046`を両targetでPASSした。全行はexact lock/target、HIP dispatchのみ、fallbackなし、
  1 token以上、cleanup count 0である。

| model/case | cross-target generated token要約 | stop | submission / kernel dispatch |
| --- | --- | --- | ---: |
| 2B `Hello` | `[11]` | max 1 | 352 / 370 |
| 2B 255/256/257 | 各`[264]` | max 1 | 352 / 370 |
| 2B Unicode max | `[85951]` | max 1 | 352 / 370 |
| 2B Unicode stop | 29 generated、末尾`248046`、visible 28 | stop token | 10,208 / 10,730 |
| 9B `Hello` | `[11]` | max 1 | 468 / 492 |
| 9B 255/256/257 | 各`[264]` | max 1 | 468 / 492 |
| 9B Unicode max | `[85951]` | max 1 | 468 / 492 |
| 9B Unicode stop | 45 generated、末尾`248046`、visible 44 | stop token | 21,060 / 22,140 |

- 9B/V620の255/256/257 tokenは248.6/246.9/248.6秒で正常完了したため、Phase 4の
  long-boundary evidence timeoutは300秒とする。負荷中の9B/V620は19,173,812,000 bytes VRAM、
  gfx 99%、unthrottled、ECC error 0を観測した。
- 共通execution/kernel source変更の回帰として、既存4B G3のprompt 1/7/255/256/257とUnicode
  stopの6ケース×2 targetを全件再実行し、既存token列、stop reason、dispatch auditと一致した。
- 実行後は全3 GPUでprocess残留なし、V620 16 MiB、R9700 257 MiBのbaselineへ戻り、両GPUとも
  unthrottled、ECC error 0だった。request-local state、retryable cleanup、durable quarantineは全行0である。
- RMSNorm、embedding、final outputのreal-weight range recipe/hashを
  [Phase 4 slice identities](../../../../models/qwen3.5-phase4-slices.md)へ固定した。raw sliceは保存していない。
- lintはworkspace全targetを`-D warnings`でPASSした。tensor roleを明示する内部2関数だけは
  `clippy::too_many_arguments`をコマンド側で限定許可し、GPU検証済みartifact hashを変える
  source属性は追加していない。
- M1〜M3の受入条件を満たしたためPhase 4を完了し、計画をarchiveした。push/releaseは依頼範囲外のため
  実施しておらず、上記treeはlocal integration identityである。

[対応する計画](../../../../plans/archive/2026/08/11-20/qwen35-2b-9b-compatibility.md)
