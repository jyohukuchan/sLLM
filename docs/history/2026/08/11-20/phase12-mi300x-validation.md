# Phase 12 MI300X validation history

## 2026-08-15: P12-A5 closeoutとVM削除 PASS

- integration reviewの指摘をgfx942 telemetry例外の限定へ反映し、focused re-review、workspace全test、clippy、
  Phase 5 llama/OpenAI runner 53 test、format、diff、JSON/schema/manifest/workflow、markdown link検査をPASSした。
- raw report、accuracy、service、performance、traceの28 fileとfixed llama.cpp比較binary 1 file、計29 file、約4 MiBを
  repository外のlocal evidence storeへ退避した。summaryが参照する20個のraw/report/trace hashはすべてlocal fileと
  一致し、比較binary SHA-256も`c4276e737c3ccb8dd59f356e7d2705e1e890ad6ddc8352dcee4dac43ffa70941`へ一致した。
- ユーザーがHot Aisle VMを削除したことを確認し、旧endpoint `23.183.40.75:22`へのSSHは8秒のconnect timeoutとなった。
  Phase 12専用のlocal SSH秘密鍵、公開鍵、known-host entryも削除した。これによりP12-A5とPhase 12を完了し、planを
  archiveへ移した。

## 2026-08-15: P12-A4 performanceとfixed llama.cpp比較 PASS

- final provider修正後の4B BF16/FNUZ FP8 direct engineをshort-odd、32/32、prefill-long、decode-longで各3 warmup＋
  10 measuredした。全sampleはexact `gfx942`、HIP-only、fallbackなし、request/workspace cleanup zeroで、modelは
  rowごとに1回だけloadした。表は中央値、括弧内はMADである。VRAMはdecimal GBで、各rowのallocator high-waterを使う。

| dtype / case | TTFT ms | E2E ms | prefill tok/s | decode tok/s | TPOT ms | resident / peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 short-odd | 38.592 (0.115) | 272.205 (0.442) | 518.07 | 72.06 | 13.874 | 8.412 / 8.541 |
| BF16 32/32 | 56.211 (0.048) | 507.186 (0.131) | 643.54 | 71.06 | 14.076 | 8.412 / 8.609 |
| BF16 prefill-long | 1244.420 (1.053) | 4587.771 (0.878) | 832.14 | 38.57 | 25.937 | 8.412 / 13.100 |
| BF16 decode-long | 56.364 (0.094) | 3998.963 (1.909) | 642.57 | 64.93 | 15.390 | 8.412 / 8.616 |
| FP8 short-odd | 43.598 (0.071) | 354.002 (0.203) | 453.83 | 54.15 | 18.462 | 4.847 / 4.976 |
| FP8 32/32 | 61.237 (0.046) | 662.063 (0.352) | 587.33 | 53.28 | 18.786 | 4.847 / 5.044 |
| FP8 prefill-long | 1240.756 (1.182) | 5387.443 (4.871) | 834.51 | 31.22 | 32.009 | 4.847 / 9.535 |
| FP8 decode-long | 61.233 (0.056) | 5228.618 (1.474) | 588.50 | 49.53 | 20.209 | 4.847 / 5.052 |

- FP8 resident VRAMはBF16比57.62%、42.38%減だが、E2EはBF16比1.174〜1.307倍、decode throughputは
  74.97〜80.94%だった。したがってMI300Xでも現candidateのFP8を性能優位とはせず、VRAM節約と精度を確認した
  opt-in pathとして扱う。
- fixed llama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`、tree
  `e9b6173953477054a4068884aa5fc9aeef6475e8`をROCm 7.14.0、exact `gfx942`、HIP Graph on、VMM offでbuildした。
  `libggml-hip.so.0.18.0`はoffload arch `gfx942`だけを含み、ROCm closureは同じ7.14 rootへ解決した。GGUFは
  model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`のBF16、SHA-256
  `636158bd8a217374134cc2455aa40603f7579366fda0f0f5efcbf8bcba37c045`で、33/33 layerをMI300Xへoffloadした。

| llama.cpp BF16 case | TTFT ms | E2E ms | prefill tok/s | decode tok/s | TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-odd | 9.492 (0.087) | 109.003 (0.300) | 1900.63 | 160.96 | 5.697 |
| 32/32 | 10.441 (0.058) | 197.433 (0.181) | 3230.55 | 165.86 | 5.738 |
| prefill-long | 76.375 (0.117) | 822.019 (1.679) | 13513.43 | 170.35 | 5.792 |
| decode-long | 10.631 (0.044) | 1511.122 (0.895) | 3177.28 | 169.95 | 5.803 |

- 同じBF16/token条件でsLLMのE2Eはllama.cppの2.50/2.57/5.58/2.65倍、llama.cppのdecode throughputは
  sLLMの2.23/2.33/4.42/2.62倍だった。FP8とllama.cpp BF16を同等dtype比較とは呼ばない。
- final resident candidateの4B BF16 `Hello` 3 tokenをrocprofv3で代表trace化した。1476 kernel、全HIP、
  fallbackなし、cleanup zeroで、generate report、trace JSON、summaryのSHA-256は順に
  `01bef9eab4cab07e647dff3c6402bbe7176493392b39bcb9bee8e29ac4d32515`、
  `96d1f923bfc87f0bb2c6b58929b3879b4706e57b6c53c38fcbeed3e6647f9e9a`、
  `27e769dbd66bbcb16bf3ebb8d1972d2aaedf0d541c91eb86332c9a8ebabc7c29`である。

## 2026-08-15: P12-A3 contiguous-resident KV/service PASS

- Hot Aisle MI300X VFではVMM capabilityがtrueだったため、最初のproduction service auditは既存の
  `virtual-contiguous`を自動選択した。これはPhase 12開始時に固定したgfx942 `contiguous-resident`条件と違うため、
  そのrunを最終evidenceへ採用しなかった。exact `gfx942`だけcreate時に明示resident providerを選び、他targetの
  capability-selected挙動を維持する修正を加えた。
- 修正後にKV state 19 caseとfull attention 16 caseをfocused rerunし、さらにOpenAI serviceでlogical capacity
  1023/1024/1025、raw non-stream/SSE、stop、Phase 5 render baseline、公式OpenAI Python client 3.1.0、
  disconnect後のrecovery、2並行requestをPASSした。全requestの`kv_memory_kind`は`contiguous-resident`、
  backendはHIP、fallbackなし、request/workspace cleanup zeroだった。shutdownのfinal current/model/request/workspace、
  retryable/durable cleanupはすべて0、GPU process数はpre/postとも0である。
- `amd-smi static`はMI300X VF、304 CU、BDF、driver、ECC block stateを取得できた。一方`amd-smi metric`は
  provider tool自身が`Namespace.partition`例外で終了したため、温度・電力・clock telemetryは0へ置換せず
  pre/postとも明示`unavailable`とした。integration reviewで、この例外許容をgfx942だけへ限定し、既存
  gfx1030/gfx1201 runnerのfail-closed telemetry条件を緩めないよう修正した。

## 2026-08-15: P12-A2 4B/9B BF16/FNUZ FP8 PASS

- 4B BF16は`Hello`、Unicode chat `こんにちは🌙`、stop commaを通し、9B BF16は`Hello`最小spotを通した。
  4B/9Bとも全dispatch HIP、target `gfx942`、fallbackなし、cleanup zeroだった。
- 4B/9B BF16からtext-linear FP8 sidecarを生成した。4B sidecar fingerprintは
  `sha256:6bf020c108ebe8deec168fc7193f97d74bf88805a40b414cc3df9266ab16d87f`、9Bは
  `sha256:ab55ebda538b5155b12c5198f403f3ab7017ba4d66429de0779b4505dbf18d0e`である。
- 初回4B FP8実行で、graph validationがOCP `F8E4M3Fn`だけを許してFNUZ resident viewを拒否する問題を検出し、
  明示scaled encodingを保った`F8E4M3FnuZ`を許可した。次にOCP byteを値としてFNUZへ再量子化しscaleを維持すると
  top-1は一致しても最大KLDが`0.407`となった。有限OCP byteは同じFNUZ byteでちょうど半値になる性質を使い、
  byteを保持、OCP negative zero `0x80`だけFNUZ zeroへ正規化、OCP NaNを拒否し、positive finite outer FP32
  scaleを2倍するresident rebasingへ変更した。全有限OCP byteの積が一致するhost testを追加した。
- 修正後4B accuracy 3 caseのKLDは`0.025996861`、`0.007520127`、`0.003122728`、9Bは
  `0.007287292`、`0.006057783`、`0.010212244`で、全caseのtop-1がBF16と一致し、fallbackはなかった。
  4B FP8のfixed/Unicode/stop generationと9B FP8の最小generationも`native-fnuz`、encoding
  `e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32`、全HIP、cleanup zeroでPASSした。

## 2026-08-15: P12-A1 model-free/operator PASS

- exact `gfx942` buildでelementwise 21 operation、attention preprocess/RoPE 8 case、KV state 19 case、full attention
  16 case、sigmoid output gate 6 caseを実行した。非整列境界255/256/257、attention length
  1023/1024/1025、state concurrency/stale/timeout/drop-cancelを含み、すべて数値oracle、native dispatch、
  fallbackなし、cleanup zeroでPASSした。
- 既存のsemantic RMSNorm G1 producerを`gfx942`へ拡張し、wave64 kernel id 2、logical symbol
  `rmsnorm.baseline.wave64.v1`、device symbol `sllm_rmsnorm_baseline_wave64_v1`を固定した。幅
  1/3/255/256/257/2560/4096の7 caseはBF16 oracle tolerance内で、最大絶対誤差0.0078125、最大相対誤差
  0.575%だった。全caseが3 allocation、3 copy、1 dispatch、cleanup zeroだった。
- model-free GDN evidence producerを追加し、実Qwen3.5 layoutの16 QK head、32 value head、head dim 128、
  convolution kernel 4を使ってtoken 1/3/17をCPU数値oracleと比較した。各caseはcausal-convolutionとrecurrent GDNの
  2 dispatch、recurrent kernel id 2、状態lengthのtransactional publication、fallbackなし、cleanup zeroを満たし、
  最大絶対誤差0.00390625、最大相対誤差1.471%だった。
- 4B BF16 13 fileと9B BF16 15 fileをVMへ取得し、VM上で全fileからlock fingerprintを再計算した。4Bは
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、9Bは
  `sha256:2d2bc642540e97d4681f8c66140e09f305f487476bb9fe238ca82a298febf893`へ一致した。
- 以上によりP12-A1をPASSとし、P12-A2の4B/9B model統合へ進む。

## 2026-08-15: P12-A0 PASSと最初のoperator実行

- VMのEd25519 host key fingerprint `SHA256:mmwFXekpBuP+T8LRcvf8Leypw5s1iE9WZ9oWO9hSRxQ`を照合してSSH接続した。
  Ubuntu 24.04、kernel `6.8.0-124-generic`、amdgpu `6.16.13`、13 CPU core、約220 GiB RAM、12 TB disk、
  MI300X VF x1を確認し、foreign GPU processはなかった。
- GPU実測tupleは`gfx942:sramecc+:xnack-`、wave64、304 CU、205,822,885,888 bytes HBM、BDF
  `0000:ff:00.0`、HIP UUID `GPU-cb0412d4d88cfa69`、NPS1/SPXだった。VMM attributeは`true`だったが、計画どおり
  `contiguous-resident`を維持する。
- provider管理のROCm 7.2.4とdriverを保持したまま、AMD packageのROCm 7.14.0/LLVM 23 exact gfx942 rootを追加した。
  logical `/opt/rocm`はcanonical `/opt/rocm/core-7.14`へ解決し、HIP runtime `7.14.60850-0000000`、
  hipBLASLt `libhipblaslt.so.1.4`を同じrootからloadした。CMake/Ninjaだけを追加し、reboot、driver交換、package削除はない。
- tiny runtimeは`41 -> 42`、1 dispatch、2 copy、1 allocationでPASSし、FNUZ outer-vector
  `m=3,k=128,n=256`はzero-workspace solutionを8件返した。rocprofv3 1.3.2のkernel/memory traceもPASSし、
  trace SHA-256は`44795361ecca31a622f116b3b17c4b8aaf51dde72931fd2c02f447405eca76d1`だった。
- source commit `a5e389be348442c4e99e97cc449fe3c356b8291f`からexact `gfx942`、code object v6、wave64、XNACK off、
  SRAM ECC onのproduction buildを作成し、offload bundleがgenericでなく`gfx942`だけを含むことを確認した。
- 最初のproduction起動は、HIPが返すfeature suffix付きdevice名をlogical `gfx942`との文字列完全一致が拒否して停止した。
  任意suffixを許さず`gfx942:sramecc+:xnack-`だけを正規化するdraft修正を加えた。これによりFNUZ FP8 hipBLASLtは
  `m=1/3,k=128,n=256`の2件をkernel id 5、fallbackなし、最大相対誤差0.373%以下、cleanup zeroでPASSした。
  BF16はwave64 MMVFとhipBLAS GEMMの17 shapeを数値oracle、17 native dispatch、fallbackなし、cleanup zeroでPASSした。
  P12-A0をPASS、P12-A1を実行中とする。

## 2026-08-15: Phase 12開始とaccess準備

- ユーザーの明示指示によりPhase 12を開始し、active planを`in_progress`へ変更した。既存の受入条件1〜6と
  Qwen3.5 4B/9B BF16/FP8、contiguous-resident KV、service、性能比較のmatrixを開始時の条件として固定した。
- 開始sourceはcommit `a5e389be348442c4e99e97cc449fe3c356b8291f`、tree
  `d0ace3d9fac29dd60375f5d6263f42355658a3bd`で、`main`と`origin/main`のahead/behindは0/0、作業treeはcleanだった。
- Phase 11 MI300X candidateのdry-runは6 profile、推定435分でPASSした。この時点では実GPU実行をまだ行っていなかった。
- repository外にPhase 12専用の短命ED25519 SSH keyを作成した。public fingerprintは
  `SHA256:YNhBwZGNGfdNnlg7yDLpXzDcic0vls6MDAGD67/PLvM`で、private keyはlocal hostだけにmode `0600`で保持する。
  VM作成後はremote endpoint/userとVM側Ed25519 host key fingerprintを照合してから接続する。
- 待機queueの実装単位Q0〜Q4は完了済みで、Q5は以前のユーザー指示により本goal対象外だった。この状態を記録し、
  local forward workからMI300X Phase 12へ切り替えた。

## 2026-08-15: VM取得延期とlocal先行queue

- ユーザーが十数時間以上MI300X cloudを継続管理できないため、本Phaseを`ready`のまま保持し、VMを起動しないことを
  固定した。
- 待機中はPhase 13以降をlocal forward queueで先行する。再開時はlatest mainからexact `gfx942` candidateを再buildする。
- Phase 12 matrixはQwen3.5 4B/9B BF16/FP8、contiguous-resident KV、service、性能比較のまま維持し、先行した
  Gemma/NVFP4/MoEを自動追加しない。

## 2026-08-14: Hot Aisle実機計画の作成

- Hot Aisle Small VMのMI300X x1をPhase 12に採用する計画を作成した。192 GB HBM3、8/13 CPU core、
  224 GB RAM、12 TB NVMeはsingle GPU/batch 1の4B/9B BF16・FP8と限定27B FP8 spotに十分と判断した。
- multi-GPU、Infinity Fabric、RCCL/RDMA、bare-metal固有挙動、別CDNA3 SKUはこの一台の証拠範囲外とした。
- 標準予定を10〜12 GPU時間、現実的な上限を16時間、必要な場合だけ追加4時間とし、2〜3時間のpreflightと
  6〜8時間のintegration/performanceを別sessionに分割する。
- VMMなしを想定し、first-hourでexact tuple、FNUZ、contiguous-resident KV、profilerをstop/go判定する。
- 詳細は[Phase 12 archive](../../../../plans/archive/2026/08/11-20/phase12-mi300x-validation.md)を正とする。
