# Phase 8 BF16最適化履歴

## 2026-08-14: 計画作成

- `sLLM.md`とmain planの開発順序を照合し、Phase 8をRDNA2/RDNA4のBF16最適化として開始する計画を作成した。
- Phase 5 baselineでは4B TTFTがexact-token llama.cppよりV620で49.4〜278.5倍、R9700で
  31.4〜742.1倍長く、最初の共通backlogがprefill GEMM、operator dispatch、同期削減であることを再確認した。
- current sourceを監査し、Matmulが出力1要素ごとのscalar K loop、causal attentionがthread 0中心の
  score/softmax、Qwen executionが多数のsubmissionを個別waitするbaselineであることを確認した。
- ROCm 7.14.0 local installにはrocBLAS/hipBLAS、hipBLASLt、CKがあり、rocBLAS target payloadは
  `gfx1030`/`gfx1201`、hipBLASLt library payloadは少なくとも`gfx1201`に存在する。library存在を
  solution supportと読み替えず、A1のruntime query/PoCで確定する。
- 実装順序をprofile/数値契約、GEMM PoC、production GEMM、vAttention上のFA2-style attention、
  host同期削減、profile-driven fusion、統合比較とした。
- Phase 5のO0〜O3 laneを再利用し、最適化ごとのfull baseline、2B/9B、llama.cpp比較を行わない。
- 一律の性能倍率は未承認のためhard gateにせず、正しさ、exact target、no-fallback、health、cleanupを
  blocking条件とした。performance thresholdは複数O2/O3履歴後の別提案とする。

## 2026-08-14: attention範囲の確認

- ユーザー指示により、Phase 8のattention実装範囲はvAttention上のFA2-styleだけとする。
- RDNA4 `gfx1200`/`gfx1201`向けFA3-likeは、FA2 profile確定後に同一数値・shape・memory契約で比較する
  非blockingな将来タスクとしてplanへ記録した。今回のPhase 8受入条件や完了をblockしない。

## 2026-08-14: P8-A0 profileと数値契約

- optimized GPU outputを見る前に`phase8-bf16-numerics-v1`を固定した。MatmulはBF16 input/weight、
  FP32 accumulation、BF16 RNE outputとし、reduction順変更を`gamma_k * sum(abs(product)) + BF16 half ULP`
  で評価する。signed zero、subnormal、large finite、NaN、正負Infと、M/K/Nの非整列・境界形状を含めた。
  attentionはBF16 Q、FP16 KV、FP32 accumulation、BF16 outputで、finite absolute tolerance 0.016と
  NaN/Inf classification一致を固定した。同candidateを見てthresholdは拡大していない。
- 4B short-oddと32/32 surrogateを3 warmup + 10 measuredで両GPU取得した。各requestはmodelを再loadせず、
  全dispatch HIP、fallbackなし、session cleanup 0だった。versioned summaryは
  `ci/matrix/phase8-profile-summary-v1.json`を正とし、raw report/rocprof traceはlocal-onlyとした。

| GPU | case | TTFT median | E2E median | prefill tok/s | decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 | short-odd 17/17 | 1.099 s | 9.653 s | 15.555 | 1.876 |
| V620 | surrogate 32/32 | 1.072 s | 17.617 s | 30.034 | 1.874 |
| R9700 | short-odd 17/17 | 0.683 s | 8.891 s | 25.102 | 1.951 |
| R9700 | surrogate 32/32 | 0.710 s | 16.595 s | 45.488 | 1.953 |

- short-oddのPhase 5 baseline比で、V620はTTFT 7.550→1.099秒、prefill 2.253→15.555 tok/s、
  decode 0.876→1.876 tok/s、E2E 25.838→9.653秒となった。R9700はTTFT 2.878→0.683秒、
  prefill 5.921→25.102 tok/s、decode 1.674→1.951 tok/s、E2E 12.445→8.891秒となった。
  residentは8,411,592,192 bytes、short-odd peakは8,540,569,292 bytes、32/32 peakは
  8,608,921,472 bytesである。Matmul library workspaceは0 bytesでweight複製を行っていない。
- focused rocprofではonline-softmax attention 10 callの平均がV620 2.846 ms、R9700 0.685 msだった。
  KV kernelは14 call平均18.134/12.214 us、copyは24 call平均7.695/5.482 usだった。通常iterationへ
  rocprofを持ち込まず、tracked summaryだけを残した。
- 固定llama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`の
  `ggml/src/ggml-cuda/common.cuh`と`ggml-cuda.cu`を再確認した。device-lifetime BLAS handle、stream設定、
  row-major weightをtranspose operandとして渡すGEMM mappingをreuse候補にしたが、sLLMのownership/C ABIへ
  sourceを直接copyせず、公開hipBLAS契約を独立実装した。vLLM/CK等からのcopyは行っていない。

## 2026-08-14: P8-A1/A2 BF16 Matmul fast path

- baseline `matmul.bf16_fp32.v1`を保持し、prefill用16x16 tiled kernel
  `matmul.bf16_fp32.tiled16.v2`、M=1用workgroup reduction
  `matmul.bf16_fp32.decode.v2`をregistryへ追加した。debug/evidenceでは
  `SLLM_MATMUL_FORCE_BASELINE=1`でbaselineを明示選択でき、実行後の黙ったfallbackはない。
- standalone hipBLAS PoCでcheckpointの`[M,K] x [N,K]^T`をweight転置・複製なし、FP32 compute、
  BF16 output、workspace 0で実行した。M=1,K=2560,N=9216のsteady-stateはV620でcustom 381.645 us対
  hipBLAS 1.815 ms、R9700でcustom 117.642 us対hipBLAS 45.751 usだった。このためV620はcustomを維持し、
  R9700だけM=1,K>=1024,N>=1024を`hipblasGemmEx` kernel ID 4へ登録した。R9700の初回callには
  約0.97秒のlibrary lazy initializationがあるが、handleはcontextで一度だけ作成・共有し、requestごとの
  create/destroyやweight repackはない。
- frozen manifestの5形状を含む17 caseを両GPUで実行し、数値budget、special classification、exact target、
  kernel ID/symbol、fallbackなし、cleanup 0をPASSした。実model形状M=1,K=2560,N=9216はV620がID 3、
  R9700がID 4を選んだ。float64 oracle修正版の最終report SHA-256は
  `gfx1030=c2e778b4f94e029df6e025195a3da11a781f9d2020727fa5fe38da3d101a60f4`、
  `gfx1201=5c8d6ae30a59be4e6f9466399e0a86332e3080d16b8032e1ca135beeecc5d7f6`である。

## 2026-08-14: P8-A3 vAttention上のFA2-style path

- scoreをthread 0で3回相当処理していたbaselineを、256-thread協調dot reductionと一pass online softmaxへ
  置換した。kernel registryは`causal_attention.online_softmax_gqa.v2`、ID 2を監査へ出す。Qwenの
  opaque KV owner、HIP VMM virtual-contiguous pointer、token-major FP16 K/V、GQA mappingは変更していない。
- prefill M=1/3/17/37/255/256/257、decodeの小境界、committed KV 1023/1024/1025、query NaNと
  value +Inf classificationの16 caseを
  両GPUで実行した。独立ordered scalar softmax/value oracle、BF16 RNE、FP16 subnormalのscore寄与、
  causal visibility、GQA mapping、metadata、fallbackなし、cleanup 0を全件PASSした。長いKVで隣接head差が
  BF16刻みに潰れた初期test inputはhead間隔を32へ修正し、kernel結果ではなくevidenceの識別力を直した。
  float64 oracle修正版の最終report SHA-256は
  `gfx1030=30474212ac0b260c87be941bbd349f6a65a4b1ebe94755b14a9ff864b0d8c2a7`、
  `gfx1201=6796727d19aff27899e8f136d0ff673fe67e5532b089ce9be6925d515ec1a0be`である。
- 今回はFA2-styleだけをproductionとし、RDNA4向けFA3-likeは計画済みの非blocking follow-upのままとした。

## 2026-08-14: P8-A4/A5 plan再利用と採用判断

- labelとtoken countが同じsemantic opの`PreparedOperation`をrequest lifetimeでcacheし、反復decodeでの
  reprepareを除いた。positionをdescriptorに含むattention preprocessはcache対象外であり、host recorderで
  2回目decodeのprepareがその8件だけになることを固定した。hipBLAS handle/solution stateはcontext lifetimeで
  共有するため、library handleはmodel/session側のownerより短いrequestへ降ろしていない。
- 同一streamのsemantic waitをまとめる案も実装・測定したが、R9700で改善せずV620ではnoiseを含む悪化となった。
  「遅いfast pathをdefaultにしない」という受入条件に従ってbatching変更はrevertし、readback/state publish/
  cancellationの既存transaction境界を維持した。A5では上位残差候補を一律fusionせず、実測で明確に勝った
  R9700大形状hipBLASだけをtarget-specific defaultへ採用した。

## 2026-08-14: P8-A6 model/API/llama.cpp統合比較

- 2B/9Bのverified lockをcache実体へ再照合してから、同じ4B optimized common pathのshort-oddを
  3 warmup + 10 measuredで実行した。全runはHIP-only、fallbackなし、cleanup 0だった。

| model | GPU | TTFT median | E2E median | prefill tok/s | decode tok/s | raw SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-2B | V620 | 0.805 s | 7.044 s | 21.211 | 2.565 | `cd8beb91db5eaba96f1df147b5fb1a2bb67d69b3e2ad1e894b868d183fc62207` |
| Qwen3.5-2B | R9700 | 0.413 s | 6.589 s | 41.469 | 2.592 | `8db8f31b04b34181485820e6ae8149c3a364a6538453fd9c70e31648f790b146` |
| Qwen3.5-9B | V620 | 1.179 s | 9.727 s | 14.492 | 1.875 | `780d35c41185a736a7af3daaf648d8663adadbfdffe57bd5887e39cea9d48396` |
| Qwen3.5-9B | R9700 | 0.827 s | 9.042 s | 20.692 | 1.950 | `d58d0d57a4922929e50fef5e91303d587d0a78f90da3dadd4e339fc93fbfc5f6` |

- fixed llama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`のexact-token wrapperを同じ
  4B model revision、BF16、17/17条件で再build・実行した。V620はTTFT 0.082 s、prefill
  208.075 tok/s、decode 41.004 tok/s、E2E 0.473 s、R9700はTTFT 0.025 s、prefill
  706.767 tok/s、decode 52.271 tok/s、E2E 0.331 sだった。raw report SHA-256は
  `gfx1030=9bfd2295e90e5cd41a04c62badaa8cc5298bfff3f5fbdf04c9c1a5830669e0b3`、
  `gfx1201=a862fedad473680fc056fc49589588fbd277aa6f2d947adb50a4a505687a7e41`である。
  Phase 8はbaseline差を大幅に縮めたが、short-odd E2Eはllama.cppよりV620で約20.4倍、R9700で
  約26.9倍遅い。dispatch/kernel数、decode GEMV、host orchestration、未実装fusionを残差backlogとする。
- optimized server binaryでOpenAI Python client 2.44.0、raw non-stream/SSE、logical capacity
  1023/1024/1025、disconnect後の回復を両GPUでPASSした。全completed requestはHIP-only、fallbackなし、
  ECC 0、server shutdown後のcurrent/request/workspace bytesとGPU processは0だった。report SHA-256は
  `gfx1030=5bc6b2bfc7af105870199d7319863158883b7f201dafe78d0e3f3b24e3b2272f`、
  `gfx1201=763567d5580e0246abb3656cad67dfd4452bd6c4f52c9c5192847dc52464d4e3`である。
- O2のproduction CLI buildはbase commit `915c1c48511aa099a1560bc3afc6ed01301a4361`、semantic tree
  `1a352a9e5f334c29068b03606737c52a57dca02e`へ固定した。binary SHA-256は
  `gfx1030=00937214cd2da3a6986377cfb8c8aae2d19f54f59140458eb0cce21e97738213`、
  `gfx1201=995c85d800141644838a6bdce42b862151b0685795f68d63766dfb5c1665206e`である。
  その後の変更はdocsと数値evidence summary/test/runnerだけで、production runtime build inputとの差が
  ないことをtree diffで確認した。

## 2026-08-14: integration reviewの数値evidence修正

- 累積reviewで、frozen manifestがMatmul/attentionの独立referenceをfloat64と定義している一方、GPU
  evidence runnerが従来のfloat32逐次oracleを使っている不整合を発見した。production kernelの問題では
  ないが、受入evidenceのcorrectness defectとしてcloseout前に修正した。
- BF16/FP16からexact decodeした積をfloat64で加算し、float64からBF16へ直接RNEするoracleへ変更した。
  Matmulは`gamma_k * sum(abs(product)) + BF16 half ULP`、attentionはfinite absolute 0.016と
  NaN/Inf分類一致をrunner自身で判定する。direct BF16 tie/subnormal/overflow unit testを追加した。
- 最初の修正版runは、fixtureのlarge-finite値が両operandに入った時のFP32 product/reduction overflowと、
  special activationのrow stride誤りを正しく検出した。finite tolerance caseが全required Kでoverflowしない
  `0x5c00`へ変更し、`row * K`へ注入位置を修正した。kernel変更は不要だった。
- 最終real-GPU runはMatmul 17/17、attention 16/16を両targetでPASSした。digestは上記A1/A2/A3節へ
  反映済みで、post ECC 0、process 0、VRAM baseline復帰を確認した。

## 2026-08-14: P8-A6 canonical O2

- semantic tree `1a352a9e5f334c29068b03606737c52a57dca02e`のtarget別production CLIを使い、canonical
  7 caseを各GPUでcorrectness 1、warmup 3、measured 10実行した。ユーザーが両GPU利用を許可しているため、
  exact GPUを分離した2 processを同時実行し、他方の同一candidateだけを明示的なcompanion workloadとした。
  全14 reportはPASS、HIP-only、fallbackなし、session cleanup 0だった。

| case | V620 TTFT / E2E | V620 prefill / decode | R9700 TTFT / E2E | R9700 prefill / decode |
| --- | --- | --- | --- | --- |
| minimum 1/1 | 0.540 / 0.544 s | 1.870 / n/a tok/s | 0.519 / 0.524 s | 1.947 / n/a tok/s |
| short-odd 17/17 | 1.110 / 9.656 s | 15.391 / 1.873 tok/s | 0.683 / 8.881 s | 25.108 / 1.953 tok/s |
| boundary 255/64 | 2.968 / 37.145 s | 86.362 / 1.845 tok/s | 2.296 / 34.551 s | 111.872 / 1.954 tok/s |
| boundary 256/64 | 2.977 / 37.125 s | 86.445 / 1.845 tok/s | 2.299 / 34.550 s | 112.131 / 1.955 tok/s |
| boundary 257/64 | 3.156 / 37.331 s | 81.846 / 1.845 tok/s | 2.472 / 34.790 s | 104.665 / 1.951 tok/s |
| prefill-long 1024/128 | 10.826 / 82.917 s | 94.734 / 1.762 tok/s | 8.234 / 74.301 s | 124.612 / 1.923 tok/s |
| decode-long 32/256 | 1.060 / 137.629 s | 30.389 / 1.867 tok/s | 0.710 / 131.156 s | 45.507 / 1.955 tok/s |

- resident VRAMは全caseで8,411,592,192 bytesだった。最大peak/workspaceはprefill-longの
  13,099,918,848 / 4,599,066,624 bytes、decode-longは8,616,261,504 / 143,720,832 bytesだった。
  requestごとのsubmission/kernel数はshort-odd 7,956/8,364、prefill-long 59,904/62,976、
  decode-long 119,808/125,952である。prepared cacheは再prepareを除いたがdispatch数自体は変えず、
  graph/fusion/host orchestrationを次の性能backlogに残す。
- pre/postのECCは両GPUとも全field 0、GPU processは0→0、VRAMはV620 16 MiB、R9700 257 MiBの
  pre-run baselineへ完全復帰した。health digestはpre/post metric
  `e241cf71f7c7ca7b051736b2378e2c5faa972c7aa2e814c69d8d9ee375414f98` /
  `eee563269f72d3d6eb18d36a73aa451158a5a2c5804d3f829dd020b9ee8c4636`、processは両方
  `109ae561bd38a16df03d9a360d6960385112827187893ea5af9f7564aaaf727e`である。全caseのraw digest、
  TPOT、VRAM、submission/kernel数は`ci/matrix/phase8-profile-summary-v1.json`を正とする。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase8-bf16-optimization.md)
