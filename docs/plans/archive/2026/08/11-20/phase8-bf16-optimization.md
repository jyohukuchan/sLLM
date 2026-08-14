# Phase 8: BF16単一リクエスト最適化（完了）

> 完了日: 2026-08-14

## 目的

Qwen3.5-4B BF16の単一GPU・batch 1経路を、canonical V620 `gfx1030`とR9700
`gfx1201`で最適化する。Phase 5で観測したllama.cppとの差を、全model/GPUへ共通適用できる
GEMM、attention、dispatch/synchronizationの順に縮める。baseline semantic opとvAttentionの
上位契約は維持し、最適化pathはkernel registryへ追加する。

最適化済みpathが同条件のllama.cppを上回ることを目標にするが、未承認の一律倍率やtoken/sを
hard gateにはしない。正しさ、exact target、no-fallback、health、cleanupはblocking条件とし、
性能はcase/GPU別に測定して採用範囲を決める。遅いfast pathをdefaultにせず、そのshape/targetでは
既存baselineまたは別の検証済みpathを明示選択する。

## 開始時点の事実

- Phase 5の4B BF16 TTFTはexact-token llama.cpp wrapperより、V620で49.4〜278.5倍、
  R9700で31.4〜742.1倍長い。差は長いprefillほど拡大した。
- 現行BF16 Matmulは出力1要素を1 threadが担当し、K方向をscalar loopするbaselineである。
  weight `[N,K]`の各行を全出力要素から繰り返し読むため、最初の共通最適化対象とする。
- 現行causal attentionはscoreとsoftmaxの大半をthread 0で実行し、同じscoreを2回計算し、
  tokenごとにworkgroup同期するbaselineである。
- Qwen executionは多くのsemantic submissionをhostで個別waitする。ASCII 8-token例では
  3,744 submission / 3,936 kernel dispatchを観測済みである。
- ROCm 7.14.0 local installにはrocBLAS/hipBLAS、hipBLASLt、CKがある。target別payloadは
  rocBLASが`gfx1030`/`gfx1201`、hipBLASLtが少なくとも`gfx1201`を持つが、実際のshape/dtype
  solution対応はruntime queryと実機PoCで確認し、libraryの存在だけから採用しない。
- vAttentionはKV memory managementでありFlashAttention系kernelと排他的でない。FP16の
  virtual-contiguous K/V pointerを維持し、Paged Attention production backendは追加しない。

## スコープ

- BF16 weight / BF16 activation / FP32 accumulation / BF16 outputのMatmul fast path。
- prefillとsingle-token decodeを分けたshape-aware kernel selection。
- vAttention上のfull-attention prefill/decode fast path。
- model-resident handle、prepared plan、workspace、solution selectionの再利用。
- single HIP stream上のdevice dependencyを使ったhost wait削減。
- profileで残った上位costに限定したfusionまたはkernel launch削減。
- canonical V620/R9700のdirect engine測定と、最終llama.cpp/service比較。

次はPhase 8に含めない。

- FP8、NVFP4、KV量子化、load-time quantization。
- multi-request batching、continuous batching、chunked prefill。
- multi-stream overlap、自動tuning DB、JIT compiler、複数GPU。
- Paged Attention production backend、prefix sharing、RadixAttention。
- Qwen vision、MTP、MoE、他model architecture。

## 非blocking follow-up

- RDNA4 `gfx1200`/`gfx1201`向けのFlashAttention-3-like kernelを、Phase 8のFA2-style pathと
  profile結果が確定した後の将来タスクとする。warp specialization、非同期data movement、
  producer/consumer pipeline等のFA3系設計要素をAMD wave/LDS/命令へ対応付けるが、NVIDIA固有実装を
  そのまま移植したとはclaimしない。
- このfollow-upではcanonical R9700 `gfx1201`でFA2-styleを同一数値・shape・vAttention契約の
  比較baselineとし、RDNA4固有pathとしてkernel registryへ追加する。V620/RDNA2へ一般化しない。
- FA3-likeの調査・実装・実GPU比較は今回のP8-A0〜A6の受入条件ではなく、Phase 8完了をblockしない。

## 実装原則

1. 多model・両GPUへ共通な変更を先に行い、RDNA2 `gfx1030`を成立させてから
   RDNA4固有solution/tuningを追加する。
2. baseline kernelは数値比較とunsupported shapeの明示選択用に残す。dispatch後のlibrary error、
   unsupported solution、数値不一致をbaselineへ黙ってfallbackしない。
3. kernel/solution選択はtarget、dtype、encoding、M/K/N、alignment、workspace上限を入力とする。
   選択したprovider、kernel/solution ID、workspace bytesをauditへ残す。
4. checkpoint weight `[N,K]`を恒久転置・複製しない。row-major viewまたはoperand/transposition設定で
   library contractへ接続し、model-resident VRAM増加を測定する。
5. optimized Matmulはreduction順変更を許し得る。独立FP32 oracleに対するop/shape/range別の
   toleranceを最適化結果を見る前にA0で固定し、同じcandidateの結果を見て拡大しない。
6. llama.cpp固定commitのAMD GEMM、graph submission、buffer/layout処理を直接reuse候補として先に調べる。
   reuseする場合はproject provenance契約に従う。CK、AITER、vLLM等はno-copy参考に限定する。
7. raw rocprof trace、binary、model、large sliceは追跡せず、versioned summary、manifest、digestだけを残す。

## 受入条件

1. A0でper-op kind/shape、kernel/solution、submission、host wait、GPU時間、workspaceを識別できる
   bounded profileを作り、4B short-oddと32/32 surrogateを両canonical GPUで取得する。
2. BF16 Matmulのoptimized数値budgetを、非整列値、M/K/Nの境界前後、signed zero、subnormal、
   large finite、NaN/Inf classificationを含む独立oracleで実装前に固定する。
3. Matmul registryがbaselineとfast pathを区別し、`gfx1030`/`gfx1201`でprevalidated providerだけを
   選択する。prefillとM=1 decodeの選択を分け、unknown target/shapeは明示的に拒否または登録済み
   baselineを選ぶ。
4. full-modelのweight viewを複製せず、handle/descriptor/algorithm/workspaceをmodel-resident lifetimeで
   再利用する。requestごとのlibrary handle生成や全weight再packを行わない。
5. FlashAttention-2-styleのonline softmax/tiled full-attention fast pathが、vAttentionの
   virtual-contiguous pointer、FP16 K/V layout、opaque KV ownerを変更せずに動作する。
   `M=1/3/17/37`とKV `1023/1024/1025`を含める。
6. 同一streamで安全な依存はdevice側へ残し、host waitをreadback、transactional state publish、
   cancellation/error境界まで遅延する。timeout/drop後のbuffer/state lifetimeと旧state保持を壊さない。
7. optimized pathのG1/G2 numerical oracle、4B G3固定/Unicode/stop、fallbackなし、exact target、
   loader root、ECC/health、process/VRAM cleanupがPASSする。
8. 通常iterationはO0/O1だけを使い、境界に影響する変更だけO1-boundaryを実行する。
   2B/9B、canonical long、llama.cpp、serviceはA6統合まで毎回再実行しない。
9. A6で両GPUの4B O2、共通pathを使う2B/9B spot check、exact-token llama.cpp比較、
   non-stream/SSE service smokeを同じsemantic/build identityへ結び付ける。
10. case別のTTFT、prefill/decode token/s、TPOT、E2E、resident/peak/workspace VRAM、submission/kernel数、
    選択kernelをbaselineと比較する。改善しないpathはdefaultへ昇格せず、残差backlogを記録する。
11. affected host/compile/GPU test、1回のintegration review、plan/history/main-plan/runtime/compatibility文書を
    同期し、完了時にこのplanをarchiveへ移す。

## 実装順序

### P8-A0: profileと数値契約

- 既存Phase 5 timingへop kind、shape、kernel/solution、workspace、host wait区間を追加する。
- rocprofv3は代表1 sampleの原因特定だけに使い、通常iterationの必須条件にしない。
- 4B short-oddと32/32 surrogateをV620/R9700で取得し、GEMM、attention、host wait、その他の
  wall time構成を固定する。
- optimized Matmul/attentionのtolerance manifestと、baseline/optimized differential fixtureを作る。
- llama.cpp固定commitのreuse候補と、rocBLAS/hipBLAS/hipBLASLtのlayout・dtype・solution queryを記録する。

### P8-A1: library PoCとkernel registry拡張

- `[M,K] x [N,K]^T -> [M,N]`をweight複製なしで呼ぶstandalone PoCを作る。
- `gfx1030`はrocBLAS/hipBLASを最初の共通候補とし、`gfx1201`も同じpathを検証する。
  hipBLASLtはruntime queryが成功する`gfx1201` shapeだけの追加候補とする。
- M=1はlibrary GEMM/GEMVとwave-reduction HIP kernelを比較し、shape別の選択境界を記録する。
- provider failure、workspace不足、unsupported algorithm、target mismatchをfail-closedに検査する。

### P8-A2: production BF16 Matmul fast path

- public Matmul ABI/semantic descriptorを維持したままprovider-neutral planを追加する。
- model load時にhandle、layout、algorithm、workspaceを準備し、全layer/requestで再利用する。
- full-attention/linear-attention/MLP/final projectionの既存Matmul consumerをregistry選択へ接続する。
- G1/G2、focused full-model、O0/O1をV620優先で実施し、その後R9700共通pathとRDNA4追加pathを検証する。

### P8-A3: vAttention上のFlashAttention-2-style fast path

- score再計算とthread-0 softmaxを廃止し、wave/workgroup reduction、online softmax、Q/K/V tileを使う。
- prefill用tiled kernelとM=1 decode用kernelを分け、head dim 256、Q heads 16、KV heads 4を最初の
  production shapeとする。
- virtual-contiguous K/V pointerを通常contiguous pointerとして受け、VMM page commit/layout/APIは変更しない。
- 1023/1024/1025 KV境界、causal mask、GQA head mapping、NaN/Inf、cancel/cleanupを独立oracleで確認する。
- 今回はFA2-styleだけをproduction対象とし、RDNA4向けFA3-likeは上記の非blocking follow-upへ残す。

### P8-A4: plan再利用とhost synchronization削減

- semantic planのprepare/cache keyをmodel-resident化し、requestごとの再prepareを除く。
- 同一HIP streamの順序保証を使い、各opのhost waitをterminal/readback/state publishへ集約する。
- completion ownerがinput/output/workspace/queueを保持し、error、timeout、drop、cancel時に先行stateだけを
  publishする既存transaction contractを維持する。
- submission数、kernel数、host wait回数とwall timeをA0 profileへ戻して効果を確認する。

### P8-A5: decodeとfusionのprofile-driven最適化

- A4後のprofileで上位に残ったcostだけを対象とする。
- 候補はM=1 GEMV tuning、gate/up grouped GEMM、residual+RMSNorm、QKV preprocess+KV append、
  SiLU multiply、argmax/readbackであり、全候補を一律実装しない。
- fusion後も元semantic opの数値budget、buffer alias、audit、cancellation境界を保持する。
- target固有tuningは共通pathの後にregistryへ追加し、別GPUへ一般化しない。

### P8-A6: 統合、比較、完了同期

- canonical両GPUで4B O2を実行し、共通pathの2B/9B short spot checkを行う。
- fixed llama.cpp wrapperと同じmodel revision、commit、target、dtype、input/output token条件で再比較する。
- direct engineを正本とし、OpenAI non-stream/SSEは同一request identityのservice overhead smokeだけを行う。
- 改善したshape/target、baselineのままの範囲、llama.cppとの差、VRAM/workspace、残差backlogを記録する。
- 累積integration reviewとaffected final gatesをPASSした後、plan/history/main planを同期してarchiveする。

## 計測lane

| lane | Phase 8での使用 |
| --- | --- |
| O0 | 変更対象GPU、4B short-odd、warmup 1 + measured 3。correctnessと方向性を確認 |
| O1 | O0 + 32/32 surrogate。通常の最適化iteration |
| O1-boundary | tile、alignment、KV/VMM境界を変えた時だけB-1/B/B+1を追加 |
| O2 | A2/A3/A4統合とA6で、変更対象GPUの4B canonical 7 caseを実行 |
| O3 | release/nightlyまたは意味変更時だけ、dual-GPU、2B/9B、llama、serviceを含める |

## 性能判定

- 数値thresholdは現在未承認であり、Phase 8開始時点ではperformance hard gateを追加しない。
- 各fast pathは対象caseでbaselineより遅ければdefaultにしない。noise範囲はmedian、p10/p90、MAD、
  repeated O1/O2で判断し、単一最良runを採用根拠にしない。
- llama.cppを上回れない場合も結果を省略せず、kernel time、host wait、dispatch、memory bandwidth、
  workspaceのどこに差が残るかを次のbacklogへ分解する。
- 複数のO2/O3履歴が揃うまでP1 regression thresholdは提案に留め、別途ユーザー承認を得る。

## 完了結果

- A0〜A6を完了し、frozen float64 numerical oracle、BF16 Matmul registry、vAttention上の
  FA2-style online softmax、prepared semantic cache、target-specific hipBLAS decode pathを実装した。
- canonical V620 `gfx1030` / R9700 `gfx1201`でMatmul 17 case、attention 16 case、4B O2 7 case、
  2B/9B spot、fixed llama.cpp比較、OpenAI non-stream/SSEをPASSした。全GPU runはexact target、
  HIP-only、fallbackなし、ECC 0、terminal process/VRAM cleanupを満たした。
- short-oddのPhase 5 baselineに対し、V620はTTFT 7.550→1.110秒、prefill 2.253→15.391 tok/s、
  decode 0.876→1.873 tok/s、E2E 25.838→9.656秒、R9700はTTFT 2.878→0.683秒、prefill
  5.921→25.108 tok/s、decode 1.674→1.953 tok/s、E2E 12.445→8.881秒となった。
- fixed llama.cppとの差はshort-odd E2EでV620約20.4倍、R9700約26.9倍残る。未改善のhost wait batchingは
  defaultへ採用せず、dispatch graph、fusion、decode GEMV、host orchestrationを後続性能backlogとした。
- RDNA4向けFA3-likeはユーザー指示どおり今回実装せず、上記の非blocking follow-upとして維持する。
- 完了evidenceとdigestは[対応するhistory](../../../../../history/2026/08/11-20/phase8-bf16-optimization.md)、
  versioned数値・性能summaryは`ci/matrix/phase8-bf16-numerics-v1.json`と
  `ci/matrix/phase8-profile-summary-v1.json`を正とする。

## Rollbackと再計画

- provider、kernel、solutionはregistry entry単位で無効化でき、baseline semantic pathを削除しない。
- numerical mismatch、silent fallback、state publication破壊、cleanup不良は該当work unitのblockerとする。
- review/検証が実装時間を超える、同じrejectが2回、1時間機能進捗なし、見積り1.5倍超、
  acceptance変更が発生した場合は追加測定を止め、同じwork unitを再計画する。

[対応するhistory](../../../../../history/2026/08/11-20/phase8-bf16-optimization.md)
