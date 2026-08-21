# Phase 36: MI300X latest-main実機再検証

> 状態: complete（Sessions A〜D PASS。2026-08-21のユーザー決定で当初のconditional extensionをscopeから削除してclose）
> 作成日: 2026-08-20
> 開始日: 2026-08-21

## 目的

Phase 12で一度実機PASSしたHot Aisle MI300X VF x1、exact `gfx942`の経路を、Phase 35後のlatest mainで再検証する。
今回は初回CDNA3移植ではなく、Phase 13以降に追加されたGGUF、low-bit KV、chunked prefill、MTP、vision、MoE、
Full Attention/GDN変更を含むcurrent runtimeの回帰・未検証範囲を段階的に確認し、実機で見つかった問題を修正する。

課金中の長い単一sessionへ全matrixを詰め込まず、GPU sessionをA〜Dへ分ける。Session Aは環境、exact artifact、
Phase 12相当のmodel-free operator、Qwen3.5-4B短生成までを完結させ、後続sessionを実行できる土台かどうかを判断する。
Session B以降はAのPASS後に進み、修正時は影響したsessionと下流の最小rowだけをfocused rerunする。

## 開始根拠

- Phase 12はUbuntu 24.04、kernel `6.8.0-124-generic`、amdgpu `6.16.13`、ROCm 7.14.0、
  MI300X VF x1、`gfx942:sramecc+:xnack-`、wave64、NPS1/SPX、VMM=trueでBF16/FNUZ FP8、4B/9B、
  contiguous-resident KV、OpenAI service、性能、llama.cpp比較をPASSした。
- Phase 16のFP8/NVFP4 KV、Phase 16Fのmixed low-bit model、Phase 17/18のMTP・vision、Phase 19のMoE、
  Phase 20のGGUF公開経路、Phase 31のchunked prefill、Phase 33〜35のattention/GDN変更は、MI300Xでは
  compile-onlyまたは未実行の範囲を含む。
- Phase 33/35のlong-context Full Attention/GDN高速providerは現在exact `gfx1030`/`gfx1201`へ限定され、
  `gfx942`は従来providerを維持する。Session Aではこのroutingを期待値として確認し、未検証providerが誤選択される
  状態を性能差ではなくcorrectness blockerとして扱う。
- Phase 12のfixed llama.cpp比較ではsLLM BF16 E2Eに2.50〜5.58倍の差が残った。性能改善自体はPhase 36の目的ではないが、
  latest mainの現状把握と将来判断のためSession Dで再取得する。

## 権限・実行境界

- ユーザーの2026-08-21の開始指示によりSessions A〜Dを実行した。A0〜A5のidentity、ROCm 7.14 exact build/load、
  tiny/profiler、99 operator、Qwen3.5-4B BF16/FNUZ FP8短生成に続き、Bのlow-bit KV/10k+、CのMTP/vision/OpenAI
  lifecycle、Dの反復性能/fixed llama.cpp/rocprofv3をPASSした。A〜DをPhase 36の完了範囲とする。
- correctness、安全なtarget routing、resource/lifetime、cleanupの問題はPhase内で修正する。
- 性能が旧値またはllama.cppより遅いことだけではcorrectness blockerにしない。profileで原因を分類し、追加最適化は
  Phase 36へ無制限に取り込まない。
- Git commitはdraft実行identityの必須条件にしない。source tree、tracked diff、build input、toolchain、model lock、binary digestから
  semantic candidateを一意にできればよい。integration closeoutでは最終candidate identityを固定する。
- provider imageのkernel/amdgpu driverを無断で交換しない。project ROCm user-spaceを追加する場合も同一rootへ閉じ、
  provider既定の別ROCmをproduction processへ混在させない。
- model、GGUF、binary、raw trace、credentialはGitへ追加しない。summary、schema、再現に必要な小さいmetadataだけを追跡する。

## 対象と非対象

### 対象

- single visible MI300X x1、exact `gfx942`、wave64、BF16およびCDNA3 E4M3FNUZ FP8。
- primary modelはfixed Qwen3.5-4B dense GGUF。BF16とFP8を主要dtypeとする。
- Phase 12 operator回帰、GGUF公開runtime、FP16/FP8/NVFP4 KV、chunked prefill、MTP、vision、OpenAI service。
- fixed llama.cppとの同一VM・同一token条件比較と代表caseのrocprofv3分類。

### 非対象

- MI300X x2以上、Infinity Fabric、P2P、RCCL、tensor/expert/pipeline parallel、RDMA/RoCE。
- MI300A、MI325X、bare metal、別cloud、別ROCm tupleへの一般化。
- 新しい性能providerの広い探索、TurboQuant、Paged Attention採用、DeepSeek V4、Responses API。
- Phase 36の実機結果だけを根拠に`experimental`から広い`supported`へ昇格すること。

## Phase全体の受入条件

1. Session Aでexact `gfx942` artifact、wave64、FNUZ/BF16基本operator、Qwen3.5-4B短生成がHIP-only、fallbackなしでPASSする。
2. Session Bでlow-bit KVと10k+ chunked prefillの数値、capacity、physical memory、cleanupを確認する。
3. Session CでMTP、vision、OpenAI non-stream/SSE/cancel/recoveryを代表条件で確認する。
4. Session Dで同一MI300X tupleのsLLM repeated performance、fixed llama.cpp、代表profileを取得する。
5. 発見したcorrectness、target routing、resource、cleanup defectを修正し、変更箇所と下流のfocused rerunをPASSする。
6. exact tuple、candidate/model/artifact identity、実行範囲、未実行範囲をcompatibility/historyへ同期し、別SKUへ一般化しない。

## Session構成

| session | 主目的 | 標準GPU時間 | 依存 |
| --- | --- | ---: | --- |
| A | identity、exact build/load、99 operator回帰、Qwen 4B短生成 | 2〜3時間、上限4時間 | なし |
| B | low-bit KV、chunked prefill、10k+ context、memory accounting | 3〜5時間 | A PASS |
| C | MTP、vision、OpenAI service lifecycle | 3〜4時間 | A PASS。BのKVを使うrowはB PASS |
| D | repeated performance、llama.cpp、rocprofv3 | 4〜6時間 | A〜Cのprimary Qwen範囲PASS |

標準の必須範囲はA〜Dで合計12〜18 GPU時間を見込む。VM準備やmodel transferが再利用でき、問題がなければ
10〜14時間へ収まる可能性がある。

## Session A詳細計画

### Aの目的

latest mainがMI300X上で安全に起動し、後続のmodel・long-context・service検証へ進める状態かを、短時間でfail-closedに判断する。
Aではlow-bit KV full matrix、10k+、MTP、vision、OpenAI service、性能比較を実行しない。operatorとQwen短生成で問題が出た場合は、
課金中に広い試行錯誤を続けず、再現に必要な最小evidenceを取得してVMを停止し、local修正へ戻る。

### Aの固定受入条件

1. 単一visible deviceがMI300X VF、logical `gfx942`、実device名`gfx942:sramecc+:xnack-`、wave64、期待するCU/HBMとして見える。
2. build/runtime ROCm、HIP、ROCr、hipBLAS、hipBLASLtがproject指定のROCm 7.14 rootへ解決し、別releaseを混在させない。
3. exact `gfx942`、Code Object V6、wave64、XNACK off、SRAM ECC onのartifactだけがloadされる。
4. Phase 12相当99 operator caseが独立数値oracle、native dispatch、fallbackなし、cleanup zeroでPASSする。
5. Qwen3.5-4B BF16/FNUZ FP8 GGUFの固定短生成がHIP-onlyで完了し、selected dtype/provider/targetをreportする。
6. 終了時にrequest/workspace allocation、GPU process、retry/quarantineが0または事前baselineへ復帰する。

### A開始前のlocal準備（GPU課金外）

- source、tracked diff、submodule/reference状態、Cargo.lock、ROCm target設定を記録し、semantic candidate IDを生成する。
- `gfx942` release buildとhost testsをlocal compile-onlyで通し、offload bundle、Code Object、feature metadataを検査する。
- Session A runnerをdry-runし、全work unit、command、timeout、出力path、expected case countが選択されることを確認する。
- Phase 12のpreflight/operator runnerをlatest public GGUF経路へ合わせ、旧safetensors/sidecar公開引数へ依存させない。
- Qwen3.5-4B BF16/FP8 GGUF、derived lock、SHA-256、必要なtokenizer/chat metadataを準備する。
- model/binaryをVM上で再取得する場合はURLとhashを用意する。credentialは短命・最小権限でrepository外へ置く。
- Session Aのraw出力保存先とVM外退避先を用意し、終了時にsummary digestだけをGitへ持ち込めるようにする。

### A0: provisionとidentity（目安20〜30分）

- MI300X x1のUbuntu 24.04 VMを優先する。instance、virtualization/SR-IOV、CPU/RAM/disk、region、provider imageを記録する。
- GPU UUID、BDF、product、actual GCN arch、wave size、CU、HBM、NPS/SPX、VMM、ECC、foreign processを取得する。
- OS point release、kernel、amdgpu、ROCm roots、compiler、HIP runtime、ROCr、hipBLAS/hipBLASLt、rocprofv3を記録する。
- `amd-smi metric`がprovider例外で取得不能な場合は`unavailable`と理由を記録する。0へ置換せず、gfx942以外へ例外を広げない。
- GPU共有、active foreign workload、ECC uncorrectable、target/feature不一致、必要disk不足があればA1へ進まない。

### A1: ROCm root・artifact build/load（目安30〜45分）

- provider driverを維持し、必要ならROCm 7.14 user-space rootを追加する。production processのloader closureを同一rootへ固定する。
- latest candidateをexact `gfx942`だけでrelease buildし、CLI/server/evidence binariesのSHA-256とsizeを記録する。
- offload bundleがgeneric `gfx9-4`や別exact targetを含まず、Code Object V6、wave64、XNACK off、SRAM ECC onであることを検査する。
- actual feature suffixをlogical `gfx942`へ既存contractどおり正規化し、`gfx942:sramecc+:xnack+`等を黙って許可しない。
- wrong-target artifactまたは期待targetの改変を使ったnegative caseを一件実行し、load前または起動時にfail closedすることを確認する。

### A2: tiny runtime・library probe（目安15〜25分）

- allocation/copy、tiny `41→42` kernel、event、stream、device synchronize、cleanupを実行する。
- hipBLAS/hipBLASLtのBF16およびFNUZ solution queryを実行し、unsupported/zero solutionをPASS扱いしない。
- OCP E4M3FNからresident E4M3FNUZへの全有限byte変換、negative zero正規化、scale rebasingの小oracleを確認する。
- rocprofv3が代表tiny kernelを取得できることを確認する。権限不足時はSession D profileだけを保留できるか分類するが、
  native runtime自体の失敗と混同しない。

### A3: Phase 12相当model-free operator matrix（目安45〜75分）

次の99 caseを一つのlogical matrixとして実行する。timeout、crash、zero selection、CPU execution、unsupported solution、
fallback、cleanup失敗はPASSにしない。

| family | case数 | 主な境界・確認 |
| --- | ---: | --- |
| FNUZ FP8 hipBLASLt matmul | 2 | production decode/prefill代表shape、resident FNUZ |
| BF16 MMVF/GEMM | 17 | M=1、2〜8、非整列K/N、production shape、wave64/provider |
| elementwise | 21 | 非整列要素数、NaN/Inf contract、alias/error |
| attention preprocess/RoPE | 8 | token/position境界、head layout、wave64 |
| KV state | 19 | 1/3/17、127/128/129、255/256/257、1023/1024/1025 |
| Full Attention | 16 | causal poison、GQA、非整列KV長、FP16 baseline |
| output gate | 6 | RMSNorm、SiLU/gate、BF16 round stage |
| RMSNorm wave64 | 7 | width 1/3/255/256/257/2560/4096 |
| GDN | 3 | token 1/3/17、Qwen実layout、state publication |
| 合計 | **99** | 独立oracle、HIP-only、fallback false、cleanup zero |

- BF16/FNUZ matmulとRMSNormはexpected kernel/provider symbolまで照合する。
- Full AttentionとGDNはSession Aではgfx942 baseline providerが選択されることを確認する。Phase 33/35の
  gfx1030/gfx1201限定providerが選択された場合はtarget routing defectとして停止する。
- Phase 35共通source変更によるcompile成功だけを数値PASSへ読み替えない。gfx942実機出力を独立oracleへ照合する。
- 最初のfailureで残り全部を盲目的に継続せず、同familyの最小case、境界両側、provider auditを取得してlocal修正へ戻す。

### A4: Qwen3.5-4B GGUF短model smoke（目安30〜45分）

- fixed Qwen3.5-4B BF16 GGUF/derived lockをloadし、固定promptの3 input / 5〜17 output、Unicode、stopを代表rowとして実行する。
- 同じmodel lineageのFP8 GGUFをloadし、OCP storageからFNUZ resident変換、selected `native-fnuz` providerを確認する。
- BF16/FP8とも全dispatch target `gfx942`、HIP-only、CPU/backend fallbackなし、model partial offloadなしを要求する。
- greedy token、top-1、または固定logit位置を既存oracleと比較する。数値変更台帳にない原因不明の差はAで停止する。
- resident/peak HBM、load時間、first request、second resident request、終了時allocationを記録する。
- Session Aでは9B、10k input、MTP、vision、service、repeated performanceを追加しない。

### A5: cleanup・summary・VM停止（目安15〜25分）

- request/model/workspace allocation、GPU process、ECC、foreign process、loader pathをpost観測する。
- serverを起動していないSession Aでも、CLI/evidence process group、temporary workspace、model mmap/file handleの残留を確認する。
- raw log、operator report、binary/model/candidate digestをVM外artifact storeへ退避し、Gitへraw trace/model/binaryを追加しない。
- runner生成summaryは`phase36-mi300x-session-a-summary-v1.schema.json`、最終evidenceは
  `phase36-mi300x-session-a-final-v1.schema.json`で検査し、tuple、99 case、model smoke、failure、未実行範囲を記録する。
- Bへ直ちに進まない場合はVMを停止する。credential/VMを保持する場合は期限と所有者を記録し、不要なら削除する。

### Session A実行結果

- A0〜A2はexact MI300X VF / gfx942 tuple、単一ROCm 7.14 loader closure、tiny/library/profilerをPASSした。最終
  release buildはlogical targetを`gfx942`に保ち、全11 artifactの唯一のdevice bundleを
  `gfx942:sramecc+:xnack-`へ固定した。全device code objectはCode Object V6 contract、ELF flags `0xE4C`
  （SRAM ECC on / XNACK off）、wave64であり、generic/別targetを含まない。`gfx1201`指定はdispatch前に拒否した。
- A3はPhase 12相当99 operatorを最終artifactで全てPASSした。family別case数は
  `2/17/21/8/19/16/6/7/3`、実dispatch数は`4/17/21/8/19/16/6/7/6`、operator summary SHA-256は
  `5daa5869932513490c50cbb9ff330cf47fb581aa333fc1133fc0261a1192222d`である。
- A4はcanonical 4B BF16/FP8 GGUFのverify、fixed、Unicode、stopをHIP-only、fallback/partial offloadなし、cleanup 0で
  PASSした。BF16/FP8 `Hello`は3 token目だけ`2688`/`1044`へ分岐し、gfx942 wave64 BF16とhipBLASLt FNUZの
  cross-provider N1差へ局所化した。Unicodeとstopは一致した。
- Phase 29のGDN wave32 treeが承認scope外のgfx942へ漏れていたため、wave64 buildはPhase 28の逐次norm順を維持するよう
  修復した。GDN 1/3/17 tokenのfocused oracleは3/3 PASSし、RDNA provider、ABI、dispatch/resourceは変更していない。
- A4のresident確認はBF16/FP8ともmodel load 1回、warmup 3、measured 10、second resident requestを含む14 requestで
  model reuseを確認した。resident/peakはBF16 `8,411,592,192`/`8,477,011,968` bytes、FP8
  `4,847,029,760`/`4,912,449,536` bytesで、model drop後は両方0だった。
- A5はGPU/sLLM process 0、model handle 0、VRAM baseline復帰、全sysfs RAS blockのCE/UE 0、
  retryable/durable cleanup 0を確認した。
  ROCm 7.14 bind mountは解除し、provider既定の`/opt/rocm-7.2.4` linkへ復元した。A closeout時点では後続実行へ備えて
  VMと専用SSH keyを保持し、当時の後続sessionを未実行としてfinal Session A summaryへ明記した。その後B〜Dを実行したが、
  Session A summaryのscopeと過去時点の記録は変更しない。Phase closeout時にVMはユーザーが削除した。
- [Session A final summary](../../../../../../ci/matrix/phase36-mi300x-session-a-final-v1.json)はSHA-256
  `9e39c0aba7bd1a11725b95df0e15f6a5728cbde2e57ec250d07bc0432ca27dd4`、
  [strict schema](../../../../../../ci/schema/phase36-mi300x-session-a-final-v1.schema.json)はSHA-256
  `b00dc2494f4aa7fe21cd27c2ab6f1e2627a5b13e1875fa1e14a2ed5d052c8def`である。

### Aのfirst-hour stop/go

最初の60分でA0〜A2を終える。次のいずれかならmodel/operatorの長いrunへ進まずVMを停止する。

- exact MI300X/gfx942/wave64/feature suffixを確認できない。
- ROCm build/runtime rootを同一releaseへ閉じられない、またはprovider driverとの組合せでtiny runtimeが失敗する。
- exact gfx942 artifactをloadできない、generic/別target artifactが黙ってloadされる。
- BF16/FNUZ hipBLASLt solutionが得られず、既存contractにないfallbackしか動かない。
- ECC uncorrectable、foreign active workload、共有/partition不明により性能以外のcorrectness evidenceも信用できない。

### Aで問題が見つかった場合の修正手順

1. failureをenvironment、runner/allowlist、artifact/loader、provider selection、kernel数値、resource/lifetime、model integrationへ分類する。
2. VM上でsourceを場当たり的に増分編集せず、再現reportと最小inputを保存してVMを停止する。
3. localでhost test、compile-only、可能なoracleを追加して修正する。既存RDNA targetへの影響がある場合だけ該当targetのfocused controlを行う。
4. 新candidate identityを作り、Session Aの失敗family、境界両側、Qwen smokeまでをfocused rerunする。
5. 修正がoperator semantic、public runtime、target routingへ波及した場合だけ、A全体またはB以降の影響rowを再実行する。

同じ原因で二回修正に失敗、Aが4 GPU時間超、またはdriver/kernel/ROCm tuple変更が必要になった場合は、追加課金runを止めて
Phase 36のsession構成を再計画する。

## Session B以降の予定概要

### Session B: low-bit KV・chunked prefill・long-context

- FP16、dynamic FP8、static FP8、NVFP4 KVのappend/read/attentionをMI300Xで実機照合する。
- Phase 16相当のFP8/NVFP4各17 case、capacity境界、nonfinite、committed byte、cleanupを実行する。
- 512/2K/4K/8K/16Kのchunk指定とMI300X自動選択を監査し、10,001 input / 2 outputをFP16/FP8で実行する。
- gfx942の`contiguous-resident`固定が192 GB HBM上で設定context全量を物理確保する影響を測り、VMM=trueの
  `virtual-contiguous`または増分commit比較が必要かを事実として整理する。provider変更は別の明示判断にする。
- peak HBM、arena high-water、GTT spill、fallback、cleanupを記録し、memory failureとkernel数値failureを分ける。

#### Session B実行結果

- FP16、dynamic FP8、static FP8、NVFP4のFull Attentionを各29 case、合計116/116でPASSし、FP16 KV stateも19/19を
  PASSした。独立NumPy oracleはquantization、nonfinite、padding、token/head/block境界とinvalid scale offset拒否を確認した。
- canonical 4B BF16 GGUFでFP16 KVとdynamic FP8 KVを、それぞれauto、512、2,048、4,096、8,192、16,384 token指定の
  6通り、
  合計12 row実行した。全rowがexact 10,001 input / 2 output、入力IDは全て`23066`、生成IDは
  `[23066,23066]`、HIP-only、fallbackなし、cleanup 0だった。MTP修正後の最終CLIでも両auto rowをfocused rerunした。
- autoは10,001 tokenを1 chunk、明示指定は期待する20/5/3/2/1 chunkへ分割した。workspace arena high-waterは
  auto/16,384指定で`5,278,049,280` bytes、512指定で`270,209,024` bytesだった。request stateはFP16
  `379,289,600` bytes、dynamic FP8 `217,961,216` bytesで、低bit化による削減を物理計測と一致させた。
- gfx942は意図どおり`contiguous-resident`を維持した。各rowでHBM peakを取得し、GTTは小さいruntime baseline増分だけで
  model spillを示さず、終了後はHBM `299,687,936` / GTT `22,695,936` bytesの共通baselineへ戻った。VMM=trueを
  `virtual-contiguous`へ変更する判断は行っていない。

### Session C: MTP・vision・OpenAI service

- Qwen3.5-4BのMTP off/on、width 2/3/4/7/8、BF16+FP16 KVとFP8+FP8 KVの代表rowを逐次target oracleへ照合する。
- accepted/rejected prefix、KV/GDN state publication、rewind/replay、通常CLIのvisible tokenを確認する。
- PNG/JPEG/WebP代表画像でvision projector、64 projected token、text generation、lazy resident、cleanupを確認する。
- OpenAI raw non-stream/SSE、official client、reasoning split、stop、seeded sampling、disconnect/recovery、cancel、
  連続request、二並行request、graceful shutdownを実行する。
- network/backpressure不具合とGPU provider不具合を分け、サービスの失敗をmodel/kernel PASSへ混ぜない。

#### Session C実行結果

- BF16 target＋FP16 KVのtarget-onlyとMTP width 2/3/4/7/8を実行し、全てtarget-onlyと同じ16 visible tokenへ一致した。
  widthごとのtarget rowsは3/4/5/8/9、proposalは14/21/28/49/56で、accepted/rejected合計、state capacity slack、
  publication/rewindをfail-closedに監査した。FP8 target＋dynamic FP8 target KVもtarget-only/width 3をPASSした。
  現行のMTP side pathは明示的にBF16 weights＋FP16 KVであり、target FP8と混同しない。
- 公開CLIのMTPがgfx1201とwidth 1へ固定されていたため、forced width 1〜8、exact gfx942 admission、bounded state slack、
  quantized GGUF plan validationを追加した。初回width 2のcapacity overflowとFP8 plan schema拒否は修正し、影響rowを
  最終CLIで再実行した。通常auto/off経路、RDNA provider、model recipeのfail-closed境界は維持する。
- 256×256のPNG/JPEG/WebPを各1枚実行し、各入力のimage-pad 64 token、同じ生成ID
  `[760,1156,6587,264]`、HIP-only、fallbackなし、cleanup 0を確認した。serverでは最初の画像でvision residentがlazyに
  増加し、2件目で再利用、shutdown後はHBM/GTT baselineへ復帰した。
- OpenAI profile v1はraw/official-clientのnon-stream/SSE、reasoning split、stop、seeded sampling、1023/1024/1025境界、
  HIP dispatch後disconnect/cancelと直後のrecovery、二並行queue、graceful shutdownをPASSした。全completed rowはexact
  gfx942/HIP、fallbackなし、request/workspace cleanup 0である。`amd-smi metric`はproviderの`partition`例外により
  `unavailable`として保持し、0へ置換していない。
- GPU実行後はprocess 0を確認してROCm 7.14 bindを解除し、provider既定の`/opt/rocm-7.2.4`へ復元した。B/C rawは
  repository外の`~/.local/share/sllm-evidence/phase36/session-{b,c}/enc1-gpuvm015-2026-08-21/raw`へ退避した。
- 最終candidate source digestは`f07b31c9a83aee326c62de3c2f0d1d2da8ff189a66085526ddf79edad2bdf94a`である。
  [Session B summary](../../../../../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)はSHA-256
  `13e4d86859191dbadae66e940bd3adfd8e1ec598fa8dba627de8f3581f6bf274`、
  [Session C summary](../../../../../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)はSHA-256
  `4fdc5e4f029e097721b2bc1dfb40b0f51282c268dc55fcbbeb4a7c66073c42f5`である。対応schemaのSHA-256はB
  `f3f0f6204655b646805b33155cd347243699367084936e4b37f8298d91dcbfce`、C
  `4ebb1d0f76a570d7b2d624a4d9f0c05aabe00e05d951dd4c2bf1533b3db20fc0`、repository外raw manifestのSHA-256はB
  `474cc874d5e832c76fa000abcb6d2a418fa1a2b73e8e1caa3e876e9474fca607`、C
  `fe2a475e9f7f9c007f08f6cca5d636100e1c0355c0cdbb3e86a7f55ae86f0412`である。

### Session D: performance・llama.cpp・profile

- Qwen3.5-4B BF16/FP8でshort-odd、32/32、prefill-long、decode-long、10,001 input / 2 outputを、
  原則3 warmup＋10 measuredで取得する。TTFT、prefill/decode tok/s、TPOT、E2E、resident/peak HBM、分散を記録する。
- fixed llama.cpp commitをexact gfx942だけでbuildし、同一VM、同じmodel revision、token IDs、dtype/KV、offload条件で比較する。
- artifact byte identityが異なる場合はE1 system-equivalentとして明示し、厳密同一比較と表記しない。
- 代表10,001/2をrocprofv3でprojection、Full Attention、GDN、MTP/other、kernel外へ分ける。
- Phase 12からの変化、current V620/R9700との差、llama.cpp差を表にする。Phase 36中の性能修正は明白なprovider誤選択、
  pathological fallback、resource defectに限定し、新規最適化探索は別Phase候補へ送る。

#### Session D実行結果

- Qwen3.5-4B BF16/FNUZ FP8の各5ケースをdirect token lane、FP16 KV、greedy、3 warmup＋10 measuredでPASSした。
  direct laneはPhase 36以降のrepository-owned evidence専用で、GGUF/derived lockだけを受け付ける。公開product CLIや
  Phase 20以前のsource lock/cache runnerを復活させるものではない。
  全rowはexact `gfx942`、HIP-only、fallbackなしで、process終了後のHBM/GTTは共通baseline
  `299,687,936` / `22,695,936` bytesへ復帰した。10,001/2 E2E中央値はBF16 `22.556130816`秒、FP8
  `22.556528472`秒、生成IDは両方`[23066,23066]`だった。
- Phase 12比のE2Eはshort-odd/32-32/decode-longでBF16 `-3.52/-2.54/-3.01%`、FP8
  `-3.47/-3.32/-2.64%`、prefill-longだけBF16 `+14.96%`、FP8 `+11.87%`だった。BF16 10,001/2は
  Phase 35のV620 `22.683165076`秒とほぼ同じ（`-0.56%`）、R9700 `65.214329776`秒より`65.41%`短かった。
- fixed llama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`をexact gfx942だけでbuildし、5ケースを
  全GPU offload、BF16 weights、F16 KV、同一token IDでPASSした。GGUFは同じupstream revisionだがsLLM用MTP含有版と
  llama.cpp用`--no-mtp`版でbytes/tensor setが異なるためE1である。10,001/2中央値は`0.8512540725`秒で、sLLM比は
  E2E `26.4975`倍だった。
- BF16 10,001/2のrocprofv3は1,263 kernel call、device total `22.539747157`秒を取得した。device time shareは
  GDN `73.95%`、Full Attention `25.12%`、projection `0.70%`、MTP/other `0.23%`で、host wall
  `35.096660866`秒からkernel interval unionを除いたexternalは`12.556934578`秒だった。性能差自体をblockerにせず、
  新規最適化は別Phase候補とする。
- rawはrepository外の
  `/home/homelab1/.local/share/sllm-evidence/phase36/session-d/enc1-gpuvm015-2026-08-21/raw`へ退避した。
  GPU/sLLM/llama/rocprof process 0、HBM/GTT baseline、RAS CE/UE 0を確認し、ROCm 7.14 bindを解除してprovider既定の
  `/opt/rocm-7.2.4`へ復元した。
  [Session D summary](../../../../../../ci/matrix/phase36-mi300x-session-d-summary-v1.json) / [schema](../../../../../../ci/schema/phase36-mi300x-session-d-summary-v1.schema.json)の
  SHA-256は`5d05db578fc6466c4dfcf355efde9cd04b0b07567300f882a24703b31bb19214` /
  `1ce037012e128750021f7323735d752f03e57b66fc6be1f3ff86799838867cbb`、raw manifestは
  `7b57db0319035da6f7fdcbc00e7369f76c43810946dfc7e24a5948a4e7e3aed0`である。

## Session間のidentity・evidence contract

- 各sessionはsource/build input、toolchain、GPU tuple、binary、model/derived lock、runner、report schemaを記録する。
- docs-only変更でsemantic identityが変わらない場合は前sessionのGPU evidenceを再利用できる。source/kernel/provider/build inputが
  変わった場合は影響sessionだけ新identityでfocused rerunする。
- raw reportとtraceはrepository外へ保持し、tracked summaryからdigest、case count、結果、保存先識別子を参照する。
- Session A〜Dを一つの巨大PASSへ丸めず、各sessionのPASS/FAIL/PARTIAL、修正、scopeを独立して残す。

## closeout

- 一回のintegration reviewを行い、findingは変更箇所だけfocused re-reviewする。
- Phase 36 summary、history、main plan、runtime、GPU/software互換性をexact MI300X tupleの範囲で同期する。
- plan/historyを相互linkしてarchiveする。A〜DのPASSとユーザー決定によるconditional extensionのscope削除を記録し、
  9B、Gemma/MoE、長時間安定性をPhase 36のPASSへ含めない。
- VM、disk、model cache、SSH key、known-host、provider credentialの保持/削除状態を明示する。

[Phase 12計画](../../../../archive/2026/08/11-20/phase12-mi300x-validation.md)
[Phase 12 summary](../../../../../../ci/matrix/phase12-mi300x-summary-v1.json)
[Phase 35計画](../../../../archive/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)
[MI300X software tuple](../../../../../compatibility/software.md)
[AMD GPU compatibility](../../../../../compatibility/amd-gpu.md)
[runtime architecture](../../../../../architecture/runtime.md)
[メイン計画](../../../../main-plan.md)
[Phase 36履歴](../../../../../history/2026/08/11-20/phase36-mi300x-current-main-validation.md)
