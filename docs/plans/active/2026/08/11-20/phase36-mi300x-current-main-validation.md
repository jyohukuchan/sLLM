# Phase 36: MI300X latest-main実機再検証

> 状態: planned（計画のみ。VM作成・GPU実行・実装修正は未開始）
> 作成日: 2026-08-20

## 目的

Phase 12で一度実機PASSしたHot Aisle MI300X VF x1、exact `gfx942`の経路を、Phase 35後のlatest mainで再検証する。
今回は初回CDNA3移植ではなく、Phase 13以降に追加されたGGUF、low-bit KV、chunked prefill、MTP、vision、MoE、
Full Attention/GDN変更を含むcurrent runtimeの回帰・未検証範囲を段階的に確認し、実機で見つかった問題を修正する。

課金中の長い単一sessionへ全matrixを詰め込まず、GPU sessionをA〜Eへ分ける。Session Aは環境、exact artifact、
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

- 本文書の作成は計画だけであり、MI300X VMの作成・起動、credential作成、GPU実行、production source変更を開始しない。
  ユーザーがPhase 36開始を指示した後に実行する。
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
- conditional extensionとしてGemma 4 mixed NVFP4、Qwen3.5-35B-A3B MXFP4 MoE。
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
5. Session Eを実行する場合、Gemma/MoEのtarget admissionと実行providerを明示し、未対応ならunsupported errorと
   実装不足を区別する。
6. 発見したcorrectness、target routing、resource、cleanup defectを修正し、変更箇所と下流のfocused rerunをPASSする。
7. exact tuple、candidate/model/artifact identity、実行範囲、未実行範囲をcompatibility/historyへ同期し、別SKUへ一般化しない。

## Session構成

| session | 主目的 | 標準GPU時間 | 依存 |
| --- | --- | ---: | --- |
| A | identity、exact build/load、99 operator回帰、Qwen 4B短生成 | 2〜3時間、上限4時間 | なし |
| B | low-bit KV、chunked prefill、10k+ context、memory accounting | 3〜5時間 | A PASS |
| C | MTP、vision、OpenAI service lifecycle | 3〜4時間 | A PASS。BのKVを使うrowはB PASS |
| D | repeated performance、llama.cpp、rocprofv3 | 4〜6時間 | A〜Cのprimary Qwen範囲PASS |
| E | Gemma mixed NVFP4、Qwen MoE MXFP4、30〜60分安定性 | 4〜8時間、conditional | A PASS。model別admission準備 |

標準の必須範囲はA〜Dで合計12〜18 GPU時間を見込む。VM準備やmodel transferが再利用でき、問題がなければ
10〜14時間へ収まる可能性がある。Session EはQwen denseのMI300X互換性claimを妨げないconditional extensionとする。

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
- `phase36-mi300x-session-a-summary-v1.json`、schema、host testへ、tuple、99 case、model smoke、failure、未実行範囲を記録する。
- Bへ直ちに進まない場合はVMを停止する。credential/VMを保持する場合は期限と所有者を記録し、不要なら削除する。

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

### Session C: MTP・vision・OpenAI service

- Qwen3.5-4BのMTP off/on、width 2/3/4/7/8、BF16+FP16 KVとFP8+FP8 KVの代表rowを逐次target oracleへ照合する。
- accepted/rejected prefix、KV/GDN state publication、rewind/replay、通常CLIのvisible tokenを確認する。
- PNG/JPEG/WebP代表画像でvision projector、64 projected token、text generation、lazy resident、cleanupを確認する。
- OpenAI raw non-stream/SSE、official client、reasoning split、stop、seeded sampling、disconnect/recovery、cancel、
  連続request、二並行request、graceful shutdownを実行する。
- network/backpressure不具合とGPU provider不具合を分け、サービスの失敗をmodel/kernel PASSへ混ぜない。

### Session D: performance・llama.cpp・profile

- Qwen3.5-4B BF16/FP8でshort-odd、32/32、prefill-long、decode-long、10,001 input / 2 outputを、
  原則3 warmup＋10 measuredで取得する。TTFT、prefill/decode tok/s、TPOT、E2E、resident/peak HBM、分散を記録する。
- fixed llama.cpp commitをexact gfx942だけでbuildし、同一VM、同じmodel revision、token IDs、dtype/KV、offload条件で比較する。
- artifact byte identityが異なる場合はE1 system-equivalentとして明示し、厳密同一比較と表記しない。
- 代表10,001/2をrocprofv3でprojection、Full Attention、GDN、MTP/other、kernel外へ分ける。
- Phase 12からの変化、current V620/R9700との差、llama.cpp差を表にする。Phase 36中の性能修正は明白なprovider誤選択、
  pathological fallback、resource defectに限定し、新規最適化探索は別Phase候補へ送る。

### Session E: Gemma・MoE・安定性（conditional）

- Gemma 4 mixed NVFP4 GGUFのadmission、W4A4/W8A8、FP8 KV、BF16/ignore recipeをMI300Xへ移植・確認する。
- Qwen3.5-35B-A3B MXFP4 MoEのtarget admission、router/top-8、routed/shared expert、40-layer full model、CLI/APIを確認する。
- 現行evidence binaryまたはproduction admissionがgfx1030/gfx1201限定の場合、runner限定とruntime未対応を分けて修正する。
- primary Qwen denseがA〜Dで完了していれば、Gemma/MoE未完をPhase全体のMI300X dense evidenceへ遡及させない。
- 対応modelで30〜60分の連続生成・cancel混在・resident連続requestを行い、HBM/ECC/process leakを確認する。

## Session間のidentity・evidence contract

- 各sessionはsource/build input、toolchain、GPU tuple、binary、model/derived lock、runner、report schemaを記録する。
- docs-only変更でsemantic identityが変わらない場合は前sessionのGPU evidenceを再利用できる。source/kernel/provider/build inputが
  変わった場合は影響sessionだけ新identityでfocused rerunする。
- raw reportとtraceはrepository外へ保持し、tracked summaryからdigest、case count、結果、保存先識別子を参照する。
- Session A〜Eを一つの巨大PASSへ丸めず、各sessionのPASS/FAIL/PARTIAL、未実行、修正、scopeを独立して残す。

## closeout

- 一回のintegration reviewを行い、findingは変更箇所だけfocused re-reviewする。
- Phase 36 summary、history、main plan、runtime、GPU/software互換性をexact MI300X tupleの範囲で同期する。
- plan/historyを相互linkしてarchiveする。実行しなかったSession Eを暗黙のPASSにせず、conditional未実行として残す。
- VM、disk、model cache、SSH key、known-host、provider credentialの保持/削除状態を明示する。

[Phase 12計画](../../../../archive/2026/08/11-20/phase12-mi300x-validation.md)
[Phase 12 summary](../../../../../../ci/matrix/phase12-mi300x-summary-v1.json)
[Phase 35計画](../../../../archive/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)
[MI300X software tuple](../../../../../compatibility/software.md)
[AMD GPU compatibility](../../../../../compatibility/amd-gpu.md)
[runtime architecture](../../../../../architecture/runtime.md)
[メイン計画](../../../../main-plan.md)
[Phase 36履歴](../../../../../history/2026/08/11-20/phase36-mi300x-current-main-validation.md)
