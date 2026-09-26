# Phase 87 段階4: W×A16の廃止と契約の整理

## 状態

2026-09-24の初回受入では、旧A16の選択経路・公開契約を整理し、Qwen3.8、Gemma 4 12B、
Gemma 4 26B-A4Bの直接artifactを両GPUで確認した。最初のQwen3.8 smokeで、旧W4A16拒否条件が
W4A4にも当たる欠陥を発見し、`metadata.nvfp4 && !metadata.nvfp4_w4a4`へ限定して再実行でPASSした。

同日のレビューで、呼び出し元0件のNVFP4 W4A16 kernel 2本・`launch_nvfp4`・`select_nvfp4_variant`が
`native/lowp`に残り、binaryにも入っていると分かった。このため初回の完了判断を差し戻した。
**WU-4Rで残存コードを削除し、段階4の受入を完了した。** 実行可能なA16経路は残さず、
過去のABI・variant数値は監査用tombstoneとして保持する。

## 対象と決定

段階4の対象は、活性値をBF16のまま読む次の低精度経路である。

| 形式 | 旧経路 | 現行状態 |
| --- | --- | --- |
| NVFP4 W4A16 | 旧Qwen／Gemma weight-only sidecar、lowp ABI value 2 | retired。ABI値と監査用provider identityは保持し、prepare／planと旧graph／upload入口はfail-closed |
| MXFP8 W8A16 | Phase 85 M=1 opt-in、ABI value 6、selector ID 101 | retired。ABI／IDは履歴用tombstone、A16 selector／launcher／evidence branchは登録しない |
| MXFP6 W6A16 | Phase 85 M=1 opt-in、ABI value 7、selector ID 102 | retired。ABI／IDは履歴用tombstone、A16 selector／launcher／evidence branchは登録しない |

この整理は現行の直接artifact経路をW×A16から移行する作業ではない。段階0で、Qwen3.8本体、Gemma 4 12B、Gemma 4 26B-A4Bの
直接実行形式はすでにW4A4相当であることを確認していた。したがって、これらはW4A4の既存経路を維持する。
MXFP8 W8A8とMXFP6 W6A6は動的activation量子化を使う現行経路として維持する。

MXFP4の公開契約はW4A6へ更新した。`LOWP_MXFP4_W4A6_V1` はweight E2M1、activation OCP MXFP6 E3M2、block 32、E8M0 scaleを
表すversioned placeholderである。実装は未対応のため、format情報の照会は契約を返すが、matmul plan／launchはunsupportedで終了する。
既存MXFP4 W4A4をW4A6として扱ったり、未対応形式を別providerへfallbackしたりしない。

## model inventory

段階0で確認した入力scaleと、段階4の扱いは次のとおりである。

| model/artifact | scale／形式のsource確認 | 段階4での扱い |
| --- | --- | --- |
| `unsloth/Qwen3.8-27B-NVFP4` | MLP gate/up/down 168 tensorがW4A4で、input-global scale 168/168 | 直接W4A4経路を維持 |
| `unsloth/gemma-4-12b-it-NVFP4` | MLP 144 tensorがW4A4で、input-global scale 144/144 | 第一級W4A4経路を維持。旧sidecar W4A16はretired |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | custom MoEのgate/up/down 11,520 projectionにinput scale 11,520/11,520 | custom W4A4相当経路を維持。generic W4A16 consumerへ戻さない |
| `nvidia/Gemma-4-31B-IT-NVFP4` | locked remote index／metadataにinput scale 180/180を確認 | **reference-only**。ローカルpayloadがなく、runtime／GPU model checkは行わない |

Qwen3.8のW4A4基準dispatchと、artifactごとのscale件数の詳細は[段階0棚卸し](../11-20/phase87-a16-inventory.md)にある。31Bのremote metadata確認は
schemaとscale inventoryの参照であり、payload取得、weight upload、実行成功を意味しない。

## source contract

現行sourceで確認できる変更は以下である。

- `native/lowp/include/lowp/lowp.h` は旧A16 ABI数値を名前付きretired valueとして保持し、公開MXFP4は
  `LOWP_MXFP4_W4A6_V1`へ更新した。
- `native/lowp/src/lowp_plan.cpp` とprovider planは、旧A16形式をunsupportedとして拒否する。過去のprovider／selector enum値は
  audit identityの読み出し用に残るが、planはA16実行variantを返さない。
- Qwen3.5の旧W4A16 sidecar graph builderとCLI／server実行分岐は削除し、旧CLI指定は直接W4A4 artifactを案内して拒否する。
  Gemma旧sidecar upload/layout入口は直接W4A4 artifactを案内してfail-closedする。
- Phase 85の共有MX evidence toolからA16の環境変数、kernel symbol、dispatch検証分岐を除去した。A8／A6の現行MX evidenceは維持する。
  旧standalone W4A16 evidence binaryは現行targetから除去し、過去の結果は履歴とartifact identityとして保持する。
- WU-4Rで旧NVFP4 W4A16のkernel 2本、launcher、variant選択、kernel名・device symbol・grid表の実行分岐を削除した。
  variant数値8／9／10は名前付きtombstoneとして保持するが、実行可能kernelへは対応付けない。
  W4A4の量子化・FP16 staging・tensor-scale epilogueと共有算術helperは参照先を確認して維持した。
  `SLLM_NVFP4_FORCE_BASELINE`は旧W4A16 selector内から削除した。現役W4A4量子化での読み出しは、
  [FORCE_BASELINE診断用参照経路](../../../../development/force-baseline-reference-oracle.md)（本番のrollback経路ではない）として保持した。

これらはsource／host contractの整理であり、GPUで全対象modelの実行が成功したことを示す証拠ではない。

## 検証状態

### 現時点で確認できるhost／compile範囲

- 旧A16形式を公開format照会・provider plan・selectorで実行可能形式として扱わないsource条件を確認した。
- 公開MXFP4 W4A6のformat descriptorと、未実装planのfail-closed条件を確認した。
- Qwen3.8とGemma 4の第一級W4A4 binding、Gemma 26B custom scale inventoryを段階0のartifact/header/index証拠へ照合した。
- 旧A16 evidence identityを新しい現行経路の数値結果へ流用しない。既存の過去evidenceは履歴参照専用とする。

### GPU model check（2026-09-24）

以下をexact `gfx1030`／`gfx1201`のrelease buildで実施した。

- 直接Qwen3.8 W4A4の通常実行で、A16 selector／providerが選択されないことをdispatch auditで確認した。
- payloadがあるGemma 4 12B W4A4とGemma 4 26B custom W4A4を両GPUで起動し、W4A4経路、HIP-only、fallbackなし、cleanup zeroを確認した。
- 旧Qwen／Gemma sidecarとMX A16 opt-inが、prepareまたはgraph constructionで明示的にunsupportedとなることを確認した。
- 各実行の生成token、必要な数値control、binary／manifest identityを保存した。

Gemma 4 31Bはpayloadがないため、このGPU確認には含めない。remote metadataの参照結果だけでruntime対応済みとは報告しない。

| model | gfx1030 | gfx1201 | 確認範囲 |
| --- | --- | --- | --- |
| Qwen3.8 direct NVFP4 | `.local-artifacts/phase87/stage4/qwen38-gfx1030-r2.log` PASS | `.local-artifacts/phase87/stage4/qwen38-gfx1201.log` PASS | W4A4 prefill 168、decode ID84 224 dispatch、HIP-only、fallbackなし、token再現、cleanup zero |
| Gemma 4 12B direct NVFP4 | `.local-artifacts/phase87/stage4/gemma12b-gfx1030.log` PASS | `.local-artifacts/phase87/stage4/gemma12b-gfx1201.log` PASS | 直接artifactのW4A4 plan、固定sampling再現、HIP-only、fallbackなし、cleanup zero。kernel identity別集計は未実施 |
| Gemma 4 26B-A4B MoE | `.local-artifacts/phase87/stage4/gemma26b-gfx1030.log` PASS | `.local-artifacts/phase87/stage4/gemma26b-gfx1201.log` PASS | custom NVFP4 MoE、17 prefill＋17 decode、active expert、cancel復旧、cleanup zero |

旧A16形式はlowp公開host testとpublic runtime host testでfail-closedを確認した。
Gemma 4 31Bはreference-only・payloadなしであり、両GPU実行対象へ数えない。

### 初回CI／host確認（WU-4R前、2026-09-24）

- RMSNorm H3 compile-only: `gfx1030`／`gfx1201` ともにPASS。
- `validate_json_manifests.py`: PASS。
- lowp boundary validator: PASS。
- 対象pytest 19件: 全件PASS。
- 更新対象のsource SHA検証: PASS。
- `git diff --check`: PASS。

このRMSNorm H3 compile-only結果はWU-4Rのsource変更前の証拠であり、後述の新しいbuildへ読み替えない。

### WU-4Rの最終確認（2026-09-24）

- exact `gfx1030`／`gfx1201`でlowp全targetとHIP runtime releaseをbuildし、lowp host 2件・GPU correctness 3件を
  各targetでPASS。最初のlowp全buildは研究用`phase87_copy_bandwidth` probeの符号変換warningで止まったが、
  明示的な32-bit変換へ修正し、両targetの全buildをPASSさせた。
- 両targetのlowp archiveと実モデルbenchmark binaryから、旧kernel
  `sllm_matmul_nvfp4_block16_packed_dequant_v1`／`sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2`が消え、
  W4A4 kernel symbolが残ることをbinary scanで確認した。host fixture testでは退役variant 8／9／10の
  logical名とdevice symbolが`nullptr`になることを確認した。scanの原票は
  `.local-artifacts/phase87/wu4r/binary-symbol-audit.json`。
- Qwen3.8の通常8192/128、MTPなし、0 warmup＋1 measuredは両GPUでPASS。
  生成token SHA-256はV620 `c9c0b4ee401b11544fe0faaec15d882cb2491c31a460a32a89dce28e437bcc0e`、
  R9700 `75d36def8ff45d155373ebb05b885d0e4f0e7197ba9adb2123adfb1fe63b189d`で段階9と一致。
  HIP-only、fallbackなし、cleanup zero。raw reportは`.local-artifacts/phase87/wu4r/qwen38-mtp-off-gfx{1030,1201}.json`、
  binary SHA-256は順に`34568f130c2f1bb9bb58e63e07f367cfa3b54b62aa0e46ff135573a824419f0c`、
  `35d5cb3523a62f28ca6fb711d5d2cfffd47de3975af036291a705172d0ab7206`。
- `sllm-core`／`sllm-hip`のlib test、H3公開runtimeのsymbol契約pytest 36件／subtest 529件、
  `validate_json_manifests.py`、C++ formatをPASS。`hip-runtime-compile`のlowp source hashと
  `rmsnorm-h3-compile`のCI契約hashを更新した。local h0は628件選択・628件PASS
  （`.local-artifacts/phase87/wu4r/h0/h0/report.json`、dirty local development evidence）。

## rollbackと証拠

rollbackは本作業単位のGit差分を戻す。dirty worktreeのため、未作成commitをsource identityやimmutable release identityとして扱わない。
過去のW4A16／MX A16 evidenceは削除せず、retired ABI／provider／selector IDに対応する履歴として参照可能に保つ。

計画: [Phase 87 Qwen3.8 NVFP4 single-request plan](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階4-wa16の廃止と契約の整理)
