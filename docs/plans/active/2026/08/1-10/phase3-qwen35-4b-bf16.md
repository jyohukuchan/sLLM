# Phase 3 Qwen3.5-4B BF16 text生成計画

## 状態

- 作成日: 2026-08-04
- 状態: active
- 上位計画: [main plan](../../../../main-plan.md)
- Stage A子計画: [model lock・RMSNorm・G2（完了・archive）](../../../../archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)
- CI正本: [CI・テスト方針](ci-test-strategy.md)

## 現行方針（2026-08-10以降）

旧来のuniversal gate、closeout、checkpointごとのfresh review規則は2026-08-10付でsupersededとする。defaultは`trusted-solo-development`で、`external-contribution`と`release`は別のinactive・nonblocking laneとする。既存のevidence段落にあるidentity、test、review結果は履歴として保持するが、そのcloseout/fresh-review機構は継続しない。

| lane | 条件 |
| --- | --- |
| draft | 影響範囲のfocused test、dirty local worktreeを許可。immutable identityと独立reviewは要求しない。 |
| integration | 影響するH/G checkとreview 1回。B5以降の個別draftはfocused test後にbatch integrationする。 |
| release/push | clean worktreeのimmutable identity、final relevant matrix、累積review 1回。 |
| docs-only | Markdown、link、consistencyだけを確認し、closeoutは行わない。semantic/build identityが不変でmappingが明示される場合はGPU evidenceを再利用できる。 |

受入条件はunit開始時にfreezeする。実際のcorrectness/security defectはblockerとし、新しいprocess requirementはユーザー承認がない限りnonblocking follow-upとする。high-risk ABI/kernelのdesign reviewはoptional、integrationは1回、指摘へのfocused re-reviewは必要時のみ、releaseは累積1回とし、checkpointごとのfresh reviewは行わない。

source/build inputs、toolchain、model lock、artifact/reportはsemantic/build identityとしてGit commit/treeと分離する。GPU evidenceのfail-closed意味、exact target/artifact/model lock、numerical oracleは弱めない。同一unitを2回reject、reviewがimplementationを超過、function progressが1時間以上ない、verification/docsが30%超、経過時間がestimateの1.5倍超、またはgate/acceptanceが変更された場合は、新規review/verificationを止めてreplanする。

## 目的と完了境界

固定した`Qwen/Qwen3.5-4B` BF16 modelを、単一AMD GPU、batch 1、text-onlyでloadし、CLIからprefillとdecodeを実行してtextを生成する。canonical exact `gfx1030`と`gfx1201`の両方で、同一semantic/build identity（Git SHA/treeは別field）のhost、compile-only、GPU preflight、kernel/ABI、model slice、end-to-end evidenceをfail-closedに集約する。

Phase 3はStage AのRMSNorm G2/P0だけでは完了しない。次をすべて満たした時だけ完了とする。

- 完全model lockとverified read-only cacheから、全text weightをfail-closedに解決・loadできる。
- Qwen3.5 text stackの32 main layer、hybrid linear/full attention schedule、state/cache、tied embedding/lm headを実行できる。
- tokenizerとtext-only chat templateをRust側で適用し、CLIが固定promptから1 token以上を生成・decodeできる。
- exact `gfx1030`と`gfx1201`で同じmodel lockとcandidateを使い、CPU・他backend・generic kernelへのfallbackなしでG3がPASSする。
- H0〜H3、G0、private diagnostic G1、必要なsemantic G1/G2、G3、runtime/dispatchへ必要なP0、実行前後health、process cleanupを同一run graphへ集約できる。
- full model、raw slice、binary、traceをGitまたはGitHub Actions artifactへ保存していない。

## Superseded pre-2026-08-10 trusted-boundary policy（historical）

以下は2026-08-10以前の運用方針とその判断根拠を保持する履歴であり、現行laneのblocking条件ではない。

2026-08-08から今後数週間は、単独maintainerによるtrusted development期間とする。Phase 3のlocal/GPU実行は、maintainerが内容を確認したrepository codeと明示commandだけを対象とし、外部PR、fork由来code、未review script、第三者binaryは専用local/GPU hostで実行しない。

この期間は、悪意ある同一UID process、敵対的fork PR、永続runner上のhostile codeに耐えるrepository内custom capsuleの完成を、Stage A〜Eの開始条件・完了条件から外す。これは安全要件の撤回ではなくtrust boundaryの限定であり、buggy codeに対する標準的な隔離としてsecret・Docker socketを渡さないこと、可能な範囲のcontainer/network隔離、timeout・resource上限、process cleanup、実行前後GPU health、candidate SHA・artifact identityの検証は維持する。dirty worktree上の実行結果は`local-development`に限定し、immutable candidateのevidenceへ昇格しない。

中断されたA0 security hardeningの部分変更は検証済み実装として継承せず、byte-for-byteの過去版復元もPhase 3の作業に含めない。trusted development中のlocal確認はdirect testと標準containerを使い、当該custom capsuleをimmutable evidence経路へ使用しない。immutable candidateのhost evidenceを取得する段階では、現行部分変更を土台にせず、必要最小限のtrusted-development baselineを新しい作業単位として作成し、通常回帰とreviewを通した新identityを固定する。

外部contributorのcodeを実行する前、または複数の信頼境界を持つ運用へ移る前に、ephemeral VM/JIT runnerまたはjob後reimageをsecurity boundaryとする設計をhard gateとして再開し、実行前にCI正本と本計画を更新する。

## 作業見積りと再計画

各unitは開始時にscope、estimate、lane、freezeした受入条件を記録する。旧来のhard-stop clockおよび`estimate + 1 hour`自動停止機構は廃止し、上記triggerで新規review/verificationを停止してreplanする。重いcommandの生存監視は継続するが、正常進行中のprocessを一律の時計で停止しない。

Stage Aの工程別実績は[archive済みStage A計画](../../../../archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)を正とする。estimateの変更、gateまたはacceptanceの変更、二度のrejectは新しいblocking条件を自動生成せず、replan対象として記録する。

## 対象

含むもの:

- `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、BF16 weight/activation。
- Linux、単一AMD GPU、単一request、batch 1、text-only。
- config、safetensors、tokenizer、chat template、model I/O、GPU weight upload。
- embedding、RMSNorm、BF16 matmul/GEMV/GEMM baseline、RoPE、full attention、linear attention、MLP、residual、logits。
- full-attention KV cache、linear-attention recurrent/convolution state、position、prefill/decode更新。
- greedyを最初の必須sampling pathとするCLI text生成。seeded samplingはgreedy完了後に独立追加できる。
- op別のhost oracle、semantic G1、real-weight G2、full-model G3。

含まないもの:

- vision encoder、image/video input、MTP/speculative decode、MoE。
- Qwen3.5-2B/9B、複数request、request batching、chunked prefill、複数GPU。
- OpenAI-compatible server/API。Phase 3完了後の開発順序で実装する。
- optimized kernel、autotuning、性能競争、P1、llama.cpp性能比較。
- quantization、FP8、NVFP4、KV cache量子化。

## 固定identityとmodel contract

- model revision、全入力file、license/base model、component分類はStage Aのmodel lockを正とする。
- model lock fingerprintはmodel bytes、source/build inputs・toolchain・artifact/reportはsemantic/build identity、Git commit/treeはsource snapshot、tuple digestは実行環境のidentityであり、相互に代用しない。docs-only差分でsemantic/build identityが不変かつmappingが明示される場合は、対応するGPU evidenceを再利用できる。
- top-level multimodal configをlockするが、実行対象は`text_config`とmain language modelだけとする。vision 297 tensorとMTP 15 tensorはknown-unconsumedとして列挙し、未知tensorや対応済みとして扱わない。
- text contractはhidden size 2560、32 main layers、intermediate size 9216、vocab 248320、BF16、RMSNorm epsilon `1e-6`、weight tying、3 linear-attention layerごとに1 full-attention layerの明示scheduleを基準とする。lock validatorが実config/indexと一致を証明するまで実装定数として黙ってfallbackしない。
- full-attentionの`q_proj`出力はheadごとの`[Q 256, gate 256]` interleaveであり、flatなQ半分/gate半分に分割しない。text-only MRoPEはrotary dim 64、sections `[11, 11, 10]`、3軸同一absolute position、NeoX pairとし、残り192 dimensionへRoPEを適用しない。
- linear-attention stateはconvolution BF16 `[3, 8192]` row-major、recurrent F32 `[32, 128, 128]` row-major、full-attention KVはFP16 `[4, T, 256]`として型・layout・lifetimeを分離する。convolutionはoldest-to-currentの4-tap、recurrent updateはrequest-local zero stateからprefill/decodeで同じ順序を使う。
- text-only CLIの停止集合は`[248046, 248044]`とし、`<|im_end|>`を`<|endoftext|>`より先に判定する。prompt tokenは停止判定せず、生成されたstop token自身をvisible outputへ含めないが、reportへtoken IDと停止理由を保持する。

## 実装stage

### Stage A: model-bound最小数値経路

[archive済みStage A子計画](../../../../archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)に従い、完全model lock、host validator、RMSNorm semantic contract、独立NumPy oracle、public HIP runtime、baseline kernel、semantic G1、real-weight G2、短いP0を完成させた。

Stage Aはpublic runtimeとmodel provenanceを固定するrollback境界であり、Phase 3全体の完了ではない。

Stage Aはcommit `ac2baa3a0734d0894353ba180259d979da5a831e`、tree `4e43a9c42c9aa2dfa6a6d438610fa54c4e482d10`に対するH0〜H3、canonical 2 GPUのG0/private G1/semantic G1/G2/P0、前後health/process、独立review 9の`PASS`により2026-08-09に完了した。次のGPU evidence refresh前に、workflow/controllerからlocal commandを導出するtracked orchestrationまたはdry-run preflightを2〜4時間で整備してからStage Bへ進む。

履歴上、この運用負債解消を2026-08-09 23:51:09 JSTに開始し、当初のhard中断時刻を2026-08-10 03:38:27 JSTとした。同作業完了後のStage B plan reviewを継続するため、ユーザー指示により2026-08-10 03:38:27 JSTから6時間延長し、その後B1を継続するため同日09:38:27 JSTからさらに6時間延長した。当時の最終全体hard中断時刻は同日15:38:27 JSTだった。受入境界は、defaultでGPU、model cache、container、buildへ触れないhost-only dry-run、既存正本から導出したcanonical 2 targetの実行plan、run identity・短いsocket root・target別build/output ownership・canonical JSONのfail-closed検証、focused host回帰、独立reviewの`PASS`だった。これらのclockとreview条件は現行laneへ持ち越さない。

同作業は2026-08-10 01:26 JSTに受入境界を満たして完了した。tracked plannerは既存workflow、matrix、G1/G2/P0総合validator、builderのpure layout helperからexact `gfx1030`→`gfx1201`の実行planを導出し、clean immutable Git identity、authority file hash、symlink component、短い未作成run root、target別path/output、schema順序をfail-closedに検証する。GPU、model cache、container、build、networkは実行せず、focused 11件、fail-closed 46件、matrix/JSON/G1/G2/P0 validator、dirty-local H0 316/316、独立reviewのHigh/Medium 0件・`PASS`を得た。既存P0 builderのsame-UID/trusted-solo output symlink安全負債は当面のtrusted development境界に従い延期したままとし、次のGPU evidence refreshではこのplannerを先行させる。

### Stage B: Rust model I/Oとtext frontend

1. model lock、config、safetensors index/headerを型付きでparseし、unknown version/architecture、hash、shape、dtype、tensor集合の不一致を拒否する。
2. verified read-only cacheをmmapまたはbounded readし、必要byte rangeだけをopaque GPU bufferへuploadする。shard全体の複製を作らない。
3. main text tensorをrequired、config-conditional、known-unconsumed、rejectedへ分類し、全required tensorの一意なconsumerを検証する。
4. tokenizer vocabulary/merges/special tokenとtext-only chat templateをload・validateする。image/video content、tooling等の未対応template branchは黙って処理せず明示unsupportedとする。
5. CLIにoffline model verificationとtokenize/render/decodeの独立入口を設け、model execution前にfrontendだけをhost testできるようにする。

B0 dependency closureで`tokenizers =0.21.4`をdefault featureなし・`onig`だけに固定し、HTTP、progressbar、`esaxx_fast`を無効にした。B3 tokenizer frontendはこの依存とB2 verified assetだけを使い、B4 typed chat rendererは任意Jinjaを実行せず、locked `chat_template.jinja`のhashと対応するQwen3.5 text-only rendererを実装する。停止policyは既存のversioned lock/schema/APIを再利用し、依存version/checksumはroot `Cargo.lock`、license/feature/MSRVはB0のtracked dependency policyとoffline validatorを正とする。

#### Stage B開始時baselineと重複禁止境界

Stage B開始時点の`model.rs`は空のstubではない。Stage Aですでに、型付き`ModelLock`/`TextConfig`、固定Qwen identity、738 tensor名・dtype・component分類、streaming hash、index/headerのgap・overlap・範囲・dtype/size検証、verified file descriptorを保持した最大16 MiBのpositional tensor range read、cache/path/inode安定性検査を実装済みである。`sllm-frontend`にもversioned generated-token停止policyがあり、`tokenizers =0.21.4`はdefault featureなし・`onig`だけでroot lockへ解決済みである。

従って後続作業は既存`verify_model_cache()`、`VerifiedCache`、`TensorDescriptor`、停止controllerを置き換えない。特に、別のsafetensors parser、別のcache hasher、共有seek cursor、shard全体を返すAPI、任意Jinja executor、同じ停止policyの別実装を追加しない。Stage Bで閉じる残差は、全738 tensorのexpected shape、frontend assetのbounded verified read、dependency/license/MSRV evidence、tokenizer/typed renderer、main-text tensorの一意consumer/load plan、CLI、opaque GPU bufferへのexact-range upload接続である。

#### Stage B独立作業単位

各unitはlane、scope、estimate、freezeした受入条件を開始時に記録する。draftはdirty local worktreeでfocused affected testだけを行い、integrationは影響するH/G checkとreview 1回、release/pushはclean immutable identityとfinal relevant matrixを行う。semantic/build identity（source/build inputs、toolchain、model lock、artifact）はGit commit/treeと分離し、docs-onlyで不変ならmapping付きGPU evidenceを再利用できる。整理でsemantic/build identityが変われば該当evidenceを取り直す。依存順は`B0 -> B1 -> B2 -> B3 -> B4 -> B5 -> B6 -> B7a -> B7b`とするが、B5以降の個別draftはfocused test後にbatch integrationする。B0〜B6はhost-onlyで、full model download、weight payloadの一括materialize/decode/mmap/upload、GPU、containerを禁止する。固定cacheを使う適用確認は、全locked fileをbounded bufferでstreaming hashし、metadata/rangeを照合するだけのlocal model-bound evidenceとして行い、CPU CIへ持ち込まない。B7a/B7bだけがHIP/runtime/backendへ影響し、canonical `0000:03:00.0`のV620とcanonical R9700を使う。spare V620は使用しない。B7のfull H/G matrixはintegration/releaseでのみ実行し、draftへ要求しない。

| ID | 所有範囲と成果物 | 予測 / lane | 受入条件・evidence・rollback境界 |
| --- | --- | --- | --- |
| B0 dependency closure | `Cargo.toml`、`Cargo.lock`、各workspace memberの`Cargo.toml`、新規`ci/dependencies/rust-workspace-v1.json`、`ci/schema/rust-dependency-policy-v1.schema.json`、`ci/tools/validate_rust_dependencies.py`、`ci/tests/test_rust_dependencies.py`と必要な`ci/matrix/{suites-v1,host-v1,path-to-suite-v1}.json`登録。通常・build・devを含む全workspace targetの解決graphをinventoryし、全registry dependencyのexact version/checksum/license、workspace memberごとの有効feature、禁止feature、Rust 1.85 offline解決を機械検証する。`tokenizers 0.21.4`はdefault featureなし・許可feature `onig`だけとし、`sllm-hip`の`static_assertions`等のfrontend閉包外も除外しない。 | 予測2〜4時間 | local crate cacheだけを使い、policyと`cargo metadata --locked --offline --format-version 1`の全package/edge/target集合が一致し、focused draft test後にintegration review 1回、release/pushでfinal relevant matrix。model/GPU/cache/networkなし。validator・manifest・Cargo/CI登録差分だけでrollbackできる。これを最初の実装単位とする。 |
| B1 tensor shape closure | `crates/sllm-core/src/model.rs`、`crates/sllm-core/tests/model_contract.rs`、`ci/tools/validate_model_lock.py`、`ci/tests/test_model_lock_contracts.py`、`docs/references/qwen3.5-phase3-full-model-reader.md`。既存738-name/dtype catalogをprivate typed expected shapeまで拡張し、lock済みconfigから導出する全main/vision/MTP shapeとheader shapeの一致をRust/Python mirror validatorの両方で検証する。`lib.rs`のpublic API、lock schema、tiny fixture、suite登録は変更しない。 | 予測5〜8時間 | 2の冪だけでなく1、3、17、境界前後を含むsynthetic config/header mutationでmissing/duplicate/wrong rank/dimension/dtype/overflowをfail-closed。draftはfocused affected test、integrationは影響H/G checkとreview 1回、release/pushはfinal relevant matrix。同一semantic/build identityの固定cache metadataを照合し、既存parser/hasher/range APIを保持する。 |
| B2 verified frontend assets | `crates/sllm-core/src/model.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-core/tests/model_contract.rs`。`VerifiedCache`の保持済みFDから、固定kindの`config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`だけをpositional readするAPIを追加する。asset全体sizeへ種類別hard capを適用し、順に1 MiB、16 MiB、256 KiB、64 KiBとする。B3はself-containedな`tokenizer.json`を消費するため、`merges.txt`と`vocab.json`はB2 APIへ公開しない。 | 予測2〜4時間 | raw pathを受け取らず、shard、任意path、symlink/hardlink、cap超過、差替え、同一inode改変を拒否し、weight shard全体を返さない。draftはfocused affected test、integrationは影響H/G checkとreview 1回、release/pushはfinal relevant matrix。GPU/model payloadなし。新APIとtestだけでrollbackできる。 |
| B3 tokenizer frontend | `crates/sllm-frontend/src/tokenizer.rs`、`crates/sllm-frontend/src/lib.rs`、新規`crates/sllm-frontend/tests/tokenizer_contract.rs`、`ci/fixtures/tokenizer-v1/**`、同fixtureをH1へ結合する`ci/matrix/path-to-suite-v1.json`の最小登録とrevision更新、対応する`ci/tools/validate_matrix.py`の期待revision同期、追加integration-test targetだけを反映する`ci/dependencies/rust-workspace-v1.json`のtarget inventory同期。B2 assetからだけ`tokenizers`を構築し、encode/decode、special-token identity、EOS集合との整合をtyped APIにする。 | 予測3〜5時間 | ASCII、Unicode、空、非整列長、未知/欠落/重複special token、malformed tokenizerをhost negative testで検証し、固定prompt token IDs/decodeがversioned fixtureと一致。draftはfocused affected test、integrationは影響H/G checkとreview 1回、release/pushはfinal relevant matrix。GPU/full modelなし。package、edge、lock、version、feature、licenseは変更せず、frontend module/fixture、H1 path登録、manifest/validator identity同期、target-only inventory同期だけでrollbackできる。 |
| B4 typed chat renderer | `crates/sllm-frontend/src/chat.rs`、`crates/sllm-frontend/src/lib.rs`、`crates/sllm-frontend/Cargo.toml`、root `Cargo.lock`のlocal frontend依存一覧、新規`crates/sllm-frontend/tests/chat_contract.rs`、`ci/fixtures/chat-template-v1/**`、`docs/references/qwen3.5-phase3-full-model-reader.md`、fixture専用H1 ownershipとrevisionを同期する`ci/matrix/path-to-suite-v1.json`・`ci/tools/validate_matrix.py`、新規integration-test targetと既存workspace固定`sha2`へのfrontend edgeを同期する`ci/dependencies/rust-workspace-v1.json`・`ci/schema/rust-dependency-policy-v1.schema.json`・`ci/tools/validate_rust_dependencies.py`。locked template identityを要求し、Qwen3.5 text-only messageをtyped rendererで生成する。 | 予測3〜5時間 | fixed `hello`とUnicodeのrendered text/token IDsを固定し、image/video/tool/unknown role、不正content、template hash不一致を明示unsupportedにする。任意Jinjaを実行しない。既存`sha2`のversion/featureを再利用し、package集合は不変、frontend edgeは1件増、Cargo.lockはlocal `sllm-frontend`依存一覧だけを同期する。draftはfocused affected test、integrationは影響H/G checkとreview 1回、release/pushはfinal relevant matrix。reader記録とH1 path/validator、target/edge inventoryを同期し、実装前に固定templateのwhitespace、thinking branch、escapingのseparated reader記録を完了する。 |
| B5 weight registry/load plan | 新規`crates/sllm-core/src/weights.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-core/tests/weight_contract.rs`。B1のdescriptorをrequired/config-conditional/known-unconsumed/rejectedへ分類し、layer/roleを含む一意consumerとexact source rangeを持つimmutable host load planを作る。各tensor rangeは既存16 MiB read上限以下の決定的chunkへ分割し、destination offsetもoverflowなく固定する。 | 予測3〜5時間 | 全required main-text tensorがconsumer 1件、tied lm-head条件が明示、vision/MTPはknown-unconsumed、unknown/missing/duplicate consumerは拒否。16 MiB境界前後をpayload allocationなしで検証し、chunk順序とplan digestを決定的にする。draftはfocused affected test、integrationはB5を含むbatchの影響H/G checkとreview 1回、release/pushはfinal relevant matrix。GPU ABIは決めずRust内部descriptorに限定する。 |
| B6 offline CLI | `Cargo.toml`、`Cargo.lock`、`crates/sllm-cli/Cargo.toml`、`crates/sllm-cli/src/main.rs`、新規`crates/sllm-cli/src/model.rs`、`crates/sllm-cli/tests/model_frontend_cli.rs`、`ci/schema/model-frontend-cli-report-v1.schema.json`、`ci/tests/test_model_frontend_cli_contract.py`、追加target/edge/countに必要な`ci/dependencies/rust-workspace-v1.json`、`ci/schema/rust-dependency-policy-v1.schema.json`、`ci/tools/validate_rust_dependencies.py`。`verify-model`、`tokenize`、`render`、`decode`をmodel executionから独立させる。 | 予測3〜5時間 | 明示lock/cache入力、offline、fail-closed exit、versioned machine-readable出力、stdout/stderr分離を固定し、tiny fixtureで全入口とnegative caseを検証する。draftはfocused affected test、integrationはB5以降batchの影響H/G checkとreview 1回、release/pushはfinal relevant matrix。doctor以外はHIP probeを起動しない。CLI/schema/dependency inventory差分だけでrollbackできる。 |
| B7a backend-neutral bounded buffer readback | `crates/sllm-core/src/execution.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-hip/src/bridge.rs`、新規`crates/sllm-hip/src/bin/sllm-execution-transfer-g1-evidence.rs`、対応するschema/matrix/validator/testと既存suite/path登録。backend adapterが広告する非zero `max_transfer_bytes`以下のowned `BufferRange`だけをqueueから非同期D2Hするbackend-neutral API、単一observer completion、terminal success後だけのcapacity一致`read_into`、session/queue/buffer identityとlifetimeを追加する。HIP adapterは既存`SLLM_HIP_MAX_TRANSFER_BYTES`（1 GiB）を広告し、既存`Queue::copy_to_host()`/versioned transfer ABIだけへlowerする。1 GiB超のbufferも単一readbackへ黙って受理せず、consumerが明示chunkへ分割する。semantic opの`Submission`/output専用`Readback`とは型を分ける。 | 予測4〜6時間 | 1、3、17、255、256、257 byteとoffset/末端境界で、既存`ExecutionSession::upload()`から新readback APIまでのexact round-trip、wrong session/queue/range、zero/overflow、早期read、短い/長いdestination、drop/shutdownをfail-closedに検証する。広告上限`B`の`B-1/B/B+1`はpayload/GPU allocationなしのfake adapter host contractで検証し、`B+1`をsubmitしない。draftはfocused affected test。integration/releaseでのみ、該当H0〜H3/G0〜G3/P0のfull relevant matrix、aggregate、pre/post health、cleanup、review 1回を実行する。新しいnative/C ABIまたは直接`Queue`利用を作らず、失敗時はB6完了commitを維持する。不足が判明した場合はscopeを拡張せず別のABI決定単位を計画する。 |
| B7b exact-range weight upload bridge | `crates/sllm-core/src/weights.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-hip/src/bin/sllm-weight-upload-g1-evidence.rs`、`ci/schema/weight-upload-g1-report-v1.schema.json`、`ci/matrix/weight-upload-semantic-g1-v1.json`、`ci/tools/validate_weight_upload_g1_contracts.py`、`ci/tests/test_weight_upload_g1_contracts.py`と既存suite/path登録。B5 load planのverified chunkをB7aでreview済みの`ExecutionSession::upload()`/buffer readbackへ順に接続し、既存HIP `ExecutionSessionAdapter`/versioned transfer ABIを通してopaque GPU bufferへuploadする。新しいHIP weight-upload wrapperや直接`Queue`経路を作らず、shardまたはtensor全体のhost複製も作らない。 | 予測4〜6時間 | 複数chunk、16 MiB境界前後、tensor/destination境界前後でexact byte/readback、wrong range/target/dtype/plan identityをreject。draftはfocused affected test。integration/releaseでのみ、該当H0〜H3/G0〜G3/P0のfull relevant matrix、B5 load-plan接続専用semantic upload G1、aggregate、pre/post health、cleanup、review 1回を実行する。generic transferの重複実装・重複証明にはしない。canonical sLLM V620とR9700だけを使用し、失敗時はB7a完了commitを維持する。 |

B0開始前にllama.cpp/vLLMの追加readerは不要である。ただしB0内では、local Cargo metadata/sourceだけを調べるread-only dependency auditorとvalidator implementerを分離する。B1開始後の監査でvision mergerとMTP orientationが追跡済みreader記録に不足し、Rustだけの変更ではPython validatorと契約が分裂することを検出したため、固定vLLMをreader、固定SGLangをcross-checkとしてshape式を追記し、B1所有範囲を修正した。B4も現行reader記録に正確なrender済みbytes、whitespace/escaping、thinking branchの全境界がないため、固定revisionのtemplate/tokenizer metadataを読むreaderとimplementerを分離する。B5はRust内部load-plan、B7a/B7bは既存versioned transfer ABIの利用に限定するため、現時点のpublic C ABI未確定事項にはblockedされない。B7aで既存ABI不足が判明した場合だけ中断し、runtime正本と互換性文書を同期する別単位を先に計画する。

#### 既存evidenceの履歴注記

以下のB0以降のevidence段落は、commit/tree、artifact digest、test結果、reject理由を含む履歴である。そこに記載されたcloseout、作業単位hard-stop clock、estimate+1h、checkpointごとのfresh reviewは2026-08-10付でsupersededであり、現行laneでは継続しない。

2026-08-10 03:40 JSTに、B0〜B7b分割の最終plan candidate commit `9d3f7d5feb27294644252c60f24984fc579e3bfe`、tree `f00f0e689256c94f32226cd0d86c68f69f7b5404`へstrict H0 316/316とfresh累積独立review High/Medium 0件の`PASS`を結合した。これをrollback baseとしてB0を同日03:40:18 JSTに開始し、予測2〜4時間、個別hard中断時刻08:40:18 JSTを固定した。read-only dependency auditorはworkspace 5、registry 85、全package 90、normal/build/devを含むedge 170を確認し、`tokenizers`のresolved featureが`onig`だけである一方、`esaxx-rs`自体はfeatureなしの通常依存として残ること、`wasip2`のRust 1.87宣言がwasm32-wasip2限定edgeでLinux x86_64のRust 1.85 authorityから外れることを記録した。

B0の最終機能candidateはcommit `a5519d89820f42a8349cf3485ee8dc37154d8507`、tree `4f6896eee85399ddc10831b752355d332960a0dd`で完了した。dependency closure本体はcommit `54fdcb76a5671a075ec0ff6e346cb78f5d3cf8a0`で固定したnormalized policy/schema、offline metadata/lock validator、Rust 1.85のLinux x86_64 targetを明示するworkspace/all-target check、renamed dependency、MSRV例外、hostile target/HIP/native/Rust compiler・flag overrideの回帰、H0 suite/path登録を維持する。Cargo実行環境はoffline・rustup自動取得禁止を強制し、B0 host-only契約外のHIP/native/Rust compiler injectionを除去する。旧candidateのH1 assertion失敗をretryで昇格せず、既存testのprocess-wide FD総数比較が並列harnessで7/200失敗することを再現し、固有fixtureのdevice/inodeだけをfail-closedに検査するparallel-safe testへ修正した。fresh独立reviewの全指摘を閉じた同一identityで、pinned Python 3.12.10 strict H0 335/335、H1 151/151、H2 35/35を各1回目にimmutable `PASS`した。model、GPU、model cache、container、networkは使用していない。本完了記録を含むdocs-only identityにもH0〜H2とfresh reviewを結合してから次のB1を開始する。

B1 tensor shape closureは、B0のdocs-inclusive完了candidate `d610b4801052f11125a9002e0b59d0d0973a86d7`、tree `04d7214f86c7069ab73bc098459972f59fb3115b`をrollback baseとして2026-08-10 06:55:54 JSTに開始した。予測5〜8時間、作業単位固有hard上限9時間は維持する。当初のeffective hard stopは全体停止上限09:38:27 JSTだったが、ユーザー指示で6時間延長し、現在は作業単位上限より先に到達する2026-08-10 15:38:27 JSTをeffective hard stopとする。固定reader記録、現行738 tensor catalog、schema/fixture/testの監査から開始し、未完了で停止する場合もB1完了やshape検証済みとは扱わない。実装・host testはmodel cacheなしで行い、同一SHAの適用確認だけは固定cache全fileのbounded streaming hashとindex/header metadata照合を行う。GPU、tensor payloadのmaterialize/range read、network、containerは使用しない。

開始後の独立監査で、main text 426件のshape式は確定した一方、vision 297件とMTP 15件のorientationが追跡済みreader記録に不足していた。固定vLLMと固定SGLangをreader/independent cross-checkに分離して全family式を確定し、visionの数値を外部source defaultへ固定せずlock済み`config.json`から厳密抽出する。Rust validatorだけでなく既存Python mirror validatorも同時に更新し、schema、fixture、public API、suite登録は変更しない。開始時readerがGDNの`linear_attn.dt_bias`と`linear_attn.norm.weight`のstorage dtypeを逆と判断した点は誤りであり、固定headerどおりBF16/F32を維持する。

B1のhost-verified functional checkpointはcommit `be098f41c903c19b3f3e62883b0af8c8201e990b`、tree `0831c0bbf9fb98edcb0a6a30991b2c2476d54e48`である。既存config parserからprivate shape inputsを1回だけ構築し、Rust/Pythonの738-name catalogをshapeまで一致させ、header rank/dimensionをdtype/byte rangeと独立に照合する。focused Rustは25 unit + 8 integration、Pythonは22件をPASSし、pinned Python 3.12.10のstrict H0 335/335、H1 154/154（collected 186、deselected 32）、H2 35/35（collected 42、deselected 7）を各attempt 1のimmutable evidenceとして得た。fresh独立reviewはHigh/Medium/Low 0件の`PASS/no findings`である。ただしdocs-inclusive checkpoint `a65b2ab3129a8a392df980e8751431f7783e331f`への固定cache照合は、readerに基づき変更したGDN 2 dtypeが実headerと逆であることをfail-closedに検出した。全738 headerの一括差分ではさらに24層の`conv1d.weight`がreaderの2次元式`[8192,4]`でなくstorage上はsingleton次元を含む`[8192,1,4]`であることを確認した。このcheckpointはB1合格candidateへ昇格せず、Rust/Python catalogを固定header契約へ戻して全evidenceを取り直す。B1全体は未完了のまま次工程を開始しない。

B1最終機能candidateはcommit `b5cc617287ec2efb97c5b06bd838621f51d547c8`、tree `e901d2fa1b33ae75a7d087c1d4323d38f9f02a00`である。固定cache 13 file・9,342,905,899 bytesのcontent-only hash、index、全738 header metadataはfingerprint `sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935`で`PASS`した。pinned Python 3.12.10 strict H0 335/335、H1 156/188（32 deselected）、H2 35/42（7 deselected）は各attempt 1のimmutable `PASS`で、precommit reviewとrollback base `d610b480...`からの累積reviewはいずれもHigh/Medium/Low 0件だった。GDN storage dtypeと`conv1d.weight=[8192,1,4]`、Rust/Python exact-catalog mutation、件数付きbounded diagnostic、production descriptor map非複製を確認し、GPU、payload materialize/range read、network、containerは使っていない。この完了記録を含むdocs-inclusive identityへ同じ固定cache/H0〜H2/fresh reviewを結合してからB1を完了扱いとし、B2を開始する。

完了記録を追加したdocs-only candidate `01dbedfa9de5e435703ef26b66fb610f194cfdd2`のstrict H0 attempt 1は、335 selected中334 PASS、semantic G1 broker client-death test 1件FAILであり、retryで昇格させない。test/helperは`b5cc6172`からbyte-identicalで、単独100/100と同じ95-test command 3/3はPASSした一方、compiler PID clear後のfailure publicationを500 ms遅延させるin-memory probeで同じ失敗を決定的に再現した。productionのfail-closed動作は変えず、testの待機条件からPIDを外して既存5秒deadlineまでfailure publicationを待ち、failure non-Noneとcompiler PID Noneの両assertionを維持する。修正後のfocused 20/20、semantic G1 95/95はPASSしたが、修復を含む新identityへ固定cache/H0〜H2/fresh reviewを再結合するまでB1は未完了とする。

B1の受入済みimmutable implementation candidateはcommit `6543098f70d8c06b5a6758becd4590ab44fb9811`、tree `b4f46f5a42c09df4e2d64aa5c1f8191620d60ce8`である。固定cache再検証はcandidate SHA/tree、前後clean、validator/lock/schema/output digest、attempt 1をsidecar付きreportへ結合し、13 file・9,342,905,899 bytes、fingerprint `sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935`、全738 header metadataを`PASS`した。同一identityのpinned Python 3.12.10 strict H0 335/335、H1 156/188（32 deselected）、H2 35/42（7 deselected）は各attempt 1、required case全PASS、skipped 0、report/sidecar一致だった。正しいrollback base `d610b4801052f11125a9002e0b59d0d0973a86d7`からのfresh累積独立reviewはHigh/Medium/Low 0件の`PASS/no findings`で、全738 tensorのRust/Python exact catalog、negative mutation、checked overflow、bounded diagnostic、descriptor map非複製、header-only semantic inspection、provenanceを確認した。broker client-death修正はtest 1行だけでproduction brokerは不変、`01dbedfa...`の334/335 FAILは非受入のまま再利用していない。B1 implementationは完了し、本結果を記録するdocs-only closeoutを検証してからB2を開始する。

B1 docs-only closeout commit `8d6018057006f8c06e8c3bac5343cc3681fcb1a2`、tree `7eecd11417e530c68f62bc83ea2ff90867bf7733`はpinned Python 3.12.10 strict H0 335/335、attempt 1、skipped 0、report/sidecar一致とfresh独立review High/Medium/Low 0件を`PASS`し、B1全体を完了した。これをrollback baseとして2026-08-10 09:58:06 JSTにB2 verified frontend assetsを開始する。予測2〜4時間、個別hard中断時刻14:58:06 JSTとし、全体停止上限15:38:27 JSTより早い個別上限を適用する。所有範囲は`crates/sllm-core/src/model.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-core/tests/model_contract.rs`に限定し、`VerifiedCache`の保持済みFDから固定frontend assetだけを種類別hard cap付きpositional readで返す。公開kindは`config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`の4種、asset全体sizeの上限は順に1 MiB、16 MiB、256 KiB、64 KiBとする。B3はself-containedな`tokenizer.json`を消費するため`merges.txt`と`vocab.json`は公開せず、raw path APIも作らない。shard、任意path、symlink/hardlink、cap超過、差替え、同一inode同一size改変をfail-closedに拒否し、tiny fixtureのH0〜H2とfresh reviewを受入条件とする。GPU、weight payload、model cache、container、networkは使用しない。

B2の受入済みimmutable implementation candidateはcommit `b2a9275cd00bae55218f5b60840e471e8bb877ff`、tree `7c8ba9fec21a720134436e0a3574db2620ba52f6`である。4種の`FrontendAssetKind`から固定filename/capだけをprivate mappingし、`VerifiedCache`がverify時から保持するFDへoffset 0のpositional readを行う。asset全体sizeはallocation/read前に1 MiB、16 MiB、256 KiB、64 KiBの各capで拒否し、成功経路はread前後にcache rootと全locked path bindingを再検証する。tiny fixtureはexact bytes、全4種mapping、cap-1/cap/cap+1、root/path/symlink/hardlink差替え、同一inode同一size改変、truncate/extend、並行whole-file read、FD multiplicity/drop cleanup、lock fingerprint再計算を検証した。同一identityのpinned Python 3.12.10 strict H0 335/335、H1 163/195（32 deselected）、H2 35/42（7 deselected）は各attempt 1、failed/skipped 0、clean worktreeで、report SHA-256 `f2c324664d7d7fc2bb289669264bfddf3c30a21c0d1c1d018c9fdd5188c32162`、`1845ad1fe381751e5ebe5e6b50677fd0c0e0395b42dd43ad3a71013c0a5f6d8f`、`cad1a7b596819f873673c1eb00062387e65d8297a1a208648e6c574080350096`はsidecarと一致した。rollback base `8d6018057006f8c06e8c3bac5343cc3681fcb1a2`からのfresh累積独立reviewはHigh/Medium/Low 0件の`PASS/no findings`だった。GPU、weight payload、model cache、container、networkは使用していない。B2 implementationを完了し、本結果を記録するdocs-only closeoutへstrict H0とfresh reviewを結合してからB2全体を完了し、B3を開始する。

B2 docs-only closeout commit `c437aab32f7fa7cd0681dd8b7db3807ac55c5984`、tree `af07a678a09ea97df7d74e03811d2765d0a5632c`はpinned Python 3.12.10 strict H0 335/335、attempt 1、failed/skipped 0、clean exact SHA/tree、report SHA-256 `aa03242e651f9e344c6608c0205e53651ee716a3150a423915763fff5079a5d7`のsidecar一致とfresh独立review High/Medium/Low 0件を`PASS`し、B2全体を完了した。これをrollback baseとして2026-08-10 10:51:30 JSTにB3 tokenizer frontendを開始する。予測3〜5時間、作業単位hard中断時刻16:51:30 JSTだが、先に到達する全体停止上限15:38:27 JSTをeffective hard stopとする。所有範囲は`crates/sllm-frontend/src/tokenizer.rs`、`crates/sllm-frontend/src/lib.rs`、`crates/sllm-frontend/tests/tokenizer_contract.rs`、`ci/fixtures/tokenizer-v1/**`、同fixtureをH1へ結合する`ci/matrix/path-to-suite-v1.json`の最小登録・revision更新、`ci/tools/validate_matrix.py`の期待revision同期、追加integration-test targetだけを反映する`ci/dependencies/rust-workspace-v1.json`のtarget inventory同期に限定する。B2のtyped verified assetだけから`tokenizers 0.21.4`を構築し、encode/decode、special-token identity、EOS集合整合、ASCII/Unicode/空/非整列長、unknown/missing/duplicate special token、malformed tokenizer、固定token IDs/decodeをhost-onlyで検証する。GPU、full model、model cache、container、networkは使用しない。

初回B3 candidate `6073d2257f3811da43aa8e380a90427630c2742a`はstrict H1 182/214とH2 35/42をPASSしたが、strict H0 335/335のうちdependency inventory validator 1件だけが`workspace_members` driftでFAILした。report SHA-256 `4841e9265a3a680eb76fd6a7aaf2dd1e26bbdea86e12d8978ac5d13441d91e4b`はsidecarと一致する。このcandidateは受け入れず、H1/H2も再利用しない。原因は新規`tokenizer_contract` integration-test targetだけがB0のall-target inventoryへ未反映だったことで、package 90、edge 170、Cargo manifest/lock、version、checksum、license、feature、MSRVは不変である。inventoryへ当該target 1件だけを同期し、新candidateでH0〜H2とfresh reviewを取り直す。

B3 implementation candidate `766bfec524b8410317e41cafa69b67f1179f3a95`、tree `3b0084c073c2fa1cab3a6a46e2ce5b0bcd866d1c`をclean worktreeで固定した。同一identityのpinned Python 3.12.10 strict H0 335/335、H1 182/214（32 deselected）、H2 35/42（7 deselected）は各attempt 1、failed/skipped 0で`PASS`し、reviewed/tested/workflow SHAとtreeがcandidateへ一致した。report SHA-256は順に`0ec565591f76963ffe756fc756016b8b74659de1d5831286b3d05e142c940db8`、`723410687e38b867724c6c90852dbb2e799d45785c5e799b2028de314b9c07c0`、`5853b9f26614bf4f80fa622486784e49e8813d6c631e84946b9f732c20ee90b4`で、各sidecarと一致する。rollback base `c437aab32f7fa7cd0681dd8b7db3807ac55c5984`からのfresh累積独立reviewもHigh/Medium/Low 0件の`PASS/no findings`だった。typed encode/decode、immutable tokenizer/special/EOS snapshot、exact fixture、negative/boundary test、H1 fixture ownershipとtarget inventory同期を受け入れる。GPU、full model、model cache、network、containerは使用していない。本記録のdocs-only identityへstrict H0とfresh reviewを結合するまでB3全体は未完了とする。

B3 docs-only closeout commit `7904a2c196628adcc138eb6499a6a04bd5ebdb56`、tree `8217e4b698a390c31c10b6ed4460f63fa8988051`はpinned Python 3.12.10 strict H0 335/335、attempt 1、failed/skipped 0、clean exact SHA/tree、report SHA-256 `7f34247470172bb964992c28c7215fc1e17d91270e5279bd6ca40dcd5373defd`のsidecar一致とfresh独立review High/Medium/Low 0件を`PASS`し、B3全体を完了した。これをrollback baseとして2026-08-10 12:48:55 JSTにB4 typed chat rendererを開始する。予測3〜5時間、作業単位hard中断時刻18:48:55 JSTだが、先に到達する全体停止上限15:38:27 JSTをeffective hard stopとする。所有範囲は`crates/sllm-frontend/src/chat.rs`、`crates/sllm-frontend/src/lib.rs`、`crates/sllm-frontend/tests/chat_contract.rs`、`crates/sllm-frontend/Cargo.toml`、`Cargo.lock`のlocal frontend依存一覧、`ci/fixtures/chat-template-v1/**`、固定Qwen3.5 templateのwhitespace・thinking branch・escaping調査を記録する`docs/references/qwen3.5-phase3-full-model-reader.md`、fixture専用H1 ownershipとrevisionを同期する`ci/matrix/path-to-suite-v1.json`・`ci/tools/validate_matrix.py`、新規integration-test target・既存workspace固定`sha2`へのfrontend edge・そのexact edge countを同期する`ci/dependencies/rust-workspace-v1.json`・`ci/schema/rust-dependency-policy-v1.schema.json`・`ci/tools/validate_rust_dependencies.py`に限定する。実装前にreaderとimplementerを分離し、任意Jinjaを実行せず、text-only typed rendererと明示unsupported境界を固定する。constructorは読み出した7,756-byte自体のSHA-256を固定値へ照合し、mutable metadata labelだけでは成功させない。readerはlock一致を確認した同frontend template assetだけをbounded readし、weight payload、full model load、GPU、network、containerは使用しない。`sha2`は既存workspace package/version/featureを再利用し、Cargo.lockはlocal `sllm-frontend`依存一覧への1項目追加だけ、package集合は不変とする。

B4 candidate `b1984e47809ed8cc428b9b817409b74470beadf6`、tree `a8b01c84eef5836bc45d2535843ec3c29e180fe2`はstrict H0 335/335、H1 195/227（32 deselected）、H2 35/42（7 deselected）を各attempt 1、failed/skipped 0、clean exact identityで`PASS`したが、fresh累積独立reviewがStage B表の古い「dependency edge/Cargo.lock不変」と実際の受入済み`sha2` edge追加との矛盾をLow 1件として検出したため受け入れない。実装指摘は0件である。表を開始記録と実装範囲へ一致させ、旧evidenceを再利用せず新candidateのH0〜H2とfresh reviewを取り直す。

表を修正したcandidate `5c8bbd5c5516891fa5708245ed2a8b522f533247`、tree `a753d87ef76575ce66350070a88b1c57121fcd86`はstrict H0 335/335、H1 195/227（32 deselected）、H2 35/42（7 deselected）を各attempt 1、failed/skipped 0、clean exact identityで`PASS`した。fresh累積独立reviewは、固定template bytesだけで別repo/revisionのlockを除外しないconstructor identity不足をMedium 1件、main planの現在状態とreader残件の陳腐化をLow 1件として検出したため受け入れない。fixed `repo_id`/`resolved_revision`検査とdirect mutation negative testを追加し、正本文書を同期した新candidateで全evidenceを取り直す。

B4 final implementation candidate `b43f2132c1afc604f2ae22ab12d55101aac7921b`、tree `559c426b1184f25da131fa10e07a3926938d299e`をclean worktreeで固定した。constructorはfixed `repo_id`、`resolved_revision`、template path/size/SHAをread前に検査し、bounded verified-cache read後の実bytesへsize、SHA-256、UTF-8を検証する。private `cfg(test)` seamは同じproduction identity/metadata/read/UTF-8経路だけを使い、public direct negative testはwrong repo/revisionとsame-size spoofを拒否する。同一identityのstrict H0 335/335、H1 197/229（32 deselected）、H2 35/42（7 deselected）は各attempt 1、failed/skipped 0で`PASS`した。report SHA-256は順に`e139ea624639a609921ebe63f8398a1948b45b8d8b1c1a49a8efad9b828b745f`、`3197aa221dbc7d72c5c292662cb4c71900b46536ded041c81fa03970df0ebf62`、`27ab4aa34fc13e5fa068d855b3b8e2cbaf7e04d517cd676a3874c6c2eb8f9e94`でsidecarと一致する。B3 rollback base `7904a2c196628adcc138eb6499a6a04bd5ebdb56`からのfresh累積独立reviewはHigh/Medium/Low 0件の`PASS/no findings`だった。GPU、weight payload、full model、model cache、network、containerは使用していない。B4 implementationを受け入れ、本記録のdocs-only identityへstrict H0とfresh reviewを結合するまでB4全体は未完了、B5は未開始とする。

B4 docs-only closeout candidate `28136d4e6a50fb6349b7cf81d063397aa136a50f`、tree `da0c7e6b4647561be9bcd9804c1b56075d718083`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `73620fbe1c402dc1b78432bc5c2d4c1cf4e5b6b072c74724aeb6b136c90762ca`のsidecar一致を`PASS`した。fresh reviewが指摘したresource記録のweight payload不使用欠落は本記録で修正する。一方、candidate自身のcontentへ自身のSHA/tree/report/review結果を埋め込む要求はcontent-derived identityの自己参照になるため、B1〜B3と同じく、固定後の外部H0/reviewを同一identityへ結合し、そのPASS結果を次単位開始記録へ同期する。修復candidateのstrict H0とfresh reviewを取り直すまでB4全体は未完了、B5は未開始とする。

closeout修復candidate `555bfef127077a74bb94bc3762cdf2984c48dbdf`、tree `55df85d4c728e807fb0035454bc35b6d4fd5084d`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `1ea590f1683a555d02bdd3571b83c199a892950f5a85cf54845e07ad86fa66c9`のsidecar一致を`PASS`した。fresh reviewは自己参照境界を認め、残る指摘はmain planのimplementation resource行だけにweight payload不使用が未同期であるMedium 1件だった。本修復で3正本を一致させ、新candidateのH0とfresh reviewを取り直すまでB4全体は未完了、B5は未開始とする。

B4 final docs-only closeout `b8a71243f7f93390630c7423d6ca082f9ec51703`、tree `caa9e69da0be67a9207c0455d628e5fecb2611d8`はpinned Python 3.12.10 strict H0 335/335、attempt 1、failed/skipped 0、clean exact SHA/tree、report SHA-256 `8c7db0493b46cac2d08d96ae27190940786a04847c379a4a0bb5db1e9dc17fe3`のsidecar一致とfresh独立review High/Medium/Low 0件の`PASS/no findings`を結合し、B4全体を完了した。これをrollback baseとして2026-08-10 14:56:38 JSTにB5 weight registry/load planを開始する。予測3〜5時間、作業単位hard中断時刻20:56:38 JSTだが、先に到達する全体停止上限15:38:27 JSTをeffective hard stopとする。所有範囲は新規`crates/sllm-core/src/weights.rs`、`crates/sllm-core/src/lib.rs`、新規`crates/sllm-core/tests/weight_contract.rs`と、追加test targetに必要なdependency inventory同期だけに限定する。B1のverified descriptorからrequired/config-conditional/known-unconsumed/rejected分類、一意consumer、exact source range、最大16 MiBの決定的chunkとoverflow-safe destination offsetを構築する。事前監査では固定reader記録と既存core APIだけを読み、weight payload read/materialize、model cache、GPU、network、containerは使用しない。

2026-08-10 policy reset後のtrusted-solo draftでB5を実装した。新規`weights.rs`はverified descriptorだけから426 main-text consumerと312 known-unconsumed entryをname順に分類し、locked shard identity、checked source/destination range、16 MiB chunk、canonical 426-byte wire vectorのSHA-256をplanへ結合する。payload、model cache、GPU、network、containerは使用していない。focused `sllm-core` unit/integration 46/46、clippy、format、dependency inventory、diff checkがPASSした。B5 batch integrationは未実施だが、現行laneではB6 draftを開始できる。

B6 offline CLIをtrusted-solo draftで実装し、明示`--lock`/`--cache`だけからproduction core/frontend APIを呼ぶ`verify-model`、`tokenize`、`render`、`decode`を追加した。成功はstdout 1行の`model-frontend-cli-report-v1`、失敗は部分成功JSONを出さずstderrとexit 2に固定し、model command経路はHIP probe、network、model execution、generationを行わない。tiny backendはproductionと同じdispatch/report経路で全4成功入口を通し、subprocess testは全4失敗入口、stdout/stderr分離、invalid lockでHIPへ落ちないことを検証する。schemaはcommand/result対応とoffline/no-GPU scopeをclosed contractにし、dependency inventoryはpackage 90のままCLI direct edge 3件を加え174 edgeへ同期した。B5+B6 batch integrationはcore 46、frontend 48、CLI 9のRust計103件、schema 2件、affected clippy、Rust 1.85 offline all-target dependency closure、format、diff checkをPASSし、integration review 1回はcorrectness/security blockerとHigh/Medium finding 0件だった。weight payload、model cache、GPU、network、containerは使用していない。次はB7a draftである。

B7a事前監査では、既存public transfer ABI、`Queue::copy_to_host()`、completion所有権、1 GiBの`SLLM_HIP_MAX_TRANSFER_BYTES`だけで必要なrange readbackを実装でき、新しいnative/C ABIは不要と確定した。core draftはadapterの非zero transfer上限、session/queue/buffer identityを検査する任意`BufferRange`の`BufferReadback`、terminal success後のexact-capacity `read_into`を追加し、uploadにも同じ上限を適用する。HIP adapterは既存D2Hだけへlowerする。fake adapterの1/3/17/255/256 byte exact upload→readback、早期/short/long destination、wrong session/queue、zero/overflow、`B-1/B/B+1`のpre-submit拒否、drop/shutdownをPASSし、core 30/30、HIP lib 64/64、affected clippy、format、diff checkがPASSした。続いて`execution-transfer-g1-report-v1`のclosed schema、canonical 2-row matrix、static/negative validator test、専用evidence binary、H0 suite/path登録を実装し、hostはcore 30/30、HIP lib 64/64、contract test 2/2、matrix/JSON/dependency closure、clippy、format、diff checkをPASSした。2026-08-10 19:01 JST、canonical V620（物理HIP index 1を論理0へ限定、`gfx1030`）とcanonical R9700（物理HIP index 2を論理0へ限定、`gfx1201`）で、1/3/17/255/256/257 byteのoffset付き6ケースを各6 allocation、6 H2D、6 D2Hでbyte exactにPASSし、fallbackなし、kernel/model/semantic opなし、cleanupゼロをschema検証した。report SHA-256は順に`8d41fefe47512d4755730dc36e899769a77f9b55eb1a661de6997be115ec8842`、`f59fcfbd3d9f7bdf307bed87ea9a916750c55ce30313df519687b7cef9e7ba59`である。dirty worktree上のfocused draft evidenceであり、clean immutable identityのintegration/release evidenceには昇格しない。ローカル環境文書のcanonical root `/opt/rocm/core-7.14`と既存HIP runtime `build.rs`が要求するlogical root `/opt/rocm`の経路差は今回の同一7.14.0/LLVM 23 buildを妨げなかったが、後続のtracked orchestrationで手書き指定を除く負債として残す。B7a draftを完了し、次はB7b exact-range weight upload bridgeである。

B7bは`VerifiedCache`、B5 planの再計算digest、tensor名/dtype/descriptor、exact destination rangeをsubmit前に結合し、verified FDから最大16 MiBを1 chunkずつ読み、generic `ExecutionSession::upload()`へ順次渡すcore bridgeとして実装した。plan-global offsetはtensor rangeへ相対変換し、identity/target/dtype/range/chunk連続性/backend上限をpayload read前にfail-closedに検査する。失敗時はreceiptを返さず部分bufferを破棄する契約とし、HIP専用wrapper、直接queue、shard/tensor全体のhost複製は作らない。20 MiB固定BF16 tensorを16 MiB＋4 MiBへ分割するevidence binary、closed report/matrix schema、static/negative validator 3件、H0 suite/path登録を追加し、core bridgeのmulti-chunk/identity/range test、stub fail-closed、clippy、matrix/JSON/dependency contractをPASSした。2026-08-10 19:14 JST、lock fingerprint `sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935`とplan digest `sha256:0820227fdc4129e5ff100e0aa87db7663d75703c9ba723bc4adc950a3af6ab66`に結合した`model.language_model.layers.0.linear_attn.in_proj_z.weight`、source range `[42435872,63407392)`をcanonical V620 `gfx1030`とR9700 `gfx1201`へ各2 H2D/2 D2Hし、全byte exact、fallback/network/model execution/semantic op/kernel dispatchなし、cleanupゼロをPASSした。report SHA-256は順に`b952740d270f72255ea804e894bb6e749d2681997276aa29bf1fd5fb989aa0c9`、`d482ced5145fab28f2cda5ea3b7be7e234dfd1d7e9195f265f3df352f3340e48`である。dirty focused evidenceでrelease昇格には使わない。B7b draftを完了し、次はB5〜B7bのStage B batch integration gateである。

B5 separated preimplementation readerは、既存`TensorDescriptor`、`VerifiedCache::tensors()`、lock/config/fingerprintだけでplan入力が足り、B1のprivate catalog、parser、hash、descriptor map、payload range readerを複製・追加公開しないと確定した。main text 426件のfamily/count/consumerを固定vLLMと固定llama.cppでconcept cross-checkし、vision 297件/MTP 15件はknown-unconsumedとした。現行lockの`tie_word_embeddings=true`ではembeddingをtied lm-head aliasとし、独立lm-headを拒否する。untied branchは現行B1契約外のためconditional型だけに留める。tensor name順、checked `[start,end)`、16 MiB以下chunk、固定binary domainのSHA-256 plan digest、payloadなしの1/3/17 byteと`B-1/B/B+1` testをreader記録へ固定した。追加CI差分は新integration-test targetのdependency inventory同期だけを予定し、Cargo manifest/lock/edgeとpath-to-suiteは変えない。B5は未実装・未完了である。

初回B5 reader checkpoint `939a1be3f48983ad9deb041b9c5f9930f7c74e64`、tree `7777f1cb5eada0393e150c7d945b052a1508293a`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `7e0ae8b7c394206e5b63f5709d330c44973c6af731f56b58d62da5b788409638`のsidecar一致を`PASS`した。fresh reviewはconsumer grammarとdigestのsource shard binding不足をHigh 2件、family算術表現、vision bias境界、binary encoding/test vector不足をMedium 3件として検出したためcheckpointを受け入れない。exact suffix grammar、relative source fileとlocked size/SHA、固定tag/width/vector framing、426-byte canonical digest vectorをreaderへ追加し、新checkpointでH0とfresh reviewを取り直す。B5は未実装・未完了である。

修復reader candidate `f73c9646f221eb92fb0fe5371e0ce8519dbedb2d`、tree `775c81ed4b78d6787c271d59f783fbb20a6eb2c4`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `df219c10bf813c1dde998cbdca906c77bf45b1d9d916c337095e42ef413d40ff`のsidecar一致を`PASS`した。fresh reviewはcanonical digestが記載規則から再現不能であるHigh 1件を検出したためcandidateを受け入れない。唯一のordered wire fields、u8 tag幅、単一entry framing、optional layer位置、locked SHAの64-byte lowercase ASCII hex stringをreaderへ固定し、独立再計算で426 bytes・SHA-256 `9a57a67384038c9e437236511c50f1b03b88a4f733cb06464d4ad3e408616bb2`を確認した。新checkpointのH0とfresh reviewは未実施で、B5は未実装・未完了である。

受入条件:

- CPU CIはtiny synthetic safetensors/tokenizer fixtureだけを使い、full modelをdownload/loadしない。
- missing/duplicate/overlap/out-of-range/wrong dtype/unknown tensor、writableまたはsymlink cache、unsupported chat contentをnegative testが拒否する。
- fixed promptのtoken IDs、special token、rendered textがversioned host fixtureと一致する。

### Stage C: baseline semantic opsとHIP kernels

operatorを依存順に小さなcandidateへ分ける。

1. BF16 buffer copy、embedding gather、RMSNorm、residual elementwise。
2. BF16 matmul/GEMV/GEMMとbiasなしlinear、SiLU gated MLP。
3. RoPE、causal mask、softmax、full attention、output gate。
4. linear attentionに必要なprojection、short convolution、L2 normalization、gate/decay、recurrent state update。
5. final RMSNorm、tied embedding projection、logitsとgreedy argmax。

各opはsemantic descriptor、capability/prepare validation、public C ABI command、native registry、baseline HIP kernelを通す。exact target、dtype、layout、alignment、shapeが適合しない場合は別backend、CPU、generic kernelへfallbackしない。

受入条件:

- NumPyのbounded independent oracleと、1、3、17、`B-1/B/B+1`、model実shapeを含むcaseを持つ。
- synthetic semantic G1とreal-weight G2をprivate diagnostic G1から分離する。
- accumulation、rounding、NaN/Inf、zero length、alias、state更新順序、unsupported条件をversioned contractへ固定する。
- exact `gfx1030`/`gfx1201`のH3 artifact検査とGPU数値evidenceが同一semantic/build identityでPASSする。

### Stage D: Qwen3.5 text graphとstate

1. token embeddingから32 main layer、final norm、tied lm headまでのexecution planをRustで構築する。
2. `layer_types`の明示listどおりにlinear/full attentionをdispatchし、intervalから推測しない。
3. full-attention KVをFP16 `[4, T, 256]`、linear-attention convolution stateをBF16 `[3, 8192]`、recurrent stateをF32 `[32, 128, 128]`の別の型・row-major layout・request lifetimeとして管理する。
4. prefillとsingle-token decodeでposition、mask、KV/stateを正しい順序で更新し、completion前のreuse/freeを禁止する。
5. vision/MTP tensorや未知componentへのdispatchを拒否する。

受入条件:

- tiny synthetic multi-layer graphでlayer順序、state初期化/更新、prefill/decode一致、early drop/error cleanupをhost/GPU testする。
- layer classごとの代表的な実weight G2がPASSし、全tensor consumer coverageが100%である。
- full model memory budget、allocation plan、peak VRAMを実GPU実測し、OOMをCPU fallbackで隠さない。

### Stage E: CLI generationとG3

1. CLIがlock fingerprint、model cache、GPU identity、promptまたはtext-only chat messages、max new tokens、greedy modeを明示的に受け取る。
2. prompt render/tokenize、prefill、反復decode、固定停止集合`[248046, 248044]`またはmax tokenによる停止、decodeを一つのpublic inference pathで実行する。停止判定は新規生成tokenだけに行い、stop token自身をvisible outputから除外する。
3. run reportへinput token IDs、output token IDs、停止理由、selected backend、exact target、dispatch/artifact identity、fallback、timing、healthを記録する。secretやraw model bytesを含めない。
4. canonical両GPUで同じfixed prompt/case-setを実行し、output tokensと停止理由を比較する。

G3の最低case:

- 短いASCII promptのgreedy generation。
- Unicodeを含むtext-only chat template prompt。
- 248046、248044、小さい`max_new_tokens`それぞれによる明示停止と、stop tokenがvisible outputへ出ないこと。
- prefill長とdecode長が境界前後になる非整列case。

G3は「processが終了した」「何らかのtextを出した」だけでPASSしない。固定model lock、同一semantic/build identity、exact GPU、全dispatch HIP、fallbackなし、1 token以上、token IDs/stop reasonのversioned expectation、両GPU一致、実行前後health、process cleanupを必須とする。外部engineの単独出力を数値oracleとせず、op別G2、cross-target一致、review済みgolden token sequenceを組み合わせる。golden確定手順はG3 schema実装前に別途reviewする。

## Verification lanes

- docs-onlyはMarkdown、link、consistencyだけを確認し、closeoutを作らない。semantic/build identityが不変でmappingが明示される場合はGPU evidenceを再利用できる。
- draftはdirty local worktreeでfocused affected testだけを行い、immutable identity、独立review、full matrixを要求しない。B5以降の個別draftはfocused test後にbatch integrationする。
- integrationは影響するH/G checkとreview 1回を行う。B7のfull H/G matrixはintegration/releaseでのみ実行する。
- release/pushはclean worktreeのimmutable semantic/build identityとfinal relevant matrixを要求する。
- Phase 3のfinal integration/releaseは、関連するH0〜H3、G0〜G3、P0、aggregate、実行前後health、process cleanupを要求する。G2/G3はmodel pathへ接続した場合、P0はruntime/dispatchへの影響がある場合に含める。
- GPU不在、timeout、crash、zero selection、CPU fallback、別target/artifact/model lock、stale reportは非PASSとする。exact target、artifact、model lock、numerical oracleを省略しない。

## Review・rollback

- high-risk ABI/kernelのdesign reviewはoptional、integration reviewは1回、指摘へのfocused re-reviewは必要時のみ、release reviewは累積1回とする。checkpointごとのfresh reviewは行わない。
- source/build inputs、toolchain、model lock、artifact/reportのsemantic/build identityをGit commit/treeと分離して記録する。Git整理でsemantic/build identityが変わった場合だけ該当evidenceを取り直す。
- draftはdirty local結果をrelease evidenceへ昇格せず、integration/releaseのfailureまたはcorrectness/security defectはpushを止め、直前の検証済みrevisionを維持する。
- GPU適用、health、state、process cleanup、resource回収が失敗した場合は結果を成功扱いにせず、必要なquarantineまたはreplanを行う。

## 未解決事項

- `tokenizers =0.21.4`のsemver解決結果をroot lockへ固定したときの全transitive dependency checksum/licenseとRust 1.85 offline build evidence。
- `ModelLock`と`VerifiedCache`の公開fingerprint fieldをcallerが書換えられる既存core APIを、互換性を整理した独立follow-upでopaqueな内部verified identityへ移行する。B3では一致をmutable labelの整合確認として扱い、暗号学的なlock/cache結合やfull-model provenanceを主張しない。
- full-model G3 golden token sequenceの独立確定方法。
- model実shapeを含むop別G2 case-setとtolerance。
- 両canonical hostのVRAM budgetとfull model peak memory。

## 完了後

Phase 3完了後にこの計画をhistoryとともにarchiveし、次の開発順序でQwen3.5-2B/9Bの同一実装確認へ進む。OpenAI-compatible APIはその後に実装する。

[対応する履歴](../../../../../history/2026/08/1-10/phase3-qwen35-4b-bf16.md)
