# Phase 3 Qwen3.5-4B BF16 text生成計画

## 状態

- 作成日: 2026-08-04
- 状態: active
- 上位計画: [main plan](../../../../main-plan.md)
- Stage A子計画: [model lock・RMSNorm・G2（完了・archive）](../../../../archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)
- CI正本: [CI・テスト方針](ci-test-strategy.md)

## 目的と完了境界

固定した`Qwen/Qwen3.5-4B` BF16 modelを、単一AMD GPU、batch 1、text-onlyでloadし、CLIからprefillとdecodeを実行してtextを生成する。canonical exact `gfx1030`と`gfx1201`の両方で、同一immutable candidateのhost、compile-only、GPU preflight、kernel/ABI、model slice、end-to-end evidenceをfail-closedに集約する。

Phase 3はStage AのRMSNorm G2/P0だけでは完了しない。次をすべて満たした時だけ完了とする。

- 完全model lockとverified read-only cacheから、全text weightをfail-closedに解決・loadできる。
- Qwen3.5 text stackの32 main layer、hybrid linear/full attention schedule、state/cache、tied embedding/lm headを実行できる。
- tokenizerとtext-only chat templateをRust側で適用し、CLIが固定promptから1 token以上を生成・decodeできる。
- exact `gfx1030`と`gfx1201`で同じmodel lockとcandidateを使い、CPU・他backend・generic kernelへのfallbackなしでG3がPASSする。
- H0〜H3、G0、private diagnostic G1、必要なsemantic G1/G2、G3、runtime/dispatchへ必要なP0、実行前後health、process cleanupを同一run graphへ集約できる。
- full model、raw slice、binary、traceをGitまたはGitHub Actions artifactへ保存していない。

## 当面のtrusted development実行境界

2026-08-08から今後数週間は、単独maintainerによるtrusted development期間とする。Phase 3のlocal/GPU実行は、maintainerが内容を確認したrepository codeと明示commandだけを対象とし、外部PR、fork由来code、未review script、第三者binaryは専用local/GPU hostで実行しない。

この期間は、悪意ある同一UID process、敵対的fork PR、永続runner上のhostile codeに耐えるrepository内custom capsuleの完成を、Stage A〜Eの開始条件・完了条件から外す。これは安全要件の撤回ではなくtrust boundaryの限定であり、buggy codeに対する標準的な隔離としてsecret・Docker socketを渡さないこと、可能な範囲のcontainer/network隔離、timeout・resource上限、process cleanup、実行前後GPU health、candidate SHA・artifact identityの検証は維持する。dirty worktree上の実行結果は`local-development`に限定し、immutable candidateのevidenceへ昇格しない。

中断されたA0 security hardeningの部分変更は検証済み実装として継承せず、byte-for-byteの過去版復元もPhase 3の作業に含めない。trusted development中のlocal確認はdirect testと標準containerを使い、当該custom capsuleをimmutable evidence経路へ使用しない。immutable candidateのhost evidenceを取得する段階では、現行部分変更を土台にせず、必要最小限のtrusted-development baselineを新しい作業単位として作成し、通常回帰とreviewを通した新identityを固定する。

外部contributorのcodeを実行する前、または複数の信頼境界を持つ運用へ移る前に、ephemeral VM/JIT runnerまたはjob後reimageをsecurity boundaryとする設計をhard gateとして再開し、実行前にCI正本と本計画を更新する。

## 時間予測と中断契約

2026-08-08の再開以降、各作業単位は開始前に予測時間の範囲と中断上限を宣言する。中断上限は予測上端に1時間を加えたwall-clockとし、上限到達時は新しい変更を開始せず、安全なrollback可能点で一旦停止して、経過時間、完了範囲、未完了理由、次の分割案を報告する。ユーザーが明示的に中断した時間はwall-clockから除外する。正常に進む重いcommandは15分ごとの監視を続けるが、この中断上限を超えて自動継続しない。

Stage Aの工程別実績は[archive済みStage A計画](../../../../archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)を正とする。Stage B以降は開始前に8時間以下の独立review・rollback可能な単位へ分割し、それぞれ同じ`予測上端 + 1時間`契約を記録する。工程の見積りを変更する場合は、旧上限へ到達する前に根拠を示して本計画を更新し、経過後に後付けで上限を延長しない。

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
- model lock fingerprintはmodel bytes、candidate SHA/treeはcode、artifact/report digestはbuild/result、tuple digestは実行環境のidentityであり、相互に代用しない。
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

この運用負債解消を2026-08-09 23:51:09 JSTに開始し、当初のhard中断時刻を2026-08-10 03:38:27 JSTとした。同作業完了後のStage B plan reviewを継続するため、ユーザー指示により2026-08-10 03:38:27 JSTから6時間延長し、現在の全体hard中断時刻を同日09:38:27 JSTとする。受入境界は、defaultでGPU、model cache、container、buildへ触れないhost-only dry-run、既存正本から導出したcanonical 2 targetの実行plan、run identity・短いsocket root・target別build/output ownership・canonical JSONのfail-closed検証、focused host回帰、独立reviewの`PASS`とする。

同作業は2026-08-10 01:26 JSTに受入境界を満たして完了した。tracked plannerは既存workflow、matrix、G1/G2/P0総合validator、builderのpure layout helperからexact `gfx1030`→`gfx1201`の実行planを導出し、clean immutable Git identity、authority file hash、symlink component、短い未作成run root、target別path/output、schema順序をfail-closedに検証する。GPU、model cache、container、build、networkは実行せず、focused 11件、fail-closed 46件、matrix/JSON/G1/G2/P0 validator、dirty-local H0 316/316、独立reviewのHigh/Medium 0件・`PASS`を得た。既存P0 builderのsame-UID/trusted-solo output symlink安全負債は当面のtrusted development境界に従い延期したままとし、次のGPU evidence refreshではこのplannerを先行させる。

### Stage B: Rust model I/Oとtext frontend

1. model lock、config、safetensors index/headerを型付きでparseし、unknown version/architecture、hash、shape、dtype、tensor集合の不一致を拒否する。
2. verified read-only cacheをmmapまたはbounded readし、必要byte rangeだけをopaque GPU bufferへuploadする。shard全体の複製を作らない。
3. main text tensorをrequired、config-conditional、known-unconsumed、rejectedへ分類し、全required tensorの一意なconsumerを検証する。
4. tokenizer vocabulary/merges/special tokenとtext-only chat templateをload・validateする。image/video content、tooling等の未対応template branchは黙って処理せず明示unsupportedとする。
5. CLIにoffline model verificationとtokenize/render/decodeの独立入口を設け、model execution前にfrontendだけをhost testできるようにする。

B1 frontendは`tokenizers =0.21.4`をdefault featureなし・`onig`だけで固定し、HTTP、progressbar、`esaxx_fast`を無効にする。任意Jinjaは実行せず、locked `chat_template.jinja`のhashと対応するtyped Qwen3.5 text-only rendererを実装する。先に停止policyをversioned lock/schema/APIへ反映し、依存version/checksumはroot `Cargo.lock`、license/feature/MSRVはtracked dependency policyとoffline validatorへ固定してRust 1.85のoffline buildを通す。

#### Stage B開始時baselineと重複禁止境界

Stage B開始時点の`model.rs`は空のstubではない。Stage Aですでに、型付き`ModelLock`/`TextConfig`、固定Qwen identity、738 tensor名・dtype・component分類、streaming hash、index/headerのgap・overlap・範囲・dtype/size検証、verified file descriptorを保持した最大16 MiBのpositional tensor range read、cache/path/inode安定性検査を実装済みである。`sllm-frontend`にもversioned generated-token停止policyがあり、`tokenizers =0.21.4`はdefault featureなし・`onig`だけでroot lockへ解決済みである。

従って後続作業は既存`verify_model_cache()`、`VerifiedCache`、`TensorDescriptor`、停止controllerを置き換えない。特に、別のsafetensors parser、別のcache hasher、共有seek cursor、shard全体を返すAPI、任意Jinja executor、同じ停止policyの別実装を追加しない。Stage Bで閉じる残差は、全738 tensorのexpected shape、frontend assetのbounded verified read、dependency/license/MSRV evidence、tokenizer/typed renderer、main-text tensorの一意consumer/load plan、CLI、opaque GPU bufferへのexact-range upload接続である。

#### Stage B独立作業単位

各単位は開始時にrollback baseのcommit SHA/tree、開始時刻、予測、hard中断時刻を固定する。実装とreview修正後のcandidateを別のcommit SHA/treeへ固定し、その同一identityに受入evidenceを結合する。整理でtreeが変われば当該単位のevidenceを最初から取り直す。依存順は`B0 -> B1 -> B2 -> B3 -> B4 -> B5 -> B6 -> B7a -> B7b`とし、後続単位は直前単位のcandidate/evidence/reviewがPASSするまで開始しない。B0〜B6はhost-onlyで、full model download、weight payloadの一括materialize/decode/mmap/upload、GPU、containerを禁止する。固定cacheを使う適用確認は、全locked fileをbounded bufferでstreaming hashし、metadata/rangeを照合するだけのlocal model-bound evidenceとして行い、CPU CIへ持ち込まない。B7a/B7bだけがHIP/runtime/backendへ影響し、canonical `0000:03:00.0`のV620とcanonical R9700を使う。spare V620は使用しない。

| ID | 所有範囲と成果物 | 予測 / hard上限 | 受入条件・evidence・rollback境界 |
| --- | --- | --- | --- |
| B0 dependency closure | `Cargo.toml`、`Cargo.lock`、各workspace memberの`Cargo.toml`、新規`ci/dependencies/rust-workspace-v1.json`、`ci/schema/rust-dependency-policy-v1.schema.json`、`ci/tools/validate_rust_dependencies.py`、`ci/tests/test_rust_dependencies.py`と必要な`ci/matrix/{suites-v1,host-v1,path-to-suite-v1}.json`登録。通常・build・devを含む全workspace targetの解決graphをinventoryし、全registry dependencyのexact version/checksum/license、workspace memberごとの有効feature、禁止feature、Rust 1.85 offline解決を機械検証する。`tokenizers 0.21.4`はdefault featureなし・許可feature `onig`だけとし、`sllm-hip`の`static_assertions`等のfrontend閉包外も除外しない。 | 2〜4時間 / 5時間 | local crate cacheだけを使い、policyと`cargo metadata --locked --offline --format-version 1`の全package/edge/target集合が一致し、H0〜H2、`cargo +1.85.0 check --workspace --all-targets --locked --offline`、fresh reviewがPASS。model/GPU/cache/networkなし。validator・manifest・Cargo/CI登録差分だけでrollbackできる。これを最初の実装単位とする。 |
| B1 tensor shape closure | `crates/sllm-core/src/model.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-core/tests/model_contract.rs`、`ci/schema/model-lock-v1.schema.json`、`ci/fixtures/model-lock-v1/**`。既存738-name/dtype catalogをtyped expected shapeまで拡張し、configから導出する全main/vision/MTP shapeとheader shapeの一致を検証する。 | 5〜8時間 / 9時間 | 2の冪だけでなく1、3、17、境界前後を含むtiny synthetic fixtureでmissing/duplicate/wrong rank/dimension/dtype/overflowをfail-closed。H0〜H2とfresh review、同一SHAの固定cache metadata照合がPASS。既存parser/hasher/range APIは保持し、この単位だけでrollbackできる。 |
| B2 verified frontend assets | `crates/sllm-core/src/model.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-core/tests/model_contract.rs`。`VerifiedCache`の保持済みFDから、固定名のtokenizer/config/template assetだけを種類別hard cap付きでpositional readするAPIを追加する。 | 2〜4時間 / 5時間 | shard、任意path、symlink/hardlink、cap超過、差替え、同一inode改変を拒否し、weight shard全体を返さない。tiny fixtureのH0〜H2とfresh reviewがPASS。GPU/model payloadなし。新APIとtestだけでrollbackできる。 |
| B3 tokenizer frontend | `crates/sllm-frontend/src/tokenizer.rs`、`crates/sllm-frontend/src/lib.rs`、新規`crates/sllm-frontend/tests/tokenizer_contract.rs`と`ci/fixtures/tokenizer-v1/**`。B2 assetからだけ`tokenizers`を構築し、encode/decode、special-token identity、EOS集合との整合をtyped APIにする。 | 3〜5時間 / 6時間 | ASCII、Unicode、空、非整列長、未知/欠落/重複special token、malformed tokenizerをhost negative testで検証し、固定prompt token IDs/decodeがversioned fixtureと一致。H0〜H2、MSRV、fresh reviewがPASS。GPU/full modelなし。frontend module/fixture単位でrollbackできる。 |
| B4 typed chat renderer | `crates/sllm-frontend/src/chat.rs`、`crates/sllm-frontend/src/lib.rs`、新規`crates/sllm-frontend/tests/chat_contract.rs`、`ci/fixtures/chat-template-v1/**`、`docs/references/qwen3.5-phase3-full-model-reader.md`。locked template identityを要求し、Qwen3.5 text-only messageをtyped rendererで生成する。 | 3〜5時間 / 6時間 | fixed `hello`とUnicodeのrendered text/token IDsを固定し、image/video/tool/unknown role、不正content、template hash不一致を明示unsupportedにする。任意Jinjaを実行しない。H0〜H2、fresh reviewがPASSし、renderer/module fixtureだけでrollbackできる。実装前に固定templateのwhitespace、thinking branch、escapingのseparated reader記録を完了する。 |
| B5 weight registry/load plan | 新規`crates/sllm-core/src/weights.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-core/tests/weight_contract.rs`。B1のdescriptorをrequired/config-conditional/known-unconsumed/rejectedへ分類し、layer/roleを含む一意consumerとexact source rangeを持つimmutable host load planを作る。各tensor rangeは既存16 MiB read上限以下の決定的chunkへ分割し、destination offsetもoverflowなく固定する。 | 3〜5時間 / 6時間 | 全required main-text tensorがconsumer 1件、tied lm-head条件が明示、vision/MTPはknown-unconsumed、unknown/missing/duplicate consumerは拒否。16 MiB境界前後をpayload allocationなしで検証し、chunk順序とplan digestが決定的でH0〜H2/fresh reviewがPASS。GPU ABIは決めずRust内部descriptorに限定し、module単位でrollbackできる。 |
| B6 offline CLI | `crates/sllm-cli/Cargo.toml`、`crates/sllm-cli/src/main.rs`、新規`crates/sllm-cli/src/model.rs`、`crates/sllm-cli/tests/model_frontend_cli.rs`、`ci/schema/model-frontend-cli-report-v1.schema.json`とvalidator test。`verify-model`、`tokenize`、`render`、`decode`をmodel executionから独立させる。 | 3〜5時間 / 6時間 | 明示lock/cache入力、offline、fail-closed exit、versioned machine-readable出力、stdout/stderr分離を固定し、tiny fixtureで全入口とnegative caseを検証する。doctor以外はHIP probeを起動しない。H0〜H2/fresh reviewがPASSし、CLI/schema差分だけでrollbackできる。 |
| B7a backend-neutral buffer readback | `crates/sllm-core/src/execution.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-hip/src/bridge.rs`、新規`crates/sllm-hip/src/bin/sllm-execution-transfer-g1-evidence.rs`、対応するschema/matrix/validator/testと既存suite/path登録。任意のowned `BufferRange`をqueueから非同期D2Hするbackend-neutral API、単一observer completion、terminal success後だけのbounded `read_into`、session/queue/buffer identityとlifetimeを追加し、HIP adapterは既存`Queue::copy_to_host()`/versioned transfer ABIだけへlowerする。semantic opの`Submission`/output専用`Readback`とは型を分ける。 | 4〜6時間 / 7時間 | 1、3、17、255、256、257 byteとoffset/末端境界で、既存`ExecutionSession::upload()`から新readback APIまでのexact round-trip、wrong session/queue/range、zero/overflow、早期read、drop/shutdownをfail-closedに検証する。H0〜H3、canonical G0/private G1、execution-transfer G1、aggregate、pre/post health、cleanup、fresh reviewが同一SHAでPASS。新しいnative/C ABIまたは直接`Queue`利用を作らず、失敗時はB6完了commitを維持する。不足が判明した場合はscopeを拡張せず別のABI決定単位を計画する。 |
| B7b exact-range weight upload bridge | `crates/sllm-core/src/weights.rs`、`crates/sllm-core/src/lib.rs`、`crates/sllm-hip/src/bin/sllm-weight-upload-g1-evidence.rs`、`ci/schema/weight-upload-g1-report-v1.schema.json`、`ci/matrix/weight-upload-semantic-g1-v1.json`、`ci/tools/validate_weight_upload_g1_contracts.py`、`ci/tests/test_weight_upload_g1_contracts.py`と既存suite/path登録。B5 load planのverified chunkをB7aでreview済みの`ExecutionSession::upload()`/buffer readbackへ順に接続し、既存HIP `ExecutionSessionAdapter`/versioned transfer ABIを通してopaque GPU bufferへuploadする。新しいHIP weight-upload wrapperや直接`Queue`経路を作らず、shardまたはtensor全体のhost複製も作らない。 | 4〜6時間 / 7時間 | 複数chunk、16 MiB境界前後、tensor/destination境界前後でexact byte/readback、wrong range/target/dtype/plan identityをreject。H0〜H3、canonical G0/private G1、B5 load-plan接続専用semantic upload G1、aggregate、pre/post health、cleanup、fresh reviewが同一SHAでPASS。generic transferの重複実装・重複証明にはしない。canonical sLLM V620とR9700だけを使用し、失敗時はB7a完了commitを維持する。 |

B0開始前にllama.cpp/vLLMの追加readerは不要である。ただしB0内では、local Cargo metadata/sourceだけを調べるread-only dependency auditorとvalidator implementerを分離する。B4だけは現行reader記録に正確なrender済みbytes、whitespace/escaping、thinking branchの全境界がないため、固定revisionのtemplate/tokenizer metadataを読むreaderとimplementerを分離する。B5はRust内部load-plan、B7a/B7bは既存versioned transfer ABIの利用に限定するため、現時点のpublic C ABI未確定事項にはblockedされない。B7aで既存ABI不足が判明した場合だけ中断し、runtime正本と互換性文書を同期する別単位を先に計画する。

2026-08-10時点でこのB0〜B7b分割はhost H0 316/316、Markdown link、diff checkまでPASSしたplan candidateであり、実装開始済みではない。3回のfresh independent review transportがいずれも45分間local出力を生成せず停止した後、commit専用reviewでB0が`tokenizers`閉包外のworkspace依存を網羅しないMedium 1件を検出した。全workspace targetの依存graphへ対象を拡張した。修正reviewでは、既存`ExecutionSession::upload()`に任意bufferのbackend-neutral readbackがなくB7のexact evidenceを満たせないHigh 1件を検出したため、readback contractをB7a、B5 load-plan接続をB7bへ分離した。同一immutable plan candidateのfresh reviewがHigh/Medium 0件でPASSするまでB0を開始しない。

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
- exact `gfx1030`/`gfx1201`のH3 artifact検査とGPU数値evidenceが同一candidateでPASSする。

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

G3は「processが終了した」「何らかのtextを出した」だけでPASSしない。固定model lock、同一candidate、exact GPU、全dispatch HIP、fallbackなし、1 token以上、token IDs/stop reasonのversioned expectation、両GPU一致、実行前後health、process cleanupを必須とする。外部engineの単独出力を数値oracleとせず、op別G2、cross-target一致、review済みgolden token sequenceを組み合わせる。golden確定手順はG3 schema実装前に別途reviewする。

## CIとcandidate gate

- 文書/schema/host parserだけのcandidate: H0〜H2。
- HIP ABI/runtime/kernelを変えるcandidate: H0〜H3、canonical G0、private diagnostic G1、該当semantic G1/G2、必要なP0。
- model graph/state/frontendを変えるcandidate: H0〜H3、canonical G0/private G1、影響opのsemantic G1/G2。full model executionへ接続後はG3も必須。
- Phase 3最終candidate: H0〜H3、canonical 2 GPUのG0、private diagnostic G1、全required semantic G1/G2、G3、runtime/dispatchに該当するP0、aggregate、health/process cleanup。
- GPU不在、timeout、crash、zero selection、CPU fallback、別SHA/artifact、stale reportは非PASSとする。

## review・rollback単位

- Stage A model lock、host loader、各semantic op/kernel、state/cache、model graph、tokenizer/frontend、CLI/G3を独立candidateにする。
- candidate identityを固定してから該当testを実行し、整理でtreeが変わればtestをやり直す。
- GPU適用またはhealthが失敗したcandidateはpushせず、直前の検証済みrevisionを維持する。
- state破損、GPU health異常、process残留、resource回収不能時は追加GPU実行を停止する。

## 未解決事項

- `tokenizers =0.21.4`のsemver解決結果をroot lockへ固定したときの全transitive dependency checksum/licenseとRust 1.85 offline build evidence。
- full-model G3 golden token sequenceの独立確定方法。
- model実shapeを含むop別G2 case-setとtolerance。
- 両canonical hostのVRAM budgetとfull model peak memory。

## 完了後

Phase 3完了後にこの計画をhistoryとともにarchiveし、次の開発順序でQwen3.5-2B/9Bの同一実装確認へ進む。OpenAI-compatible APIはその後に実装する。

[対応する履歴](../../../../../history/2026/08/1-10/phase3-qwen35-4b-bf16.md)
