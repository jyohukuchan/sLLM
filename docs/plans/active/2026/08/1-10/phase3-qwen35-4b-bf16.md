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

### Stage B: Rust model I/Oとtext frontend

1. model lock、config、safetensors index/headerを型付きでparseし、unknown version/architecture、hash、shape、dtype、tensor集合の不一致を拒否する。
2. verified read-only cacheをmmapまたはbounded readし、必要byte rangeだけをopaque GPU bufferへuploadする。shard全体の複製を作らない。
3. main text tensorをrequired、config-conditional、known-unconsumed、rejectedへ分類し、全required tensorの一意なconsumerを検証する。
4. tokenizer vocabulary/merges/special tokenとtext-only chat templateをload・validateする。image/video content、tooling等の未対応template branchは黙って処理せず明示unsupportedとする。
5. CLIにoffline model verificationとtokenize/render/decodeの独立入口を設け、model execution前にfrontendだけをhost testできるようにする。

B1 frontendは`tokenizers =0.21.4`をdefault featureなし・`onig`だけで固定し、HTTP、progressbar、`esaxx_fast`を無効にする。任意Jinjaは実行せず、locked `chat_template.jinja`のhashと対応するtyped Qwen3.5 text-only rendererを実装する。先に停止policyをversioned lock/schema/APIへ反映し、依存version、checksum、license、MSRVをroot `Cargo.lock`へ固定してRust 1.85のoffline buildを通す。

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
