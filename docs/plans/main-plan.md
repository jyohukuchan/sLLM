# sLLM メイン計画

## この文書の役割

- Git管理外の `sLLM.md` にある要件定義・開発方針・重要な決定を、開発に必要な範囲で追跡可能な形へ同期する。
- この文書には重要なproduct・architecture・compatibility上の決定、開発計画と順序、進捗、未解決事項だけを記録する。恒久的な実行手順は各正本文書へ置き、ここには重複させない。
- `sLLM.md` とこの文書に方針上の差異が生じた場合は、推測で統合せずユーザーへ確認する。
- 角括弧内の項目は、初期バージョンでは対応しない将来機能を表す。
- project内の権限順は、現在の明示的なユーザー指示、`sLLM.md`、`AGENTS.md`、この文書の承認済み決定、active planの作業固有条件、historyに残す過去の事実、とする。下位文書とhistoryは上位方針を上書きせず、新しい完了条件やblockerを作らない。

## プロジェクトの目的と方針

- 最新のモデルと推論機能を、コンシューマーハードウェアや非NVIDIA環境でも早期に利用可能にする。
- 比較的広いハードウェア互換性で、vLLM、SGLang、ATOM、TensorRT-LLMとの差別化を図る。
- 最新モデル・機能への追従速度でllama.cppとの差別化を図る。
- 古すぎる、または実用性能を得られないハードウェアは対応対象に含めない。
- Vulkanは対応対象に含めず、CUDA、ROCm等の専用機能を利用できるbackendを優先する。
- INT4/INT8+scale系の一般的なllama.cpp量子化形式は原則サポートしない。
  - 例: Q8_0、Q4_K、UD-Q4。
  - 低bitでも十分な精度と実用性を両立する方式が確認できた場合は再検討する。
- プロジェクトライセンスはMITとする。
- reset前の履歴は、現行`main`から到達可能なancestryと外部backup/archiveの両方に現状のまま保持する。旧Apache-2.0版の許諾は遡及的に変更せず、orphan化、force push、共有履歴の書換えは行わない。

## 初期バージョンの主要要件

- Linuxのみを対象とする。
- safetensors形式のモデルを読み込む。
- GUI以外の全機能をCLIから利用可能にする。
- AMD GPUを最初のbackendとし、RDNA2、RDNA4、CDNA3を対象候補とする。
- GPU操作、device memory、queue/event、operator dispatch、kernelはC++/HIPで実装する。
- frontend、model/config/tokenizer、scheduler、sampling、execution planはRustで実装する。
- OpenAI-compatible APIを提供する。
  - 初期仕様は `sLLM OpenAI-compatible Chat Completions profile v1` とする。
  - llama.cpp serverは実装参考・差分比較対象であり、仕様の正本にはしない。
  - [Responses APIに対応する。]
- 最適化済みの単一リクエストでは、同一条件のllama.cppより高速であることを一つの基準とする。
  - 比較条件はモデルrevision、GPU target、入力長、出力長、数値型、llama.cpp commitを記録する。
  - 一律の必達倍率は設けず、TTFT、TPOT、token/s、peak VRAMを記録する。
- 複数リクエストのリクエストバッチ処理に対応する。
- [WebUIから管理できるようにする。]
- デバッグ・正しさ確認用の標準実装はPython+NumPyとする。
  - NumPyでは時間または計算効率上の限界がある場合にJAXを使用する。
  - PyTorchは使用しない。
  - Tritonは将来のNVIDIA backendに限って使用可能とし、AMD backendには使用しない。

## 初期の実装スコープ

- 最初の縦切り実装は次に限定する。
  - Qwen/Qwen3.5-4Bの固定revision。
  - BF16 weight / BF16 activation。
  - 単一AMD GPU。
  - 単一リクエスト、batch=1。
  - text-only。visionとMTPは含めない。
  - safetensors、config、tokenizer、chat templateの読み込み。
  - CLIからprefillとdecodeを実行し、テキストを生成する。
- 初期縦切りでは、動的backend plugin、JIT compiler、汎用graph optimizer、自動tuning DB、multi-stream scheduling、RDMA、複数GPUを実装しない。
- 後付けが高コストになる次の抽象化は初期実装から含める。
  - semantic op descriptor。
  - backend capability query。
  - tensor dtypeとquantization encodingの分離。
  - buffer access modeと非同期lifetime。
  - KV layout abstractionとoptional block table。
  - shape、alignment、gfx capabilityに基づくkernel selection。

## 対応予定の詳細機能

- Infinity Fabric対応。
- その他RDMA protocolは、ユーザーがbackendを追加できる拡張点を設ける。
- FP8対応GPUではFlash Attention 4相当のattention実装を目標とする。
- リクエストバッチ処理。
- chunked prefill。
- KV cache、会話、model lock fingerprintをstorageへ保存し、起動時に再開できる簡易永続化。
  - model lock fingerprintは、使用する各model fileのSHA-256を含むlock全体の識別子とする。
  - 旧要件の`model sha256`は、このmodel lock fingerprintへ包含する。
- [LMCache。]
- [RadixAttention。]
- [ロード時量子化。]

### モデルアーキテクチャ

- DeepSeek v4: MoE、DFlash。
- Qwen3.5: Dense、MoE、MTP。
- Gemma4: Dense、MoE、MTP、[Diffusion]。
- MiniMax M3。
- 列挙順は実装優先順位を表さない。

### KV cache数値形式

- TurboQuant。
  - Key Value。
  - K4V4。
  - [K3V3。]
  - [論文準拠K2.5V2。]
  - [論文準拠K3.5V2。]
- NVFP4。
- [MXFP4。]
- FP8。
- [MXFP8。]
- FP16。

### モデル数値形式

- Weight:
  - NVFP4。
  - [MXFP4。]
  - FP8。
  - [MXFP8。]
  - BF16。
- Activation:
  - FP8。
  - [MXFP8。]
  - BF16。
- CDNA3では、e4m3fn modelをVRAMへロードする際にe4m3fnuzへ変換する。
- 混乱を避けるため、テスト専用のe4m3fnuz量子化modelは作成しない。
- NVFP4ではtensor scaleをtensor表現とkernel contractに含める。

## GPU互換性方針

- SKU名ではなく、binary compatibilityとkernel capabilityを分けて管理する。
- AMDの正規識別子はHIPが報告するexact `gfx target` とする。RDNA/CDNAは表示用の世代名として扱う。
- 配布target、code object version、wave size、`xnack`、`sramecc`等のcodegen条件をbinary keyに含める。
- matrix engine、数値形式、FP8 encoding、LDS等の能力をcapability profileとして別管理する。
- 対応候補を選ぶ初期resource gateは、次の未確定条件を出発点とする。
  - INT8とFP16の両方、またはFP4を1 TOPS以上で実行可能。
  - 専用メモリ16 GB以上。
  - 理論メモリ帯域250 GB/s以上。
  - 同一アーキテクチャの製品が十分に普及していること。例外判断には根拠を記録する。
- 上記は対応候補を選ぶ条件であり、kernel binary互換性やmodel起動時の空きmemory判定とは分離する。
- project lifecycleは `supported`、`experimental`、`planned`、`unsupported` を使用する。
- 根拠は `vendor-supported`、`project-verified`、`unverified` を別軸で記録する。
- 初期候補:
  - RDNA2: exact `gfx1030`〜`gfx1036`、配布候補 `gfx10-3-generic`。
  - RDNA4: exact `gfx1200`、`gfx1201`、配布候補 `gfx12-generic`。
  - CDNA3: exact `gfx942`。FP8 fast pathではgeneric targetを使用しない。
- 将来候補としてRDNA3、RDNA3.5、MI50、CDNA1/2/4/5、CPU、NVIDIA等の他社acceleratorを`planned`として管理する。
- NVIDIA等の将来backendでも、marketing architectureだけで分類しない。
  - 例: Turing GTX 16とRTX 20はともに`sm_75`だが、Tensor Coreの有無を別capabilityとして扱う。
- 詳細は `docs/compatibility/gpu.md` と `docs/compatibility/amd-gpu.md` を正とする。

## ソフトウェア互換性とtoolchain

- 主開発環境はUbuntu 24.04とする。
- Ubuntu 26.04等は、クラウド環境で必要になった時点で別の検証済みtupleとして追加する。
- OS、kernel、ROCm、compiler、GPU targetを独立した範囲で保証せず、組み合わせ単位で状態を記録する。
- 初期toolchain:
  - Rust edition 2024。
  - MSRV Rust 1.85.0。
  - 開発用Rust 1.97.1。`rust-toolchain.toml`で固定する。
  - Cargo resolver 3、applicationとして`Cargo.lock`をcommitする。
  - C++17。
  - ROCm 7.14.0同梱の`amdclang++`とLLVMを使用する。
  - CMake 3.21以上。
  - H0〜H2 host CI用Python 3.12.10。直接依存versionは`ci/requirements-host.txt`で固定する。
- ROCm compiler/runtime/libraryは同一releaseへ揃える。
- local開発環境の有効化とfail-closedな確認は`docs/development/environment.md`および`scripts/dev`を正本とする。
- toolchainで実装上の問題が確認された場合は、互換性文書とこの計画を更新して変更する。
- 詳細は `docs/compatibility/software.md` を正とする。

## RustとC++/HIPの境界

- Rust workspaceをtop-level buildとprocessの主体にする。
- C++/HIP backendはCMakeでstatic libraryとしてbuildし、Cargo build scriptからlinkする。
- Rust上位層は`Backend` traitでbackendを抽象化する。MVPでは静的登録のみとし、安定した外部plugin ABIは作らない。
- Rust/C++境界はHIP専用のversioned C ABIとする。
  - opaque context、queue、buffer、event handleを使用する。
  - C++例外とRust panicを境界越しに伝播させない。
  - 固定幅整数、status code、caller-owned error sinkを使用する。
  - 拡張可能structには`struct_size`とversionを持たせる。
- TensorはRust所有のBuffer viewとし、allocationを直接所有しない。
- Bufferはopaque C++ allocationをRust `Arc`で管理する。
- 非同期submitはcompletion eventと使用buffer参照を保持し、完了前の解放を禁止する。
- Backend registry、semantic Op registry、HIP Kernel registryの三層に分離する。
- 詳細は `docs/architecture/runtime.md` を正とする。

## モデル取得と再現性

- Hugging Face modelはbranch/tag名だけで固定しない。
- model lockに次を記録する。
  - `repo_id`と`repo_type`。
  - 要求したrevision。
  - 解決済みの完全なcommit SHA。
  - 実際に使用する全ファイルのSHA-256とsize。
  - Hub blob IDとLFS OID。
  - license、model card、base model、変換系列。
- 量子化や形式変換を行ったmodelでは、変換元lock fingerprint、変換toolのrepositoryとcommit、引数・設定、実行環境、出力SHA-256を記録する。
- weight shardだけでなく、index、config、tokenizer、chat template、generation/processor configもlock対象とする。
- model aliasは特定のlock fingerprintへ結び付ける。
- 詳細は `docs/models/model-lock.md` を正とする。

## 外部実装の参照とコード流用

- llama.cppとvLLMから、実装前に技術上の要点を抽出する。
- local `reference/` の公式origin、version、完全commit SHA、取得状態は[参照source固定マニフェスト](../references/source-lock.md)を正とし、固定sourceの参照範囲と採用判断は[推論engine参照](../references/inference-engines.md)へ記録する。
- 2026-08-02の追加調査対象からはLMDeployとKTransformersだけを正式なlocal参照sourceとして採用する。MLC LLM、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIは今回未採用とし、採用予定に置かない。
- vLLM等からコードを直接流用しない。参照sourceの表現を実装へ持ち込まないようreader記録とimplementation phaseを分離するが、別subagentの使用は必須にしない。
- llama.cppからの直接流用は許可するが、トップレベルLICENSEへの曖昧な追記だけで済ませない。
- 直接流用する場合は、copyright/license noticeを保持し、upstream URL、完全commit SHA、upstream/local path、hash、exact/adapted/ported区分、変更内容、import commitを記録する。
- 実際にimportした時点で`THIRD_PARTY_NOTICES.md`を作成・更新し、コピー先から参照できるようにする。
- 詳細は `docs/provenance/README.md` を正とする。

## 開発・最適化の優先順位

- 多くのモデル・GPUへ共通適用できる変更から行う。
  1. 異種モデル・異種GPUで共通。
  2. 異種モデル共通、またはGPU共通。
  3. モデルアーキテクチャ内共通、またはGPUアーキテクチャ内共通。
  4. モデル固有、またはGPU固有。
- baseline kernelとsemantic op contractを先に固定し、最適化kernelはregistryへ追加する。
- 対応、動作、native高速path、変換、emulationを同じ意味で使わない。
- 性能計測ではInferenceXと比較可能な種類のデータを収集し、グラフを作成する。
- 単一リクエストのllama.cpp比較では、model revision、llama.cpp commit、GPU target、数値型、入力長、出力長を記録する。

## 正しさ確認方針とCI・テスト

### 決定済み

- モデルアーキテクチャ共通の変更は、原則としてその系列の最小modelから確認する。
- 量子化評価にはtop-1一致率、KLD、modelの一部を切り出したBF16比誤差を使用する。
- CPUで数時間以上を要する確認は極力避ける。
- 2の冪や特定サイズだけでなく、非整列値と境界前後を含める。

### CI・テスト方針

- GPU kernel、GPU-scale GEMM/attention、full model推論、GPU性能をCPU emulationで証明しない。
- CPU CIはhost contract、極小NumPy oracle、HIP compile-onlyに限定し、full modelのdownload・load・forward・generationを行わない。
- compile成功、実GPU実行、数値一致、model slice、end-to-end、性能を別々の証拠として記録する。
- GPU不在時のCPU fallback、timeout、crash、test未収集を成功扱いにしない。
- public forkの`pull_request`からself-hosted GPU runnerを直接使用しない。GPU実行はdefault branch上の信頼済みworkflowと隔離・使い捨て可能なrunnerを基本とする。
- PR必須CPU workflowは15分以内を初期目標とし、実GPUtestは変更影響と明示tupleに基づいて選択する。
- 数値toleranceはop、入力範囲、accumulation dtype、出力dtypeごとに根拠を持って定義し、全op共通の緩い既定値を置かない。
- performanceに影響する境界`B`は実GPUで`B-1/B/B+1`を測定し、backend、dispatch、fallback、artifact hashとともに記録する。初期G3 smokeは`255/256/257`を含める。
- HIP/runtime/backend/dispatch/native buildのdraft developmentでは、影響箇所のfocused host/GPU testを行う。integrationまたはreleaseでGPUの正しさを主張するときだけ、意味上のbuild identityが一致するG0/G1/G2/P0等の該当evidenceをfail-closedに集約する。
- H0/H1/H2はintegration/releaseで選択された場合の並列rowとし、`host-required`へ集約する。draftへ全rowを一律要求しない。required workflowはp95 10分以内、hard上限15分とする。
- 初期GPU evidenceは専用local hostのexact `gfx1030` 1台と`gfx1201` 1台で直列実行し、public fork PRからGPU runnerを直接使わない。
- 詳細な方針と実装順序は[CI・テスト方針策定計画](active/2026/08/1-10/ci-test-strategy.md)を参照する。

## 開発運用上の決定

- Gitで追跡するのはsource、文書、小さなfixture、manifest、hash、summaryとし、model、binary、raw trace/profile、large model slice、生成物は追跡しない。詳細は[repository hygiene方針](../development/repository-hygiene.md)を正本とする。
- registered worktreeは有効な並行開発・evidence用途を持つため、個数だけで作業やpushを停止しない。9個以上、
  missing/prunable registration、clean・unlocked・非mainで14日超の候補は整理を促す警告とし、自動削除しない。
- 無人での進行を優先しつつsecret exposureを最小化する。専用local hostでは`homelab1`への`NOPASSWD: ALL`を意図的なtrade-offとして受容し、main agentがtask scope内で`sudo -n`を使う。恒久方針は[credential方針](../security/credentials.md)を正本とする。
- 現在の既定profileは`trusted-solo-development`とし、外部contribution実行時とrelease時の要件を分離する。使っていないprofileの要件は現在の開発をblockしない。
- main agentは調査・実装を直接行える。subagentは並列化、分離、専門的contextに効果がある場合だけ任意に使い、subagent利用や特定の`codex exec`実行方式を完了条件にしない。
- 各Phaseは受入条件、検証、plan/history closeout後に、そのPhaseだけを必要最小限のcommitへ整理してcurrent GitHub branchへ
  pushする。次Phaseの変更を同じcommitへ混ぜず、共有済み履歴の書換えやforce pushを行わない。
- 作業単位は独立してreview・rollbackしやすい範囲とするが、細分化、immutable identity、独立review、全matrix実行を各draft checkpointの完了条件にしない。draft、integration、release/push、docs-onlyのlaneと実行手順は`AGENTS.md`を正本とする。
- AIがhard gate、独立review必須化、広範/GPU再実行、security boundary、reuse制限、blocking stage、作業単位の追加分割、immutable evidence拡張を提案する場合、明示的なユーザー承認まではorigin・scope・cost・expiryを持つnonblocking proposalとして扱う。
- 受入条件は作業単位の開始時に固定する。実際のcorrectness/security defectはblockerにできるが、review中に新しく作られたprocess要件は承認なしに遡及適用しない。
- source/build input、toolchain、model lock、artifact digestから成る意味上のidentityをGit commit identityと区別する。docs-only変更で意味上のidentityが変わらないことを確認できればcode/GPU evidenceを再利用し、docs-only closeoutやfresh独立reviewを行わない。
- deploy可能なservice/runtimeが対象にある場合だけ適用後smoke/healthを要求する。独立した適用先がないlibrary、tool、文書はpush可能である。
- 同じ単位の2回reject、review時間が実装時間超過、1時間以上の機能進捗停止、検証・文書が30%超、見積り1.5倍超、gate/受入条件変更のいずれかで、新規review・検証を停止し、ユーザーへ報告して計画を見直す。

## 開発順序

1. プロジェクト開始準備。
   - 既存の問題・不明点を確認する。
   - GPU、software、runtime、provenance、model lock、API profileを文書化する。
   - CI・テスト方針を検討し、文書化する。
   - repository hygieneとcredential方針を確定し、governance baselineを機能codeより先に公開する。
2. repository skeletonと初期CI・test harnessを構築する。
   - H0〜H2のhost CIを導入する。
   - ROCm固定toolchainによるH3 compile-onlyを導入する。
   - 利用可能な実機に合わせてG0 GPU preflightを導入する。
3. Qwen/Qwen3.5-4B BF16を動作させる。
   - RDNA4。
   - RDNA2。
   - visionとMTPは含めない。
   - 実装と同時にG1 kernel/ABI、G2 model slice、G3 end-to-endを追加する。
4. Phase 4として、Qwen3.5-2B、9Bでも同一実装が動作することを確認する。
5. Phase 5として、エンジンレベルの性能ベンチマーク／ベースラインを取得する。
   - ダイレクトなエンジンでprefill/TTFT、decode TPOT/token/s、end-to-end latency、peak VRAMを測定する。
   - model-resident lifecycleとrequest-local stateを分離し、model-resident lifecycleは再利用する。
   - OpenAI-compatible API実装後はservice/API overheadも追加測定する。
6. Phase 6として、KV memory方式を選定し、OpenAI-compatible Chat Completions profile v1を実装する。
   - AMD GPU上のvAttention再現性と簡単なstandalone HIP PoCを最優先で実施する。
   - PoC結果からvAttentionまたはPaged Attentionを選び、上位serviceから内部表現を隠すKV契約を先に確定する。
   - 選択したKV方式の上にgeneration service、sampling、HTTP/SSE、cancellationを実装する。
7. CI/CDをnightly、release、compatibility、performanceへ拡張する。
8. BF16を最適化する。
   - RDNA2。
   - RDNA4。
9. 実行エンジンの構造最適化を行う。
   - dtype非依存のdecode graph/segment実行とhost同期削減。
   - BF16 M=1 GEMV/MMVF、Qwen3.5 GDN、MLP・RMSNorm等のprofile-driven fusion。
   - prefill providerを実shapeで再評価し、llama.cppとの差をhost、launch、kernel、memoryへ再分解する。
10. model本体のFP8 W8A8に対応する。
   - RDNA4。
   - RDNA2。
11. FP8/BF16実装をCDNA3へ移植する。
12. MI300X単体でCDNA3実機確認を行う。
13. モデル非依存のprepared execution制御へ移行する。
   - `QwenExecutionCore`に残るprepared semantic cache、same-stream segment owner、completion集約を
     model-neutralなexecution plan/transition層へ抽出する。
   - Qwen固有graph、attention preprocess、GDN、model stateはadapter側に残し、共通制御へ混ぜない。
   - model-neutral fixtureと既存Qwen pathで、別model adapterが同じ高速な実行骨格を利用できることを確認する。
14. google/gemma-4-12Bへ対応する。
15. Weight NVFP4へ対応する。
16. KV cache FP8/NVFP4へ対応する。
17. MTP、visionへ対応する。
18. Gemma4またはQwen3.5のMoEへ対応する。
19. 残りの初期バージョン機能を実装する。
20. 人間がREADMEを整備し、発表する。

Phase 12のMI300Xを管理できない期間は、Phase番号と依存関係を維持したままPhase 13以降のlocal-only workを
先行できる。現在のGitHub CI不整合は製品Phaseを繰り下げず、Phase 12待機中のremediation subphase
`Phase 12R`としてPhase 13より先に修復する。実行順序、停止条件、Gemma 4後の共通RDNA性能bridge、枯渇防止tailは
[Phase 12待機中のローカル先行実行キュー](active/2026/08/11-20/phase12-wait-local-forward-queue.md)を正とする。
Phase 12は`ready`のまま残し、再開時にlatest mainからexact `gfx942` candidateを再buildする。

## Phase概要と進捗

詳細な作業単位、checkpoint、commit・tree・report digest、試行錯誤、review結果は各phaseの
計画・history・Git履歴を正とし、ここでは目的、到達点、現在の順序だけを管理する。

### Phase 0: 開始準備（完了）

- product、GPU/software互換性、runtime、model lock、provenance、API、CI/test、
  repository hygiene、credentialの初期方針を確定した。
- governance baselineと固定参照sourceを公開し、reset前の履歴を保持する方針を確定した。

### Phase 1: repository skeletonとhost CI（完了）

- Rust workspace、CMakeでbuildするC++/HIP backend、versioned C ABI、H0〜H2 host CI、
  fail-closedなschema・matrix・aggregateを構築した。
- 実行方法は[host build and test entry points](../development/testing.md)、CI方針は
  [CI・テスト方針策定計画](active/2026/08/1-10/ci-test-strategy.md)を正とする。

### Phase 2: HIP compile・model-free GPU path（完了）

- ROCm固定toolchainによるH3 compile-onlyと、canonical `gfx1030`/`gfx1201`でのG0、
  model-free G1実行経路を構築した。
- H3 required昇格の観測はnonblocking follow-upであり、後続phaseを停止しない。
- 詳細は[Phase 2 archive](archive/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)を正とする。

### Phase 3: Qwen3.5-4B BF16 text generation（完了）

- 固定Qwen3.5-4B revisionをBF16、単一GPU、batch 1、text-onlyでloadし、typed frontend、
  model/weight plan、semantic op、request-local state、prefill/decode、CLI text generationへ接続した。
- host contract、real-weight G2、end-to-end G3、canonical `gfx1030`/`gfx1201`の数値・dispatch・
  fallbackなし・health/cleanupを確認し、2026-08-11に完了した。
- 正本は[Phase 3 archive](archive/2026/08/1-10/phase3-qwen35-4b-bf16.md)、詳細履歴は
  [Phase 3 history](../history/2026/08/1-10/phase3-qwen35-4b-bf16.md)とする。

### Phase 4: Qwen3.5-2B・9B互換性確認（完了）

- 4Bのshape-driven load/graph/executionを複製せず2B/9Bへ適用し、9B untied LM headと
  allocation前の空きVRAM preflightを含む単一pathを完成した。
- 2B lock fingerprintは`sha256:304e19f8b8ef78bab1848a6cfb46ac619a8ca5c8fd052cac1c43fc3f4d6dcdb3`、
  9Bは`sha256:2d2bc642540e97d4681f8c66140e09f305f487476bb9fe238ca82a298febf893`である。
- canonical両GPUのreal-weight G2、fixed/Unicode/255/256/257/max/stop、4B G3全回帰、
  health/cleanupをPASSし、integration worktree tree `16282f9014186042580fc927e47750947216d694`で
  2026-08-11に完了した。
- 正本は[Phase 4 archive](archive/2026/08/11-20/qwen35-2b-9b-compatibility.md)、詳細は
  [Phase 4 history](../history/2026/08/11-20/qwen35-2b-9b-compatibility.md)とする。

### Phase 5: エンジン性能baseline（完了）

- model-resident lifecycleを再利用するdirect engineでcanonical V620/R9700のTTFT、TPOT、token/s、
  end-to-end、peak VRAMを取得し、direct 22/22、render 2/2、固定llama.cpp dedicated wrapper 14/14を
  fail-closed aggregateまでPASSした。
- sLLMの4B TTFTはexact-token llama.cpp wrapperよりV620で49.4〜278.5倍、R9700で31.4〜742.1倍長く、
  最初の共通最適化候補はprefill GEMM、operator dispatch、同期削減である。255/256/257の大きなcliffは
  観測しなかった。
- 最適化iterationは4B short-oddと32/32短縮case、warmup 1 + measured 3を基本とし、canonical long、
  model scaling、llama比較はintegration/release/nightlyまたは意味変更時へ限定する。
- 計画とevidenceは[Phase 5 archive](archive/2026/08/11-20/engine-performance-baseline.md)、詳細値は
  [Phase 5 history](../history/2026/08/11-20/engine-performance-baseline.md)を正とする。

### Phase 6: KV memory方式選定とOpenAI-compatible Chat Completions profile v1（完了）

- A0のAMD vAttention standalone HIP PoCとA1の方式比較・最小production pathはcanonical V620 `gfx1030`と
  R9700 `gfx1201`でPASSした。初期方式はHIP VMM virtual-contiguous KV（vAttention型）、storageは
  FP16 token-major `[capacity, kv_heads, head_dim]`である。
- vAttentionとFlashAttentionは排他的ではない。vAttentionは連続virtual addressを通常のKV pointerとして
  kernelへ渡すため、contiguous-KV FlashAttention系kernelを同じaddressingで利用できる。AMD実測は
  upstream FA2/CKそのものではなくFA2-style proxyであり、FA3/4はNVIDIA対象の設計比較に限定した。
- A1のQ=37/KV=1025ではvAttention proxyがpaged proxyよりV620で約17.0%、R9700で約31.3%短いp50を示し、
  通常contiguous allocationとは概ね同等だった。実測と制約は
  [KV memory decision](../architecture/kv-memory.md)を正とする。
- 選択方式を隠すKV allocation/layout/lease境界を確定した。その上にshared generation service、sampling、
  strict JSON、models/chat completions、non-stream/SSE、usage、backpressure、disconnect cancellationを実装する。
- llama.cppのsampler/testはprovenance付き直接reuse候補とし、vLLM、SGLang、TensorRT-LLM、
  LMDeploy、Microsoft vAttentionはno-copyの設計・検証参考とする。
- A2は2026-08-13に完了した。current OpenAPIとの差は`ModelIdsShared`への`gpt-5.5`追加だけでprofile pinは
  不変である。llama.cpp再利用unitと4 engine facts-only readerをexact commit/pathへ固定し、
  `sllm-server`のHTTP runtime closureを132 package、308 edgeとしてoffline policyへ固定した。
- A3は2026-08-14に完了した。CLIとserverで共有するtransport非依存generation service、必要時だけの
  full-vocabulary logits readback、temperature/top-p/presence/frequency sampling、incremental UTF-8 stop、
  usage/finish reason/cancellation境界を実装した。temperature 0は既存device argmax経路を維持し、V620の
  focused full-model smokeでもgreedy/non-greedyのHIP-only実行とcleanupを確認した。
- A4/A5は2026-08-14に一つの実装バッチとして完了した。strict DTO/model registry、non-stream/SSE、
  bounded FIFO/event channel、timeout/shutdown/disconnect cancellation、error envelopeを実装し、A3へvisible
  delta sinkを追加した。raw HTTP/SSE matrixとOpenAI Python SDK 3.0.0のmodels/non-stream/stream smokeをPASSした。
  実GPU full-model service、HIP/VRAM/process cleanup、differential、service overheadはA6へ分離して完了した。
- A6は2026-08-14に完了した。pinned profile fixture、provenance付きllama.cpp adapted test、
  vLLM/SGLang/llama.cpp differential、production Qwen backend/serverを追加し、OpenAI Python client 2.44.0と
  raw HTTP/SSEをPASSした。canonical V620/R9700でQwen3.5-4B non-stream/stream/stop/disconnect、
  1023/1024/1025 KV capacity、physical commitment、HIP/VRAM/process cleanupをfail-closedに確認した。
- initial serverはstable GPU UUIDを一台だけ可視化し、論理device 0を使う。Phase 5 `chat-hello`と同一の
  13 input/17 outputをservice経由で測り、JSON residualはV620 0.788 ms、R9700 0.533 msだった。
- 計画は[Phase 6 archive](archive/2026/08/11-20/openai-chat-completions-v1.md)、詳細は
  [Phase 6 history](../history/2026/08/11-20/openai-chat-completions-v1.md)を正とする。

### Phase 7: CI/CDの定期・互換性・性能・release拡張（完了）

- daily、weekly、releaseのversioned profileとGitHub Actions lifecycle workflowを追加し、
  GitHub-hosted host/compile jobとtrusted self-hosted GPU jobを分離した。GIMPS終了後の運用変更により、
  dailyはcanonical V620 `gfx1030`とR9700 `gfx1201`の短い観測を選択する。
- compatibility compileは`gfx1030`〜`gfx1036`、`gfx1200`、`gfx1201`、`gfx942`の10 exact targetを
  独立rowとし、Code Objectのexact targetとROCm rootを検査する。compile-only結果を実機互換性に
  読み替えない。
- daily/weekly artifactは30日、release evidenceは90日保持する。性能は観測値であり、
  承認済み閾値がないためhard gateにしない。
- Phase 12Rでself-hosted GPU jobのautomatic triggerを廃止した。profile定義はlocal daily/weekly/release controllerの正本として
  維持し、GitHub lifecycle workflowはmanual control-planeだけを受け付ける。
- 計画は[Phase 7 archive](archive/2026/08/11-20/phase7-ci-cd-expansion.md)、詳細は
  [Phase 7 history](../history/2026/08/11-20/phase7-ci-cd-expansion.md)を正とする。

### Phase 8: BF16単一リクエスト最適化（完了）

- frozen float64 numerical oracle、BF16 Matmul registry、M>1 tiled/M=1 reduction kernel、R9700大形状の
  target-specific hipBLAS、vAttention上のFA2-style online softmax、prepared semantic cacheを実装した。
  baseline kernel、checkpoint weight layout、opaque KV owner、virtual-contiguous FP16 K/Vは維持した。
- canonical V620 `gfx1030` / R9700 `gfx1201`でMatmul 17 case、attention 16 case、4B O2 7 case、
  2B/9B spot、fixed llama.cpp、OpenAI non-stream/SSEをPASSした。全runはHIP-only、fallbackなし、
  ECC 0、terminal cleanup 0である。
- short-oddのPhase 5比はV620でTTFT 7.550→1.110秒、prefill 2.253→15.391 tok/s、E2E
  25.838→9.656秒、R9700で2.878→0.683秒、5.921→25.108 tok/s、12.445→8.881秒となった。
  llama.cppとの差はE2Eで約20.4/26.9倍残り、dispatch graph、fusion、decode GEMV、host orchestrationを
  後続性能backlogとする。
- 今回のattention実装はユーザー指示どおりFA2-styleだけとした。RDNA4向けFA3-likeは同じvAttention・
  数値・shape契約でFA2と比較する非blockingな将来タスクとして維持し、Phase 9をblockしない。
- 計画は[Phase 8 archive](archive/2026/08/11-20/phase8-bf16-optimization.md)、詳細は
  [Phase 8 history](../history/2026/08/11-20/phase8-bf16-optimization.md)を正とする。

### Phase 9: 実行エンジン構造最適化（完了）

- true completion境界の固定1 ms sleepをadaptive pollへ置換し、same-stream submission ownerをKV/terminal
  segmentまで保持した。境界後の各opはblocking waitせずterminal queryだけを行い、transactional state、
  vAttention owner、prepared semantic planを維持した。
- kernel-only/hipBLAS混在HIP Graph PoCを両targetでPASSした。productionはrequestごとのinstantiateを避け、
  KV appendを明示cutとするsegment pathを選択した。full production graph replayは残差共通最適化backlogとする。
- llama.cpp固定commitからboundedにadaptしたBF16 MMVF v3をM=1へ採用し、V620だけQwen GDN recurrent
  stateをwave-coalesced配置にした。R9700のM>1 prefillはhipBLAS GEMMEx、V620はtiled16を選択した。
- 4B short-odd中央値はV620でTTFT 0.306秒、E2E 0.855秒、prefill 56.91、decode 29.69 tok/s、R9700で
  0.051秒、0.490秒、377.46、37.20 tok/sとなった。Phase 8比E2Eは約11.3倍/18.2倍、固定llama.cppとの差は
  約1.81倍/1.48倍まで縮小した。2B/9B spot、32/32、Matmul 17 case、OpenAI non-stream/SSEもPASSした。
- 残差は主にmemory-bound M=1 matvecとhost launchであり、full attentionは支配要因ではない。次はPhase 10の
  model本体FP8 W8A8へ進み、production graph/command-listとMLP fusionはfresh profile後の共通backlogとする。
- 詳細は[Phase 9 archive](archive/2026/08/11-20/phase9-engine-structural-optimization.md)、判断と実績は
  [Phase 9 history](../history/2026/08/11-20/phase9-engine-structural-optimization.md)を正とする。

### Phase 10: model本体FP8 W8A8（完了）

- verified Qwen3.5-4B BF16 lockからper-output-row scale付きの再現可能なOCP E4M3FN sidecarを作り、
  weight/activation FP8、FP32 accumulation、BF16 outputのlinear contractを実装した。
- RDNA4 exact `gfx1201`はnative FP8 provider、RDNA2 exact `gfx1030`はW8A8 emulationと明示BF16 conversionを
  別providerとして実装した。RDNA2 pathをnative FP8と表記しない。
- 公式Qwen3.5 FP8は27B以上が中心であるため、小型第三者checkpointを基準にせず、公式27B FP8はPhase 12の
  追加interop spotとする。
- R9700 32/32でresident VRAMをBF16比約42.4%削減したが、prefill/decode/E2Eは低下したためdefaultへ
  昇格しない。native FP8はopt-in、V620 emulationはcorrectness-only、V620 `converted-bf16`は明示pathとする。
- 詳細は[Phase 10 archive](archive/2026/08/11-20/phase10-fp8-w8a8.md)を正とする。

### Phase 11: FP8/BF16のCDNA3移植（完了）

- exact `gfx942`、wave64へBF16 kernel/providerを移植し、OCP E4M3FN model storageをVRAM load時に
  E4M3FNUZへ数値変換してhipBLASLt FNUZ providerへ渡す。generic targetやraw byte reinterpretを使わない。
- MI300XではVMMなしが想定されるため、opaque KV/attention契約を維持する`contiguous-resident` providerを
  追加する。VMM対応targetのvAttentionは維持し、Paged Attentionへの選定変更やsilent fallbackは行わない。
- exact gfx942 compile/link、全byte FNUZ oracle、wave64 BF16 provider、capability-selected contiguous-resident KV、
  production `native-fnuz` graph/service、MI300X dry-run runnerを完成した。実機PASSと性能値はPhase 12で取得する。
- 詳細は[Phase 11 archive](archive/2026/08/11-20/phase11-cdna3-port.md)を正とする。

### Phase 12: Hot Aisle MI300X単体実機確認（計画済み）

- Hot AisleのMI300X x1 Small VMを用い、exact `gfx942`のBF16/FNUZ FP8、wave64、contiguous-resident KV、
  4B/9B model、service、性能、llama.cpp比較をfail-closedに確認する。
- 192 GB HBM3の一台で現行single GPU/batch 1の検証には十分である。multi-GPU、Infinity Fabric、RCCL/RDMA、
  bare-metal固有挙動、別CDNA3 SKUは証拠範囲外とする。
- 利用時間はclean candidateで合計10〜12 GPU時間、現実的な上限16時間とする。2〜3時間のpreflightと
  6〜8時間のintegration/performanceを別sessionにし、必要な場合だけ追加4時間を別日に使う。
- 詳細は[Phase 12 active plan](active/2026/08/11-20/phase12-mi300x-validation.md)を正とする。

### Phase 12R: CI portability repairとlocal/remote verification整理（完了）

- GitHub-hosted CIをtracked checkoutだけで完結するH0〜H3 portability/compile laneとして修復し、実GPU、model、
  llama.cpp実体比較、性能はtrusted local laneへ分離する。
- current H0のC++ format、Git管理外llama reference依存、Rust dependency closure driftと、public-runtime/RMSNorm H3の
  hipBLAS/hipBLASLt link不足、self-hosted GPU push pendingを修正する。
- Phase 12の完了やMI300X PASSを意味せず、既存Phase 13〜20は繰り下げない。
- tracked-only H0/H1/H2、hipBLAS/hipBLASLtを含むH3 link、manual self-hosted trigger、registry-driven local entrypointを
  2026-08-15に実装した。core/public-runtime/RMSNorm H3のcanonical両targetはcompile-only PASSであり、GPU PASSではない。
- 詳細は[Phase 12R archive](archive/2026/08/11-20/phase12r-ci-portability-repair.md)を正とする。

### Phase 13: モデル非依存prepared execution制御（完了）

- Phase 9で`QwenExecutionCore`内へ実装したprepared operation再利用、same-stream segment owner、
  completion集約、transactional publication境界を、model固有graphから独立した共通execution層へ移す。
- Qwen3.5は最初のadapterとして同じ意味・性能pathを維持する。model-neutral fixtureでQwen symbolや固定shapeを
  参照せず同じ制御を利用できることを確認し、Phase 14のGemma 4 adapterが再実装せず利用できる境界を固定する。
- 2026-08-15に共通plan/transition/cache/segment/boundary/audit/transaction、Qwen adapter移行、host fixture、
  canonical RDNA2/RDNA4の2B/4B smoke、OpenAI service smokeを完了した。4B short-oddのsubmission/kernelはPhase 9から
  増加せず、fallback/cleanupも維持した。
- 詳細は[Phase 13 archive](archive/2026/08/11-20/phase13-model-neutral-execution-control.md)を正とする。

### Phase 14: google/gemma-4-12B Dense text-only（完了）

- Phase 13のmodel-neutral executorへ二つ目のproduction model adapterとして接続し、Qwen固有のwait/cache制御を
  複製しない。
- immutable model lock、architecture inventory、weight/graph、固有semantic op、R9700 full model、V620 bounded
  evidence、CLI/OpenAI serviceを順に実装する。
- 2026-08-15にsource lock/frontend/weight graph、Gemma semantic provider、両RDNA exact targetのoperator/real-weight
  slice、R9700 exact `gfx1201`の48-layer full modelを完了した。CLIとOpenAI non-stream/SSE、stop、disconnect recovery、
  既定sampling、同一resident連続requestをshared generation pathへ接続し、fallbackなしとcleanup 0を確認した。
- R9700 direct-engine profileでは`3/17`と`32/32`を同一resident uploadで計測した。最終host evidenceはH0 `513/513`、
  H1 `421/421`、H2 `36/36` PASSであり、1回のintegration reviewとfindingだけのfocused re-reviewを完了した。
- 詳細は[Phase 14 archive](archive/2026/08/11-20/phase14-gemma4-dense.md)を正とする。

### Phase 15: Weight NVFP4（完了）

- Phase 14後のQwen/Gemma fresh profileと共通RDNA2/RDNA4最適化bridgeは2026-08-15に完了した。Gemmaの
  request workspace/prepared semantic再利用と、両model/両GPU共通のM=1 BF16 matvec streaming loadを採用した。
  R9700ではGemma `3/17`と`32/32`がfresh baseline比`+3.07%/+3.89%`、Qwen3.5-2B short-oddが
  `+1.62%`で、V620にも明確な退行はなかった。attention非支配のためFA3-likeは除外した。
- weight-only NVFP4としてvalue、block scale、tensor scale、packingをencoding/sidecar/loader/providerへ保持し、
  native、packed-dequant、emulation、converted pathを区別する。
- 2026-08-15にTransformer Engine v2.18をformat sourceとして固定し、E2M1、K-axis block 16 OCP E4M3FN scale、
  FP32 tensor scaleのconverter/sidecar/loaderと、BF16 activationを使うpacked-dequant providerを実装した。
  Qwen3.5-2B full sidecarは186 tensor、772,236,184 byteでbyte-identical再生成を確認した。
- Qwen full-modelは両exact targetでtop-1 3/3一致したが最大KLD `0.2637523`が既定budget `0.05`を超えた。
  Gemma 4-12B layer 0 gate sliceもtop-1 2/3だったため、thresholdを緩めず両targetとも
  `correctness-only opt-in`とした。providerはnative FP4ではない。
- V620ではresident 3,763,686,080 byteから1,790,406,056 byteへ52.43%削減した。CLIとOpenAI
  non-stream/SSE/stop/Unicode/連続request/disconnect/cleanupをR9700で通した。
- 詳細は[Phase 15 archive](archive/2026/08/11-20/phase15-weight-nvfp4.md)を正とする。

## 現在の状態と次の作業

- Phase 15 Weight NVFP4まで完了した。hardware検証順ではPhase 12のMI300X実機確認が残るが、2026-08-15の
  ユーザー明示指示により現在のgoalはPhase 15完了を終端とする。
- MI300Xを管理できない期間はPhase 12を`ready`で保持し、local forward queueに従ってPhase 12R、Phase 13、
  Phase 14、共通RDNA性能bridge、Phase 15の順に先行する。Phase 12RでGitHub host/compileとtrusted local GPUの
  verification境界を修復し、Phase 13で共通execution制御を抽出し、Phase 14でGemma 4 production pathを完了した。
  共通RDNA性能bridgeとPhase 15まで完了した。Phase 16以降は別の明示指示で再開する。
- Phase 9のdtype非依存completion/segment骨格とtarget別BF16 providerを再利用し、Phase 10でFP8 encoding、
  sidecar/loader、native/emulation/conversion providerを追加した。Phase 13でモデル非依存層へ抽出し、
  Phase 15開始前にもfresh profileで
  memory-bound matvec、production graph/command-list、MLP fusionの優先順位を再確認する。RDNA4 FA3-likeは
  attentionが支配要因になった時の非blocking follow-upとして別管理する。
- Phase 11でMI300XのVMMなしに備える`contiguous-resident` KV providerを実装した。Phase 12はHot Aisle MI300X x1
  Small VMを標準10〜12 GPU時間、上限16時間の二回構成で検証する。単一VMの性能証拠をmulti-GPU、bare metal、
  MI300A/MI325Xへ一般化しない。
- Phase 7完了後のAPI拡張として、opt-in Qwen thinking、`reasoning_content`と最終`content`の
  non-stream/SSE分離、strictと分けたOpenWebUI `max_tokens`互換profileを追加した。互換範囲は
  [OpenAI compatibility profile](../api/openai-compatibility.md)を正とする。
- H3 required昇格はnon-requiredのまま観測し、20回以上・7日以上の条件を満たした時点で
  昇格だけをreviewする。Phase 12Rでcurrent H3 linkとtrigger境界を修復したがrequiredへ昇格していない。
- 現行運用はtrusted-solo-developmentとし、draft/integration/release/docs-onlyの扱いは
  `AGENTS.md`を正とする。過去のcheckpoint固有運用を現行gateへ読み替えない。

## 未解決事項

- AMD consumer RDNA2を含む各exact gfx targetの実機検証範囲。
- ROCm 7.14.0とHWE kernel 6.17のmixed V620/R9700 tupleについて、長時間安定性と正式な
  compatibility statusを判断できるだけの実測が揃っていない。
- 追加op・shape・入力範囲の数値toleranceと、複数のO2/O3履歴run・分散・再現性が揃った後に定める
  性能回帰threshold。
- resource gateの1 TOPS、16 GB、帯域の定義と例外承認基準。
- Infinity Fabric、他RDMA protocol、KV永続化の詳細設計。
- 量子化形式ごとのlayout、scale granularity、accumulator、fallback表。
- sudo以外の既存平文credentialの失効・rotationとsecret managerへの移行状況。
