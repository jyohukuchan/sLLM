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
- 無人での進行を優先しつつsecret exposureを最小化する。専用local hostでは`homelab1`への`NOPASSWD: ALL`を意図的なtrade-offとして受容し、main agentがtask scope内で`sudo -n`を使う。恒久方針は[credential方針](../security/credentials.md)を正本とする。
- 現在の既定profileは`trusted-solo-development`とし、外部contribution実行時とrelease時の要件を分離する。使っていないprofileの要件は現在の開発をblockしない。
- main agentは調査・実装を直接行える。subagentは並列化、分離、専門的contextに効果がある場合だけ任意に使い、subagent利用や特定の`codex exec`実行方式を完了条件にしない。
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
4. Qwen3.5-2B、9Bでも同一実装が動作することを確認する。
5. OpenAI-compatible Chat Completions profile v1を実装する。
6. CI/CDをnightly、release、compatibility、performanceへ拡張する。
7. BF16を最適化する。
   - RDNA2。
   - RDNA4。
8. model本体のFP8 W8A8に対応する。
   - RDNA4。
   - RDNA2。
9. FP8/BF16実装をCDNA3へ移植する。
10. MI300X単体でCDNA3実機確認を行う。
11. google/gemma-4-12Bへ対応する。
12. Weight NVFP4へ対応する。
13. KV cache FP8/NVFP4へ対応する。
14. MTP、visionへ対応する。
15. Gemma4またはQwen3.5のMoEへ対応する。
16. 残りの初期バージョン機能を実装する。
17. 人間がREADMEを整備し、発表する。

## 当面のPhase 1概要

- 目的は、推論機能を実装する前に、Rust主体のtop-level build、CMakeでbuildするC++/HIP backend、両者を結ぶversioned C ABI、テストとCIの拡張可能な雛形を確立すること。
- repository skeletonとして、source、公開header、contract/reference/fixture/API test、CI schema・matrix・共通scriptの配置と責務を定め、最小構成をbuild・link・testできる状態にする。実モデルのload・推論、数値kernel、性能最適化は含めない。
- CIの正本としてtest result schema、compatibility tuple schema、suite registry、host matrix、path-to-suite mappingを作り、未知test、未知marker、期待test 0件、result欠落を成功扱いしない。
- H0静的検証、H1 host contract、H2 tiny NumPy oracleを独立して並列実行し、`host-required`へfail-closedで集約する。GPU処理をCPUで模倣するtestやfull modelのdownload・実行は追加しない。
- Phase 1の完了条件は、clean checkoutからdocumented commandで雛形を再現buildでき、H0〜H2と集約checkが時間予算内で成功し、意図的なlint・test・schema・収集件数の異常を確実にfailureへできること。
- ROCm 7.14.0によるH3 compile-only、専用local hostでのGPU evidenceとG0 preflightはPhase 1の完了後に行う。
- Phase 1は完了した。実装path、再現command、CPU-only境界は[host build and test entry points](../development/testing.md)を正とする。

## 当面のPhase 2前半

- 対象はROCm 7.14.0固定toolchain、exact `gfx1030`/`gfx1201`のH3 compile-only、専用local hostのGPU evidenceとG0、model-freeの最小GPU実行経路までとする。数値op、model load・推論、性能最適化、対応GPUの昇格は含めない。
- 最初に、未構築のG0/G1/G2/G4/P0をH3自身へ要求するbootstrap循環を解消し、変更が実際に触れる範囲だけを同一immutable SHAで要求する段階的gateをCI正本へ同期する。
- H3はnon-requiredで開始し、20回以上かつ7日以上の観測はrequired昇格だけの条件とする。観測中もG0とmodel-free GPU pathの実装を並行し、後続開発を停止しない。
- 初期runtime evidenceは専用local hostのcanonical `gfx1030` 1台と`gfx1201` 1台を直列実行し、完全tuple、artifact identity、CPU fallback未使用、実行前後のdevice healthを記録する。
- model-free最小経路は`Cargo -> sllm-hip -> versioned C ABI -> native HIP -> GPU`を通してallocation、copy、単一diagnostic kernel、completion、copy-back、解放を検証する。推論opまたはGPU対応済みの証拠にはしない。
- 詳細な作業単位、受入条件、evidence、rollback境界は[Phase 2 H3・G0・model-free GPU path計画](archive/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)を正とする。

## Phase 3概要

- 目的は、Qwen/Qwen3.5-4Bの固定revisionをBF16、単一GPU、batch 1、text-onlyでloadし、CLIからprefill/decodeしてtextを生成し、model-free G1からmodel slice G2とend-to-end G3へ進むこと。
- 対象は、完全なmodel lock、固定参照実装のreader記録、config・safetensorsのfail-closedな読み込み、最初のsemantic opであるRMSNorm、public HIP実行経路、synthetic RMSNormのsemantic G1、real-weight sliceのG2、短いRMSNorm P0 smokeとする。
- 最初のRMSNormは、BF16 raw weight / BF16 activation、FP32 accumulation、連続した最終次元をbaseline contractとする。Qwen3.5 HF weightは実効scaleを`1 + raw_weight`とするoffset-one variantとして扱う。数値toleranceはNumPy oracleとcanonical GPUで測定してop・shape・入力範囲ごとに固定し、一律の緩い既定値を置かない。
- RMSNorm公開opはleading dimensionをrowへflattenし、prepared planを再利用可能・同一planのin-flightを1件に限定する。既存generic completionを再利用し、nonfinite BF16 payloadはhost scanせずIEEE classificationを伝播する。baseline kernelは`N <= 4096`、wave32、256 threads、fallbackなしとし、additiveなexecute/dispatch情報ABIと専用RMSNorm H3 artifactを使う。
- semantic G1のcompiler actionはparent-issued exact actionとし、認証済みclient observationと最終sealed compiler環境を分離する。固定recipe全件の一回だけのissue/consume、結果delivery ACK、resource/header/device/runtime input closureのlive再検証をrequired evidenceとする。
- BF16出力の初期acceptance budgetは`tolerance_id=rmsnorm-bf16-f32-output-v1`、`atol=0.0078125`、`rtol=0.015625`としてGPU結果の前に固定する。finite caseは`abs(actual-reference) <= atol + rtol*abs(reference)`、NaN/Infはclassificationを比較し、同一candidateの結果を見て閾値を拡大しない。
- Phase 3 text-onlyのrequest stateは、linear-attention convolution stateをBF16 `[3, 8192]` row-major、recurrent stateをF32 `[32, 128, 128]` row-major、full-attention KVをFP16 `[4, T, 256]`として型とlifetimeを分離する。prefillとdecodeは同じstate transitionを使い、request間で共有しない。
- CLIの生成停止集合は固定metadataとchat templateから`[248046, 248044]`（`<|im_end|>`を先、`<|endoftext|>`を後）とする。prompt tokenは停止判定せず、新規生成tokenをargmax直後に判定し、stop token自身をvisible outputへ含めない。停止token IDと理由はreportへ保持する。
- text frontendは`tokenizers =0.21.4`をdefault featureなし・`onig` featureだけで固定し、HTTP、progressbar、`esaxx_fast`を無効にする。chat templateは任意Jinjaを実行せず、locked template hashを検証したtyped Qwen3.5 text-only rendererとする。停止policyをversioned lock/schema/APIへ反映し、全依存version/checksumはroot `Cargo.lock`、license/feature/MSRVはtracked dependency policyとoffline validatorへ固定してからfrontendを実装する。
- G2は固定model lockから実行時に抽出するRMSNorm weightと、独立生成したactivationを使う。raw model、weight slice、traceはGit管理せず、source fingerprint、tensor名、offset・shape、抽出recipe、artifact SHA-256を記録する。
- Phase 3のintegration/release candidateでは、意味上の同一build identityに対してH0〜H3、canonical `gfx1030`/`gfx1201`のG0、private diagnostic G1、semantic G1、G2、必要なP0、数値oracle、fallbackなし、実行後healthをfail-closedに集約する。個別draft checkpointでは影響範囲のfocused testだけを行う。G2/P0完了だけでfull model推論、一般GPU対応または性能最適化済みとは主張しない。
- Stage Aではattention、RoPE、MLP、KV/state、tokenizer/chat template実行、prefill/decode、CLI生成、G3をまだ含めないが、これらはPhase 3全体の後続Stageで実装する。性能最適化とP1はPhase 3に含めない。
- H3 required昇格の20回・7日観測は引き続き並行follow-upであり、Phase 3の開始条件・完了条件にはしない。
- Phase 3全体の作業単位、受入条件、evidence、rollback境界は[Phase 3 Qwen3.5-4B BF16 text生成計画](active/2026/08/1-10/phase3-qwen35-4b-bf16.md)を正とし、完了した最初のmodel-bound数値経路は[Stage A model lock・RMSNorm・G2計画](archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)に記録する。

## 現在の状態と次の作業

- 2026-08-10に開発policyをresetした。この節で以下に残す各checkpointのstrict H0〜H3、fresh独立review、docs-only closeout、同一Git SHAへの再実行、予測上端+1時間hard stopは当時の実績・運用記録であり、現在の完了条件ではない。以後のB5以降はdraftでfocused testを行い、まとまったintegration candidateで影響するhost/GPU evidenceと1回のreviewを取得し、最終releaseでclean immutable identityと累積reviewを固定する。
- 現在: Phase 3 Stage Aをcommit `ac2baa3a0734d0894353ba180259d979da5a831e`（tree `4e43a9c42c9aa2dfa6a6d438610fa54c4e482d10`）で完了した。同一immutable identityに対しH0 305/305、H1 151/151、H2 35/35、base/RMSNorm H3、canonical `gfx1030`/`gfx1201`のpre/post G0、private G1、sealed-controller semantic RMSNorm G1、real-weight G2、P0、全aggregate、health/process cleanupが`PASS`した。G2は固定Qwen3.5-4B cacheのlocked 5120-byte sliceを使用して各target 6 HIP dispatch、P0は各target 130 HIP dispatchをfallbackなしで記録し、P0はperformance threshold・最適化・他engine比較を主張しない`review_required` dispositionとした。独立review 9はfull 5-file差分と最終evidenceをhigh/medium/low 0件で`PASS`した。A5運用負債もhost-only evidence planner、H0 316/316、独立review `PASS`により解消した。Phase 3全体とfull text生成/G3は未完了である。Stage B B0〜B7bは機能実装とbatch integrationを完了した。B5/B6はweight planとoffline CLI、B7aは既存transfer ABIへlowerするgeneric readback、B7bはverified cache/load-planからgeneric uploadへ接続する16 MiB bounded bridgeである。integration candidate `806b524d5fac31cf11c866bf7bba095c0dc35e9d`（tree `ad3b252b4de671096d3e75b10b1e9a2a93a74092`）はclean host integrationとcanonical V620/R9700のB7a/B7b exact evidenceをPASSし、review 1のidentity固定指摘も閉じた。これはStage B integration根拠でありPhase 3 release昇格ではない。次はStage Cのbaseline semantic ops/kernelsである。H3 required昇格の20回・7日観測は引き続きnonblocking follow-upとする。
- Stage C最初のunit C1a contiguous BF16 copy/residual addとC1b single-GPU embedding gatherはdirty draft実装、focused dual-GPU evidence、workspace/native hostのbatch回帰まで完了した。C1aはcopy bit-exact、add FP32加算後BF16 RNEを1/3/17/255/256/257/2560要素で各14 dispatchしてPASSした。C1bはI32 token IDの範囲をdispatch前に検査し、同じhidden境界のsynthetic caseと固定Qwen3.5-4B embedding先頭3 rowを各8 dispatchしてbyte exact、fallbackなし、cleanupゼロをPASSした。C1b report SHA-256は`gfx1030=c6eac50609229bfd9c51cc7867430a26b1a2cf8d262c9b1ebd37ecd28a2bb601`、`gfx1201=4e9e1b3788a97b4658e1aab115e7762bb1a17860b995298b5277bfa9653a0d36`。release-laneのfixed H3 manifest再固定と昇格はまだ行わず、既存RMSNormを再利用してC2へ進んだ。
- C2 BF16 linear/SiLU gated MLPのdirty draftを完了した。checkpoint storageのweight `[N,K]`を直接使うmatmul core contractと、独立`SiluMul` semantic opを固定した。SiLU multiplyはpublic elementwise ABI/native HIP/Rust bridgeへ実装し、copy/addと合わせた1/3/17/255/256/257/2560要素のcanonical dual-GPU各21 dispatchがexact、fallbackなし、cleanupゼロでPASSした。report SHA-256は`gfx1030=c120f3edf272f880ca9af4479b9092cbf14b60747a18ac8ca7dbb08b2bd03e66`、`gfx1201=3bb7e41d9081d7ec7f5ff3889988df73e422cda0724a2e32d596d8be32364978`。matmulはversioned public C ABI、native HIP kernel、Rust owner/lifetime wrapper、owned execution bridge、bounded G1 runnerへ接続した。M/K/Nの1/3/17と各軸の255/256/257を13 non-Cartesian caseで覆い、K昇順の独立FP32 scalar oracleからBF16 RNEへのbit完全一致、signed zero/subnormal/large finite/NaN/Inf、各target 13 dispatch、fallbackなし、cleanupゼロをcanonical V620 `gfx1030`とR9700 `gfx1201`でPASSした。matmul report SHA-256は`gfx1030=bed009b40b589f491a8eaf98aec900a451ef71c340a3f7f9170c506e67d90f8d`、`gfx1201=338ca4ef18780bc3b4b40fa1d460bdc9a209c2229cb74b4bce697b228b14d6b7`。これはdirty focused evidenceでありreleaseへは昇格せず、次はC3 RoPE/full attentionへ進む。
- C3 separated readerでQ/gateのhead-wise packing、Q/K RMSNorm、text-only partial NeoX RoPE、FP16 KV state、causal GQA softmax、distinct sigmoid output gateの順序と分割を固定した。C3a0では`attention_bias=false`、dropout `0`、output gate有効、max position 262144、cache有効、RoPE type/theta/partial factor/interleaved/sectionsをmodel lockのtyped contractへ追加し、raw configと同値であることをRust/Python双方でfail-closedに検証する。現行lock fingerprintは`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、tiny fixtureは`sha256:1065d7427922cf9f2e37c18e18b7434b7c3e63cda0dc75236273050170292415`へ更新した。旧fingerprintに結合したStage A/B result/report digestと履歴記述はhistorical identityとして保持し、新identityへ読み替えない。再実行用active schema/matrix/runnerは現行lockへ同期した。C3a1はhead-wise Q/gate split、Q/K head RMSNorm、text-only partial NeoX RoPEをpublic HIP実行経路まで実装し、canonical V620/R9700の各8 dispatchでgate byte exact、Q/K最大1 BF16 ULP、特殊値分類・符号一致、fallbackなし、cleanupゼロをPASSした。report SHA-256は`gfx1030=fa7e7d69b5ad3062e0de21445207877ee580525eba393c47f8f3619df48e713c`、`gfx1201=d7aa13d5634046e6f7f5cfaae0d66c523d44c914c9553e5f77abf57420d9ab7c`である。C3a2はK/V別FP16 `[4,capacity,256]` state、transactional BF16 append、opaque snapshot/lifetime、成功時だけのlength/generation publicationをcore/native/Rustへ実装し、private copy-only readbackで19 bounded caseの全storageを独立oracleと厳密照合した。canonical V620/R9700はspecial/rounding/untouched領域、stale/in-flight/timeout/drop cancel、fallbackなし、cleanupゼロをPASSし、report SHA-256は`gfx1030=f72683685d523c0dc7ef9c75fdeb69d9318f366c5980d1a4e6f2e07a7a41015e`、`gfx1201=daa995c5341a32517872632126f4d06052310f51852daa47baf503b889b87ae8`である。C3bはBF16 Qとcommitted FP16 K/Vをcausal GQA stable softmaxへ接続し、canonical V620/R9700のprefill/decode各10 caseでordered scalar oracle、causal visibility、GQA mapping、fallbackなし、cleanupゼロをPASSした。report SHA-256は`gfx1030=b368cefea6c812193b34ca30abb1d2eb98d80d6842d0145096c6948857b62a64`、`gfx1201=4d3fad33ee4c60c878d9860eb6c698b60e2fd22b72b7bcd873a1f6cd76c13b3b`である。C3cは独立`SigmoidMul`、BF16 `[M,16,256]`、FP32 sigmoid multiply、最終BF16 RNE、zero-copy `[M,4096]` `o_proj` handoffを実装し、canonical V620/R9700のM `1/3/17/255/256/257`各6 caseでbit/classification、SiLUとの差、fallbackなし、cleanupゼロをPASSした。report SHA-256は`gfx1030=5764442af2716327291050ea64365b27077340036c79ac46422f795d52aa46b7`、`gfx1201=3b1a9eaa1bff2b9508fe19543f2d4f7e4cc7f8bee33ab61309495dfc1058ddd9`である。Stage C integration reviewのMedium 5/Low 2を修正し、指摘限定再reviewは全7件PASSした。修正後hostはPython 461件＋1086 subtest、workspace/all-target Rust、HIP 83件、native CTest 2件、clippy、Rust 1.85 closure、format/matrix/diffをPASSした。次はimmutable integration candidateへ固定し、同一commitのexact dual-GPU evidenceを再取得する。
- C1〜C3 integration機能candidateはcommit `a408f4e0b83dfe8a3f9e639ec2b4d7d54c7da5f4`、tree `476b30959cf852bb05b3870e8ced18747ce7467a`へ固定し、local-development H0 340/340、H1 275/307、H2 35/42、exact target別7 runnerのcanonical V620/R9700実機再検証を全てPASSした。各binaryは8 HIP bundleが対象targetのみを含み、全14 reportはHIP backend、fallbackなし、cleanupゼロである。full attentionはM=1以外の9 caseで非一様softmaxを観測し、修正後report SHA-256は`gfx1030=208b780316caadf69e2749a4458a29e91b06b16bc9feec54b6c6b18527607ba0`、`gfx1201=8188860d213efd89b72248b8e2ca7a7889129f0e327743dfdaace4bca95ae301`となった。実行後の全GPU use/VRAMとKFD残留は0である。host Pythonは3.12.3で固定3.12.10 strict evidenceではないため、これはpush可能なC1〜C3 integration checkpointでありmerge-ready/Stage C完了とは扱わない。次はStage C残りのlinear attentionとfinal norm/tied output/logitsである。
- C4 linear attentionのdirty draftをcore state contract、public C ABI、native HIP kernel、Rust owned-execution bridgeまで実装した。BF16 convolution state `[3,8192]`とF32 recurrent state `[32,128,128]`を二重bufferで保持し、成功完了時だけslot/length/generationを公開し、cancel/dropでは旧stateを維持する。projectionと`out_proj`は既存Matmulを再利用する。workspace/all-target Rust、clippy、format、ABI、native fake-HIP CTest、exact `gfx1030`/`gfx1201`全native buildをPASSした。focused実機runnerはcanonical V620/R9700でM=3 prefillからM=1 decodeへ同じstateを継続し、異なる4 tap、head/dimension別入力、Q/K repeat factor 2、token順FP32 recurrenceを独立scalar oracleへ全出力照合して両targetとも最大0 BF16 ULP、fallbackなし、終了後全GPU use/VRAM 0だった。これはdirty focused evidenceであり、次はfinal RMSNorm、tied output projection、logits/argmaxを実装してStage C integrationへまとめる。
- C5 final outputのdirty draftを完了した。final RMSNormとtied output projectionは既存RMSNorm/Matmulおよび`EmbeddingAndTiedOutput` weight planを再利用し、新規greedy Argmaxだけをcore/native/Rust/public HIP pathへ追加した。BF16 `[M,V]`、I32 `[M]`、1 row 256 thread、最小index tie、`+Inf`、all `-Inf`、signed zero、NaN `-1` sentinel、`V <= 1048576`、非alias、fallback禁止を固定し、主reviewで実kernel symbolとmetadataを一致させた。focused hostのcore 62 unit＋既存integration 18件、HIP 83件、all-target check/clippy、ABI、format/diff、native CTest 3/3をPASSした。各target 9 bundleはexact `gfx1030`または`gfx1201`のみを含み、canonical V620/R9700でM `1/3/17`、V `1/3/17/255/256/257/248320`の9 caseを独立oracleへ照合してPASSした。fallbackなし、終了後全GPU use/VRAM 0、runner残留なしである。C4/C5はdirty focused evidenceのため、次は両unitを含むStage C integration candidateを固定し、累積review 1回と同一identityのhost/exact dual-GPU evidenceを取得する。
- C4/C5を含むStage C累積reviewはHigh 0、Medium 1、Low 4だった。Qwen固定shape・consumerとowned bindingを結ぶ型付きhost contractを追加し、embeddingとtied projectionの同一BF16 `[248320,2560]` buffer・byte view共有、final RMSNorm→Matmul→Argmaxのzero-copy handoffを検査する。Argmax timing、C4 exact dispatch metadata、atomic monotonic generation、backend owner cleanup前のsingle-flight再開も修正した。修正後はcore 65件、HIP 86件、fake-HIP Argmax public C ABI timing、all-target check、affected clippy、format/diffをPASSし、5指摘だけのfocused再reviewも全件PASSした。次はこのtreeをimmutable Stage C integration candidateへ固定し、同一identityのhost/exact dual-GPU evidenceを取得する。
- Stage C integration code candidate `c7ecbd5a6bf64ff55d234a1102d60fc5b1bd6eb0`、tree `e177a0860d4a4e4fdabf4d2dd928e0bfaf422243`は同一clean identityのworkspace/all-target host検証、exact `gfx1030`/`gfx1201` build、全18 binaryのtarget-only bundle検査、canonical V620/R9700の各9 runner、実行後health/process cleanupをPASSした。C4は両targetで最大0 BF16 ULP、C5はmodel実vocabを含む9 caseをPASSし、fallbackはない。Stage C integrationを完了し、docs-only記録はこのsemantic/build identityへ結合する。
- 2026-08-08時点では今後数週間を単独のtrusted development期間とし、外部PR、未review code、第三者binaryを専用local/GPU hostで実行しない。期間中は悪意ある同一UID process、fork PRによるrunner侵害、永続runner上の敵対codeを防ぐrepository内custom capsuleの完成をPhase 3の前提から外し、将来のrunner隔離作業へ延期する。local/GPU実行はmaintainerが十分に確認したcodeと明示commandに限定し、secret・Docker socketを渡さず、可能な範囲のcontainer隔離、network遮断、timeout・resource上限、process cleanup、実行前後GPU health、candidate/artifact identityは維持する。外部contributorのcodeを実行する前、または複数の信頼境界を持つ運用へ移る前に、ephemeral VM/JIT runnerまたはjob後reimageをsecurity boundaryとする設計を必須で再開する。
- host capsule独立reviewの修復中に中断された`ci/tools/execution_capsule.py`の部分変更（現行file SHA-256 `44801a0832756f0e6966cb7b23bd25653d6cfba91d6816ededf7f9fe63239ac9`）は未検証であり、host/GPU evidenceに使用しない。2026-08-08のユーザー判断により、直前版SHA-256 `a1464bcf5ae1407aaa91b984a6782af0a06d574447afcd052864440f7faedbba`へのbyte-for-byte復元は打ち切り、A0 security hardeningの部分変更を放棄する。Phase 3のsemantic実装はdirect testと標準containerによる`local-development`確認で再開し、immutable host evidenceが必要になる前に、現行部分変更を継承しない最小のtrusted-development baselineを新規作成・reviewして新identityを固定する。
- 2026-08-08にPhase 3 Stage Aを再開し、2026-08-09に完了した。当時は各工程へ予測上端+1時間の中断上限と8時間以下の分割を適用したが、この一律運用は2026-08-10のpolicy resetで廃止した。Stage Aの実績は[archive済みStage A計画](archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)を正とする。

### 2026-08-10以前の詳細経緯（履歴）

以下は当時の判断とevidenceを保存する履歴であり、記載されたfresh review、closeout、hard-stop clock、同一Git SHA gateを現行作業へ再適用しない。
- 再開後A1 fresh reviewは2026-08-08 15:50:49 JSTに完了し、direct host回帰はPASSしたが、semantic G1のnonfinite raw-scale case欠落と、保存証拠から数値比較を独立再計算できないraw-response非保持の2件で`FAIL`とした。A2はこの2件だけを2〜4時間（中断上限5時間）で修復し、延期済みcustom capsule hardeningは継承しない。
- A2実装は2026-08-08 16:18:23 JSTに完了し、raw-scale非有限値3件とbounded raw-response/sidecarの保存・identity結合・offline再計算を追加した。implementerのdirect host回帰はPASSした。A2の20:51:54 JST中断上限を維持したままfresh独立reviewを実行し、そのPASSまではA2完了またはG1修復完了とは扱わない。
- A2 fresh独立reviewは2026-08-08 16:34 JSTに`FAIL`とした。元の2 blockerは閉じたが、semantic G1のreport/artifact/aggregate schemaにschema単体で未知keyを受理するopen objectが計23箇所あり、focused回帰もCMake compiler-broker client認証で1件失敗した。A2の20:51:54 JST中断上限を延長せず、この2件の修復と再reviewを続ける。
- A2継続修復は2026-08-08 16:33:46 JSTに`workspace-write` sandboxで正常起動した。schema閉鎖性と既存CMake broker回帰だけを対象とし、20:51:54 JST中断上限は維持する。
- 同`workspace-write` sessionは最初のread後にbubblewrap `RTM_NEWADDR`が連続再発したため、編集・test前に中断した。2026-08-08 16:36:27 JSTから同一scopeの`danger-full-access` fallbackで継続し、20:51:54 JST中断上限はリセットしない。
- A2継続修復は2026-08-08 17:05 JSTに完了し、semantic G1 report/artifact/aggregate schemaの全object境界の閉鎖、stdlib validatorと未知key拒否回帰を追加した。implementerのdirect host回帰はPASSしたが、CMake回帰をtest側のfd=30固定で通した変更が有効なfd=10〜29の拒否を隠していないかを未解決論点とする。同17:05 JSTから30分〜1時間予測のfresh独立再reviewを開始し、A2のhard中断時刻20:51:54 JSTは維持する。
- 同再reviewの`read-only` sandboxはmain plan初回read後にbubblewrap `RTM_NEWADDR`で停止したため、検証前に中断した。2026-08-08 17:06:46 JSTから同じ非変更・禁止事項の`danger-full-access` transport fallbackで継続し、review予測とA2 hard中断時刻はリセットしない。
- A2再reviewは2026-08-08 17:22 JSTに`FAIL`で完了した。A1由来の2件、schema object閉鎖、全direct host回帰はPASSしたが、production CMakeが有効な継承fd 10〜29を拒否してtestのfd=30固定がこれを隠すことと、stdlib validatorが`properties`/`patternProperties`重複および`prefixItems: [false]`でDraft 2020-12と不一致になることをblockerとした。同17:22 JSTからこの2件だけを30分〜1時間で修復し、30分〜1時間の再reviewへ進む。A2 hard中断時刻20:51:54 JSTは維持する。
- 同修復の`workspace-write` sessionはreadを完了したが、`apply_patch`がbubblewrap `RTM_NEWADDR`で編集前に2回失敗したため中断した。2026-08-08 17:26:42 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測とA2 hard中断時刻はリセットしない。
- A2最終修復は2026-08-08 17:41 JSTに完了した。production CMakeの全canonical fd>=3受理と明示境界test、stdlib validatorのDraft 2020-12 parity修復、影響するRMSNorm H3 source identity更新を行い、implementerのsemantic G1 90件、reference 26件、Rust 116件、Python/C++/全validator回帰はPASSした。同17:41 JSTから30分〜1時間予測の最終fresh独立reviewを開始し、A2 hard中断時刻20:51:54 JSTは維持する。
- 同最終reviewの`read-only` sandboxは初回read後にbubblewrap `RTM_NEWADDR`が再発したため、検証前に中断した。2026-08-08 17:43:33 JSTから同じ非変更条件の`danger-full-access` transport fallbackで継続し、review予測とA2 hard中断時刻はリセットしない。
- A2最終reviewは2026-08-08 17:58 JSTに機能gate 1〜5をすべてPASSしたが、17:43:46 JSTにmain agentがmain planとStage A active planへ同期したstatus行をrepair scope違反と解釈し、scope gateだけを`FAIL`とした。この編集は17:41:18 JST完了のrepair implementer変更ではなく、AGENTS.mdがmain agentへ割り当てる計画同期である。同17:58 JSTからこのprovenance・ownershipだけを10〜20分予測で独立再判定し、A2 hard中断時刻20:51:54 JSTは維持する。
- A2 scope再判定は2026-08-08 18:01 JSTに`PASS`で完了し、sole P1を撤回した。機能gate 1〜5の最終独立PASSと合わせ、A2は15:51:54開始から約2時間9分で完了した。同18:01:23 JSTからA3a G2 host contract・実行経路を開始し、予測2〜4時間、hard中断時刻23:01:23 JSTとする。A3aはhost-only schema/matrix/case-set/negative test、slice extractor、dedicated binary/runner/aggregateを対象とし、canonical GPU G2実行は後続A5まで行わない。
- A3aの`workspace-write` sessionは正本read後にbubblewrap `RTM_NEWADDR`が連続したため編集前に中断し、2026-08-08 18:04:49 JSTから同一scopeの`danger-full-access` transport fallbackで継続する。A3aの開始時刻、予測、23:01:23 JST hard中断時刻はリセットしない。
- A3a実装は2026-08-08 18:40 JSTに完了した。closed G2 schema/matrix、synthetic-only slice extractor、dedicated public RMSNorm evidence binary、artifact builder、runner、aggregate、host suite登録とnegative testを追加し、candidate/prerequisite/report/artifact identityをcanonicalに結合した。implementerのG2 14件、semantic G1 90件、H3 19件、model-lock 21件、reference 26件、Rust workspace 116件、Python/C++/全validator回帰はPASSし、非GPU hostのrelease binaryは`HIP unavailable`で失敗した。同18:40 JSTから30分〜1時間予測のfresh独立reviewを開始し、A3a hard中断時刻23:01:23 JSTは維持する。
- 同reviewの`read-only` sandboxは正本read後にbubblewrap `RTM_NEWADDR`でshell監査を開始できず、検証前に中断した。2026-08-08 18:42:22 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測とA3a hard中断時刻はリセットしない。
- A3a fresh独立reviewは2026-08-08 19:03 JSTに`FAIL`で完了した。matrix/case-set、slice recipe、public runtime専用binary、host stub fail、host-only登録と広範回帰はPASSしたが、stdlib schema parity、runtime slice実bytesのSHA結合、binary/source/sidecar実identity、candidate/case/prerequisite/health/error/nonzero hashのfail-closed検証に4 blockerを確認した。同19:03:46 JSTからこの4件だけを1〜2時間で修復し、その後30分〜1時間のfresh再reviewを行う。A3a hard中断時刻23:01:23 JSTは維持する。
- 同修復の`workspace-write` sandboxはread前にbubblewrap `RTM_NEWADDR`で停止し、変更はなかった。2026-08-08 19:06:10 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測とA3a hard中断時刻はリセットしない。
- A3aの4 blocker修復は2026-08-08 19:36 JSTに完了した。stdlib schema parity、実5120-byte slice、実binary/sidecar/source/build source-set、candidate/prerequisite/report/aggregate/health/case evidenceのfail-closed結合を追加し、A5 parser/oracle以前の数値PASS昇格を禁止した。implementerのG2 22件、semantic G1 90件、model-lock 21件、reference 26件、G1 29件、H3関連28件と広域host回帰はPASSした。同19:36 JSTから30分〜1時間予測のfresh独立再reviewを開始し、A3a hard中断時刻23:01:23 JSTは維持する。
- 同再reviewの`read-only` sandboxはrepository access前にbubblewrap `RTM_NEWADDR`で停止し、変更はなかった。2026-08-08 19:40 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測とA3a hard中断時刻はリセットしない。
- A3a fresh独立再reviewは2026-08-08 19:56 JSTに`FAIL`で完了した。nullable schema parity、実slice結合、focused/broad host回帰、strict Git CLI、A5以前のPASS禁止はPASSしたが、canonical名へ改名したG1 binaryをG2 artifactとして受理することと、build source-set/path registrationが実Cargo/native入力11件を漏らすことをblockerとした。同19:56 JSTからこの2件だけを1〜2時間で修復し、30分〜1時間のfresh再reviewへ進む。A3a hard中断時刻23:01:23 JSTは維持する。
- 同修復の`workspace-write` sandboxはrepository access前にbubblewrap `RTM_NEWADDR`で停止し、変更はなかった。2026-08-08 20:00 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測とA3a hard中断時刻はリセットしない。
- 最初の`danger-full-access` processは自processを別subagentと誤認して待機へ入り、repositoryを編集していないことを確認して2026-08-08 20:05 JSTに中断した。同20:05:46 JSTからnested processを禁止したdirect implementerで再開し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3aの2 blocker修復は2026-08-08 20:37 JSTに完了した。43 fileのcanonical build-input manifestとbuild-time生成identity、専用G2 binaryのno-HIP identity queryを追加し、builder・validator・runnerでsource-setを独立再計算してG1/H3/任意実行file、symlink、非regular file、identity不一致を拒否する。G2 focused 26件、locked offline Cargo build/check、format、matrix/contracts、実binary identity、1行query、5120-byte memfd、host HIP unavailable nonzero、G1回帰、`git diff --check`はPASSした。H3 3件は共有`build.rs`に対する既存dirty stateのhash/parser期待値不整合として残り、独立再reviewで帰属を再確認する。同20:38 JSTから30分〜1時間予測のfresh独立再reviewを開始し、A3a hard中断時刻23:01:23 JSTは維持する。
- 同fresh再reviewの`read-only` sandboxは`pwd`を含む全commandがbubblewrap `RTM_NEWADDR`でprocess開始前に停止し、file read・test・変更はなかった。2026-08-08 20:40 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a fresh独立再reviewは2026-08-08 21:04 JSTに`FAIL`で完了した。改名実G1拒否、43/43 source closure・Cargo rebuild/path登録、実debug G2のexact identity queryとhost fail、G2 26件、G1 134件、model-lock 21件、reference 32件、Rust workspace、C++ host staticはPASSしたが、任意Python executable/最小C ELFのidentity偽造受理、query helperの非canonical空白/改行許容、A3a変更後のH3 hash 2件・rerun parser 1件をblockerとした。H3合同回帰が内部で`rocm-smi` health照会を開始したため直ちに中断し、GPU kernel/model実行はない。同21:05 JSTから3系統だけを45〜75分で修復し、20〜40分のfresh再reviewを行う。23:01:23 JST hard中断時刻は延長しない。
- 同修復の`workspace-write` sandboxは正本とdirty baselineをread後、`echo`や`/tmp`を含む全commandがbubblewrap `RTM_NEWADDR`でprocess開始前に停止し、編集・testはなかった。2026-08-08 21:13 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a最終blocker修復は2026-08-08 21:30 JSTに完了した。fixed locked/offline Cargo buildとcanonical builder outputのbyte identityへstaged artifactを結合し、exact 1-line queryを共通化した。任意Python executable、最小C ELF、改名G1、file/sidecar/source-set/query負例をbuilder・validator・runnerで拒否し、H3 `build.rs` hash/source-setとrerun parser互換を同期した。G2 29件、G2 contracts、safe H3 static、Rust 116件、reference 26件、model-lock 21件、G1 23件、formatと`git diff --check`はPASSした。fresh debug G2 SHA-256は`5f1c1f37cb64b24362c79889010e897936cf2ccc155e1981ad6c9affff1350f3`、host通常実行は`HIP unavailable`でexit 1だった。同21:31 JSTから20〜40分予測のfresh独立再reviewを開始し、23:01:23 JST hard中断時刻は維持する。
- 同fresh再reviewの`read-only` sandboxは全commandがbubblewrap `RTM_NEWADDR`でprocess開始前に停止し、file read・test・変更はなかった。2026-08-08 21:32 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a最終fresh独立再reviewは2026-08-08 21:47 JSTに`FAIL`で完了した。実fixed offline build、G2 29件、safe H3 static 27件、Rust 116件、C++ host static、43/43 source closure、matrix/path登録、exact queryと負例、偽造binary拒否、host fail-closedはPASSしたが、public `build_artifact()`直接呼び出しがowned buildを迂回できることと、ambient `CARGO_TARGET_DIR`でCargo実出力とbuilder返却pathが分離することをblockerとした。同21:47 JSTからこの2点だけを20〜35分で修復し、10〜20分のfocused再reviewを行う。A3aの23:01:23 JST hard中断時刻は延長しない。
- 同修復の`workspace-write` sandboxは正本read後もbubblewrap `RTM_NEWADDR`が再発し、code確認・編集前に2026-08-08 21:49 JSTに中断した。同21:49:49 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a builder ownership修復は2026-08-08 21:58 JSTに完了した。public `build_artifact()`へfresh owned buildを強制し、CLIはowned resultを一度だけmanifest化する。Cargo子processの`CARGO_TARGET_DIR`をrepo-local `target`へ固定し、caller copyとambient redirectを拒否・無効化した。G2 runner 16件、schema/slice/aggregate 16件、contracts/matrix、実ambient-target probe、Cargo build/query、host fail-closed、Rust/Python/diff checksはPASSした。同21:59 JSTから10〜20分予測のfresh独立再reviewを開始し、23:01:23 JST hard中断時刻は維持する。
- A3a focused独立再reviewは2026-08-08 22:06 JSTに`FAIL`で完了した。public APIのowned buildとambient target固定は静的に成立したが、module-level private helperが通常のPython callでowned buildを迂回できることをsole blockerとした。read-only transport再発でfocused test独立再実行は未完了である。同22:06 JSTからhelper完全除去だけを5〜10分で修正し、10〜15分のfocused再reviewを行う。23:01:23 JST hard中断時刻は延長しない。
- 同修正の`workspace-write` sandboxはmain plan read後にbubblewrap `RTM_NEWADDR`が再発し、編集前に2026-08-08 22:07 JSTに中断した。同時刻から同一scopeの`danger-full-access` transport fallbackで継続し、予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a helper完全除去修正は2026-08-08 22:13 JSTに完了した。module-level helperを削除し、public `build_artifact()`内でowned buildを一度だけ実行、CLIもpublic APIだけを一度呼ぶ。G2 focused 32件、contracts/matrix、実ambient build/query、host fail-closed、Rust/Python/diff checksはPASSした。同22:13 JSTから10〜15分予測の最終focused独立reviewを開始し、23:01:23 JST hard中断時刻は維持する。
- 同最終reviewの`read-only` sandboxはmain/active plan read後にbubblewrap `RTM_NEWADDR`が再発し、code/test確認前に2026-08-08 22:15 JSTに中断した。同時刻から同一非変更scopeの`danger-full-access` transport fallbackで継続し、予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a最終focused独立reviewは2026-08-08 22:22:15 JSTに`PASS`で完了した。module-level bypass/helper不在、public build/CLI one-build、copied binary拒否、ambient target固定、G2 32件、contracts/matrix、実Cargo query、host fail-closed、Python/Rust/diff checksを確認した。A3aは18:01:23開始から約4時間21分で予測上端を約21分超えたが、23:01:23 hard中断時刻前に完了した。canonical GPU数値実行はA5まで未実行である。
- A3b P0 host contract・実行経路は2026-08-08 22:23:00 JSTに開始する。予測2〜4時間、hard中断時刻は2026-08-09 03:23:00 JSTとする。P0 case-set、closed schema、runner、aggregate、versioned review disposition、host-only negative testを対象とし、GPU/model/cache/network/container、`rocm-smi`、deferred capsule、canonical P0数値実行は対象外とする。
- A3bの`workspace-write` sessionは必須文書read前にbubblewrap `RTM_NEWADDR`が連続し、変更・testなしで2026-08-08 22:24:48 JSTに安全停止した。同22:25:04 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、A3b開始時刻、予測、2026-08-09 03:23:00 JST hard中断時刻はリセットしない。
- A3b実装は2026-08-08 23:06:36 JSTに完了した。P0 matrix/review policy、7 closed schema、validator/runner/2-row aggregate、negative test、host suite/path/manifest登録を追加し、非整列、hidden 2560、B=256の255/256/257、5 warmup・21 measured、kernel/wall median・MAD、public RMSNorm source-setとidentityを固定した。P0 18件、隣接contract 20件、matrix/manifest、Python 87 file compile/static、diff checksはPASSした。A5 producer/parser、immutable artifact、canonical 2 GPU、実health/process、review disposition以前のnumeric PASSは拒否する。同23:08 JSTから30分〜1時間予測のfresh独立reviewを開始し、2026-08-09 03:23:00 JST hard中断時刻は維持する。
- 同fresh reviewの`read-only` sandboxは初回一括read後、9回連続のbubblewrap `RTM_NEWADDR`で分割read・差分監査・testを開始できず、reviewerはcode不合格ではなく監査基盤障害として2026-08-08 23:12:26 JSTに`FAIL`で安全停止した。実装者artifactからPASSを推定せず、同23:13 JSTから同一非変更scope・禁止事項の`danger-full-access` transport fallbackでfresh reviewを再実行する。review予測とA3b hard中断時刻2026-08-09 03:23:00 JSTはリセットしない。
- A3b fresh独立reviewは2026-08-08 23:23:51 JSTに`FAIL`で完了した。P0 18件、隣接static 28件、validator 3件、独立negative probe 15件、AST/diff checksはPASSし、A5以前のnumeric PASS拒否とproducer/parser延期は成立したが、P0 source-set/path ownershipがpublic RMSNorm経路のCargo/native入力を少なくとも34件漏らし、`op.rs`変更でもdigestが変わらないことをsole P1 blockerとした。同23:24 JSTからsource closureだけを30〜60分で修復し、20〜40分のfresh再reviewを行う。A5/GPU/model等へscopeを広げず、A3b hard中断時刻2026-08-09 03:23:00 JSTは延長しない。
- 同修復の`workspace-write` sandboxは必須文書read前にbubblewrap `RTM_NEWADDR`で停止し、変更・testなしで2026-08-08 23:26 JSTに安全停止した。同23:27 JSTから同一scope・禁止事項の`danger-full-access` transport fallbackで継続し、修復予測とA3b hard中断時刻はリセットしない。
- source closure修復は2026-08-08 23:43 JSTまでに45 source pathのversioned manifest、exact source-order/bytes digest、P0 path ownership、negative testを実装し、focused P0 21件、隣接G1/G2 26件、manifest/Python static/diff checksは実装中にPASSした。ただし担当processが禁止範囲の`rocm-smi`を起動した形跡を15分監視で検出したため、追加変更を止めて同23:43 JSTに中断した。processは終了済みでGPU kernel/model実行や成果物生成は確認していない。未完了の最終自己申告には依存せず、現在のworktreeを20〜40分予測のfresh独立reviewで再検証する。A3b hard中断時刻2026-08-09 03:23:00 JSTは延長しない。
- fresh read-only reviewは2026-08-08 23:43〜23:51 JSTに、manifest/validatorが中断直前にG2専用binary・build manifestを含む47 path案へ遷移したのにfocused testは45 pathかつ`src/bin`不在を要求する自己矛盾をblockerとして確定した。test起動前にbubblewrap `RTM_NEWADDR`が連続し、監視でscope外の`rocm-smi` processも検出したためreviewerを停止した。同23:52 JSTからP0 host contract + existing public pathの45 pathへ戻しG1/G2専用producerを除外する整合修正だけを予測20〜40分で行う。substep hard中断時刻は2026-08-09 01:32 JST、A3b全体は03:23 JSTのままとする。
- 同整合修正の`workspace-write` sandboxは正本read途中からbubblewrap `RTM_NEWADDR`が連続し、編集・test前に2026-08-08 23:54 JSTで停止した。同23:54 JSTから同一scopeの`danger-full-access` transport fallbackへ切り替える。substep/A3bの中断時刻はリセットしない。
- 45 path整合修正は2026-08-09 00:00:06 JSTに完了した。P0 manifest/validator/artifact schemaの3 fileだけを修正し、移行途中fieldとG2専用binary/build manifestをP0 identityから除外、45 pathのcanonical digest/schemaを一致させた。P0 21件、隣接G1/G2 static 26件、validator、Python各87 file、diff/ownership probeはPASSした。同00:01 JSTから15〜30分予測のfresh独立reviewを開始し、review hard中断時刻01:31 JST、A3b全体03:23 JSTを維持する。
- fresh reviewの`read-only` sandboxは指定file読了後のtest起動がbubblewrap `RTM_NEWADDR`で連続失敗し、出力更新も停止したため2026-08-09 00:06 JSTに監査基盤`FAIL`として中断した。同00:07 JSTから同一非変更scopeの`danger-full-access` transport fallbackで再実行し、review/A3b中断時刻はリセットしない。
- 訂正: 23:43/23:51監視で担当process/reviewerによるscope違反と記録した`rocm-smi`はPhase 3 processの子ではなく、作業開始前から別terminal配下で稼働するPID 16827の`watch -n 0.25 rocm-smi`が生成したprocessだった。両subagentが起動したという判定を撤回する。23:43停止後に見つかった47/45 path不整合は独立したcode findingとして修正・再reviewを継続し、所有外の既存watchは変更しない。
- A3b final fresh独立reviewは2026-08-09 00:15 JSTにblockerなしの`PASS`で完了した。P0 focused 21件、隣接G1/G2 static 26件、P0/matrix/JSON validator、Python compile/static各87 file、45 path・5代表source mutation・omission/reorder/path/digest・symlink/nonregular拒否を含む独立negative/temp-copy probe、`git diff --check`がPASSした。GPU/HIP runtime、model/cache/raw slice、network/container、deferred capsule、broad host suite、Rust/native build、commit/pushは未実行である。A3bは22:23:00開始から約1時間52分で予測内、03:23:00 hard中断時刻前に完了した。
- A4 immutable evidence用の最小baselineは2026-08-09 00:16:00 JSTに開始する。予測2〜4時間、hard中断時刻は2026-08-09 05:16:00 JSTとする。中断A0の未検証`execution_capsule.py`を継承・実行せず、trusted-development期間に必要な最小host evidence経路とcandidate identity境界をreviewして固定可能にする。GPU/model/cache/raw slice/network、canonical evidence、commit/pushは対象外とする。
- A4 read-only調査はbubblewrap `RTM_NEWADDR`反復と単純probe停止により変更なしで中断し、同一非変更scopeの`danger-full-access` fallbackを2026-08-09 00:29:10 JSTに完了した。現行host runnerはHEAD/indexにない未追跡`execution_capsule.py`と`process_containment.py`をimportし、network guardもcapsule markerを要求するためA5へ進めない。`run_host_suite.py`、`network_guard.py`、`test_fail_closed.py`のA0由来部分だけを外し、review済みpre-A0 direct runnerのregistered-command、network namespace、timeout/output/RSS/count、process-group cleanup、identity/aggregate境界を再利用する。実装45〜90分、fresh review 30〜60分、A4 hard中断時刻05:16 JSTは維持する。
- A4 direct baseline実装は2026-08-09 00:50:34 JSTに開始から34分34秒で自己検証PASSした。変更は`ci/tools/run_host_suite.py`、`ci/tools/network_guard.py`、`ci/tests/test_fail_closed.py`だけで、focused test 14/14、runner wrapper count 14/14、`self_test.py`、Python compile/static各87 file、matrix/JSON validator、diff check、禁止参照scanがPASSし、未検証`execution_capsule.py`と`process_containment.py`のSHA-256も不変だった。実装担当のPASSだけではA4を完了扱いにせず、30〜60分見込みのfresh独立reviewを継続する。A4 hard中断時刻05:16 JSTはリセットしない。
- A4 fresh独立reviewのread-only transportは、repository access前にbubblewrap `RTM_NEWADDR`で全commandが失敗し、2026-08-09 00:52:41 JSTに変更なし・未判定で終了した。これはcode findingではない。同じ非変更・host-only・offline範囲を`danger-full-access` transport fallbackで再実行し、review見込み30〜60分とA4 hard中断時刻05:16 JSTは維持する。
- A4 fresh独立reviewのfallbackは2026-08-09 01:04:21 JSTに開始から10分19秒で`FAIL`を確定した。capsule/containment参照除去、禁止2 fileのhash不変、focused 14/14、wrapper 14/14、self-test、matrix/JSON、Python compile/static各87、negative probe 10/10、diff checkはPASSした。一方、(1) 全`.py` commandをunittest候補とするため登録済み29 command中13 validatorを拒否する、(2) row output上限ちょうどで未完了のbreach flagがresult validatorと不整合、(3) network-isolation setupがcommand timeout外、(4) malformed route回帰caseを削り過ぎ、の4件をblockerとした。この4件だけを30〜60分で修復し、30〜60分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 上記4件のA4修復は2026-08-09 01:05:17 JSTに`workspace-write`で開始したが、readの一部だけが通った後、`apply_patch`がbubblewrap `RTM_NEWADDR`で起動拒否された。source変更前に停止し、同一2 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修復見込み30〜60分とA4 hard中断時刻05:16 JSTはリセットしない。
- A4 review blocker修復のfallbackは2026-08-09 01:19:17 JSTに開始から14分01秒で自己検証PASSした。変更は`run_host_suite.py`と`test_fail_closed.py`だけで、focused/wrapper 17/17、登録29 commandのunittest wrapper 13/direct 16（validator 13 direct）、self-test、matrix/JSON、Python compile/static各87、diff check、禁止参照0、禁止2 file hash不変がPASSした。実装担当のPASSだけでは閉じず、30〜60分見込みのfresh独立再reviewを続ける。A4 hard中断時刻05:16 JSTはリセットしない。
- A4 fresh独立再reviewは2026-08-09 01:30:12 JSTに9分25秒で`FAIL`を確定した。登録29 command分類、focused/wrapper 17/17、self-test、各validator、禁止参照/hash、独立negative 95件は成立したが、(1) HEADにあったIPv4 9項目・IPv6 8項目のsemantic route mutation回帰がtestへ復元されていない、(2) `verify_parent_restored()`がdeadlineを先に検査し、期限切れとparent namespace不一致が同時発生すると復旧失敗をmaskする、の2件が残った。この2件だけを15〜30分で修復し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- A4最終2 blocker修復は2026-08-09 01:37:20 JSTに開始から5分38秒で自己検証PASSした。IPv4 9項目・IPv6 8項目のsemantic route mutation、counter-onlyとmalformed route回帰を復元し、parent namespace不一致をdeadline切れより先に報告する復旧検査順序と両組合せのtestを追加した。変更は`ci/tools/network_guard.py`と`ci/tests/test_fail_closed.py`だけで、focused/wrapper 19/19、独立probe、self-test、matrix/JSON、Python compile/static各87、禁止参照0、禁止2 file hash不変、diff checkがPASSした。実装担当のPASSだけではA4を閉じず、同01:37 JSTから20〜40分見込みのfresh独立再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- A4最終fresh reviewの`read-only` transportは最初のsafe commandがshell起動前のbubblewrap `RTM_NEWADDR`で失敗し、文書・code・testを読めない監査基盤`FAIL`として2026-08-09 01:39:32 JSTに変更なしで停止した。これはcode findingではない。同一非変更scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- A4最終fresh reviewのfallbackは2026-08-09 01:49:16 JSTに約10分で`FAIL`を確定した。focused/wrapper 19/19、self-test、matrix/JSON、Python compile/static各87、登録29 command分類、route全semantic field、row境界、process-group cleanup、禁止参照/hashはPASSした。一方、(1) isolation内部probeとchild namespace検査の完了後にabsolute deadlineを再確認せず期限後の成功を許す、(2) registryにない`python -m unittest ci.tests.test_fail_closed`別名を受理する、の2系統をblockerとした。この2系統だけを15〜30分で修復し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2系統修復の`workspace-write` sessionは正本と対象codeを読了した後、最初の`apply_patch`がbubblewrap `RTM_NEWADDR`で対象fileを開けず、2026-08-09 01:55 JSTに変更・testなしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替え、修復15〜30分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2系統修復のfallbackは2026-08-09 02:10:19 JSTに開始から約15分19秒で自己検証PASSした。未登録module aliasをexact registry identityから除外し、isolation probe・child検査・外部接続検査・command wrap後にabsolute deadlineを再確認して期限後の成功とprocess launchを防止した。変更は`ci/tools/run_host_suite.py`、`ci/tools/network_guard.py`、`ci/tests/test_fail_closed.py`だけで、focused/wrapper 24/24、登録29 command（unittest 13/direct 16、validator 13 direct）、delayed deadline・route・row境界・process cleanupの独立probe、self-test、matrix/JSON、Python compile/static各87、禁止参照0、禁止2 file hash不変、diff checkがPASSした。実装担当のPASSだけではA4を閉じず、20〜40分見込みのfresh独立再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- A4 fresh独立再reviewの`read-only`実行は2026-08-09 02:20:30 JSTに`FAIL`を確定した。bubblewrap `RTM_NEWADDR`により実行testと保護file hashの独立再確認は不能だったが、静的監査で、(1) `run_bounded_process()`内の`Popen()`直前にabsolute deadlineを再確認せず期限後のchild起動を許す、(2) `child_main()`の最終deadline検査が`NetworkIsolationError`をcleanに捕捉しない、(3) registry外のdirect `python -m pytest`や`cargo`等を`execution_argv()`が受理する、の3件をblockerとした。この3件だけを20〜40分で修復し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同3 blocker修復の`workspace-write` sessionは対象確認後、全workspace commandがbubblewrap `RTM_NEWADDR`で起動不能となり、2026-08-09 02:24 JSTに編集・test・hash再確認なしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替え、修復20〜40分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 最初の同fallbackは自身をmain役と再解釈してread-only調査、続いて実装用`codex exec`を再帰起動し、対象sourceを編集せず入れ子化したため、2026-08-09 02:30 JSTにmainが中断した。3対象fileのhashは02:10版から不変である。実装担当自身が再委譲せず直接修正するよう役割を明示して同fallbackを再起動し、修復20〜40分とA4 hard中断時刻05:16 JSTはリセットしない。
- 直接実装担当による同3 blocker修復は2026-08-09 02:42:57 JSTに自己検証PASSした。変更は3対象fileだけで、host commandを登録29件の完全argv一致allowlistへ閉じ、既存absolute deadlineをbounded runnerへ渡して環境構築後・`Popen()`直前に期限切れをFAIL/timed-outへ変換し、child環境構築から`execvpe()`までのdeadline例外をcleanなexit 2へ変換した。`test_fail_closed.py` 27/27、登録29/unittest 13/direct 16/validator 13、未登録variant 57拒否、deadline/child独立probe、self-test、matrix/JSON、対象3 file compile、保護2 fileを除外したPython compile/static 85 file、禁止参照0、diff check、保護hash不変がPASSした。途中の通常Python validator呼出しは保護2 fileもAST read対象にした可能性があるためevidenceから除外し、最終束は明示除外して再実行した。実装担当のPASSだけではA4を閉じず、20〜40分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportはmain plan初回read後、続くcommandがbubblewrap `RTM_NEWADDR`で起動不能となり、2026-08-09 02:47 JSTに変更なし・code判定なしで停止した。同一非変更scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 02:55:24 JSTに約8分で`FAIL`を確定した。focused/wrapper 27/27、登録29 commandと未登録variant 136拒否、self-test、matrix/JSON、保護対象を除外したPython compile/static 85 file、child期限切れclean exit、route・row/resource境界、禁止参照0、保護2 file hash不変はPASSした。一方、`run_bounded_process()`の起動期限切れ例外で`verify_parent_restored()`が0回となり、parent namespace不一致が未確認・deadline FAILにmaskされることを独立再現した。またfork descendantがrunner return直後に一時観測され、process cleanupも完全には証明できていない。この2点を15〜30分で修正・決定的test化し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2点修正の`workspace-write` sessionは計画読取の一部だけ成功した後、bubblewrap `RTM_NEWADDR`が反復してsourceへ到達できず、2026-08-09 03:00 JSTに編集・testなしで停止した。同一2 file中心・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正15〜30分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2点修正のfallbackは2026-08-09 03:08:24 JSTに自己検証PASSした。`run_host_suite.py`と`test_fail_closed.py`だけを変更し、isolation plan成立後の親namespace復元確認を必ず1回実行して不一致を起動期限切れより優先し、process-group leaderをreapせず保持したまま残存group memberへTERM/KILL、残存時はfail-closedとした。focused/wrapper 29/29、self-test、matrix/JSON、保護対象を除外したPython compile/static各93、禁止参照0、diff check、保護hash不変がPASSした。実装担当のPASSだけでは閉じず、20〜40分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは全commandがshell起動前のbubblewrap `RTM_NEWADDR`で失敗し、2026-08-09 03:10:22 JSTに変更なし・code判定なしで停止した。同一非変更scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 03:23:01 JSTに10分13秒で`FAIL`を確定した。direct/wrapper 29/29、self-test、matrix/JSON、保護対象除外Python 85 file、allowlist未登録247拒否、route・schema境界、fork cleanup 12/12、leader未reap、保護hashはPASSした。一方、(1) `/proc`列挙・stat読取不能を空集合としてcleanup成功扱いする、(2) `Popen()`後selector初期化前の例外でchildを残す、(3) 同一PGIDのreparent済みdouble-fork descendant RSSを計上しない、(4) 通常execution timeoutとparent namespace不一致の同時発生で`INFRA_ERROR`を`FAIL`へ上書きする、の4 blockerを独立再現した。この4点を20〜40分で修正・negative test化し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同4点修正の`workspace-write` sessionは計画読取途中からbubblewrap `RTM_NEWADDR`が反復し、2026-08-09 03:26 JSTに編集・testなしで停止した。同一2 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正20〜40分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同4点修正のfallbackは2026-08-09 03:35:44 JSTに自己検証PASSした。`run_host_suite.py`と`test_fail_closed.py`だけを変更し、`/proc`列挙・stat異常をfail-closed、個別PIDの`ENOENT`/`ESRCH`だけを一時消失扱い、同一PGID全memberのRSS/stateをsnapshot集計、selector構築・登録失敗を含む`Popen()`後のcleanupとleader reap、parent namespace復元不一致の`INFRA_ERROR`優先を実装・回帰化した。direct/wrapper 35/35、登録29 command分類、self-test、matrix/JSON、保護対象を除外したPython compile/static各85 file、diff check、禁止参照0、保護2 file hash不変がPASSした。実装担当のPASSだけではA4を閉じず、03:36 JSTから20〜40分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは正本の初回read後、全shell再起動がbubblewrap `RTM_NEWADDR`で失敗しcode/test監査へ進めず、2026-08-09 03:38:57 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 03:52 JSTに`FAIL`を確定した。前回4 blocker、direct/wrapper 35/35、self-test、matrix/JSON、保護対象除外Python compile/static各85、allowlist未登録535拒否、実`/proc` scan 200回、alternate double-fork RSS、output境界、禁止参照0、保護hashはPASSした。一方、(1) stdout/stderrの非一時的`EIO`をEOF扱いして出力と上限超過を失いexit 0にできる、(2) 空のIPv4 route入力をmissing headerとして拒否せず空snapshotとして受理する、の2 blockerを独立再現した。この2点だけを10〜20分で修正・negative test化し、15〜30分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2 blocker修正の`workspace-write` sessionは初回read前にbubblewrap loopback設定で失敗し、再試行も進捗なく2026-08-09 03:53:40 JSTに編集・testなしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正10〜20分、再review15〜30分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2 blocker修正のfallbackは2026-08-09 03:59:18 JSTに自己検証PASSした。非一時的pipe `OSError`を再送出し、`EAGAIN/EWOULDBLOCK`だけを一時状態として継続、IPv4空入力をmissing headerとして拒否し正しいheaderのみの空tableと区別した。EIO cleanup/reapと空routeのnegative testを追加し、direct/wrapper 37/37、独立repro各1、self-test、matrix/JSON、保護対象除外Python compile/static各85、allowlist 29/13/16/13、diff check、禁止参照0、保護hashがPASSした。実装担当のPASSだけではA4を閉じず、04:00 JSTから15〜30分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは正本初回read後のsandbox初期化が3回連続でloopback設定に失敗し、2026-08-09 04:01:03 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、15〜30分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 04:13 JSTに`FAIL`を確定した。既知6 blocker、direct/wrapper 37/37、allowlist 29/13/16/13、pipe EIO/EINTR/EAGAIN、procfs fail-closed、Popen後cleanup、double-fork RSS、route semantic field、restoration mismatch優先、self-test、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、保護hash不変はPASSした。一方、(1) sudo fallbackの新規network namespaceではloopback初期化前の`/proc/net/route`が正当に空であり、空入力拒否により実hostのnetwork isolation self-testを失敗させる、(2) 通常execution timeout後も期限切れdeadlineをparent restoration検査へ渡し、正常復元でも`FAIL`を`INFRA_ERROR`へ誤分類する、の2 blockerを独立再現した。この2点だけを5〜15分で修正・回帰化し、10〜20分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2 blocker修正の`workspace-write` sessionはrepository読取前からbubblewrap loopback初期化に失敗し、代替読取確認も進捗しないため2026-08-09 04:15:53 JSTに編集・testなしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正5〜15分、再review10〜20分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2 blocker修正のfallbackは2026-08-09 04:21:45 JSTに自己検証PASSした。sudo fallbackで固定system toolによりloopbackをupにしてから既存`setpriv`権限drop・capability除去・no-new-privilegesへ移行し、通常execution timeout後のparent namespace復元検査を期限非依存で必ず実行するようにした。正常復元は`FAIL/timed_out=true`、実際の不一致併発は`INFRA_ERROR`を維持する。変更は3対象fileだけで、focused direct/wrapper各39/39、実host network guard self-test、`self_test.py`、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、diff check、保護hash不変がPASSした。実装担当のPASSだけではA4を閉じず、10〜20分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは正本初回read後、`true`を含む全shell起動がbubblewrap loopback初期化に失敗し、2026-08-09 04:24:24 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、10〜20分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 04:32:12 JSTに`FAIL`を確定した。通常timeoutの実host probeは`FAIL/timed_out=true/復元検査1回`、不一致併発の`INFRA_ERROR`優先、direct/wrapper 39/39、実host network guard self-test、self-test、matrix/JSON、allowlist変異132/132拒否、既知blocker回帰、保護対象除外Python compile/static各85、禁止参照0、保護hash不変がPASSした。一方、sudo fallbackのroot側`unshare/sh/ip/setpriv`をambient `PATH`の`shutil.which()`で選び、repository-controlled pathをroot実行prefixへ受理できる1 blockerを独立再現した。固定absolute system pathとownership/permissionをfail-closedに検査する最小修正を5〜10分、fresh再reviewを10〜15分で行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同1 blocker修正の`workspace-write` sessionは正本読取の一部とstatus確認だけ成功した後、bubblewrap loopback初期化が連続失敗し、2026-08-09 04:34:22 JSTに編集・testなしで停止した。同一2 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正5〜10分、再review10〜15分、A4 hard中断時刻05:16 JSTはリセットしない。
- 同1 blocker修正のfallbackは2026-08-09 04:44:44 JSTに自己検証PASSした。sudo fallbackを固定absolute候補へ限定し、canonical実体、regular/executable、root所有、group/world非writable、trusted親directoryを検査し、symlink解決不能と異なるinodeへの候補分岐を拒否、同一inodeのsystem aliasだけを許可した。PATH改変、全5 tool missing、非root所有、group/world writable、symlink、非regular、inode曖昧性を回帰化した。変更は`network_guard.py`と`test_fail_closed.py`だけで、direct/wrapper 46/46、実host network self-test、self-test、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、diff check、保護hash不変がPASSした。10〜15分見込みのfresh独立reviewを行い、A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportはmain plan初回read後の次commandがbubblewrap `RTM_NEWADDR`で失敗し、2026-08-09 04:46:04 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、10〜15分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 04:52:52 JSTにblocker 0の`PASS`で完了した。固定absolute system tool、canonical実体、root ownership、permission、trusted parent chain、same-inode alias許可、異inode・missing・symlink異常・非regular・非executable拒否、adversarial `PATH`からroot prefixへのrepository path非混入を独立確認した。direct/wrapper各46/46、実host sudo network-isolation self-test、通常timeoutの`FAIL/timed_out=true`とparent復元検査1回、self-test、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、保護hash不変、diff checkがPASSした。A4は00:16:00開始から4時間36分52秒で、予測上端4時間を36分52秒超えたが、hard中断時刻05:16:00 JST以内に完了した。これにより中断A0を実行経路から外したtrusted-development baselineはreview済みとなり、候補identityを固定可能である。A5は同一immutable candidate SHAを必要とするため、local checkpoint commit等のidentity固定をユーザーが許可するまで開始しない。
- ユーザー指示によりA5 canonical 2 GPU evidenceを2026-08-09 13:43:40 JSTに開始した。予測3〜6時間、hard中断時刻は同日20:43:40 JSTで、再試行してもリセットしない。工程内をA5.0 candidate scope監査・local checkpoint固定（30〜60分、hard 15:43:40）、A5.1同一SHAのhost/H3/preflight（30〜60分、開始から2時間で中断）、A5.2 `gfx1030` evidence（45〜90分、開始から2時間30分で中断）、A5.3 `gfx1201` evidence（45〜90分、開始から2時間30分で中断）、A5.4 aggregate・前後health・独立review（30〜60分、開始から2時間で中断）へ分割する。各子工程のhard時刻とA5全体hard時刻の早い方で停止する。中断A0の未追跡`execution_capsule.py`と`process_containment.py`はcandidateへ含めず、読取・実行もしない。
- A5.0 candidate-scope reviewの`read-only` transportはbubblewrap loopback初期化に連続失敗し、2026-08-09 13:46 JSTに文書・diff未読、変更なしで停止した。同じ非変更scopeを`danger-full-access` transport fallbackで再実行し、A5.0 hard中断時刻15:43:40 JSTはリセットしない。
- A5.0 candidate-scope review fallbackは2026-08-09 13:53:56 JSTに、現行差分から`.gitignore`と中断A0の保護2 fileを除く155 pathをcheckpoint候補として`PASS`判定した。candidateはmodified 38、新規117、最大blob 173852 bytes、新規content 2699460 bytes、全candidate content 3523675 bytesでhygiene上限内、diff check、plan/history相互link、禁止path・artifact・credential scanがPASSした。`.gitignore`には`/passwords.txt` ignore削除、Phase 3外の`/.agents/skills/update/`追加、既存`.local-artifacts`規則再編が混在し、既存行変更の許可を確認できないためcandidateから除外する。元worktreeの`.gitignore`と保護2 fileを変更・削除せず保持し、155 pathだけをlocal checkpoint commitへ固定した後、そのSHAからclean linked worktreeを作成してstrict evidenceを取得する。A5.0 hard中断時刻15:43:40 JSTは維持する。
- A5のP0 Cargo build hardeningは最終candidate `ac2baa3a0734d0894353ba180259d979da5a831e`、tree `4e43a9c42c9aa2dfa6a6d438610fa54c4e482d10`へ固定した。900秒timeout、combined 4 MiB output上限、private session/process group、TERM・2秒grace・KILL、bounded leader reap、同一group消滅確認、独立resource closeをartifact contractへ結合し、required CPython 3.12.10を含むfocused 31件と独立再reviewを`PASS`した。
- 同candidateのfresh H0 305/305、H1 151/151、H2 35/35、fixed-container base H3、RMSNorm H3、pre-GPU G0、private G1、sealed-controller semantic G1、read-only real-weight G2、P0、post-GPU G0はcanonical 2 targetで全て`PASS`した。G2は各target 6 case・6 dispatch、P0は各target 5 case・130 dispatchでfallbackなし・health OK・process cleanを記録した。P0 dispositionはthreshold未承認、最適化・他engine比較・performance hard gateの主張なしを維持する。
- A5 review 9のread-only transportはbubblewrap初期化前に停止して判定へ使わず、同一非変更scopeのfresh unrestricted transport fallbackを実行した。fallbackはfull `986c8b86..ac2baa3a`差分、host/H3/GPU evidence、57 sidecar、G2/P0 validator、cleanup、focused 15 test、diff checkを確認し、2026-08-09 23:16 JSTにhigh/medium/low 0件の`PASS`を確定した。以上によりStage Aを完了し、次はA5運用負債を解消した後にStage Bへ進む。
- A5で手書きlocal commandが現行workflow contractからずれて複数回fail-closedになった運用負債は、次のGPU evidence refresh前に2〜4時間の独立作業単位で解消する。workflow/controllerを正本としてtracked orchestrationまたはdry-run preflightからcommandを導出し、container mount path、target別build ownership、workflow run ID、UNIX socket path、canonical JSON、builder-owned outputを手作業で複製しない。
- 上記A5運用負債の解消作業を2026-08-09 23:51:09 JSTに開始した。予測2〜4時間とし、当初のhard中断時刻は2026-08-10 03:38:27 JSTとした。同作業完了後のStage B plan reviewを継続するため6時間、B1を継続するためさらに6時間をユーザー指示で延長し、現在の全体hard中断時刻は同日15:38:27 JSTとする。defaultはGPU、model cache、container、buildを実行しないhost-only dry-runとし、既存workflow/controllerの正本contractからcanonical command、target別path、run identity、socket/output制約を導出する。hard中断時刻へ到達した場合は新しい変更を開始せず、検証済みの独立rollback可能点で停止する。
- A5運用負債の解消は2026-08-10 01:26 JSTに開始から約1時間35分で完了した。tracked host-only plannerとclosed schemaを追加し、既存workflow、matrix、G1/G2/P0総合validator、G1 pure build-layout helperからexact `gfx1030`→`gfx1201`のH3/G1/G2/P0 planを導出する。CLIはclean immutable commit/treeを必須とし、authority hash、全path componentのsymlink、repo containment、短い未作成run root、target順・output ownershipをfail-closedに検証する。focused 11件、fail-closed 46件、matrix/JSON/G1/G2/P0 validator、Python compile、diff check、dirty-local H0 316/316、独立review High/Medium 0件の`PASS`を得た。GPU、model cache、container、build、networkは実行せず、canonical V620/R9700 evidence identityも更新していない。既存P0 builderのsame-UID/trusted-solo output symlink安全負債はユーザー承認済みの延期境界を維持する。
- Stage B開始時監査ではStage Aですでにlock/config/index/header、738 tensor名・dtype分類、verified cache/range read、generated-token停止controller、`tokenizers 0.21.4`のroot lockが存在することを確認したため、これらを再実装しない。Stage Bの残差をB0 dependency closure、B1全tensor shape、B2 verified frontend asset、B3 tokenizer、B4 typed renderer、B5 consumer/load plan、B6 offline CLI、B7a backend-neutral buffer readback、B7b exact-range weight upload bridgeへ分割した。各単位は2〜8時間予測、上端+1時間hard stop、独立review/rollbackを持つ。B0〜B6はhost-only、B7a/B7bだけcanonical sLLM V620とR9700を使い、spare V620は使用しない。詳細はPhase 3全体計画を正とする。
- Stage B分割planは、B0の全workspace依存閉包、B7a readback分離、1 GiB transfer上限の3件を修正した最終candidate `9d3f7d5feb27294644252c60f24984fc579e3bfe`へstrict H0 316/316とfresh累積独立review High/Medium 0件の`PASS`を結合した。B0は2026-08-10 03:40:18 JSTに開始し、最終機能candidate `a5519d89820f42a8349cf3485ee8dc37154d8507`、tree `4f6896eee85399ddc10831b752355d332960a0dd`で完了した。全90 package・170 edgeのnormalized policy/schema、offline metadata/lock照合、Rust 1.85 Linux x86_64 workspace/all-target check、renamed dependency、MSRV例外、ambient target/HIP/native/Rust compiler・flag override除去、rustup自動取得禁止、H0 suite/path登録を固定した。旧candidateのH1 assertion失敗はretryで昇格せず、process-wide FD総数を使う既存testの並列競合を7/200で再現して、固有fixtureのdevice/inodeをfail-closedに検査するtestへ修正した。同一identityのstrict H0 335/335、H1 151/151、H2 35/35は各1回目に`PASS`し、累積reviewとFD修正のfresh独立reviewも全指摘を閉じて`PASS`した。model、GPU、model cache、container、networkは使用していない。次の独立単位はB1 tensor shape closureであり、開始時刻とrollback baseを別途固定する。
- B1 tensor shape closureはB0 docs-inclusive candidate `d610b4801052f11125a9002e0b59d0d0973a86d7`、tree `04d7214f86c7069ab73bc098459972f59fb3115b`をrollback baseとして2026-08-10 06:55:54 JSTに開始した。予測5〜8時間・作業単位hard 9時間を維持し、当初のeffective hard stopだった全体停止上限2026-08-10 09:38:27 JSTはユーザー指示で6時間延長した。現在は作業単位上限より先に到達する同日15:38:27 JSTをeffective hard stopとする。現行738 tensor catalogと固定reader記録から全main/vision/MTP expected shapeをconfig導出し、schema/tiny fixture/非整列・境界negative test、同一SHAの固定cache metadata照合、H0〜H2、fresh独立reviewまで完了しない限りB1完了とは扱わない。実装とhost testはmodel cacheなし、適用確認は固定cacheのbounded streaming hashとmetadata照合に限定し、GPU、payload materialize/range read、network、containerを使わない。
- B1開始後の独立監査で、vision/MTP shape orientationのreader記録不足とPython mirror validatorの所有範囲漏れを検出した。固定vLLM/SGLangから全family式をreader/cross-check分離で確定し、B1をRust/Python双方のprivate config-derived expected-shape検証へ修正した。vision値は外部source defaultへ固定せずlock済みconfigから抽出し、schema、fixture、public API、suite登録は変えない。開始時readerがGDN 2件のstorage dtypeを逆と判断した点は後の固定cache照合で誤りと判明したため、固定headerの`dt_bias=BF16`、`norm.weight=F32`を正とする。
- B1 functional checkpoint `be098f41c903c19b3f3e62883b0af8c8201e990b`、tree `0831c0bbf9fb98edcb0a6a30991b2c2476d54e48`は、全738 tensorのconfig-derived shape/dtype/class検証、Rust/Python parity、header rank/dimension照合を実装し、strict H0 335/335、H1 154/154、H2 35/35を各attempt 1でimmutable `PASS`、fresh独立reviewも指摘0件で`PASS`した。しかしdocs-inclusive checkpoint `a65b2ab3129a8a392df980e8751431f7783e331f`の固定cache照合がGDN 2 dtypeと24層の`conv1d.weight` singleton次元に関するreader誤りを検出したため、同checkpointは不合格とし、固定header契約へ戻したcandidateで全evidenceを取り直す。B1全体は未完了、B2は開始しない。
- B1最終機能candidate `b5cc617287ec2efb97c5b06bd838621f51d547c8`、tree `e901d2fa1b33ae75a7d087c1d4323d38f9f02a00`は、固定cache 13 file・9,342,905,899 bytesのcontent-only hashと全738 header metadata、strict H0 335/335、H1 156/188（32 deselected）、H2 35/42（7 deselected）を各attempt 1で`PASS`した。GDN storageは`A_log=F32 [32]`、`dt_bias=BF16 [32]`、`norm.weight=F32 [128]`、`conv1d.weight=BF16 [8192,1,4]`へ固定し、Rust/Pythonのexact-catalog mutation、bounded diagnostic、descriptor map非複製を累積review High/Medium/Low 0件で確認した。GPU、payload materialize/range read、network、containerは使っていない。完了記録を含むdocs-inclusive identityへ同じ受入evidenceとfresh reviewを結合するまでB1完了またはB2開始とは扱わない。
- B1完了記録を含むdocs-only candidate `01dbedfa9de5e435703ef26b66fb610f194cfdd2`のstrict H0 attempt 1は335 selected中334 PASS、1 FAILであり、candidateを昇格しない。唯一の失敗はsemantic G1 broker client-death testがcompiler PID clearとfailure publicationの中間状態を観測する既存raceだった。対象test単独100/100と95-test suite 3/3はPASSし、500 msのin-memory遅延で同じ失敗を決定的に再現した。productionは変更せず、testが既存5秒deadlineまでfailure publicationを独立に待つ1行修正とし、focused 20/20、semantic G1 95/95をPASSした。修復を含む新candidateのH0〜H2、固定cache、fresh reviewを最初から取り直す。
- B1最終implementation candidate `6543098f70d8c06b5a6758becd4590ab44fb9811`、tree `b4f46f5a42c09df4e2d64aa5c1f8191620d60ce8`は、candidate SHA/tree・前後clean・validator/lock/schema/output digestへ結合した固定cache reportで13 file・9,342,905,899 bytes・全738 header metadataを`PASS`した。同一identityのstrict H0 335/335、H1 156/188（32 deselected）、H2 35/42（7 deselected）は各attempt 1、skipped 0で、正しいrollback base `d610b4801052f11125a9002e0b59d0d0973a86d7`からのfresh累積reviewもHigh/Medium/Low 0件の`PASS/no findings`だった。broker修正はtest 1行だけでproductionは不変であり、失敗した`01dbedfa...`のevidenceは再利用していない。これによりB1 implementationを完了し、別docs-only closeoutで本記録を検証してからB2を開始する。
- B1 docs-only closeout `8d6018057006f8c06e8c3bac5343cc3681fcb1a2`、tree `7eecd11417e530c68f62bc83ea2ff90867bf7733`はstrict H0 335/335、attempt 1、skipped 0とfresh独立review High/Medium/Low 0件を`PASS`し、B1全体を完了した。これをrollback baseとして2026-08-10 09:58:06 JSTにB2 verified frontend assetsを開始した。予測2〜4時間、個別hard中断時刻14:58:06 JSTとし、全体停止上限15:38:27 JSTより早い個別上限を適用する。`VerifiedCache`の保持済みFDから固定frontend assetだけを種類別hard cap付きpositional readで返す。公開kindは`config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`の4種、asset全体sizeの上限は順に1 MiB、16 MiB、256 KiB、64 KiBとする。B3はself-containedな`tokenizer.json`を使うため`merges.txt`と`vocab.json`は公開せず、raw path APIも作らない。shard・任意path・link・差替え・同一inode改変・cap超過を拒否する。GPU、weight payload、model cache、container、networkは使わない。
- B2 implementation candidate `b2a9275cd00bae55218f5b60840e471e8bb877ff`、tree `7c8ba9fec21a720134436e0a3574db2620ba52f6`は、公開raw pathを持たない4種の`FrontendAssetKind`と、保持済みverified FDだけを使うwhole-file positional readを実装した。種類別capをallocation/read前に適用し、read前後のcache root・全locked path binding再検証、同一inode同一size改変、truncate/extend、symlink/hardlink/path/root差替え、cap境界、並行read、FD lifetimeをtiny fixtureで検証した。同一identityのpinned Python 3.12.10 strict H0 335/335、H1 163/195（32 deselected）、H2 35/42（7 deselected）は各attempt 1、failed/skipped 0、clean worktreeで、report/sidecar SHA-256も一致した。rollback base `8d6018057006f8c06e8c3bac5343cc3681fcb1a2`からのfresh累積独立reviewは`PASS/no findings`だった。GPU、weight payload、model cache、container、networkは使用していない。implementationを受け入れ、別docs-only closeoutへstrict H0とfresh reviewを結合するまでB2全体は未完了、B3は未開始とする。
- B2 docs-only closeout `c437aab32f7fa7cd0681dd8b7db3807ac55c5984`、tree `af07a678a09ea97df7d74e03811d2765d0a5632c`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report/sidecar一致とfresh独立review High/Medium/Low 0件を`PASS`し、B2全体を完了した。これをrollback baseとして2026-08-10 10:51:30 JSTにB3 tokenizer frontendを開始した。予測3〜5時間、作業単位hard中断時刻16:51:30 JSTだが、全体停止上限15:38:27 JSTを今回のeffective hard stopとする。B2 assetからだけ`tokenizers`を構築するtyped encode/decode API、special-token identity、EOS整合、versioned tiny fixtureとnegative/boundary testを所有範囲とし、GPU、full model、model cache、container、networkは使用しない。precommit監査で新規`ci/fixtures/tokenizer-v1/**`は汎用`ci/**`規則からH1等を選択する一方、fixture固有の明示的H1 ownershipがないことを確認したため、B3の最小CI所有範囲へ`ci/matrix/path-to-suite-v1.json`の専用H1登録を追加する。依存、suite command、test tierは変更しない。`ModelLock`と`VerifiedCache`のfingerprint公開fieldはcallerが書換え可能なため、B3はその一致を暗号学的lock結合とは主張せず、core検証済みcache由来bytesが渡されたlockのtokenizer意味契約を満たすことだけを保証する。opaqueな内部identityへの移行はtrusted B3をblockしない別core follow-upとする。
- 初回B3 candidate `6073d2257f3811da43aa8e380a90427630c2742a`はprecommit review、strict H1 182/214、H2 35/42をPASSしたが、strict H0 335/335のdependency inventory validator 1件だけが`workspace_members` driftでFAILしたため受け入れず、H1/H2も再利用しない。原因は新規`tokenizer_contract` integration-test targetだけがB0のall-target inventoryへ未反映だったことで、package 90、edge 170、Cargo manifest/lock、version、checksum、license、feature、MSRVは不変である。`ci/dependencies/rust-workspace-v1.json`へtarget 1件だけを同期し、新candidateでH0〜H2とfresh reviewを取り直す。GPU evidenceは不要で、B2 rollback baseは維持する。
- B3 implementation candidate `766bfec524b8410317e41cafa69b67f1179f3a95`、tree `3b0084c073c2fa1cab3a6a46e2ce5b0bcd866d1c`は、pinned Python 3.12.10 strict H0 335/335、H1 182/214（32 deselected）、H2 35/42（7 deselected）を各attempt 1、failed/skipped 0、clean exact SHA/treeで`PASS`した。3 report SHA-256 `0ec565591f76963ffe756fc756016b8b74659de1d5831286b3d05e142c940db8`、`723410687e38b867724c6c90852dbb2e799d45785c5e799b2028de314b9c07c0`、`5853b9f26614bf4f80fa622486784e49e8813d6c631e84946b9f732c20ee90b4`はsidecarと一致し、B2 rollback baseからのfresh累積独立reviewもHigh/Medium/Low 0件の`PASS/no findings`だった。B3 implementationを受け入れ、docs-only closeout identityのstrict H0とfresh reviewがPASSするまでB3全体は未完了とする。GPU、full model、model cache、network、containerは使用していない。
- B3 docs-only closeout `7904a2c196628adcc138eb6499a6a04bd5ebdb56`、tree `8217e4b698a390c31c10b6ed4460f63fa8988051`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report/sidecar一致とfresh独立review High/Medium/Low 0件を`PASS`し、B3全体を完了した。これをrollback baseとして2026-08-10 12:48:55 JSTにB4 typed chat rendererを開始した。予測3〜5時間、個別hard中断時刻18:48:55 JSTだが、全体停止上限15:38:27 JSTをeffective hard stopとする。固定Qwen3.5 templateのseparated reader記録、任意Jinjaを実行しないtext-only typed renderer、versioned host fixture、unsupported境界、fixture専用H1 path/validator identity、新規test targetと既存workspace固定`sha2`へのfrontend edgeおよびCargo lock/schema/validatorのexact edge count同期に限定する。constructorは読み出した7,756-byte自体のSHA-256を固定値へ照合し、mutable metadata labelだけでは成功させない。readerはlock一致済みの同frontend assetだけをbounded readし、weight payload、full model load、GPU、network、containerは使用しない。
- B4 candidate `b1984e47809ed8cc428b9b817409b74470beadf6`、tree `a8b01c84eef5836bc45d2535843ec3c29e180fe2`はstrict H0 335/335、H1 195/227（32 deselected）、H2 35/42（7 deselected）を各attempt 1、failed/skipped 0、clean exact identityで`PASS`したが、fresh累積独立reviewはStage B表に残った「dependency edge/Cargo.lock不変」と実際の`sha2` frontend edge追加との矛盾をLow 1件とした。実装指摘は0件だがcandidateは受け入れず、表を実際の受入範囲へ修正して旧evidenceを再利用せず取り直す。
- 表を修正したB4 candidate `5c8bbd5c5516891fa5708245ed2a8b522f533247`、tree `a753d87ef76575ce66350070a88b1c57121fcd86`はstrict H0 335/335、H1 195/227（32 deselected）、H2 35/42（7 deselected）を各attempt 1、failed/skipped 0、clean exact identityで`PASS`した。fresh累積独立reviewは、template bytesが一致しても別repo/revisionのlockを受理できるconstructor identity不足をMedium 1件、現在状態とreader残件の陳腐化をLow 1件として検出したためcandidateを受け入れない。fixed repo/revision検査とdirect mutation test、正本文書同期を行い、旧evidenceを再利用せず新candidateの全evidenceを取り直す。
- B4 final implementation candidate `b43f2132c1afc604f2ae22ab12d55101aac7921b`、tree `559c426b1184f25da131fa10e07a3926938d299e`は、固定repo/revisionとtemplate path/size/SHA、bounded read後のraw-byte SHA/UTF-8を順に検証するtyped rendererを確定した。同一identityのstrict H0 335/335、H1 197/229（32 deselected）、H2 35/42（7 deselected）は各attempt 1、failed/skipped 0、clean exact identityで`PASS`し、report SHA-256 `e139ea624639a609921ebe63f8398a1948b45b8d8b1c1a49a8efad9b828b745f`、`3197aa221dbc7d72c5c292662cb4c71900b46536ded041c81fa03970df0ebf62`、`27ab4aa34fc13e5fa068d855b3b8e2cbaf7e04d517cd676a3874c6c2eb8f9e94`はsidecarと一致した。B3 rollback baseからのfresh累積独立reviewはHigh/Medium/Low 0件の`PASS/no findings`だった。GPU、weight payload、full model、model cache、network、containerは使用していない。implementationを受け入れ、別docs-only closeoutのstrict H0とfresh reviewがPASSするまでB4全体は未完了とする。
- B4 docs-only closeout candidate `28136d4e6a50fb6349b7cf81d063397aa136a50f`、tree `da0c7e6b4647561be9bcd9804c1b56075d718083`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `73620fbe1c402dc1b78432bc5c2d4c1cf4e5b6b072c74724aeb6b136c90762ca`のsidecar一致を`PASS`した。fresh reviewのうち、resource記録からweight payload不使用が欠落したMediumは採用して同期する。candidate自身のcontentへcandidate SHA/tree/report/reviewを埋め込むHigh要求はcontent-derived Git identityでは自己参照となるため採用せず、B1〜B3と同様に、closeout commitを固定して外部evidence/reviewを結合し、その結果を次単位開始記録へ同期する。修復したdocs-only candidateのH0とfresh reviewを取り直すまでB4全体は未完了である。
- docs-only closeoutは自身のcontentへ自身のcommit SHA/treeや未実行review結果を要求しない。commit固定後のstrict evidenceとfresh reviewを外部で同一identityへ結合し、PASS後に次単位の開始記録へcloseout identity、report digest、review結果を同期する。これはGitの自己参照loopを避けつつ、B1〜B3で用いたimmutable boundaryを維持する恒久規則とする。
- closeout修復candidate `555bfef127077a74bb94bc3762cdf2984c48dbdf`、tree `55df85d4c728e807fb0035454bc35b6d4fd5084d`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `1ea590f1683a555d02bdd3571b83c199a892950f5a85cf54845e07ad86fa66c9`のsidecar一致を`PASS`した。fresh reviewは自己参照境界とactive/historyのresource記録を認めたが、main planのimplementation行だけweight payload不使用が未同期であるMedium 1件を検出したため受け入れない。本行を同期した新candidateのH0とfresh reviewを取り直す。
- B4 final docs-only closeout `b8a71243f7f93390630c7423d6ca082f9ec51703`、tree `caa9e69da0be67a9207c0455d628e5fecb2611d8`はstrict H0 335/335、attempt 1、failed/skipped 0、clean exact identity、report SHA-256 `8c7db0493b46cac2d08d96ae27190940786a04847c379a4a0bb5db1e9dc17fe3`のsidecar一致とfresh独立review High/Medium/Low 0件の`PASS/no findings`を結合し、B4全体を完了した。これをrollback baseとして2026-08-10 14:56:38 JSTにB5 weight registry/load planを開始した。予測3〜5時間、個別hard中断時刻20:56:38 JSTだが、全体停止上限15:38:27 JSTをeffective hard stopとする。B5はB1 descriptorからmain-text tensorのrequired/config-conditional/known-unconsumed/rejected分類、一意consumer、exact source range、最大16 MiBの決定的chunkとdestination offsetを構築するhost-only Rust内部planに限定し、weight payload read/materialize、model cache、GPU、network、containerは使用しない。
- B5 separated preimplementation readerは、B1の公開`TensorDescriptor`/`VerifiedCache::tensors()`でproduction入力が足り、private catalog/parser/hash/map/range readerの追加公開や複製が不要と確定した。main text 426件を一意consumerへ、vision 297件/MTP 15件をknown-unconsumedへ分類する。現行`tie_word_embeddings=true`ではembeddingをtied lm-head aliasとし、独立lm-headを拒否する。untied branchはB1 lock/catalog外なので型上のconditional表現だけに留める。name順、checked half-open range/destination、16 MiB以下chunk、固定binary domainのSHA-256 digest、metadata-only境界testをreader記録へ固定した。B5は未実装・未完了である。
- 初回B5 reader checkpoint `939a1be3f48983ad9deb041b9c5f9930f7c74e64`、tree `7777f1cb5eada0393e150c7d945b052a1508293a`はstrict H0 335/335、attempt 1、failed/skipped 0、report SHA-256 `7e0ae8b7c394206e5b63f5709d330c44973c6af731f56b58d62da5b788409638`のsidecar一致を`PASS`したが、fresh reviewがconsumer grammarとdigest shard bindingのHigh 2件、family算術、vision bias境界、binary encoding/test vector不足のMedium 3件を検出したため受け入れない。exact name grammar、relative source fileとlocked size/SHA、固定tag/width/framing、canonical digest vectorをreaderへ追加した新checkpointで取り直す。B5は未実装・未完了である。
- 修復reader candidate `f73c9646f221eb92fb0fe5371e0ce8519dbedb2d`、tree `775c81ed4b78d6787c271d59f783fbb20a6eb2c4`はstrict H0 335/335、attempt 1、failed/skipped 0、report SHA-256 `df219c10bf813c1dde998cbdca906c77bf45b1d9d916c337095e42ef413d40ff`のsidecar一致を`PASS`したが、fresh reviewは426-byte canonical digestが記載規則から再現不能であるHigh 1件を検出したため受け入れない。domainからentry/chunkまでの唯一のwire順、tag幅、entry countの単一framing、optional layer位置、locked SHAのASCII hex string表現を固定し、独立再計算で426 bytes・SHA-256 `9a57a67384038c9e437236511c50f1b03b88a4f733cb06464d4ad3e408616bb2`を確認した。新checkpointのstrict H0とfresh reviewは未実施で、B5は未実装・未完了である。
- 後続A3は8時間の一括工程にせず、G2 host contract・実行経路をA3a（2〜4時間、中断上限5時間）、P0 host contract・実行経路をA3b（2〜4時間、中断上限5時間）へ分割する。各工程は開始時刻とhard中断時刻を別々に固定し、未完了時間を次工程へ移し替えない。

### 現行status summary

- 完了:
  - プロジェクト名を`sLLM`へ変更し、CLI、Rust crate、C ABI、環境変数、CI、文書、source directoryを新名称へ統一。
  - プロジェクトライセンスをMITへ統一。
  - Qwen3.5のMTP表記とBF16階層の矛盾を修正。
  - reset前のGit履歴を現状維持すると決定。
  - 初期MVP、GPU分類、toolchain、Rust/C++境界、model lock、provenance、API profileの草案を作成。
  - CI・テストの失敗事例、参照推論engine、GitHub Actions、AMD GPU runner運用を調査し、方針策定計画を作成。
  - 再出発レビューを検証し、repository hygiene、credential、performance cliff、GPU merge gate、fail-closed集約の対策を作成。
  - governance baselineをcommit `2764e73ebc45c8bbd209a426ca93ce341ed5d860`として`origin/main`へ公開。
  - Rust 1.97.1/MSRV 1.85.0、C++17、公式system package版ROCm 7.14.0/LLVM 23の開発環境を構築し、legacy ROCm user-spaceを除去したうえで、exact `gfx1030,gfx1201`の最小HIP smokeを実GPU 3台で確認。
  - 専用local hostで`homelab1`への`NOPASSWD: ALL`設定を完了し、無人進行を優先する明示的なrisk trade-offとして受容。
  - llama.cpp、vLLM、SGLang、TensorRT-LLM、ROCm/ATOM、LMDeploy、KTransformersの2026-08-02時点の安定releaseを完全commit SHAで`reference/`へ固定し、7件の再現manifestを作成。
  - 固定した7件のexact revisionを一次sourceとしてCI・testを再調査し、採用・不採用項目を記録。
  - H0〜H2の時間予算とfail-closed集約、schema/markerの必須概念、H3昇格条件、初期GPU evidenceのtarget・台数・基本隔離方針を確定。
  - Rust workspace、4 crate、CMake C++17 static host stub、versioned C ABI、checked-in bindingsを追加し、Cargoからbuild・link・testできるrepository skeletonを構築。
  - `test-result-v1`、compatibility tuple、suite/host/path matrix、tracked/local hygiene、共通host runner、fail-closed aggregatorを実装。
  - H0/H1/H2を独立rowとして実装し、実収集・選択件数、row/command resource、network namespace、clean SHA identityを記録・検証。意図的なformat/test/schema/0件/missing/duplicate/unknown/stale/hash/identity/resource異常をfailureへできることをself-testで確認。
  - opaque queue/buffer/event、access mode、completion ownership、TensorView/NVFP4境界、semantic op arity、C/Rust ABI layout parity、error sink truncation contractをPhase 1の非実行contractとして追加。
  - Python 3.12/Linux x86_64 host dependencyをtransitive dependencyとartifact SHA-256まで固定し、test中の外部networkをnamespaceで遮断。
  - GitHub-hosted CPUだけを使う`host-required` workflowを追加し、official Actionsを完全commit SHAへ固定。H3/GPU/self-hosted runnerは含めていない。
  - Phase 2 bootstrap gateを変更scope別へ整理し、H3 required昇格観測をG0/model-free pathと並行する正本へ同期。
  - 公式ROCm 7.14.0 imageをimmutable digestで固定し、exact `gfx1030`/`gfx1201`のnon-required H3 matrix、host bundleと抽出device code objectを分離したartifact metadata schema、fail-closed validatorとnegative testを追加。
  - 明示HIP CMake pathと、固定image内のpinned `amdclang++`だけを使うH3 direct compile/link runnerを追加し、bundle保持host objectからdevice code objectを抽出してCode Object V6、target別ELF flags、wave32、symbol、artifact/sidecar identityを検査する経路を構築。
  - H3 2 rowのreport・metadata・device artifact・sidecar・candidate identityをfail-closed集約する独立checkと、digest/configを検査したROCm containerを`--network none`で使うnon-required workflowを追加。GPUまたは生成executableは実行しない。
  - H3実装をcommit `03f90be1ad85145e3abee86e67615c1e17f552b4`として公開し、GitHub上のexact 2 compile rowがPASSすることを確認。初回aggregateでworkflow run identityのcontainer伝播漏れを検出し、fail-closedに失敗した。
  - canonical GPUをV620 `0000:03:00.0` / `GPU-76a08c022586fed6`とR9700 `0000:47:00.0` / `GPU-a8e9ddefa2d60f55`へ固定し、spare V620を必須rowから除外した。
  - model-free G1をpublic inference ABIから分離したprivate evidence ABIと専用Rust binaryとして実装。1、3、17、255、256、257 byteを各2 device allocation、2 HIP transfer、1 diagnostic dispatchでbyte exact検証し、host stub、CPU fallback、model、semantic opをPASS経路から除外した。
  - G1 artifactのexact target、Code Object V6、wave32、ELF flags、kernel symbol、runtime loader path、candidate identityを検査し、timeout/output/process cleanup、stale/symlink/target差し替えをfail-closedにするrunner・2 row aggregateを追加した。
  - immutable candidate `f393d688a051d2b73c8773d8a930a711592609bc`でH0 106件、H1 42件、H2 9件、H3 exact 2 target、canonical 2 GPUのG0/G1を集約までPASSさせた。G1は各GPUで1、3、17、255、256、257 byteをbyte exactに検証し、CPU fallback、model、semantic opを使用していない。
  - generated-token停止policy導入前のPhase 3 model-lock candidate `sha256:89ba8a6b2e1b7c0324090ddf15ce0e673ff4c3dc242c4127690d490056d8efd1`を独立監査までPASSさせた。111-byteの有効なsafetensors fixture、同size path差し替え・同一inode改変・gap/overlap/trailing/overflow・FD cleanupを含む15件のhost contractを固定し、実Qwen cache全13 fileのcontent-only hashと738 tensorのindex/header/slice照合を再検証した。このfingerprintは過去candidateのidentityであり、現行runtime入力には使用しない。
  - versioned generated-token停止policyをmodel lockへ固定した旧Qwen lock fingerprintは`sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935`であり、Stage A/B evidenceのhistorical identityとして保持する。C3a0 typed attention/RoPE contract追加後の現行fingerprintは`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`とする。fingerprint domainは`schema_version`と完全な`model`のまま変更しない。
- 次:
  1. Stage C integrationは完了した。次はStage Dで、token embeddingから32 main layer、final norm、tied lm headまでのRust execution planとrequest-local state orchestrationを実装する。
  2. 最初のStage D draft unitは、`layer_types`の明示list、weight consumer、既存C1〜C5 opを結ぶhost-only graph builderとし、GPU kernelを追加しない。tiny multi-layer graphで順序、shape、state allocation、vision/MTP拒否をfocused testする。
  3. H3はnon-requiredのまま観測を継続し、20回以上・7日以上の条件を満たした時点でrequired昇格だけをreviewする。

## 未解決事項

- AMD consumer RDNA2を含む各exact gfx targetの実機検証状態。
- ROCm 7.14.0 system package環境の最小smokeを越える数値kernel、model、長時間安定性、性能と、HWE kernel 6.17上のmixed V620/R9700 tupleを正式な互換性対象にできるか。
- Qwen3.5-4Bのop・shape・入力範囲ごとのGPU数値toleranceとfull-model G3 golden token sequence。固定metadataから決めた停止policyをgoldenから逆算しない。
- resource gateの1 TOPSに用いる精度・operation数、16 GBの単位とdevice-local memoryの定義、帯域の算出方法、対応例外を承認する基準。
- Infinity Fabric、RDMA、KV永続化の詳細設計。
- 量子化形式ごとのlayout、scale granularity、accumulator、fallback表。
- Qwen3.5-4B baselineで測定して決めるop別数値toleranceと性能回帰閾値。
- sudo以外の既存平文credentialの失効・rotationとsecret managerへの移行状況。
