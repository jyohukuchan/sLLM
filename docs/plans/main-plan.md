# uLLM メイン計画

## この文書の役割

- Git管理外の `uLLM-project.md` にある要件定義・開発方針・重要な決定を、開発に必要な範囲で追跡可能な形へ同期する。
- この文書には重要なproduct・architecture・compatibility上の決定、開発計画と順序、進捗、未解決事項だけを記録する。恒久的な実行手順は各正本文書へ置き、ここには重複させない。
- `uLLM-project.md` とこの文書に方針上の差異が生じた場合は、推測で統合せずユーザーへ確認する。
- 角括弧内の項目は、初期バージョンでは対応しない将来機能を表す。

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
  - 初期仕様は `uLLM OpenAI-compatible Chat Completions profile v1` とする。
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
- vLLM等からコードを直接流用しない。reader subagentとimplementer subagentを分離し、要点だけを渡す。
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
- HIP/runtime/backend/dispatch/native buildへ影響する変更は、同じreviewed immutable SHAに対するG0/G1/G2/P0とfail-closed集約をmerge条件とする。
- H0/H1/H2は並列required rowとし、`host-required`へ集約する。required workflowはp95 10分以内、hard上限15分とする。
- 初期GPU evidenceは専用local hostのexact `gfx1030` 1台と`gfx1201` 1台で直列実行し、public fork PRからGPU runnerを直接使わない。
- 詳細な方針と実装順序は[CI・テスト方針策定計画](active/2026/08/1-10/ci-test-strategy.md)を参照する。

## 開発運用上の決定

- Gitで追跡するのはsource、文書、小さなfixture、manifest、hash、summaryとし、model、binary、raw trace/profile、large model slice、生成物は追跡しない。詳細は[repository hygiene方針](../development/repository-hygiene.md)を正本とする。
- 無人での進行を優先しつつsecret exposureを最小化する。専用local hostでは`homelab1`への`NOPASSWD: ALL`を意図的なtrade-offとして受容し、main agentがtask scope内で`sudo -n`を使う。恒久方針は[credential方針](../security/credentials.md)を正本とする。
- 作業単位は独立してreview・rollbackできる範囲とする。immutable identityの固定から検証、適用、適用後確認、rollback/fail-stop、`push` skillによる公開までの実行手順は`AGENTS.md`を正本とする。

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
- model-free最小経路は`Cargo -> ullm-hip -> versioned C ABI -> native HIP -> GPU`を通してallocation、copy、単一diagnostic kernel、completion、copy-back、解放を検証する。推論opまたはGPU対応済みの証拠にはしない。
- 詳細な作業単位、受入条件、evidence、rollback境界は[Phase 2 H3・G0・model-free GPU path計画](active/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)を正とする。

## 現在の状態と次の作業

- 現在: Phase 1を完了し、Phase 2のbootstrap gate同期、H3静的contract、CMake/build接続、固定image内`amdclang++`によるexact `gfx1030`/`gfx1201` compile-only runner、artifact検査・fail-closed集約、non-required workflowを実装済み。次は同一immutable candidateをdigest固定ROCm imageで検証してH3観測を開始し、待機せずG0へ進む。
- 完了:
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
- 次:
  1. H3実装を同一immutable candidateとしてdigest固定ROCm imageのexact 2 rowで検証し、non-required観測を開始する。
  2. H3観測と並行して、専用local hostのGPU evidence実行・集約とG0 preflightを構築する。
  3. canonical `gfx1030`/`gfx1201`でmodel-free最小GPU実行経路を同一candidate SHAに対して検証する。

## 未解決事項

- AMD consumer RDNA2を含む各exact gfx targetの実機検証状態。
- ROCm 7.14.0 system package環境の最小smokeを越える数値kernel、model、長時間安定性、性能と、HWE kernel 6.17上のmixed V620/R9700 tupleを正式な互換性対象にできるか。
- Qwen3.5-4Bで使用する完全commit SHAとlock manifest。
- resource gateの1 TOPSに用いる精度・operation数、16 GBの単位とdevice-local memoryの定義、帯域の算出方法、対応例外を承認する基準。
- Infinity Fabric、RDMA、KV永続化の詳細設計。
- 量子化形式ごとのlayout、scale granularity、accumulator、fallback表。
- Qwen3.5-4B baselineで測定して決めるop別数値toleranceと性能回帰閾値。
- sudo以外の既存平文credentialの失効・rotationとsecret managerへの移行状況。
