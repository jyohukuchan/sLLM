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

## 現在の状態と次の作業

- 現在: Phase 0のCI・テスト方針を確定し、repository skeletonと初期CI・test harnessの設計開始待ち。
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
- 次:
  1. test result schema、compatibility tuple manifest、suite registry、path mapping、repository skeletonを設計する。
  2. tracked tree H0、local hygiene command、H1〜H2のCPU CIをrepository skeletonと同時に実装する。
  3. ROCm 7.14.0によるH3 compile-only CIを追加する。
  4. 専用local hostのGPU evidence実行・集約とG0 preflightを構築する。

## 未解決事項

- AMD consumer RDNA2を含む各exact gfx targetの実機検証状態。
- ROCm 7.14.0 system package環境の最小smokeを越える数値kernel、model、長時間安定性、性能と、HWE kernel 6.17上のmixed V620/R9700 tupleを正式な互換性対象にできるか。
- Qwen3.5-4Bで使用する完全commit SHAとlock manifest。
- resource gateの1 TOPSに用いる精度・operation数、16 GBの単位とdevice-local memoryの定義、帯域の算出方法、対応例外を承認する基準。
- Infinity Fabric、RDMA、KV永続化の詳細設計。
- 量子化形式ごとのlayout、scale granularity、accumulator、fallback表。
- Qwen3.5-4B baselineで測定して決めるop別数値toleranceと性能回帰閾値。
- sudo以外の既存平文credentialの失効・rotationとsecret managerへの移行状況。
