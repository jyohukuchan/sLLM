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
- 初期実装ではsafetensors形式のモデルを読み込む。最終的な公開runtimeのモデル入力と
  配布artifactはGGUFへ統一し、safetensorsは変換・開発用の入力へ移す。
- GUI以外の全機能をCLIから利用可能にする。
- AMD GPUを最初のbackendとし、RDNA2、RDNA4、CDNA3を対象候補とする。
- GPU操作、device memory、queue/event、operator dispatch、kernelはC++/HIPで実装する。
- frontend、model/config/tokenizer、scheduler、sampling、execution planはRustで実装する。
- OpenAI-compatible APIを提供する。
  - 初期仕様は `sLLM OpenAI-compatible Chat Completions profile v1` とする。
  - llama.cpp serverは実装参考・差分比較対象であり、仕様の正本にはしない。
  - [Responses APIに対応する。]
- model artifactの`max_position_embeddings`等はnative/公式推奨contextとして扱い、runtimeの品質hard gateにはしない。
  serverの実行上限はユーザーが`--context-length`で自由に指定でき、省略時だけmodel推奨値を既定値にする。
  推奨値を超える場合は起動時に一度だけ、設定値と公式推奨token数を警告する。追加opt-inやoverride flagは要求しない。
  requestのprompt tokenと要求output tokenの合計は設定した実行上限以内とし、32-bit位置表現、kernel dispatch、VRAM等の
  実装・資源制約によるfail-closed errorはmodel品質gateと分離する。推奨外の品質は保証せず、RoPE scaling等を明示指定する
  将来拡張とは別に管理する。
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

### ユーザー向けモデルコンテナ

- 2026-08-15のユーザー明示決定により、最終的な公開runtimeのモデル入力と配布artifactを
  GGUFへ統一する。ホビーユーザーにsafetensors shard、量子化sidecar、tokenizer等の
  複数artifactを個別管理させず、推論に必要なweight、scale、model metadata、tokenizer、
  vocabulary、chat templateを原則として単一GGUFへ収容する。
- 初期縦切りで実装したsafetensors direct loadと現在の量子化sidecarは、GGUF変換が完了するまでの
  開発・移行経路として扱う。最終的な公開runtimeではGGUFを正本とし、safetensorsは変換toolの
  入力として残せる。runtime内部のderived cacheは許容するが、別のユーザー管理artifactにはしない。
- GGUFコンテナへの統一は、Q8_0、Q4_K等の一般的なllama.cpp量子化形式を自動的に対応対象へ
  加える決定ではない。対応するtensor encodingと実行経路は別に決定する。
- safetensorsからGGUFへ変換する場合は、変換元lock fingerprint、変換toolのrepositoryとcommit、
  引数・設定、出力全体のSHA-256を記録する。runtimeのmodel lockはGGUF本体、metadata、tensor inventoryを
  検証対象とする。標準GGUFとの互換性を優先し、独自metadataまたはtensor typeが必要な場合は明示的に
  versioningする。

## 外部実装の参照とコード流用

- llama.cppとvLLMから、実装前に技術上の要点を抽出する。
- local `reference/` の公式origin、version、完全commit SHA、取得状態は[参照source固定マニフェスト](../references/source-lock.md)を正とし、固定sourceの参照範囲と採用判断は[推論engine参照](../references/inference-engines.md)へ記録する。
- 2026-08-02の追加調査対象からはLMDeployとKTransformersだけを正式なlocal参照sourceとして採用する。MLC LLM、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIは今回未採用とし、採用予定に置かない。
- vLLM等からコードを直接流用しない。参照sourceの表現を実装へ持ち込まないようreader記録とimplementation phaseを分離するが、別subagentの使用は必須にしない。
- llama.cppからの直接流用は許可するが、トップレベルLICENSEへの曖昧な追記だけで済ませない。
- MTPのspeculative decode/verify制御はllama.cpp実装を一括移植しない。llama.cpp issue
  [#25618](https://github.com/ggml-org/llama.cpp/issues/25618)で、量子化targetに対するdraft-model型speculationが
  greedy target-only生成から分岐する問題が報告されているため、同issueは回帰caseのsourceとしてのみ扱う。
  sLLMでは通常の逐次target decodeを数値oracleとし、draft tokenを順番に承認し、target-onlyと同じ計算結果を得る
  独自contractをPhase 18で実装・検証した。
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
- 性能candidateの採用単位を`adoption scope S`とする。`S`は同じproviderへrouteされるproduction入力の集合で、実行前に評価できる
  stable dispatch keyから定義する。
- dispatch keyはexact target、dtype/encoding、semantic op、shape/layout/alignment、request mode、mechanism上意味のあるcontext境界等で
  構成する。benchmark case名、prompt内容、実測後の結果、個別token列をkeyにしたoverfit分岐は作らない。
- candidateは次の全条件を満たすscope `S`へ採用できる。
  1. `S`の代表full-model caseの少なくとも一つがbaseline比5%以上改善する。
  2. `S`に属する全validation caseでstableな悪化を残さない。
  3. `S`外はbaseline providerへrouteされ、provider identityとselection overhead込みの性能にstableな悪化がない。
  4. correctness、fallback、resource、cleanup条件を全経路で満たす。
- `shared adoption`は`S`が固定matrix全体の場合、`scoped adoption`は`S`がその真部分集合の場合とする。管理性のためsharedを優先するが、
  scope外のcandidate単体悪化を理由に、安全に分離できる5%以上のscoped improvementを棄却しない。
- 数値範囲やcontext閾値をkeyにする場合は境界`B-1/B/B+1`とscope内の複数代表値を検証する。単一benchmark点しか裏付けないscopeは
  production採用しない。scopeのkey、代表case、境界、baseline complementをfinal performance run前にmanifestへ固定する。
- Phase 29だけは2026-08-18のユーザー明示決定により、上記1の5% thresholdをfull-modelではなくcommitted decode step内の
  GDN recurrent family全kernelのdevice時間へ適用する。full-model値はdiagnosticとし、このPhase固有例外を他Phaseへ一般化しない。
- 数値実装変更は[数値・出力影響変更台帳](../compatibility/numerical-output-changes.md)へ一元記録する。変更前とtoken列が異なっても、
  real-number semanticを維持し、差の原因が説明可能で、解析上の誤差boundまたは期待誤差が非増加となるN1変更は数値gateを自動承認する。
  既存tolerance内でも誤差が僅かに増加するN2変更は人間判断とし、原因不明・非有界・非決定のN3変更は採用しない。
- N1自動承認は数値互換性だけに適用し、性能、state/fallback、resource、cleanup、ABI、security/correctness defectのhard conditionは維持する。
  N1の定常承認に専用FP64/high-precision providerを要求せず、解析が曖昧なN2/N3の解消時だけ任意で作成する。

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
   - Phase 15Oとして、model本体のFP8/NVFP4量子化pathをdecodeとprefillに分け、Phase 16より前に最適化する。
     - decodeはdynamic FP8 activation量子化とFP8/NVFP4 M=1 providerを最適化する。
     - prefillはFP8 hipBLASLt solutionとNVFP4 packed-dequant tiled providerを最適化する。
   - Phase 15Qとして、Unsloth NVFP4 checkpointを使い、現行NVFP4の品質差を量子化algorithm、format ceiling、
     format mapping/runtimeへ切り分ける。
   - 提供元が公開するNVFP4 PTQ/QAT checkpointと、MXFP4/MXFP8でQATまたはnative公開されたmodelをfirst-class model
     inputとして扱う。sLLMがBF16から生成するPTQ converterの品質判定と、提供元quantized/native modelのsupport判定を分ける。
16. KV cache FP8/NVFP4へ対応する。
   - FP8 KVを先に、同じopaque KV encoding/layout境界へNVFP4を追加する。
   - append時に新規tokenだけを量子化し、attentionは全cacheのFP16/BF16 mirrorを作らず直接消費する。
   - 詳細は[Phase 16 archive](archive/2026/08/11-20/phase16-kv-cache-fp8-nvfp4.md)を正とする。
   - Phase 16Fとして、提供元NVFP4/MXFP4 modelをfirst-class inputへ統合する。
     - Phase 16のFP8 KV後に、Unsloth Gemma 4 12BのW4A4 MLP、W8A8 attention、FP8 KV、BF16/ignoreという
       mixed recipeをartifact metadataどおりに実行する。
     - NVIDIA Gemma 4 31B NVFP4はlocal 32 GiBへworkspace込みで収まらないためschema/reference対象、OCP MXFP4/MXFP8と
       Kimi K3はencoding/import対象とし、未実装MoE/architectureをFP4非対応と混同しない。
     - 詳細は[Phase 16F archive](archive/2026/08/11-20/phase16f-first-class-fp4-model-input.md)を正とする。
17. MTP、visionへ対応する。
   - fixed Qwen3.5-4BのMTP text-onlyを先に完成させ、draft/verify/accept、greedy同値、stochastic sampling、
     accepted prefixだけのKV publicationをgeneration serviceへ統合する。
   - MTP単独closeout後に同じmodelのprocessor、vision encoder/projector、multimodal prompt、Chat Completions image inputを実装する。
   - 詳細は[Phase 17 archive](archive/2026/08/11-20/phase17-qwen35-mtp-vision.md)を正とする。
18. MTPを通常生成と数値的に同一な内部高速化経路として完成させる。
   - fixed Qwen3.5-4Bの通常逐次decodeをoracleにし、draftを逐次承認する。量子化targetを含め、MTP on/offで
     target logits、visible token、commit済みKV、sampling/stop/usageの結果を一致させる。
   - 数値順序を変える一般的なmulti-token verifyを無条件採用せず、single-token decodeと同じrow arithmeticを保つ
     serial-equivalent batch、device-side orchestration、staged KVによってtarget launch/synchronizationを減らす。
   - target別の反復MTP off/on計測でnoise envelopeを越える改善を確認し、倍率を記録する。詳細は
     [Phase 18 archive](archive/2026/08/11-20/phase18-mtp-exact-sequential-speedup.md)を正とする。
19. Qwen3.5のMoEへ対応する。
   - `Qwen3.5-35B-A3B`の単一GPU text-only pathをprimaryに、router、top-8 routed expert、shared expert、
     decode/prefill別expert provider、CLI/OpenAI APIを実装する。
   - 詳細は[Phase 19 archive](archive/2026/08/11-20/phase19-qwen35-moe.md)を正とする。
20. ユーザー向けモデル入力と配布artifactをGGUFへ統一する。
   - safetensorsと量子化sidecarから、推論に必要な情報を収容した単一GGUFへの変換経路を用意する。
   - 公開runtimeはGGUFを正本として読み込み、変換元と出力をmodel lockで再現可能に固定する。
   - Phase 20はGGUF converter、loader/runtime、metadata/tensor type、model lock、移行・互換性のcloseoutだけを扱う。
     request batching、chunked prefill、簡易永続化、残るmodel/KV形式をPhase 20の範囲に含めない。
21. 通常decodeのsegment同期を限定的に最適化する。（完了、candidate棄却）
   - 通常modeのper-op timingを無効化し、同一streamの非空segmentを最大1 terminal completion eventへ集約する。
   - semantic op、kernel、provider、state publicationは変更せず、改善しないcandidateはdefaultへ採用しない。
   - 17 ownerを1 fence eventへ集約する構造削減は成立したが、final dual-GPU比較は0.14%/0.18%遅くnoise内だった。
     production defaultはprofiledへ戻した。詳細は[Phase 21 archive](archive/2026/08/11-20/phase21-limited-decode-sync-optimization.md)を正とする。
22. profile-guidedに通常decodeのBF16 `M=1` matvecを限定最適化する。（完了、candidate棄却）
   - Phase 21でwall改善を示さなかったevent同期candidateは拡張せず、current v4をbaselineにexact targetと実`K/N`を使う
     shape-aware providerを一候補ずつ比較する。
   - 最初のwork unitはQwen3.5-4B dense BF16のMLP `2560→9216` / `9216→2560`とし、vocabulary projectionは
     独立controlとして測る。DeepSeek V4、TurboQuant、fusion、batching、graph replayは混ぜない。
   - 8 shapeのfresh profileとwave32x8 candidateを評価したが、最終counterbalanced V620 wallは0.52%遅かった。
     candidateを除去しcurrent v4を維持した。詳細は
     [Phase 22 archive](archive/2026/08/11-20/phase22-profile-guided-decode-matvec-optimization.md)を正とする。
23. 既存推論engineとの差分と細粒度critical-path計測から、見落としている最適化余地を探索する。（完了）
   - Qwen3.5-4B BF16とcanonical V620/R9700でcold load、direct/API、prefill、decode、concurrencyを計測し、
     固定llama.cppとの比較可能性をE1/E2へ分離した。vLLM/SGLangはmatched runtimeがないためfacts-only比較に限定した。
   - 最大の新規発見は、prefillで最終行だけを消費するのに全`M`行のvocabulary projection/argmaxを実行することだった。
     256-token E2E Amdahl上限はV620 13.06%、R9700 37.92%である。
   - Phase 24候補をlast-row projection、projection-family fusion/shared load/plan replay、continuous batchingの順に絞った。
     production最適化はPhase 23へ含めていない。詳細は
     [Phase 23 archive](archive/2026/08/11-20/phase23-cross-engine-differential-performance-discovery.md)を正とする。
24. 通常prefillのterminal LM head/Argmaxを最終行だけへ限定する。（完了、shared candidate採用）
   - final RMSNorm後のchecked last-row viewで255 token以上の通常prefillのterminal projection/Argmaxを一行へした。
     short request、明示all-logits、MTP target/draftはall-rowを維持する。
   - 固定10 target/patternはすべて非悪化で、V620のP1/P2/P3は13.14%/12.08%/12.73%改善した。R9700も全patternで
     0.09〜0.49%改善したため、改訂後の全pattern非悪化かつ任意pattern 5%以上を満たした。
   - physical workspaceを126,644,220 bytes縮小し、dual-GPU oracle、sampling、MTP、profileをPASSした。
     target固有問題が残らなかったためgfx1030/gfx1201分岐は追加していない。詳細は
     [Phase 24 archive](archive/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)を正とする。
25. decode projection familyをbatch-compatibleに最適化する。（完了、A0でcandidateなし）
   - Phase 24後のfresh profileでprojection shareはV620 86.48%、R9700 79.23%だったが、支配分は異なるweightの必須readだった。
   - gate/upの共有可能activationはweight trafficの0.00543%で、profiler observer effect込みのlaunch完全除去上限も
     TPOTの0.94%/2.60%だった。5% full-model条件へ届くcredible work unitがないためproduction実装を開始せず完了した。
     詳細は[Phase 25 archive](archive/2026/08/11-20/phase25-batch-compatible-projection-family-optimization.md)を正とする。
26. continuous request batchingを実装する。（完了、candidate棄却）
   - waiting/decode-ready、checked row map、compatibility class、round-robin、backpressureのhost contractはPASSした。
   - 現行KV/GDN stateとpositionはrequestごとのscalarであり、既存`M>1`は一request内の連続token専用だった。独立requestへ
     流用せず、GPU `B>1`とthroughput改善を未達としてcandidateを棄却した。詳細は
     [Phase 26 archive](archive/2026/08/11-20/phase26-continuous-request-batching.md)を正とする。
27. exact decode差分からprojectionのweight-stream/providerを最適化する。（完了、candidateなし）
   - current sLLMと固定llama.cppをfresh比較したが、GGUF bytes、token stream、timing boundaryが一致しないためE1に限定した。
   - projectionはV620でsLLMが6.76%速く、R9700で12.53%遅かった。projection除外残差はprefill/MTP境界を含むcoarse値で、
     decode-onlyのcross-engine比率には使わない。
   - 全target非悪化かつ任意pattern 5%以上へ届く共通projection candidateを固定できず、production変更なしで完了した。詳細は
     [Phase 27 archive](archive/2026/08/11-20/phase27-exact-decode-projection-weight-stream-provider-optimization.md)を正とする。
28. decodeのprojection外device処理を限定最適化する。（完了、明示例外採用）
   - committed decodeを再計測し、最大familyのGDN recurrent state passを統合した。
   - full-model改善はV620 1.54%、R9700 2.97%で通常の5%基準未達だが、ユーザー明示例外としてshared pathへ採用した。
   - 5%規則自体は維持する。詳細は
     [Phase 28 archive](archive/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)を正とする。
29. GDNのuseful-workgroup並列化を限定最適化する。（完了、N1 shared candidate採用）
   - 現行32 workgroupに対する16/64/128/256の単純分割は、head直列化、idle thread、Q/K norm重複、output finalize追加により
     全variantが悪化した。workgroup数とCU数を一致させるだけの探索は繰り返さない。
   - Q/K/output normのwave32 reductionはV620の全patternを2.15〜2.20%、R9700を8.10〜9.21%短縮し、GDN-only性能条件を満たした。
   - output 128の5/6 target/patternでbaselineと生成token列が分岐したが、非負二乗和の逐次深さ127を固定tree深さ概ね8へ減らす
     解析的誤差低減N1変更として数値gateを自動承認した。性能条件も満たすためfull wave candidateをshared採用した。詳細は
     [Phase 29 archive](archive/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)と
     [bounded summary](../../ci/matrix/phase29-gdn-device-summary-v1.json)を正とする。

数値roadmapから独立した`Phase X`として、Qwen3.5系GDNのllama.cpp AMD性能調査・修正・sLLM還元を完了した。
Phase XはPhase 20の状態、完了条件、実行順を変更しない。

Phase番号を割り当てない将来タスクとして、READMEの整備と人間による発表がある。発表時期は未定であり、
Phase 19/20/21/22/23/24/25/26/27/28/29の完了条件、直後の作業、または番号付きPhaseとして割り当てない。

Phase 12のMI300Xを管理できない期間は、Phase番号と依存関係を維持したままPhase 13以降のlocal-only workを
先行できる。現在のGitHub CI不整合は製品Phaseを繰り下げず、Phase 12待機中のremediation subphase
`Phase 12R`としてPhase 13より先に修復する。実行順序、停止条件、Gemma 4後の共通RDNA性能bridge、枯渇防止tailは
[Phase 12待機中のローカル先行実行キュー](archive/2026/08/11-20/phase12-wait-local-forward-queue.md)を正とする。
待機queueのQ0〜Q4完了後、2026-08-15のユーザー明示指示でPhase 12を開始し、latest mainからexact `gfx942`
candidateを再確認した。

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
  昇格しない。当時の内部evidenceではR9700 native、V620 emulation、V620 `converted-bf16`を別provider scopeとして記録した。
  これは最終CLIへ異なる許可操作を要求する区分ではない。
- 詳細は[Phase 10 archive](archive/2026/08/11-20/phase10-fp8-w8a8.md)を正とする。

### Phase 11: FP8/BF16のCDNA3移植（完了）

- exact `gfx942`、wave64へBF16 kernel/providerを移植し、OCP E4M3FN model storageをVRAM load時に
  E4M3FNUZへ数値変換してhipBLASLt FNUZ providerへ渡す。generic targetやraw byte reinterpretを使わない。
- MI300XではVMMなしが想定されるため、opaque KV/attention契約を維持する`contiguous-resident` providerを
  追加する。VMM対応targetのvAttentionは維持し、Paged Attentionへの選定変更やsilent fallbackは行わない。
- exact gfx942 compile/link、全byte FNUZ oracle、wave64 BF16 provider、capability-selected contiguous-resident KV、
  production `native-fnuz` graph/service、MI300X dry-run runnerを完成した。実機PASSと性能値はPhase 12で取得する。
- 詳細は[Phase 11 archive](archive/2026/08/11-20/phase11-cdna3-port.md)を正とする。

### Phase 12: Hot Aisle MI300X単体実機確認（完了）

- Hot AisleのMI300X x1 Small VMを用い、exact `gfx942`のBF16/FNUZ FP8、wave64、contiguous-resident KV、
  4B/9B model、service、性能、llama.cpp比較をfail-closedに確認する。
- 192 GB HBM3の一台で現行single GPU/batch 1の検証には十分である。multi-GPU、Infinity Fabric、RCCL/RDMA、
  bare-metal固有挙動、別CDNA3 SKUは証拠範囲外とする。
- 利用時間はclean candidateで合計10〜12 GPU時間、現実的な上限16時間とする。2〜3時間のpreflightと
  6〜8時間のintegration/performanceを別sessionにし、必要な場合だけ追加4時間を別日に使う。
- P12-A0は実測identity、ROCm 7.14同一root、VMM/FNUZ/profiler/tiny runtime、exact gfx942 build/loadをPASSした。
  P12-A1はfeature suffix付きMI300X device名のfail-closed正規化、wave64 RMSNorm、GDNを含むoperator matrixを
  native数値oracleでPASSした。P12-A2は4B/9B BF16/FNUZ FP8、top-1/KLD、fixed/Unicode/stop generationをPASSした。
- P12-A3は実測VMM=trueでもPhase固定条件どおりexact gfx942を`contiguous-resident` KVへ明示固定し、1023/1024/1025、
  OpenAI raw/SSE/client/reasoning/disconnect/並行request、shutdown zeroをPASSした。P12-A4は4B BF16/FP8の4 caseを
  3 warmup＋10 measuredし、FP8のresident VRAM 42.4%減とE2E 17〜31%低下を記録した。同じBF16/token条件の
  fixed llama.cppにはsLLM E2Eで2.50〜5.58倍の差が残る。integration review、focused re-review、文書/evidence監査、
  repository外への証拠退避を完了し、ユーザー管理VMの削除と旧endpointの到達不能を確認してPhase 12を完了した。
- 詳細は[Phase 12 archive](archive/2026/08/11-20/phase12-mi300x-validation.md)を正とする。

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
  sLLM製PTQ converter candidateを採用しなかった。providerはnative FP4ではなく、このmodel品質判定とruntime correctnessは分ける。
- V620ではresident 3,763,686,080 byteから1,790,406,056 byteへ52.43%削減した。CLIとOpenAI
  non-stream/SSE/stop/Unicode/連続request/disconnect/cleanupをR9700で通した。
- sidecarのdevice payloadは量子化対象BF16 weight比`0.281250271`で、理論`4.5/16`と一致する。全residentには
  encoding非依存の1,018,251,968 byteがあるため、全体削減率52.43%は正常である。
- 両GPUのQwen3.5-2B short-odd・32/32をBF16/NVFP4各3 warmup + 10 measuredで比較した。NVFP4 decodeは
  V620で約21〜22%、R9700で約20〜22%低下し、R9700 prefill/TTFTは大幅退行した。memory削減だけでdefaultへ昇格しない。
- 詳細は[Phase 15 archive](archive/2026/08/11-20/phase15-weight-nvfp4.md)を正とする。

### Phase 15O: FP8/NVFP4 model量子化path最適化（完了）

- 2026-08-15のユーザー明示指示により、Phase 16より前に実行するbridge phaseとして追加した。
- FP8はdynamic activation量子化をwave reduction/native pair conversionへ更新し、R9700の代表M=1/M=32 shapeで
  7.48〜29.36%低遅延、Qwen3.5-4B 32/32でprefill `+5.89%`、decode `+10.69%`、E2E `-9.27%`となった。
- NVFP4 decode候補は改善せず棄却して従来device kernelを維持した。prefillはpacked weight K tileを最大8 M rowで
  共有するproviderを採用し、M=32 operatorでR9700 59.29〜59.51%、V620 51.21〜56.68%低遅延となった。
- resident/peak、sidecar、数値/fail-closed contractは不変。FP8のtarget別provider優先順位は維持し、NVFP4はaccuracy
  budget超過のsLLM製PTQ converter candidateだけを不採用とした。historical status labelはユーザー向け起動modeではない。
- 詳細は[Phase 15O archive](archive/2026/08/11-20/phase15o-model-quant-path-optimization.md)を正とする。

### Phase 15Q: Unsloth NVFP4品質要因の切り分け（完了）

- `google/gemma-4-12B-it`と`unsloth/gemma-4-12b-it-NVFP4`の固定artifact間で、共有BF16 source 349 tensorが
  byte-identicalであることを確認した。Unsloth MLP 144 tensorのpacked E2M1、E4M3 block scale、reciprocal global scaleを
  独立decoderで確定し、同じW4A16 sLLM providerへlosslessにimportした。
- 32 prompt・96 logit位置のmatched比較で、Unsloth `imatrix_mse` payload U0はsLLM min-max S0よりmedian KLDを
  R9700 `0.3315→0.1619`、V620 `0.3715→0.1736`へ改善した。top-1一致も`61.46%→79.17%`、
  `62.50%→76.04%`へ改善し、activation-aware calibrationの寄与を確認した。
- U0のweight MSEは全144 tensorでS0より悪く、改善位置もR9700 66/96、V620 61/96に留まった。最大KLDは
  `9.1781`/`7.5777`で既存budget `0.05`を超えたため、原因はalgorithmだけでも数学的な型限界だけでもない`mixed`と判定した。
  O0 weight-MSE searchも採用せず、S0/U0/O0をsLLM製PTQ converter candidateとして採用しない。この判定を提供元QAT/native
  checkpoint、NVFP4/MXFP4 encoding、または同じquantized artifactを正しく実行するruntime providerのsupport判定へ転用しない。
- 詳細は[Phase 15Q archive](archive/2026/08/11-20/phase15q-unsloth-nvfp4-quality-attribution.md)を正とする。

### FP4 model inputと内部状態の製品方針（決定済み、Phase 16F計画済み）

- NVFP4とOCP MXFP4を公式model input経路へ置く。NVFP4、MXFP4、対応するFP8/MXFP8 activation、model固有のmixed-precision
  recipeは別encodingとしてfail-closedに解釈し、異なるscale/group/layoutを同じ「FP4」として推測しない。
- sLLMがBF16 sourceから生成するPTQ artifactは、対応するBF16とのKLD、top-1、task品質でconverter採否を決め、既存budgetを
  暗黙に緩和しない。提供元PTQ/QAT checkpointは同じquantized artifactのreference runtime、提供元評価、task oracleで判定する。
  BF16が正本として存在しないnative low-bit modelへBF16 KLD gateを要求しない。
- `default`、`opt-in production`、`correctness-only opt-in`はユーザーに選択、確認、警告を要求するCLI modeにしない。
  量子化artifactを選ぶこと自体をユーザーの選択とし、低bitの一般的trade-offを理由とする警告を通常出力へ追加しない。
- 内部ではruntime成熟度、target別provider優先順位、converter品質、model/evidence scopeを独立に保持する。通常起動はartifact metadataと
  exact targetからproviderを自動選択し、開発・benchmark用の明示provider overrideだけを任意に残す。破損artifact、未対応encoding、
  実行不能targetは警告付き継続ではなくerrorにする。
- 最終GGUFではBF16、FP8、NVFP4、MXFP4に同じユーザー操作を使い、encoding/providerはloader内部で解決する。現行の
  safetensors＋sidecar＋provider引数は移行中の開発interfaceであり、最終UX contractではない。
- 2026-08-16に残タスクの依存関係を見直し、first-class FP4 full-model integrationをPhase 16Fとして詳細計画化した。
  Unsloth primary artifactがFP8 KVを要求するためPhase 16の後、MTP/visionの前に置く。詳細は
  [Phase 16F archive](archive/2026/08/11-20/phase16f-first-class-fp4-model-input.md)を正とする。

### Phase 16: KV cache FP8/NVFP4（完了）

- 同じopaque state、VMM virtual-contiguous/contiguous-resident、transaction境界へFP8/NVFP4 value/scale plane、append、
  packed attention direct consumptionを追加した。全cache FP16/BF16 mirrorや別encoding fallbackは作らない。
- exact `gfx1030`/`gfx1201`で各encoding 17 case、独立NumPy oracle、capacity、nonfinite、cleanup、実committed byteをPASSした。
  `gfx942`はcompile/link-onlyでありGPU PASSではない。詳細は
  [Phase 16 archive](archive/2026/08/11-20/phase16-kv-cache-fp8-nvfp4.md)を正とする。

### Phase 16F: first-class FP4 model input（完了）

- `unsloth/gemma-4-12b-it-NVFP4`をprimaryに、W4A4 MLP、W8A8 attention、Phase 16 FP8 KV、BF16/ignoreの
  mixed recipeをmodel pathだけから自動検出し、同じCLI/server操作で実行する。
- NVIDIA Gemma 4 31Bはschema/model-lock/reference、OCP MXFP4/MXFP8とKimi K3はencoding/import boundaryへ固定する。
  Kimi full modelはMoE/architectureとhardware capacityの後続課題であり、encoding supportと分ける。
- exact Unsloth artifactのdirect upload、W4A4/W8A8/static-FP8 KV full graph、通常CLI/server自動検出をcanonical
  V620/R9700でPASSした。same-artifact NVIDIA reference未実行のためmodel evidenceは`experimental`であり、通常UXは変えない。
- 詳細は[Phase 16F archive](archive/2026/08/11-20/phase16f-first-class-fp4-model-input.md)を正とする。

### Phase 17: Qwen3.5 MTP、vision（完了）

- fixed Qwen3.5-4BのMTP 15 tensorをcomponent manifest/graphへ昇格し、greedy/stochastic verify、opaque
  transaction、real-weight deterministic GPU component evidenceを実装した。canonical 2 targetのrunnerはdraftごとに
  target forwardを逐次実行してtarget-onlyより遅く、通常generation serviceへの正確な高速化統合とMTP off/on倍率は
  未確認である。この残差をPhase 18へ移し、MTP用の許可flagや品質警告は追加しない。
- vision 297 tensor、bounded PNG/JPEG/WebP/non-animated GIF decode、locked processor、multimodal embedding/mRoPE、
  lazy vision resident、CLI local image、Chat Completions Base64 data URLを実装した。HTTP(S) fetch/Files APIは行わない。
- V620/R9700でvision→64 projected token→text prefill/decodeをHIP-only、fallbackなし、deterministic、cleanup 0でPASSし、
  R9700の実CLI画像生成もPASSした。量子化text artifactのvision weight量子化は本Phaseへ広げていない。
- 詳細は[Phase 17 archive](archive/2026/08/11-20/phase17-qwen35-mtp-vision.md)を正とする。

### Phase 18: MTP逐次承認・target-only数値同一・最低限の高速化（完了）

- M=2..8のserial-equivalent target block、最初のrejectまでの逐次承認、KV/linear-state rewindとaccepted-prefix replay、
  既存one-token generation loop用adapterを実装した。sampled requestは同じpublic target sampler/RNGを保つ内部target-only選択とした。
- V620/R9700でBF16+FP16 KVとFP8 W8A8+static FP8 KVのM=`2/3/4/7/8`を実行し、token/hidden、M=8 raw logits、
  accepted-prefix K/V payloadを逐次M=1へbit/byte exact照合した。HIP-only、fallbackなし、cleanup 0だった。
- R9700 BF16 width 1は3 warmup + 10 measuredで中央値`1.0355x`、MAD`0.0028`、p10/p90 `1.0242/1.0448`となり、
  fixed Qwen3.5-4B text-only greedyで内部auto-selectする。V620 width 1は`0.9990x`でnoise内のため同じUXのtarget-onlyを維持する。
- 通常CLIとOpenAI non-stream/SSE/cancel/recovery/shutdownをR9700で実機PASSした。MTP用flag、opt-in、品質警告は追加していない。
- 詳細は[Phase 18 archive](archive/2026/08/11-20/phase18-mtp-exact-sequential-speedup.md)を正とする。

### Phase 19: Qwen3.5 MoE text-only production path（完了）

- `Qwen3.5-35B-A3B`をprimaryに、256 routed expertからtokenごとのtop-8と1 shared expertを実行するsparse MoEを
  単一AMD GPUのtext-only generationへ統合する。Qwen3.5 Denseのfull attention/GDNとPhase 16Fの
  low-bit recipe descriptorを再利用する。
- primary artifactは`amd/Qwen3.5-35B-A3B-MXFP4` revision
  `2e19c6576db91e5d5a93455415619262218bf8a1`、architecture/lineage controlは
  `Qwen/Qwen3.5-35B-A3B-FP8` revision `9d1823d2dee688a6b25e77009dc727688c44936e`に固定した。
- router/stable top-8、OCP MXFP4 routed expert、shared-expert gate、weighted combineをNumPy oracleとHIPで照合し、
  decodeはselected 8 + shared、prefillはactive pairのexpert別grouped executionとする。
- exact R9700/V620で22,009,574,016 byte resident、22,230,758,892 byte peak、40層full-model
  prefill/decode、CLI/OpenAI non-stream/SSE/cancel/recovery/seeded sampling/shutdownをHIP-only、fallbackなしでPASSした。
  full-model active pairはprefill 960、decode 320であり、256 expert全件実行ではない。
- integration reviewで検証後のshard path再openを除去し、verified descriptorからのpositional readへ固定した。
  additive MoE C ABI layoutをC/Rust probeへ追加し、actual 24.6 GB artifactとR9700 full-modelをfocused再検証した。
- Phase 19はMoE vision/MTP、batching、expert/tensor parallel、CPU offload、GGUF writer/readerを含まない。
- 詳細は[Phase 19 archive](archive/2026/08/11-20/phase19-qwen35-moe.md)を正とする。

### Phase 20: GGUF統一（完了）

- safetensorsと量子化sidecarを変換・開発入力へ移し、ユーザー向けのモデル入力と配布artifactを
  BF16、FP8、NVFP4、MXFP4で共通の単一GGUFへ統一する。
- Phase 20の範囲はGGUF converter、loader/runtime、standard/extension metadataとtensor type、model lock、
  移行・互換性の検証とcloseoutだけとする。その他の残機能をPhase 20へ混ぜない。
- bounded GGUF v3 reader、deterministic writer、`derived-gguf-lock-v1`、BF16/FP8/NVFP4/MXFP4 converterを実装し、
  公開CLI/serverを`--gguf`と`--derived-lock`だけの単一経路へ移行した。旧cache/sidecar/provider引数とdirect benchmark laneは
  公開parserから削除し、source importerはconverter・開発用に限定した。
- canonical V620 `gfx1030`とR9700 `gfx1201`でQwen BF16、Gemma mixed NVFP4、Qwen MoE MXFP4をsource経路と同じtop-1へ照合し、
  R9700ではQwen FP8も実行した。全実行はHIP-only、fallbackなし、cleanup 0で、MoE OpenAI server lifecycleもPASSした。
- 完了監査で検出したrank-5 tensor、GGUF公開経路のQwen vision/MTP、A5 timing evidence、旧help表記を修正した。
  pinned standard readerで4 artifactをmax rank 4としてparseし、canonical V620/R9700でvision、MTP、再生成FP8/MoEと
  3 warmup + 10 measured timing laneを確認した。詳細は
  [Phase 20 archive](archive/2026/08/11-20/phase20-gguf-unification.md)を正とする。

### Phase 21: 限定decode segment同期最適化（完了、candidate棄却）

- 単一request、batch 1の通常text decodeで、semantic opごとに作成・record・query・destroyしているcompletion/timing eventを、
  既存model-neutral execution segmentのterminal completionへ集約する。
- 通常modeのper-op timingを無効化し、profile/evidence modeだけでper-op HIP timingを維持する。非空segment当たりの
  terminal completion eventを最大1個とし、terminal成功後にownerを個別HIP queryなしでfinalizeする。
- semantic op、kernel、provider、tensor layout、transactional state publication、public standalone completion ABIは変更しない。
- token/position H2D統合、KV publication変更、HIP Graph/command-list、event pool、GEMV、batching、DeepSeek V4、TurboQuant、
  multi-GPUはPhase 21へ含めない。
- Qwen3.5-4B BF16 GGUFをprimary laneとしてcanonical V620/R9700でfresh baselineとcandidateを比較した。
  17 ownerを1 fence eventへ集約する構造削減、token/audit/cleanup一致はPASSしたが、final counterbalanced中央値は
  V620/R9700でcandidateが0.14%/0.18%遅く、いずれもnoise内だった。
- 固定した採用条件に従ってproduction candidateを棄却し、Qwen/GemmaはPROFILED defaultを維持する。
  deferred ABI/core primitiveはfault-testedな実験基盤として残すが、性能成果とは表記しない。詳細は
  [Phase 21 archive](archive/2026/08/11-20/phase21-limited-decode-sync-optimization.md)を正とする。

### Phase 22: profile-guided decode M=1 matvec最適化（完了、candidate棄却）

- Phase 21のevent集約は構造削減だけが成立してwall改善を示さなかったため、同じ同期micro-optimizationを継続せず、
  直近profileでGPU時間の主因だったdense BF16 `M=1` matvec本体を限定対象にする。
- P22-A0でcurrent providerを調査し、BF16 decodeが`M`とexact targetだけでsingle v4/wave64 variantを選び、`K/N`を
  選択に使っていないことを確認した。Qwen3.5-4Bの1-token graphは249 matvecで、MLP gate/up 64回とdown 32回を含む。
- 最初の比較familyをMLP `K=2560,N=9216` / `K=9216,N=2560`へ固定した。tied vocabulary
  `K=2560,N=248320`は同じfresh profileの独立controlとし、測定なしに同variantへ束ねない。
- 8 distinct shapeのfresh profileを両GPUで取得し、wave32x8 candidateを比較した。V620 gate/upはoperatorで約32%短縮したが、
  downは約35%、R9700主要2 shape加重値は約13%悪化した。gate/upだけへ絞った最終V620 wallも0.52%遅かった。
- 固定した採用条件に従ってcandidateを棄却し、新kernel ID/symbol/selectionを最終sourceから除去した。
  current v4/wave64をproduction defaultに維持し、8-shape profile evidenceと18-case f64 oracle拡張だけを残した。
- DeepSeek V4、TurboQuant、量子化path、fusion、H2D統合、event pool、graph replay、batchingは非対象とする。詳細は
  [Phase 22 archive](archive/2026/08/11-20/phase22-profile-guided-decode-matvec-optimization.md)を正とする。

### Phase 23: cross-engine differential performance discovery（完了）

- Qwen3.5-4B BF16のcanonical V620/R9700でcold load、warm direct/API、256-token prefill、128-token decode、
  concurrency=2を取得した。全production laneはHIP-only、fallbackなし、cleanup 0だった。
- 固定llama.cpp peerとの256-token prefillはE1 system-equivalentでsLLMが6.44x/6.60x長かった。fresh decodeはtoken列と
  出力長が異なるためE2に限定し、勝敗ratioを作らなかった。vLLM/SGLangもmatched runtime不在のためfacts-onlyとした。
- prefillの全行LM head/argmaxという見落としを新規発見した。LM-head-shaped workはdevice timeの13.48%/46.92%、
  normal E2E Amdahl上限は13.06%/37.92%だった。
- Gemma 4 R9700 controlはmatvec device share 83.67%を示し、decode matrix familyのmodel横断性を確認した。
  同時2 API要求はほぼ完全に直列化し、HTTP/SSE residualは約0.5〜0.6 msに留まった。fresh-process model-readyは
  10.53/11.60 sで、full-file hashと直列upload completionがcold path候補になった。
- Phase 24 shortlistは`P23-O1` last-row-only prefill projection、`P23-O2` projection-family fusion/shared load/plan replay、
  `P23-O3` continuous batchingである。production source/default/API/model formatは変更していない。詳細は
  [Phase 23 archive](archive/2026/08/11-20/phase23-cross-engine-differential-performance-discovery.md)、
  [bounded summary](../../ci/matrix/phase23-performance-discovery-summary-v1.json)、
  [technical note](../references/phase23-inference-engine-performance-differential.md)を正とする。

### Phase 24: prefill terminal-row projection optimization（完了、shared candidate採用）

- Phase 23の最上位`P23-O1`について、final RMSNorm後のlast-row viewから`[1,vocab]` logitsと`[1]` Argmaxを作る
  Qwen bounded candidateを実装し、明示all-logits/MTP pathを全行のまま分離した。
- 改訂後のP0/P1/P2/P3/D0 dual-GPU matrixは全10組で非悪化だった。V620 P1/P2/P3は13.14%/12.08%/12.73%、
  R9700の全5 patternも0.09〜0.49%改善し、任意pattern 5%以上の採用条件を満たした。
- one-row pathは通常requestの255 token以上に限定し、short request、明示all-logits、MTP target/draftはall-rowを維持した。
  host 19 test、dual-GPU distinctive-row oracle、sampling 3 profile、MTP幅2、profiler mechanism proofをPASSした。
- workspace high-waterは1,149,766,656 bytesから1,023,122,436 bytesへ縮小し、model-resident bytesは不変だった。
  correctness defectも性能悪化も残らなかったためgfx1030/gfx1201で共通経路を採用し、target分岐は追加していない。
  詳細は[Phase 24 archive](archive/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)と
  [bounded summary](../../ci/matrix/phase24-terminal-row-summary-v1.json)を正とする。

### Phase 25: batch-compatible projection-family optimization（完了、A0 negative discovery）

- fresh profileのprojection device shareはV620 86.48%、R9700 79.23%だったが、familyで除去できない各weight readが大半だった。
- gate/up pairの共有可能activation readはweight trafficの0.00543%で、32 launch/tokenを完全に除去するprofiler上限も
  V620 0.94%、R9700 2.60%だった。linear attentionは既にpacked、terminalは現行decode provider、prepared descriptorはcache済みである。
- 5% full-model条件へ届くcredible removable fractionがないため、A0でcandidateなしと判定した。production source、default、
  target分岐は変更していない。詳細は[Phase 25 archive](archive/2026/08/11-20/phase25-batch-compatible-projection-family-optimization.md)と
  [bounded summary](../../ci/matrix/phase25-projection-family-summary-v1.json)を正とする。

### Phase 26: continuous request batching（完了、candidate棄却）

- fresh C2完了時刻はV620 0.457/0.908秒、R9700 0.327/0.646秒で、現行productionの直列性を再確認した。
- `C=1,2,3,4,7,8`、compatibility class、row map、fairness、backpressure、cancel/errorを持つhost plannerを追加し、
  focused 5 testとserver全32 testをPASSした。in-flight cancelはcompletionまでactive resourceを保持して結果を非公開にする。
  ただしmodel/device resourceを所有せずproductionへ未接続である。
- 現行coreの`M>1`は同一request内の連続tokenで、KV/GDN length/stateはper-owner scalarである。独立requestを流用すると
  causal stateを共有するため、GPU `B>1`へ接続しなかった。必要なmulti-sequence state ABIは見積り1.5倍超として再計画対象にした。
- production scheduler/backend mutex/defaultは維持し、GPU batching/throughput改善を主張しない。詳細は
  [Phase 26 archive](archive/2026/08/11-20/phase26-continuous-request-batching.md)と
  [bounded summary](../../ci/matrix/phase26-continuous-request-batching-summary-v1.json)を正とする。

### Phase 27: exact decode projection weight-stream/provider optimization（完了、negative discovery）

- current sLLMの28-token prompt / 128-token decodeはV620 32.38 tok/s、R9700 37.00 tok/s、固定llama.cppのdecode-onlyは
  48.94/53.96 tok/sだった。GGUF bytes、token列、timing boundaryが異なるためE1 system-equivalentに限定し、engine勝敗は主張しない。
- mandatory projection 8.41 GB/tokenのdevice timeはV620でsLLM/llama.cpp 17.71/18.99 ms、R9700で17.85/15.86 msだった。
  V620ではsLLM projectionが6.76%速く、R9700だけ12.53%遅いため、両target共通のprojection provider gapではなかった。
- projectionを除くcoarse residualは両targetに残ったが、prefillの非projection kernel、R9700 MTP内部work、unmatched timing boundaryを
  含むためdecode-onlyのpeer比率には使えない。Phase 22で反対targetを悪化させた同系providerとPhase 25で棄却済みのfusionを再開せず、
  全target非悪化かつ任意pattern 5%改善へ届く共通candidateは`NO_COMMON_PROJECTION_CANDIDATE`とした。
- production source/default/target splitを変更せずnegative discoveryとして完了した。詳細は
  [Phase 27 archive](archive/2026/08/11-20/phase27-exact-decode-projection-weight-stream-provider-optimization.md)と
  [bounded summary](../../ci/matrix/phase27-weight-stream-summary-v1.json)を正とする。

### Phase 28: decode projection外device処理の限定最適化（完了、明示例外採用）

- Phase 27のprojection除外残差は、prefill projectionだけを除いたaggregateをnominal decode stepで割っており、prefillの
  GDN/attention/normとR9700のMTP内部workが残っていた。3.80倍/3.54倍というdecode-only比較を撤回し、A0/A1で再計測する。
- execution transactionのcommitted output stepを正規境界に、prefill、target decode、MTP draft/verify/replayを分離する。
  evidence/profile laneだけでdispatchをcomponent/familyへ写像し、production defaultへper-op timing overheadを追加しない。
- GDN recurrent、attention preprocess、causal attention/KV、RMSNorm、elementwise、Argmax、device copy/fillを分解し、
  fixable contributionが最大のcoherent work unitを一つ固定する。projectionとhost residualは計測controlに限定する。
- committed decode再計測ではprojection外がV620 3.584 ms、R9700 3.328 ms/token、最大familyはGDN recurrentだった。
  state copy/decay/projectionのpass統合でGDNを23.18%/56.23%、full-modelを1.54%/2.97%改善した。
- 5%規則は変更していないが、ユーザー明示承認により本candidateだけを例外採用した。token IDs、all-HIP、fallback、cleanupを確認し、
  target splitなしのshared pathとした。詳細は[Phase 28 archive](archive/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)と
  [bounded summary](../../ci/matrix/phase28-nonprojection-summary-v1.json)を正とする。

### Phase 29: GDN useful-workgroup並列化最適化（完了、N1 shared candidate採用）

- Phase 28 productionの32-workgroup fused GDNをbaselineにする。探索した16 workgroupはV620/R9700のGDN device時間を
  135.6%/373.9%、64/128/256 workgroupの単純row splitも最低3.17%/11.82%悪化させた。
- 原因はworkgroup数不足だけではなく、16でのvalue-head直列化、64以上でのidle thread、Q/K norm・gate・decay重複、
  block間同期に伴うoutput finalize追加である。CU数だけからworkgroup総数を選ぶruntime policyは作らない。
- prepared/tiled構造とbounded variantを比較し、追加launch/scratchなしのwave32 reductionを最終候補に固定した。
  projection、causal convolution、attention、scheduler、GGUF/KV formatは変更していない。
- ユーザー明示決定により、Phase 29の採用可否はfull-modelではなくcommitted decode stepのGDN recurrent family全kernelの
  device p50で判定する。scope内全validation pattern非悪化かつ任意代表pattern5%以上ならshared/scoped adoptionできる。
  full-model TPOT/E2Eはdiagnosticとして記録するが採否を変更しない。
- 14 request × 16 output、B0/B1/B2、各variant 3 processを一度に一GPUだけで取得した。full wave candidateはV620 2.15〜2.20%、
  R9700 8.10〜9.21%短縮し、baseline endpoint driftは最大1.34%だった。
- model-free oracleと16-token列は一致し、output 128では6 pattern中5 patternが20〜112 token目で分岐した。同一provider repeatは再現した。
  対象が非負二乗和で、実数式不変のまま逐次深さ127を固定tree深さ概ね8へ減らすためN1へ分類し、token差を台帳へ記録して
  数値gateを自動承認した。GDN性能条件も満たすfull wave candidateをshared採用し、target splitは追加していない。詳細は
  [Phase 29 archive](archive/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)と
  [bounded summary](../../ci/matrix/phase29-gdn-device-summary-v1.json)を正とする。

### Phase X: Qwen3.5系GDNのllama.cpp AMD性能調査・修正・sLLM還元（完了）

- Qwen3.8-27B/Qwen3.5 architectureで観測したllama.cpp HIPの長prompt prefill崩れと低decode性能を、
  GDN prefill、GDN decode、MTP、KV/context memory、Harness/API overheadへ分解した。
- 根因はGDNではなく、HIP buildの`GGML_CUDA_FA_ALL_QUANTS=OFF`によりQ5_1 K/VがFlash Attentionから外れたことだった。
  `ON`のfresh buildで9,435-token promptのprefill/decode中央値はV620が340.80/33.42、R9700が
  779.06/41.93 tok/sとなり、旧HIP build比5.59x/4.91x、11.21x/3.35xへ改善した。
- Qwen exact shapeのQ5_1 Flash-Attention numerical testを`gfx1030`/`gfx1201`で各18/18 PASSし、CPU/backend fallbackと
  GTT spillがないことを確認した。当時のspare V620 local subagentを修正buildへ切り替え、後続決定で同buildをTP2へ拡張した。
- local subagentの現行起動・Pi接続・main agent利用契約は
  [Local Qwen3.8 subagent](../development/local-qwen-subagent.md)を正とする。2026-08-17の追加決定により、現行運用は
  V620×2 tensor split `1,1`、parallel 2、non-unified KV、actual context 491,520/slot、983,040 total、全model layer
  GPU offload、Q5_1 target/draft KV、MTP幅3である。単一V620構成へfallbackしない。
- boundedな委譲ではlocal Qwen subagentを優先し、Piまたは互換Harness processを合計2つまで同時実行する。Qwenが利用不能・不適切、2 slotが
  使用中、または追加並列性が有用なら、Qwen待ちで直列化せずnative Codex subagentを使用する。subagent利用自体は従来どおり
  完了条件ではない。main taskがsLLM GPU作業でいずれかのV620を必要とする間はQwenを利用不能として扱い、idle serviceを
  停止してpairを解放し、Codex subagentを使用する。
- post-closeout multi-GPU比較では独立V620 server 2基が最大aggregate throughputだったが、単一endpointで2 subagentと
  約0.5M context/slotを両立する運用要件を優先し、V620×2 tensorを524,288から491,520/slotへ縮小して通常起動へ昇格した。
  R9700+V620×2 layer split `5,2,2`はR9700を占有するため非運用のままとする。比較詳細は [multi-GPU selection
  summary](../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json)を正とする。
- VulkanはHIP原因を見分ける比較controlだけであり、sLLMの対応backendへ追加しない。Q5_K_XL/Q5_1も
  llama.cpp診断条件に限定し、sLLMの一般INT量子化support方針を変更しない。
- sLLMはFP16 KVを使用し、`linear_attention.gdn.v1`は原因でなかったためsource変更と新規provenance eventは不要と判断した。
  Phase Xは数値roadmapから独立したまま完了し、Phase 20のGGUF統一条件を変更しない。詳細は
  [Phase X archive](archive/2026/08/11-20/phase-x-qwen35-gdn-amd-performance.md)を正とする。

## 残タスクと改訂した実行順序

| 順序 | Phase | 主要成果 | 先行理由・依存 |
| ---: | --- | --- | --- |
| 完了 | Phase 16 | FP8/NVFP4 KV append・attention・capacity・quality | first-class Unsloth mixed recipeのFP8 KV依存を満たした |
| 完了 | Phase 16F | NVFP4 full mixed artifact、MXFP4/MXFP8 encoding/import | faithful provider artifact経路とGGUF handoffを固定した |
| 完了 | Phase 17 | Qwen3.5 MTP component、vision、multimodal CLI/API | MTP性能統合は未完、vision実機PASS |
| 完了 | Phase 18 | MTP逐次承認、target-only数値同一、最低限の高速化 | R9700でexact MTPを内部採用、V620はtarget-only維持 |
| 完了 | Phase 19 | Qwen3.5-35B-A3B MoE text-only production path | R9700/V620の通常CLI/APIへ統合済み |
| 完了 | Phase 20 | GGUF統一のみ | hobby user向けmodel inputと配布artifactを単一containerに固定した |
| 完了・棄却 | Phase 21 | 通常decodeのper-op timing無効化とsegment terminal completion集約 | 構造削減はPASSしたがdual-GPU wall差がnoise内のためproduction defaultへ不採用 |
| 完了・棄却 | Phase 22 | shape-aware BF16 M=1 matvec providerの限定比較 | operator局所改善がwallへ転化せずcurrent v4を維持した |
| 完了 | Phase 23 | cross-engine差分と細粒度critical-path計測による最適化余地探索 | prefill全行LM head、decode projection family、service serializationを上位候補として抽出した |
| 完了・採用 | Phase 24 | prefill terminal LM head/Argmaxのlast-row限定 | 固定10組非悪化、V620の3 prefill caseで12.08〜13.14%改善しshared pathを採用した |
| 完了・候補なし | Phase 25 | batch-compatible projection-family最適化 | fresh profileで5%へ届くcredible removable fractionがなく、production変更なしでnegative closeoutした |
| 完了・棄却 | Phase 26 | continuous request batching | host plannerはPASSしたがscalar KV/GDN ABIを安全なGPU `B>1`へ接続できずproduction未採用 |
| 完了・候補なし | Phase 27 | fresh decode差分とprojection weight-stream/provider調査 | 比較はE1に限定し、共通projection candidateなしでproduction変更せず完了した |
| 完了・例外採用 | Phase 28 | committed-step単位のprojection外device短縮 | GDN state pass統合を両GPU共通pathへ採用。通常の5%規則は維持 |
| 完了・採用 | Phase 29 | GDN useful-workgroup並列化 | 解析的誤差低減N1としてtoken差を記録し、GDN性能条件を満たすshared wave reductionを採用した |
| 完了・限定採用 | Phase 30 | RDNA4 native attention/KV hardware-path最適化 | gfx1201 native FP8 readとwave providerをM=1/M>=32へ採用、append encodeとmatrix候補は不採用 |
| 完了・採用 | Phase 31 | low-bit KV通常運用向けchunked prefill・workspace memory基盤 | arenaを約86.79%縮小し、両targetの10k+ FP16/FP8とgfx1201の16,385-token 2-chunkを成立させた |
| 完了 | Phase X | Qwen3.5系GDNのllama.cpp AMD性能調査・修正・sLLM還元 | Q5_1 HIP Flash Attention build coverageを修正し、local subagentへ採用 |

Phase 23は残る性能候補を実装せず、既存engineとのmatched comparisonと細粒度計測から再評価して完了した。
最上位last-row projectionをPhase 24、projection-family最適化をPhase 25、continuous request batchingをPhase 26へ、
残るexact decode weight-stream/provider差分をPhase 27、committed decodeのprojection外device短縮をPhase 28、
GDNのuseful-workgroup並列化をPhase 29へ
ユーザー指示で割り当てた。2026-08-19のlong-context KV調査で、FP16/FP8 causal attentionがgfx1201でも同じ
scalar/vector kernelを使い、native FP8 conversion、packed dot、WMMA/SWMMACを使用していないことを確認したため、
RDNA4 native attention/KV hardware-path最適化をPhase 30へ割り当てた。Phase 30はgfx1201 native E4M3FN readと
wave32 causal attentionを`M=1`/`M>=32`へ限定採用し、Qwen3.5-4B BF16の4108 inputでTTFT 9.60%、decode throughput
7.86%の3-process中央値改善を確認して完了した。native append encodeはchunk 256悪化、matrix/FlashAttention providerは
bounded N0/N1 work unitを越えるため不採用とした。low-bit KVの通常運用に必要な10k+ full-model検証を成立させるため、
chunked prefillとliveness-aware workspace memory基盤をPhase 31へ割り当てた。Phase 31はvAttention型
`virtual-contiguous` providerを維持したまま完了した。completion-boundary liveness slotにより10,001-token workspaceを
39.95 GB相当から5.28 GBへ縮小し、gfx1030/gfx1201の10,001-token FP16/dynamic FP8、gfx1201の16,385-token
FP16/dynamic FP8 2-chunkをHIP-onlyでPASSした。CLI/serverへ明示的なFP16/dynamic FP8/static FP8/NVFP4選択を追加したが、
defaultはFP16を維持する。Paged Attention、native FP8 append encode再検証、low-bit KVのdefault昇格は含めない。
詳細なacceptanceと結果は[Phase 31 archive](archive/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)および
[bounded summary](../../ci/matrix/phase31-chunked-prefill-summary-v1.json)を正とする。
その他に残る将来項目はKV/会話/model lockの簡易永続化、TurboQuantを含む残りKV形式、残るmodel family、
multi-GPU/Infinity Fabric/RDMA、README整備、
人間による発表である。これらには現時点でPhase番号を割り当てない。Responses API、LMCache、RadixAttention、
将来MX形式等の角括弧項目は初期versionの完了条件へ読み替えない。完了済みのPhase 18へ後続範囲を逆流させない。

### 性能最適化backlog

- 2026-08-18のcurrent source、直近full-model profile、性能historyを横断して確認できた明確な最適化余地を、
  Phase Xへ切り出したQwen3.5系GDN/llama.cpp AMD調査とPhase 21で棄却した限定segment同期を除き、
  この節で管理する。dense BF16 `M=1` matvecの最初のwork unitをPhase 22へ割り当て、Phase 23でinventoryを
  cross-engine差分、critical-path share、Amdahl上限から再分類した。prefill last-row projectionをPhase 24、
  batch-compatible projection-family optimizationをPhase 25、continuous request batchingをPhase 26、
  exact decode weight-stream/provider optimizationをPhase 27、projection外device短縮をPhase 28へ割り当て、
  cold loaderは未割当のまま維持する。
  このinventoryは完了Phaseのscopeを拡張せず、個別taskの受入条件、
  実装順、対象targetは着手時のfresh profileで固定する。
  一般論だけの候補をhard gateにしない。
- runtime dispatch・同期:
  - 現行native runtimeはsemantic opごとにcompletion owner、HIP completion event、timing event、registry handleを
    生成し、同一streamのsegment末尾で各ownerを個別queryする。Phase 21でsegment単位completionを比較したがwall改善が
    noise内だったため通常defaultへ採用しなかった。event/completion pool、
    registry lock削減、parameter更新可能なnative command-listまたはproduction graph replayはPhase 21へ含めず、
    未割当backlogとして維持する。requestごとの素朴なgraph instantiateは再導入しない。
  - decodeのtoken IDとpositionを別々の同期付きH2Dにせず、一つのstaging transferまたはdevice-side position生成へ
    まとめる。terminal argmax完了と4-byte token readbackも一つのstream boundaryへ含める。
  - full-attention layerごとのKV append host waitは、accepted stateだけを公開するtransaction contractを維持したまま、
    stream-ordered publicationまたはstaged stateで集約できるか測る。
- attention・KV hardware path:
  - 現行generic causal attentionはFP16 ID 2とpacked-KV ID 3を報告するが、実体はencoding引数で分岐する同一kernelであり、
    gfx1201 code objectにもnative FP8 conversion、packed dot、WMMA/SWMMACがない。Phase 16のcorrectness/memory baselineとしては正しいが、
    RDNA4 hardware性能を評価するproviderではない。
  - Phase 30でgfx1201 native FP8 codec、decode wave tile、prefill matrix attentionを別work unitとして比較した。native FP8 readと
    wave32 reductionは採用し、gfx1030と`M=2..31`はbaseline controlを維持する。native append encodeとmatrix providerは不採用である。
    10000+ full-model prefillは現行workspaceが利用可能VRAMを超えるため、chunked-prefill/resource workをPhase 31へ割り当てた。
  - Q/PのFP16/FP8化、softmax順序、accumulator変更は数値台帳のN0〜N3へ分類し、N2を性能だけで自動採用しない。
- Dense BF16 execution:
  - Phase 27のfresh E1比較では、V620 projectionはpeerより6.76%速く、R9700だけ12.53%遅かった。両target共通の
    projection provider gapではなく、全target非悪化かつ任意pattern 5%改善へ届くwork unitを固定できなかった。
  - Phase 27のprojection除外coarse residualはprefill非projection workとR9700 MTP内部stepを含んでいたため、peerに対する
    3.80倍/3.54倍claimを撤回した。Phase 28でcommitted output step単位にfamilyを分解し、device処理だけを短縮する。
  - production graph/command-list、gate/up+SiLU等のfamily fusion、R9700限定projection providerは未割当backlogとして維持する。
    後者はR9700のstable adoption scopeで5%以上改善し、gfx1030等をbaselineへ確実にrouteできる共通registry keyがあれば再検討できる。
- FP8、NVFP4、MXFP4 model path:
  - Q/K/Vやgate/up等、同じBF16 activationを消費する複数linear間でdynamic FP8/NVFP4/MXFP4 activationとscaleを
    共有する。RMSNorm等のproducerからの直接量子化、quantize+matmul融合、M=1専用quantizerも比較する。
  - FP8 hipBLASLtはzero-workspace・単一heuristicに限定せず、有限workspace、複数solutionの実測、shape/target別
    algorithm cache、queue/stream別handleを比較する。gfx942 FNUZは旧activation quantizer固定を解消できるか実機で測る。
  - W4A4 quantizerはblock 16/32当たりのthread利用率、packed load/store、scale reductionを改善する。decodeとprefillで
    実質同じscalar bodyを使う経路を分離し、prefillのM/N/K tile共有、利用可能なtargetのnative matrix path、
    native pathを持たないtargetのpacked-dequant tiled GEMMを実装候補とする。
  - low-bit prepared planごとの`hipMalloc` workspaceをrequest arenaへ集約し、同時実行しないlinear間で再利用する。
- sparse MoE:
  - 現行MXFP4 routed expertとBF16 shared expertのscalar K loopを、expertごとにtokenを実際にbatch化するgrouped GEMM、
    target別packed/native matrix providerへ置き換える。現行のexpert別groupingはblock順序を整えるだけであり、
    weight tileを複数tokenで共有するGEMMにはなっていない。
  - routed gate/up、SiLU、intermediate quantization、down、routing weight、shared expert combineの境界をprofileし、
    数値順序を保てる範囲でfusionとweight/activation tile再利用を行う。shared expertは既存BF16 providerを再利用する。
  - routerのstable top-8、softmax、expert groupingはthread-0/全pair再走査からwave-parallel reduction、stable selection、
    prefix-sum/compactionへ移す。router projection、status初期化、top-k/group metadata生成の融合も候補とする。
- exact MTP:
  - draft argmaxしか使わない経路のfull-vocabulary logits D2Hを除去し、target/MTP hidden stateをhost `Vec`経由で
    D2H/H2Dせずdevice residentのまま接続する。MTP prompt prefillのtoken-by-token host loopもdevice側でまとめる。
  - draft、serial-equivalent verify、逐次acceptをdevice-side orchestrationへ寄せ、reject時のaccepted-prefix replayを
    staged/COW state commitへ置き換えられるか検討する。通常逐次生成とのtoken、logits、KV、sampling結果の一致は維持する。
  - overhead削減後にacceptance率とtarget別profileからdraft幅を自動選択する。R9700の幅拡大、量子化path、sampled path、
    V620再評価は同じ内部UXで行い、現行の遅い幅を無条件に有効化しない。
- sampling、frontend、service:
  - non-greedy requestのfull-vocabulary BF16 logits D2H、CPU F32変換、全語彙penalty・sortをGPU samplingへ移し、
    hostへは選択tokenだけを返す。CPU samplingを残す場合もcandidate buffer、token count、partial selectionを再利用する。
  - schedulerの単一workerとgeneration全体を保持するbackend mutexをcontinuous batchingへ置き換え、decode batch、
    chunked prefillとのinterleave、per-sequence state、queue/stream別library handleを設計する。
  - bounded SSE event channelの`blocking_send`でGPU generationまで停止しないよう、boundedな内部ringとnetwork writerを
    分離する。disconnect cancellation、backpressure上限、visible output順序は維持する。
  - generationごとに全既出tokenを複製してprefix全体をdecodeし、全文snapshotを保持するhost O(n^2)経路を、
    byte-fallbackを保つincremental decoderと短いrollback windowへ置き換える。
- request state、memory、long context:
  - requestごとのgraph再構築、dynamic tensor単位のdevice allocation、KV/GDN state、prepared cacheの作り直しを、
    graph template cache、liveness arena、tensor alias、request owner/state pool、decode M=1 plan再利用へ移す。
  - prefix token列、model lock fingerprint、KV encodingをkeyにしたprefix/KV cacheとvAttention page共有/COWを検討する。
    KV、会話、model identityの簡易永続化は再起動後の再prefill削減にも利用する。
  - Phase 31ではchunked prefillによりprefill workspaceをselected chunkへboundedとし、同時liveでないrequest-owned
    intermediateだけをliveness arenaで再利用する。automatic defaultはtotal VRAM `<=16 GiB`で512、`>16 GiB`で
    16K/8K/4K/2Kを大きい順にfit判定する。vAttention型`virtual-contiguous` providerをproduction defaultとして維持し、
    Paged Attentionはopaque KV state下の別physical-layout providerとして後続比較へ残す。
  - chunked prefillは長promptのlatency/peak memoryとrequest間fairnessを改善し、現行matmul一dispatchのM上限
    `65,536`を超える設定contextを実行可能にする境界として実装する。Phase 31の直接の採用目的はまず10k+ full-modelの
    memory feasibilityとlow-bit KV検証成立であり、5%速度改善を要求しない。
  - gfx942実機はVMM capabilityがtrueだったため、長い設定contextで全capacityを物理確保する
    `contiguous-resident`固定と、virtual-contiguousまたは増分commit providerを再比較する。
- model load、GGUF、vision:
  - shard/artifact検証後の再読込とtensor/chunkごとの同期uploadを、mmap、並列hash、disk read/CPU変換/H2Dの
    double buffering、複数transferの集約waitへ移す。検証済みidentity cacheは内容検証contractを弱めず利用する。
  - GGUF converterは単一container化だけでなく、runtimeのrow-major/transposed packed weight、scale plane、MoE layer blobを
    execution-readyに配置し、起動時repack、sidecar join、FNUZ等のtarget変換を減らせる余地を残す。ただしこの性能項目を
    Phase 20の追加完了条件にはしない。
  - visionはlazy residentのcold start、複数画像の逐次実行、vision embeddingのhost readback/text graph再uploadを対象に、
    preload、image batch、device-to-device binding、image digest cache、greedy multimodalの不要logits readbackを検討する。
- context・target依存の条件付き候補:
  - 短contextではfull attentionは支配要因でない。long-context fresh profileで支配的になった場合だけ、full/sliding-window別
    tiled online softmax、FlashAttention系provider、quantized KVのvectorized unpack/scale共有、GQA head間KV tile共有を扱う。
  - gfx942はBF16で固定llama.cpp比の大きな差が残るため、実機再取得時にwave64 MMVF、launch replay、GEMM solution、
    FNUZ quantizerを分離して再profileする。単一MI300X VMの結果を別CDNA SKUへ一般化しない。
  - multi-GPU、expert/tensor/pipeline parallel、Infinity Fabric/RCCL/RDMAはcapacity・batch throughput候補とし、
    単一requestやPCIe構成では通信費を含む実測後に採否を決める。
- 現時点でそのまま再提案しない候補:
  - V620の一般的なM>1 hipBLAS切替、R9700のtransposed GDN state、既存weight-only NVFP4 decodeの複数N列・
    scale broadcast、V620で現状のままMTP幅2を有効化する案は既存実測で改善しなかったため再採用しない。
  - 短contextでのfull attention/FA3-like最優先化とrequestごとのproduction HIP Graph instantiateも再採用しない。
    前提となるprofileまたは実装構造が変わった場合だけ、新しいcandidateとして別に測定する。

## 現在の状態と次の作業

- Phase 29はGDN-only採用指標で性能条件を満たすwave reductionを得た。長生成token差は非負二乗和の固定tree化による説明可能な
  解析的誤差低減N1として台帳へ記録し、数値gateを自動承認してshared production pathへ採用した。target splitはない。詳細は
  [Phase 29 archive](archive/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)と
  [bounded summary](../../ci/matrix/phase29-gdn-device-summary-v1.json)を正とする。
- Phase 28はGDN state pass統合を明示例外として完了し、Phase 29のproduction baselineとする。Phase 28自身の通常5%規則は維持する。詳細は
  [Phase 28 archive](archive/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)を正とする。
- Phase 27はfresh E1比較とweight-stream accountingを完了した。V620のprojectionはpeerより速く、R9700だけに約7.36%の
  full-model楽観上限があったため、全target非悪化かつ任意pattern 5%以上へ届く共通projection candidateはなかった。
  production sourceとtarget分岐を変更せずnegative completionとした。詳細は
  [Phase 27 archive](archive/2026/08/11-20/phase27-exact-decode-projection-weight-stream-provider-optimization.md)を正とする。
- Phase 25はPhase 24後のfresh dual-GPU profileで5%へ届くprojection-family候補がないことを確認し、production sourceを
  変更せずnegative completionとした。Phase 26もfresh C2 baselineとhost plannerを完了したが、現行scalar KV/GDN ABIでは
  独立requestのGPU `B>1`が安全に表現できずcandidateを棄却した。batching再開にはmulti-sequence state ABIの独立計画が必要である。
- Phase 24は改訂後の採用条件を満たして完了した。固定10 target/patternはすべて非悪化で、V620 P1/P2/P3は
  13.14%/12.08%/12.73%改善した。shared last-row pathとphysical one-row allocationを採用し、gfx1030/gfx1201分岐は
  追加していない。short request、明示all-logits、MTP target/draftはall-rowを維持する。
- Phase 21は17 ownerを1 fenceへ集約する構造削減をPASSしたが、canonical V620/R9700のE2E中央値が
  0.14%/0.18%遅くnoise内だったためcandidateを棄却し、production defaultをPROFILEDへ戻して完了した。
- Phase 22は8 distinct shapeのfresh profile、wave32x8 candidate、dual-GPU oracle、counterbalanced full-model比較を完了した。
  V620 gate/upのoperator改善はwallへ転化せず、最終candidateが0.52%遅かったため棄却した。current v4/wave64を維持し、
  profile evidenceと18-case oracle拡張だけを残した。DeepSeek V4、TurboQuant、fusion、H2D統合、event pool、graph replay、
  batchingはPhase 22へ含めていない。
- Qwen3.5系GDNのllama.cpp AMD性能調査・修正・sLLM還元は数値roadmapから独立したPhase Xとして完了した。
  Q5_1 HIP Flash Attentionをall-quant buildで有効化してlocal V620 subagentへ採用し、sLLM GDN source変更は不要と判断した。
  GGUF統一の完了条件へ含めず、上記inventoryのうちPhase 25/26へ割り当てていないproduction実装は未割当backlogとして維持する。Phase 23は
  inventoryの計測・再分類だけを担当する。
- 2026-08-16のユーザー決定により、提供元NVFP4/MXFP4 QAT/native modelを公式入力とし、低bit形式を理由とする追加opt-in、
  起動コマンド差、通常警告を最終UXへ設けない。内部状態とconverter品質は上記FP4製品方針に従って分離する。
- 最終的なユーザー向けモデル形式をGGUFへ統一するPhase 20は完了し、safetensors direct loadと量子化sidecarは
  converter・開発入力へ移行した。Phase 21/22はGGUF/model-lock形式を変更しない。
- README整備と人間による発表は時期未定の将来タスクであり、番号付きPhaseやPhase 19/20/21/22/23/24/25/26/27の完了条件に割り当てない。
- MI300Xを管理できなかった期間はPhase 12を`ready`で保持し、local forward queueに従ってPhase 12R、Phase 13、
  Phase 14、共通RDNA性能bridge、Phase 15の順に先行する。Phase 12RでGitHub host/compileとtrusted local GPUの
  verification境界を修復し、Phase 13で共通execution制御を抽出し、Phase 14でGemma 4 production pathを完了した。
  共通RDNA性能bridgeとPhase 15を完了後、MI300X Phase 12も完了した。
- Phase 9のdtype非依存completion/segment骨格とtarget別BF16 providerを再利用し、Phase 10でFP8 encoding、
  sidecar/loader、native/emulation/conversion providerを追加した。Phase 13でモデル非依存層へ抽出し、
  Phase 15開始前にもfresh profileで
  memory-bound matvec、production graph/command-list、MLP fusionの優先順位を再確認する。RDNA4 FA3-likeは
  attentionが支配要因になった時の非blocking follow-upとして別管理する。
- Phase 11でMI300XのVMMなしに備える`contiguous-resident` KV providerを実装した。Phase 12ではHot Aisle MI300X x1
  Small VMで検証し、単一VMの性能証拠をmulti-GPU、bare metal、MI300A/MI325Xへ一般化しない範囲で記録した。
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
