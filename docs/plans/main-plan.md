# sLLM メイン計画

## この文書の役割

- Git管理外の `sLLM.md` にある要件定義・開発方針・重要な決定を、開発に必要な範囲で追跡可能な形へ同期する。
- この文書には重要な製品・アーキテクチャ・互換性上の決定、開発計画と順序、進捗、未解決事項だけを記録する。恒久的な実行手順は各正本文書へ置き、ここには重複させない。
- `sLLM.md` とこの文書に方針上の差異が生じた場合は、推測で統合せずユーザーへ確認する。
- 角括弧内の項目は、初期バージョンでは対応しない将来機能を表す。
- プロジェクト内の権限順は、現在の明示的なユーザー指示、`sLLM.md`、`AGENTS.md`、この文書の承認済み決定、進行中計画の作業固有条件、履歴に残す過去の事実、とする。下位文書と履歴は上位方針を上書きせず、新しい完了条件や阻害条件を作らない。

### 表記方針

- 説明文と状態名は可能な限り日本語で記述する。
- API名、型名、コマンド、ファイル名、GPU識別子、数値形式、規格上の名称、証拠に記録した文字列は、検索性と実装との一致を保つため原綴りを維持する。
- `PASS`、`fail-closed`、`baseline`など証拠や契約に現れる語は、初出または文脈で日本語の意味が分かるようにする。

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
- リセット前の履歴は、現行`main`から到達可能な履歴系列と外部バックアップ／保管庫の両方に現状のまま保持する。旧Apache-2.0版の許諾は遡及的に変更せず、孤立化、強制push、共有履歴の書換えは行わない。

## 初期バージョンの主要要件

- Linuxのみを対象とする。
- 初期実装ではsafetensors形式のモデルを読み込む。最終的な公開実行環境のモデル入力と
  配布成果物はGGUFへ統一し、safetensorsは変換・開発用の入力へ移す。
- GUI以外の全機能をCLIから利用可能にする。
- AMD GPUを最初のバックエンドとし、RDNA2、RDNA4、CDNA3を対象候補とする。
- GPU操作、device memory、queue/event、operator dispatch、kernelはC++/HIPで実装する。
- フロントエンド、モデル設定、tokenizer、スケジューラ、サンプリング、実行計画はRustで実装する。
- OpenAI-compatible APIを提供する。
  - 初期仕様は `sLLM OpenAI-compatible Chat Completions profile v1` とする。
  - llama.cpp serverは実装参考・差分比較対象であり、仕様の正本にはしない。
  - [Responses APIに対応する。]
- モデル成果物の`max_position_embeddings`等は公式推奨contextとして扱い、実行環境の品質に関する厳格な必須条件にはしない。
  サーバーの実行上限はユーザーが`--context-length`で自由に指定でき、省略時だけモデル推奨値を既定値にする。
  推奨値を超える場合は起動時に一度だけ、設定値と公式推奨token数を警告する。追加opt-inやoverride flagは要求しない。
  要求のprompt tokenと要求output tokenの合計は設定した実行上限以内とし、32-bit位置表現、kernel dispatch、VRAM等の
  実装・資源制約による安全側の失敗はモデル品質判定と分離する。推奨外の品質は保証せず、RoPE scaling等を明示指定する
  将来拡張とは別に管理する。
- 最適化済みの単一リクエストでは、同一条件のllama.cppより高速であることを一つの基準とする。
  - 比較条件はモデルrevision、GPU target、入力長、出力長、数値型、llama.cpp commitを記録する。
  - 一律の必達倍率は設けず、TTFT、TPOT、token/s、peak VRAMを記録する。
- 複数要求のバッチ処理に対応する。
- [WebUIから管理できるようにする。]
- デバッグ・正しさ確認用の標準実装はPython+NumPyとする。
  - NumPyでは時間または計算効率上の限界がある場合にJAXを使用する。
  - PyTorchは使用しない。
  - Tritonは将来のNVIDIA backendに限って使用可能とし、AMD backendには使用しない。

## 初期の実装スコープ

- 最初の縦切り実装は次に限定する。
  - Qwen/Qwen3.5-4Bの固定revision。
  - BF16重み／BF16活性値。
  - 単一AMD GPU。
  - 単一要求、`batch=1`。
  - 文章のみ。visionとMTPは含めない。
  - safetensors、config、tokenizer、chat templateの読み込み。
  - CLIからprefillとdecodeを実行し、テキストを生成する。
- 初期縦切りでは、動的バックエンドプラグイン、JITコンパイラ、汎用グラフ最適化、自動調整DB、複数streamのスケジューリング、RDMA、複数GPUを実装しない。
- 後付けが高コストになる次の抽象化は初期実装から含める。
  - semantic op descriptor。
  - バックエンド能力問い合わせ。
  - tensor dtypeとquantization encodingの分離。
  - bufferアクセス方式と非同期生存期間。
  - KV配置の抽象化と任意のblock table。
  - 形状、整列、gfx能力に基づくkernel選択。

## 対応予定の詳細機能

- Infinity Fabric対応。
- その他RDMA protocolは、ユーザーがbackendを追加できる拡張点を設ける。
- FP8対応GPUではFlash Attention 4相当のattention実装を目標とする。
- 要求バッチ処理。
- chunked prefill。
- KV cache、会話、モデル固定指紋を保存領域へ記録し、起動時に再開できる簡易永続化。
  - モデル固定指紋は、使用する各モデルファイルのSHA-256を含む固定情報全体の識別子とする。
  - 旧要件の`model sha256`は、このモデル固定指紋へ包含する。
- [LMCache。]
- [RadixAttention。]
- [ロード時量子化。]

### モデルアーキテクチャ

- DeepSeek v4: MoE、DFlash。
- Qwen3.5: Dense、MoE、MTP。
- Gemma4: Dense、MoE、MTP、[Diffusion]。
- MiniMax M3。
- 列挙順は実装優先順位を表さない。

### KV cacheの数値形式

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

### モデルの数値形式

- 重み:
  - NVFP4。
  - [MXFP4。]
  - FP8。
  - [MXFP8。]
  - BF16。
- 活性値:
  - FP8。
  - [MXFP8。]
  - BF16。
- CDNA3では、e4m3fnモデルをVRAMへ読み込む際にe4m3fnuzへ変換する。
- 混乱を避けるため、テスト専用のe4m3fnuz量子化モデルは作成しない。
- NVFP4ではtensor scaleをtensor表現とkernel契約に含める。

## GPU互換性方針

- SKU名ではなく、バイナリ互換性とkernel能力を分けて管理する。
- AMDの正規識別子はHIPが報告する厳密な`gfx target`とする。RDNA/CDNAは表示用の世代名として扱う。
- 配布target、code object版、wave幅、`xnack`、`sramecc`等のコード生成条件をバイナリ識別子に含める。
- 行列演算器、数値形式、FP8 encoding、LDS等の能力を能力プロファイルとして別管理する。
- 対応候補を選ぶ初期資源条件は、次の未確定条件を出発点とする。
  - INT8とFP16の両方、またはFP4を1 TOPS以上で実行可能。
  - 専用メモリ16 GB以上。
  - 理論メモリ帯域250 GB/s以上。
  - 同一アーキテクチャの製品が十分に普及していること。例外判断には根拠を記録する。
- 上記は対応候補を選ぶ条件であり、kernel binary互換性やモデル起動時の空きmemory判定とは分離する。
- プロジェクトの対応状態は `supported`、`experimental`、`planned`、`unsupported` を使用する。
- 根拠は `vendor-supported`、`project-verified`、`unverified` を別軸で記録する。
- 初期候補:
  - RDNA2: 厳密な`gfx1030`〜`gfx1036`、配布候補`gfx10-3-generic`。
  - RDNA4: 厳密な`gfx1200`、`gfx1201`、配布候補`gfx12-generic`。
  - CDNA3: 厳密な`gfx942`。FP8高速経路ではgeneric targetを使用しない。
- 将来候補としてRDNA3、RDNA3.5、MI50、CDNA1/2/4/5、CPU、NVIDIA等の他社acceleratorを`planned`として管理する。
- NVIDIA等の将来backendでも、marketing architectureだけで分類しない。
  - 例: Turing GTX 16とRTX 20はともに`sm_75`だが、Tensor Coreの有無を別capabilityとして扱う。
- 詳細は `docs/compatibility/gpu.md` と `docs/compatibility/amd-gpu.md` を正とする。

## ソフトウェア互換性とツールチェーン

- 主開発環境はUbuntu 24.04とする。
- Ubuntu 26.04等は、クラウド環境で必要になった時点で別の検証済みtupleとして追加する。
- OS、kernel、ROCm、compiler、GPU targetを独立した範囲で保証せず、組み合わせ単位で状態を記録する。
- 初期ツールチェーン:
  - Rust edition 2024。
  - MSRV Rust 1.85.0。
  - 開発用Rust 1.97.1。`rust-toolchain.toml`で固定する。
  - Cargo resolver 3。アプリケーションとして`Cargo.lock`をコミットする。
  - C++17。
  - ROCm 7.14.0同梱の`amdclang++`とLLVMを使用する。
  - CMake 3.21以上。
  - H0〜H2ホストCI用Python 3.12.10。直接依存versionは`ci/requirements-host.txt`で固定する。
- ROCmのコンパイラ、実行環境、ライブラリは同一リリースへ揃える。
- ローカル開発環境の有効化と安全側の確認は`docs/development/environment.md`および`scripts/dev`を正本とする。
- ツールチェーンで実装上の問題が確認された場合は、互換性文書とこの計画を更新して変更する。
- 詳細は `docs/compatibility/software.md` を正とする。

## RustとC++/HIPの境界

- Rustワークスペースを最上位ビルドと処理の主体にする。
- C++/HIPバックエンドはCMakeで静的ライブラリとしてビルドし、Cargoビルドスクリプトからリンクする。
- Rust上位層は`Backend` traitでバックエンドを抽象化する。MVPでは静的登録のみとし、安定した外部プラグインABIは作らない。
- Rust/C++境界はHIP専用のversioned C ABIとする。
  - 不透明なcontext、queue、buffer、event handleを使用する。
  - C++例外とRust panicを境界越しに伝播させない。
  - 固定幅整数、状態コード、呼出側所有のエラー出力先を使用する。
  - 拡張可能structには`struct_size`とversionを持たせる。
- TensorはRust所有のBuffer viewとし、割当てを直接所有しない。
- Bufferは不透明なC++割当てをRust `Arc`で管理する。
- 非同期投入は完了eventと使用buffer参照を保持し、完了前の解放を禁止する。
- バックエンド台帳、semantic Op台帳、HIP Kernel台帳の三層に分離する。
- 詳細は `docs/architecture/runtime.md` を正とする。

## モデル取得と再現性

- Hugging Faceモデルはbranch/tag名だけで固定しない。
- モデル固定情報に次を記録する。
  - `repo_id`と`repo_type`。
  - 要求したrevision。
  - 解決済みの完全なcommit SHA。
  - 実際に使用する全ファイルのSHA-256とsize。
  - Hub blob IDとLFS OID。
  - ライセンス、model card、基底モデル、変換系列。
- 量子化や形式変換を行ったモデルでは、変換元の固定指紋、変換ツールのリポジトリとコミット、引数・設定、実行環境、出力SHA-256を記録する。
- 重みshardだけでなく、index、設定、tokenizer、chat template、generation/processor設定も固定対象とする。
- モデル別名は特定の固定指紋へ結び付ける。
- 詳細は `docs/models/model-lock.md` を正とする。

### ユーザー向けモデルコンテナ

- 2026-08-15のユーザー明示決定により、最終的な公開実行環境のモデル入力と配布成果物を
  GGUFへ統一する。ホビーユーザーにsafetensorsのshard、量子化sidecar、tokenizer等の
  複数成果物を個別管理させず、推論に必要な重み、scale、モデル情報、tokenizer、
  語彙、chat templateを原則として単一GGUFへ収容する。
- 初期縦切りで実装したsafetensorsの直接読込みと現在の量子化sidecarは、GGUF変換が完了するまでの
  開発・移行経路として扱う。最終的な公開実行環境ではGGUFを正本とし、safetensorsは変換ツールの
  入力として残せる。実行環境内部の派生cacheは許容するが、別のユーザー管理成果物にはしない。
- GGUFコンテナへの統一は、Q8_0、Q4_K等の一般的なllama.cpp量子化形式を自動的に対応対象へ
  加える決定ではない。対応するtensor encodingと実行経路は別に決定する。
- safetensorsからGGUFへ変換する場合は、変換元の固定指紋、変換ツールのrepositoryとcommit、
  引数・設定、出力全体のSHA-256を記録する。実行環境のモデル固定情報はGGUF本体、metadata、tensor一覧を
  検証対象とする。標準GGUFとの互換性を優先し、独自metadataまたはtensor typeが必要な場合は明示的に
  版管理する。

## 外部実装の参照とコード流用

- llama.cppとvLLMから、実装前に技術上の要点を抽出する。
- ローカルの`reference/`に置く公式origin、version、完全commit SHA、取得状態は[参照元固定マニフェスト](../references/source-lock.md)を正とし、固定した参照元の調査範囲と採用判断は[推論エンジン参照](../references/inference-engines.md)へ記録する。
- 2026-08-02の追加調査対象からはLMDeployとKTransformersだけを正式なローカル参照元として採用する。MLC LLM、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIは今回未採用とし、採用予定に置かない。
- vLLM等からコードを直接流用しない。参照元の表現を実装へ持ち込まないよう調査記録と実装段階を分離するが、別subagentの使用は必須にしない。
- llama.cppからの直接流用は許可するが、トップレベルLICENSEへの曖昧な追記だけで済ませない。
- MTPの投機decode／検証制御はllama.cpp実装を一括移植しない。llama.cpp issue
  [#25618](https://github.com/ggml-org/llama.cpp/issues/25618)で、量子化targetに対するdraft-model型speculationが
  greedyなtarget-only生成から分岐する問題が報告されているため、同issueは回帰事例の参照元としてのみ扱う。
  sLLMでは通常の逐次target decodeを数値oracleとし、draft tokenを順番に承認し、target-onlyと同じ計算結果を得る
  独自契約をフェーズ18で実装・検証した。
- 直接流用する場合は、著作権・ライセンス表示を保持し、upstream URL、完全commit SHA、upstream/local path、hash、exact/adapted/ported区分、変更内容、取込みcommitを記録する。
- 実際に取り込んだ時点で`THIRD_PARTY_NOTICES.md`を作成・更新し、コピー先から参照できるようにする。
- 詳細は `docs/provenance/README.md` を正とする。

## 開発・最適化の優先順位

- 多くのモデル・GPUへ共通適用できる変更から行う。
  1. 異種モデル・異種GPUで共通。
  2. 異種モデル共通、またはGPU共通。
  3. モデルアーキテクチャ内共通、またはGPUアーキテクチャ内共通。
  4. モデル固有、またはGPU固有。
- 基準kernelとsemantic op契約を先に固定し、最適化kernelはregistryへ追加する。
- 対応、動作、ネイティブ高速経路、変換、emulationを同じ意味で使わない。
- 性能計測ではInferenceXと比較可能な種類のデータを収集し、グラフを作成する。
- 単一リクエストのllama.cpp比較では、モデルrevision、llama.cpp commit、GPU target、数値型、入力長、出力長を記録する。
- 性能候補の採用単位を`adoption scope S`（採用範囲）とする。`S`は同じproviderへ送られる実運用入力の集合で、実行前に評価できる
  安定したdispatch keyから定義する。
- dispatch keyは厳密なtarget、dtype/encoding、semantic op、shape/layout/alignment、要求方式、仕組み上意味のあるcontext境界等で
  構成する。benchmark事例名、prompt内容、実測後の結果、個別token列をkeyにした過適合分岐は作らない。
- 性能候補に固定の改善率閾値または全pattern一律非悪化条件を置かない。担当AIが範囲`S`ごとに、
  演算子／モデル全体の改善量と絶対時間、測定の確からしさ、改善・悪化の一貫性、利用頻度と対象範囲、正しさ、資源、
  target分岐、実装・検証・将来保守費用、既存アーキテクチャとの整合、将来の再利用性と差戻し容易性を総合し、採用が妥当かを決める。
- 担当AIは採否理由、既知の改善と悪化、測定限界、採用範囲、基準経路で補完する範囲、再検討条件を計画・履歴・要約へ
  明記する。局所改善がモデル全体の測定雑音未満でも、bit exact、全範囲で一貫した改善、実装が単純、hardware-native化や将来利用価値が高い等の
  理由があれば採用できる。反対に大きな局所改善でも、寄与が小さく保守費用や分岐が大きければ棄却できる。
- 正しさ・security上の欠陥、原因不明の数値差、fallback・資源・後始末の破壊、未対応targetへの誤送信は引き続き阻害条件とする。
  性能上の安定した悪化は自動的な阻害条件ではないが、隠さず定量化し、範囲分離または利益とのtrade-offを説明する。
- `shared adoption`は`S`が固定matrix全体の場合、`scoped adoption`は`S`がその真部分集合の場合とする。管理性のため共通採用を優先するが、
  範囲外での候補単体の悪化を理由に、安全に分離できて採用利益が保守費用を上回る限定改善を棄却しない。
- 数値範囲やcontext閾値をkeyにする場合は境界`B-1/B/B+1`と範囲内の複数代表値を検証する。単一benchmark点しか裏付けない範囲は
  実運用へ採用しない。範囲のkey、代表事例、境界、基準経路で補完する範囲を最終性能測定前にmanifestへ固定する。
- 2026-08-19以前のフェーズで使った5%閾値とフェーズ29のGDN限定例外は当時の歴史的決定として維持するが、
  新規採否およびユーザーが明示的に再評価を求めた候補へは上記の担当AI裁量規則を適用する。
- 数値実装変更は[数値・出力影響変更台帳](../compatibility/numerical-output-changes.md)へ一元記録する。変更前とtoken列が異なっても、
  real-number semanticを維持し、差の原因が説明可能で、解析上の誤差boundまたは期待誤差が非増加となるN1変更は数値gateを自動承認する。
  既存tolerance内でも誤差が僅かに増加するN2変更は人間判断とし、原因不明・非有界・非決定のN3変更は採用しない。
- N1自動承認は数値互換性だけに適用し、性能、状態／fallback、資源、後始末、ABI、security／正しさ上の欠陥に関する厳格な条件は維持する。
  N1の定常承認に専用FP64/high-precision providerを要求せず、解析が曖昧なN2/N3の解消時だけ任意で作成する。

## 正しさ確認方針とCI・テスト

### 決定済み

- モデルアーキテクチャ共通の変更は、原則としてその系列の最小modelから確認する。
- 量子化評価にはtop-1一致率、KLD、modelの一部を切り出したBF16比誤差を使用する。
- CPUで数時間以上を要する確認は極力避ける。
- 2の冪や特定サイズだけでなく、非整列値と境界前後を含める。

### CI・テスト方針

- GPU kernel、GPU規模のGEMM／attention、モデル全体の推論、GPU性能をCPU emulationで証明しない。
- CPU CIはホスト契約、極小NumPy oracle、HIPコンパイル専用検査に限定し、モデル全体のdownload・load・forward・generationを行わない。
- compile成功、実GPU実行、数値一致、モデル断片、end-to-end、性能を別々の証拠として記録する。
- GPU不在時のCPU代替実行、timeout、crash、test未収集を成功扱いにしない。
- 公開forkの`pull_request`からself-hosted GPU runnerを直接使用しない。GPU実行は既定branch上の信頼済みworkflowと隔離・使い捨て可能なrunnerを基本とする。
- PR必須CPU workflowは15分以内を初期目標とし、実GPU testは変更影響と明示tupleに基づいて選択する。
- 数値toleranceはop、入力範囲、accumulation dtype、出力dtypeごとに根拠を持って定義し、全op共通の緩い既定値を置かない。
- 性能に影響する境界`B`は実GPUで`B-1/B/B+1`を測定し、backend、dispatch、fallback、成果物hashとともに記録する。初期G3 smokeは`255/256/257`を含める。
- HIP／実行環境／backend／dispatch／native buildの下書き開発では、影響箇所に絞ったホスト・GPU testを行う。統合またはreleaseでGPUの正しさを主張するときだけ、意味上のbuild identityが一致するG0/G1/G2/P0等の該当証拠をfail-closedに集約する。
- H0/H1/H2は統合・releaseで選択された場合の並列行とし、`host-required`へ集約する。下書きへ全行を一律要求しない。必須workflowはp95 10分以内、厳格上限15分とする。
- 初期GPU証拠は専用ローカルホストの厳密な`gfx1030` 1台と`gfx1201` 1台で直列実行し、公開forkのPRからGPU runnerを直接使わない。
- 詳細な方針と実装順序は[CI・テスト方針策定計画](active/2026/08/1-10/ci-test-strategy.md)を参照する。

## 開発運用上の決定

- Gitで追跡するのはソース、文書、小さなfixture、manifest、hash、要約とし、モデル、binary、生のtrace/profile、大きなモデル断片、生成物は追跡しない。詳細は[リポジトリ衛生方針](../development/repository-hygiene.md)を正本とする。
- 登録済みworktreeは有効な並行開発・証拠取得用途を持つため、個数だけで作業やpushを停止しない。9個以上、
  missing/prunable登録、clean・unlocked・非mainで14日超の候補は整理を促す警告とし、自動削除しない。
- 無人での進行を優先しつつsecret露出を最小化する。専用ローカルホストでは`homelab1`への`NOPASSWD: ALL`を意図的なtrade-offとして受容し、main agentが作業範囲内で`sudo -n`を使う。恒久方針は[認証情報方針](../security/credentials.md)を正本とする。
- 現在の既定profileは`trusted-solo-development`とし、外部contribution実行時とrelease時の要件を分離する。使っていないprofileの要件は現在の開発を阻害しない。
- main agentは調査・実装を直接行える。独立して進められる範囲限定のコーディング、調査、絞り込んだtest、要約、反復作業は
  subagentへ積極的に委譲し、資源または依存上の理由がなければ利用可能な並列枠で同時実行する。通常のnative coding workerは
  速度に優れるxhighのLunaを優先する。Terra/SolはLunaとmain agentで効率的に扱えない横断調査、反復失敗後の上位対応、
  または特に深い専門推論が必要な場合だけ使う。main agentは編集確認、共有作業領域の競合解消、関連検査に責任を持ち、
  subagent利用や特定の`codex exec`実行方式を完了条件にしない。
- 各フェーズは受入条件、検証、計画・履歴の完了処理後に、そのフェーズだけを必要最小限のcommitへ整理して現在のGitHub branchへ
  pushする。次フェーズの変更を同じcommitへ混ぜず、共有済み履歴の書換えや強制pushを行わない。
- 作業単位は独立してreview・rollbackしやすい範囲とするが、細分化、不変identity、独立review、全matrix実行を各下書き時点の完了条件にしない。下書き、統合、release/push、文書のみの作業区分と実行手順は`AGENTS.md`を正本とする。
- AIが厳格な必須条件、独立review必須化、広範／GPU再実行、security境界、再利用制限、阻害段階、作業単位の追加分割、不変証拠の拡張を提案する場合、明示的なユーザー承認までは提案元・範囲・費用・期限を持つ非阻害提案として扱う。
- 受入条件は作業単位の開始時に固定する。実際の正しさ・security上の欠陥は阻害条件にできるが、review中に新しく作られた手続き上の要件は承認なしに遡及適用しない。
- ソース／build入力、toolchain、モデル固定、成果物digestから成る意味上のidentityをGit commit identityと区別する。文書だけの変更で意味上のidentityが変わらないことを確認できればコード／GPU証拠を再利用し、文書だけの完了処理や新しい独立reviewを行わない。
- 適用可能なservice／実行環境が対象にある場合だけ適用後smoke／healthを要求する。独立した適用先がないlibrary、tool、文書はpush可能である。
- 同じ単位の2回reject、review時間が実装時間超過、1時間以上の機能進捗停止、検証・文書が30%超、見積り1.5倍超、gate/受入条件変更のいずれかで、新規review・検証を停止し、ユーザーへ報告して計画を見直す。

## フェーズ一覧と進捗

詳細な作業単位、試行錯誤、コミット識別子、証拠のダイジェスト、レビュー結果は、各フェーズの
保存済み計画・履歴・Git履歴を正本とする。この節では、全体の順序、主要な到達点、現在の状態だけを管理する。

状態は日本語で統一する。「完了」は採用・棄却を含めてそのフェーズの判断が閉じた状態、
「ホスト準備可能」は実機なしで準備を進められる状態、「計画済み」は未着手、
「計画済み・次」は未着手のうち次の既定優先対象、「再編済み・未着手」は実装前に範囲を別フェーズへ移した状態、
「要承認」は開始前にユーザーの明示承認が必要な状態を表す。再編済みは完了や棄却を意味しない。

| 状態 | フェーズ | 主な範囲・到達点 |
| --- | --- | --- |
| 完了 | 0 | 製品、互換性、実行環境、モデル固定、来歴、API、CI、リポジトリ管理の初期方針を確定 |
| 完了 | 1 | Rustワークスペース、C++/HIPバックエンド、版管理C ABI、ホストCIを構築 |
| 完了 | 2 | 固定ROCmによるHIPコンパイル専用検証と、モデル非依存GPU実行経路を構築 |
| 完了 | 3 | Qwen3.5-4B BF16の単一GPU・単一要求・文章生成を実装 |
| 完了 | 4 | 同一実装をQwen3.5-2B/9Bへ拡張し、VRAM事前検査を追加 |
| 完了 | 5 | V620/R9700と固定llama.cppの基準性能を取得 |
| 完了 | 6 | 仮想連続KVメモリ方式とOpenAI互換Chat Completions v1を実装 |
| 完了 | 7 | 定期・互換性・性能・リリース向けCI/CDを整備 |
| 完了 | 8 | BF16の行列積・attention・キャッシュを単一要求向けに最適化 |
| 完了 | 9 | 同期削減、区間実行、M=1 MMVFを含む実行エンジン構造を最適化 |
| 完了 | 10 | Qwen BF16からのFP8 W8A8経路とRDNA2/RDNA4別実装を追加 |
| 完了 | 11 | BF16/FNUZ FP8、wave64、常駐連続KVをCDNA3 `gfx942`へ移植 |
| 完了 | 12 | Hot Aisle MI300Xで演算子、4B/9B、API、性能、後始末を実機確認 |
| 完了 | 12R | 追跡済みファイルだけで閉じるCI移植性とローカル実GPU検証の分離を修復 |
| 完了 | 13 | モデル固有グラフから、モデル非依存の準備済み実行制御を分離 |
| 完了 | 14 | Gemma 4 12B Dense文章生成を共通実行層へ統合 |
| 完了 | 15 | Weight NVFP4の形式、読み込み、実行、品質判定を実装 |
| 完了 | 15O | FP8/NVFP4のdecode・prefill経路を計測し、採用候補を限定 |
| 完了 | 15Q | Unsloth NVFP4の品質差を形式・量子化・実行経路へ分解 |
| 完了 | 16 | FP8/NVFP4 KVの追記・attention・容量・品質を実装 |
| 完了 | 16F | 提供元FP4/MXFP形式を第一級モデル入力として統合 |
| 完了 | 17 | Qwen3.5 MTP、vision、複数形式画像のCLI/API経路を実装 |
| 完了 | 18 | MTPを逐次target生成と数値的に一致する内部高速経路として統合 |
| 完了 | 19 | Qwen3.5-35B-A3B MoE文章生成を単一GPUの通常CLI/APIへ統合 |
| 完了 | 20 | 公開モデル入力と配布成果物を単一GGUFへ統一 |
| 完了・不採用 | 21 | decode区間の完了イベント集約を評価。壁時計差が雑音内のため既定経路へ不採用 |
| 完了・不採用 | 22 | 形状別BF16 M=1 matvecを評価。局所改善が全体時間へ転化せず不採用 |
| 完了 | 23 | 他エンジンとの差と詳細計測から、prefill最終行・projection・直列化を抽出 |
| 完了・採用 | 24 | prefill終端LM head/Argmaxを最終行へ限定する共通経路を採用 |
| 完了・候補なし | 25 | projection群の共有可能量が小さく、実装候補なしで完了 |
| 完了・不採用 | 26 | 継続要求バッチのホスト計画器を実装。GPU `B>1`へ安全に接続できず不採用 |
| 完了・候補なし | 27 | decode projection差を再計測。両GPU共通候補なしで完了 |
| 完了・例外採用 | 28 | GDN状態処理統合を共通経路へ採用。従来の5%規則は維持 |
| 完了・採用 | 29 | GDNのwave reductionをN1数値変更として記録し共通採用 |
| 完了・限定採用 | 30 | RDNA4のnative FP8 KV読出しとwave attentionを対象形状へ限定採用 |
| 完了・採用 | 31 | chunked prefillと生存期間対応作業領域で10k超のKV経路を成立 |
| 完了・限定採用 | 32 | RDNA4 native FP8 KV追記を低保守費用の範囲へ限定採用 |
| 完了・限定採用 | 33 | decode split-KVとGQA K/V共有のFull Attention経路を限定採用 |
| 完了・限定採用 | 34 | V620長行prefillの対象形状を既存hipBLASへ送り、10,001-token全体を61.14%短縮 |
| 完了・限定採用 | 35 | 長文Full AttentionとGDNを構造最適化し、V620/R9700を34.93%/13.45%短縮 |
| 完了 | 36 | MI300X最新`main`で99演算子、4B BF16/FNUZ FP8、低bit KV、10,001/2、MTP、vision、API、反復性能を確認 |
| 再編済み・未着手 | 37 | 旧MI300X GDN・Full Attention計画。実装前にフェーズ49〜51へ吸収 |
| 再編済み・未着手 | 38 | 旧MI300X残差計画。実装前にフェーズ51へ吸収 |
| 完了 | 39 | 稼働性、認証、可観測性、TLS/CORS、再開可能SSEを実装 |
| 完了 | 40 | token選択、grammar、構造化生成、logprobsをホスト/API/HIPへ統合 |
| 完了 | 41 | prefix/KV再利用、session状態、checkpoint、context shift、speculationを統合 |
| 完了 | 42 | Completions、Embeddings、Rerank、token操作、infillを公開API/CLIへ追加 |
| 完了 | 43 | Responses、Anthropic Messages、function/tool protocolを実装。tool実行は分離 |
| 完了 | 44 | 汎用template、reasoning制御、対話CLI、reverse promptを実装 |
| 完了（ホスト＋RDNA GPU、MI300X保留） | 45 | LoRA/control vector、複数モデル台帳、動的load/unload/cacheを実装 |
| 計画済み | 46 | 変換、量子化、imatrix、分割・結合、ベンチマーク、品質・デバッグ用ツール |
| 要承認 | 47 | 組込みtool/MCP実行。別worker/sandboxと信頼境界の承認前は開始しない |
| 計画済み | 48 | 公開APIだけを使う最小WebUI・管理画面 |
| 完了・限定採用 | 49 | V620 `gfx1030`でGQA P32を限定採用し、long-prefill v2とHIP Graphを棄却。通常5行の退行確認まで完了 |
| 完了・限定採用 | 50 | R9700 `gfx1201`でPhase 49変更を採否し、MI300X `gfx942`向けwave64引継ぎを準備 |
| 計画済み・次 | 51 | フェーズ49/50の採用内容をMI300X `gfx942`へwave64対応で適用し、同じ7行で検証 |
| 完了 | X | llama.cpp HIPのQ5_1 Flash Attention構成を修正し、ローカルQwen補助エージェントへ反映 |

直近の性能経路は既定順をフェーズ49→50→51とする。フェーズ49はV620の全7行同等達成を後続GPUの開始条件にせず、
GQA P32を限定採用、long-prefill v2とHIP Graphを棄却し、採用経路の正しさ・資源・通常5行の退行確認を終えて完了した。
フェーズ50はR9700 `gfx1201`の限定採用とMI300X `gfx942` wave64引継ぎ準備を終え、実機性能検証をフェーズ51へ引き継ぐ。
R9700の同等達成もMI300X開始の必須条件にはしない。
フェーズ46〜48は予約済みの機能経路として保持するが、ユーザーが優先順位を変更しない限り性能経路の後に扱う。
フェーズ37〜38はコード変更や実機検証へ着手する前にこの性能経路へ再編した。
フェーズ37以降の詳細な依存関係と受入条件は
[フェーズ37以降の進行中計画](active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を正本とする。
フェーズXの詳細は
[フェーズX保存済み計画](archive/2026/08/11-20/phase-x-qwen35-gdn-amd-performance.md)を正本とする。

### フェーズ36: MI300X最新`main`実機再検証（完了）

- Hot Aisle MI300X VF x1、`gfx942:sramecc+:xnack-`、wave64、ROCm 7.14という厳密な組合せで、
  フェーズ35後の`main`を再検証した。複数GPU、別CDNA製品、長時間安定性への一般化は行わない。
- セッションAでは99演算子、Qwen3.5-4B BF16/FNUZ FP8短生成、固定・Unicode・停止条件、資源解放を確認した。
  公開FP8 GGUFのOCP→FNUZ常駐変換、GDN wave32経路の`gfx942`への誤適用、code objectのfeature未固定を修正し、
  最終成果物をCode Object V6、ELF flags `0xE4C`、wave64へ固定した。
- セッションBでは4種KV encodingのFull Attention `116/116`、FP16 KV状態`19/19`、
  FP16/dynamic FP8 KVと`auto/512/2K/4K/8K/16K` chunkの10,001入力／2出力12行をPASSした。
  全行でHIPのみ、fallbackなし、後始末0を確認し、`contiguous-resident` KVを維持した。
- セッションCではMTP幅1〜8、vision、OpenAI非stream/SSE、reasoning、停止、seed、取消し回復、
  2要求並行、正常終了を確認し、`gfx942`のMTP admissionと報告を修正した。
- セッションDではBF16/FP8各5事例を3回warmup＋10回測定し、固定llama.cppとrocprofv3を取得した。
  10,001/2のE2E中央値はsLLM BF16/FP8が`22.5561/22.5565`秒、固定llama.cppが`0.8513`秒で、
  E1比は`26.50x`だった。device時間はGDN `73.95%`、Full Attention `25.12%`、projection `0.70%`、その他`0.23%`であり、
  当時のMI300X優先候補をGDNとFull Attentionに固定した。この候補は再編後のフェーズ51へ引き継ぐ。
- A〜D完了後、当初の条件付き拡張から9B、Gemma/MoE、長時間安定性をユーザー決定で外し、VM削除を確認して完了した。
  正本は[フェーズ36保存済み計画](archive/2026/08/11-20/phase36-mi300x-current-main-validation.md)、
  [セッションA要約](../../ci/matrix/phase36-mi300x-session-a-final-v1.json)、
  [セッションB要約](../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)、
  [セッションC要約](../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)、
  [セッションD要約](../../ci/matrix/phase36-mi300x-session-d-summary-v1.json)とする。
- 完了後の独立R9700比較では、同じ`23066`×10,001入力／2出力、BF16、FP16 KV、greedy、3＋10測定で、
  sLLM `3.936429665`秒、固定llama.cpp `2.063845785`秒、E1比`1.90733x`だった。
  詳細は[R9700 E2E履歴](../history/2026/08/21-31/r9700-sllm-llama-e2e-comparison.md)と
  [追跡済み要約](../../ci/matrix/r9700-sllm-llama-e2e-v1.json)を正本とする。

### フェーズ39: service運用性（完了）

- health/readiness、上限付きPrometheus metrics、非阻害の実行時memory snapshot、秘匿化したprops/slots、管理者取消し、複数user/admin keyと更新、厳密CORS、Rustls、明示的に有効化する再開可能SSEをホスト側へ実装した。
- serverの全target test 62件とclippy warning 0を確認した。GPU kernelは変更しておらず、GPUの正しさや性能はこのフェーズの成果として主張しない。正本は[保存済み計画](archive/2026/08/21-31/phase39-service-operability.md)と[履歴](../history/2026/08/21-31/phase39-service-operability.md)とする。

### フェーズ40: token選択・grammar・構造化生成（完了）

- legacy互換を含む順序付きsampler chain、logprobs、bounded GBNF／JSON Schema、構造化`response_format`、`n=1..=8`の選択状態、厳密なAPI/SSE形式を実装した。
- HIP `TokenSelect`は文法mask、bias、履歴由来の加算値を扱い、選択結果だけを16 byteで返す。GPU非対応samplerへ暗黙にfallbackせず、従来のArgmaxとABIを維持する。
- ホスト・ABI検査、V620 `gfx1030`／R9700 `gfx1201`の境界語彙を含む選択契約、Qwen/Gemmaの構造化生成をPASSした。`gfx942`はwave64固定compile／経路だけを確認し、MI300X実行は保留した。正本は[保存済み計画](archive/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)、[履歴](../history/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)、[GPU要約](../../ci/matrix/phase40-token-selector-gpu-summary-v1.json)とする。llama.cppのコードは直接流用していない。

### フェーズ41: prefix・session状態・speculation（完了）

- 固定identity、最長prefix検索、lease/LRU、Qwen/Gemmaの不透明状態forkを実装した。VMMでは読取り専用page共有と末尾COW、連続状態では同一device内コピーを使い、物理bytesを重複計上しない。
- 全KV encoding planeとQwen GDN／linear、Gemmaのfull/sliding層を版管理checkpointへ保存し、厳密なfilesystem・checksum・quota検証後に新しいownerへtransactional restoreする。実運用はstatelessなprompt境界の保存・読込みに限定し、生成途中の再開や暗黙の大域sessionは対応済みと主張しない。
- context shift、絶対位置、assistant prefill、MTP／external／ngram共通draft契約を統合し、V620／R9700の全plane fork・COW・export/importをPASSした。`gfx942`はwave64固定compileのみで、MI300X実行は保留した。正本は[保存済み計画](archive/2026/08/21-31/phase41-prefix-session-speculation.md)、[履歴](../history/2026/08/21-31/phase41-prefix-session-speculation.md)、[GPU要約](../../ci/matrix/phase41-state-gpu-summary-v1.json)とする。llama.cppのコードは直接流用していない。

### フェーズ42: 推論方式・基本公開endpoint（完了）

- OpenAI subsetのCompletions／Embeddings、sLLM独自Rerank、tokenize／detokenize／apply-template／input-tokens、能力確認付きFIM/infillを共通frontend、scheduler、HTTP、CLIへ実装した。未対応field、範囲外値、非finite値、上限超過、未対応能力はGPU投入前に拒否する。
- Embeddingsは最終RMSNorm後のhidden rowを平均・L2正規化し、Rerankは同じvectorの内積を順位に使う。現在のQwen/Gemma固定モデルは検証済みFIM templateを持たないため、infillをfail-closedにする。
- V620／R9700でQwen3.5-4BとGemma-4-12Bのモデル全体embeddingをPASSし、検証中にGemma static-FP8のscale-plane参照と常駐bytes計上を修正した。MI300X実行は保留した。正本は[保存済み計画](archive/2026/08/21-31/phase42-inference-modes-public-endpoints.md)、[履歴](../history/2026/08/21-31/phase42-inference-modes-public-endpoints.md)、[GPU要約](../../ci/matrix/phase42-inference-gpu-summary-v1.json)とする。llama.cppのコードは直接流用していない。

### フェーズ43: Responses・Anthropic Messages・function/tool protocol（完了）

- OpenAI ResponsesとAnthropic Messagesを別々の固定仕様へ結び付け、`/v1/responses`と`/v1/messages`の厳密な解析、非stream／SSE、安定ID、usage、stop、bounded replayを共通schedulerへ接続した。Chat Completionsの別名やprovider共通wire形式にはしない。
- 順序付きmessage／call／result、tool定義・選択、直列／並列方針、生成envelopeを実装し、Qwenだけでフェーズ40のJSON Schema grammarを明示有効化する。Gemmaと未広告backendはfail-closedにする。
- 実装範囲はtool call生成とclient所有結果の往復までで、process、network、filesystem、secret、credential、MCP等の実行経路はない。組込みtool/MCP実行は明示承認が必要なフェーズ47へ残す。正本は[保存済み計画](archive/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)、[履歴](../history/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)、[machine profile](../../tests/fixtures/phase43_protocol_profiles_v1.json)とする。

### フェーズ44: template・reasoning・対話UX（完了）

- MiniJinja `2.24.0`を固定したboundedな汎用template、型付きadapter、kwargs、digest identityを実装した。include/import/extends、symlink、不正UTF-8／NUL、非finite値等をbackend初期化前に拒否し、既存Qwen/Gemmaの出力を暗黙に置換しない。
- reasoning mode／budgetを既存の生成制御へ統合し、強制終了、出力不足、grammar衝突等を投入前に検査する。Chat、Responses、CLIは同じloweringを共有し、Anthropic thinkingとGemma/raw-textは未対応のままにする。
- `chat` CLIへprompt file、対話stdin、型付き履歴、reverse prompt、JSONL event、成功turnだけのtransactional publishを追加し、フェーズ41の不透明checkpointへ接続した。WebUIはフェーズ48、組込みtool/MCP実行はフェーズ47、生成途中・wire sessionの再開は後続へ残す。正本は[保存済み計画](archive/2026/08/21-31/phase44-template-reasoning-interactive-ux.md)と[履歴](../history/2026/08/21-31/phase44-template-reasoning-interactive-ux.md)とする。

### フェーズ45: adapter・動的モデル管理（完了）

- `sllm-model-manifest-v1`のoffline事前検査、LoRA／control vectorの派生identity、別名だけを操作する管理面、registry lease、draining／quarantine／LRU、API拡張、`sllm models` CLIを実装した。
- V620 `gfx1030`／R9700 `gfx1201`のrelease buildでQwen BF16の無効・LoRA・control・併用をbitwise一致でPASSし、HIPのみ、fallbackなし、資源の基準値復帰を確認した。BroadcastAdd単体も両targetでPASSした。
- `gfx942`／MI300X実行だけを保留し、VM再確保後の独立経路にする。正本は[保存済み計画](archive/2026/08/21-31/phase45-adapter-dynamic-model-lifecycle.md)、[履歴](../history/2026/08/21-31/phase45-adapter-dynamic-model-lifecycle.md)、[machine profile](../../tests/fixtures/phase45_adapter_lifecycle_v1.json)、[GPU要約](../../ci/matrix/phase45-adapter-lifecycle-gpu-summary-v1.json)とする。

### llama.cppとの差分棚卸しと割当状況

- 2026-08-21に、固定参照llama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`と
  現行sLLMを、公開CLI／HTTP機能と実行環境上の意味で比較した。llama.cpp serverは実装参考・比較対象であり、
  sLLMのAPI仕様は引き続き[OpenAI互換profile](../api/openai-compatibility.md)を正とする。
- モデルアーキテクチャ／family、hardware／backend／precision／codegen、並列／複数利用者／継続batch、
  複数GPU／Infinity Fabric／RCCL／RDMA、性能provider探索はこの棚卸しから除外した。Vulkanと一般的なllama.cpp
  INT4/INT8+scale量子化は既存方針上の意図的除外であり、機能不足として未割当一覧へ加えない。
- 2026-08-21のユーザー指示により、次の差分をフェーズ39〜48へ依存順に割り当てた。各フェーズ開始時に現行要件、
  外部仕様の固定、security、互換性、再利用可能なllama.cpp参照元と来歴、受入条件を固定する。
  詳細は[フェーズ37以降の進行中計画](active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を正とする。

| 分類 | 固定llama.cppにある主な機能 | 現行sLLMの状態と未割当範囲 |
| --- | --- | --- |
| 公開API・用途 | Responses、Completions、Embeddings、Rerank、Anthropic Messages、tokenize/detokenize、apply-template、infill、専用input-token-count endpoint | フェーズ42でCompletions、Embeddings、sLLM独自Rerank、4つのutility endpoint、能力確認付きInfillを、フェーズ43でResponsesとAnthropic Messagesの厳密なsubsetを共通実行環境へ実装した。現在のQwen/Gemmaには検証済みFIM能力がないためInfillをfail-closedにする |
| 制約生成・tool | GBNF／JSON Schema制約decode、構造化出力、function/tool calling、組込みtool/MCP実行、logit bias、logprobs | フェーズ40でbounded GBNF／JSON Schema、構造化`response_format`、logit bias、mask適用後のlogprobsを、フェーズ43でgrammar制約付きfunction/tool callとclient所有結果の往復を実装した。組込みtool/MCP実行だけはフェーズ47の明示承認待ちである |
| sampling | 構成可能sampler chain、top-k、min-p、typical、Mirostat、DRY、XTC、adaptive/dynamic temperature、ignore-EOS | フェーズ40で版管理した順序付きsampler chainと追加samplerを実装した。GPU TokenSelectは対応subsetだけへ明示的に送り、高度な全候補filterはホスト経路へ残す。既存のGPU sampling性能課題とは分ける |
| prompt・context・状態 | context shift、prompt/KV再利用、session/slot checkpoint保存・復元、assistant prefill、FIM/infill、external draft/ngram speculation | フェーズ41でidentity-safeなprefix/KV再利用、stateless prompt checkpoint、context shift、assistant prefill、MTP/external/ngram共通契約を実装し、フェーズ42で検証済み能力に限るFIM/infillを追加した。生成途中・wire sessionの再開と外部executor提供は残る |
| adapter・読込み管理 | 事前読込みLoRAのscale／要求切替、control vector、モデルcache／offline制御、routerによるload/unload/cache | フェーズ45で固定情報・成果物の事前検査、順序付きLoRA/control選択、別名だけを扱う動的registry、load/unload/LRU/quarantineをホスト・API・CLIへ実装した。V620／R9700のモデル全体とBroadcastAddはPASS、MI300X実行は保留である |
| template・対話UX | 任意Jinja／custom templateとkwargs、reasoning制御、実行中reasoning制御API、対話、reverse prompt、prompt file、WebUI | フェーズ44でsandbox化したMiniJinja汎用template、bounded kwargs／digest identity、reasoning制御、`chat`の型付き履歴・reverse prompt・prompt fileを実装した。フェーズ41checkpointへ接続し、既存Qwen/Gemmaと一回実行の`generate`を維持する。WebUIはフェーズ48、生成途中・wire sessionの再開は後続である |
| service運用・可観測性 | HTTP health/readiness、任意Prometheus metrics、props/slots、再開可能stream、CORS/TLS、key file／複数key、server UI | フェーズ39でhealth/readiness、上限付きmetrics／実行時memory、秘匿化props/slots、管理者取消し、任意の再開可能SSE、厳密CORS、Rustls、複数user/admin keyとrotationを実装した。server UIだけをフェーズ48に残す |
| 周辺tool・品質評価 | 汎用HF-to-GGUF、quantize/imatrix、GGUF split/merge、LoRA conversion、llama-bench、perplexity/KL/task評価、debug dump | 固定converter、モデル固定、範囲限定benchmark／証拠は実装済み。汎用変換・評価toolは未割当である。未対応量子化形式を自動的に製品範囲へ追加しない |

- この棚卸しは機能差の事実を残し、後続フェーズへの割当は上記進行中計画で管理する。割当はフェーズ36の範囲変更、
  完了済みフェーズの再開、全機能の一括実装、組込みtool/MCP実行のsecurity承認を意味しない。
  フェーズ36セッションCは[保存済み計画](archive/2026/08/11-20/phase36-mi300x-current-main-validation.md)に固定したprofile v1のservice、reasoning、stop/sampling、連続・二並行要求、
  lifecycle matrixをそのまま実行し、上表の未実装機能を未実行FAIL、追加受入条件、またはフェーズ36の阻害条件として扱わない。

KV／会話／モデル固定のstateless prompt checkpointはフェーズ41、Responses APIはフェーズ43で完了した。WebUIはフェーズ48へ割り当てた。
TurboQuantを含む残りKV形式、残るモデルfamily、複数GPU／Infinity Fabric／RDMA、README整備、人間による発表、
LMCache、RadixAttention、将来MX形式には現時点でフェーズ番号を割り当てない。これらを初期versionの完了条件へ
読み替えず、完了済みのフェーズ18へ後続範囲を逆流させない。

### 性能最適化の残課題

- 2026-08-18の現行ソース、直近モデル全体profile、性能履歴を横断して確認できた明確な最適化余地を、
  フェーズXへ切り出したQwen3.5系GDN/llama.cpp AMD調査とフェーズ21で棄却した限定segment同期を除き、
  この節で管理する。dense BF16 `M=1` matvecの最初の作業単位をフェーズ22へ割り当て、フェーズ23で一覧を
  cross-engine差分、critical-path share、Amdahl上限から再分類した。prefill last-row projectionをフェーズ24、
  batch-compatible projection-family optimizationをフェーズ25、continuous request batchingをフェーズ26、
  exact decode weight-stream/provider optimizationをフェーズ27、projection外device短縮をフェーズ28、フェーズ33後の
  V620長行BF16 prefill providerをフェーズ34、フェーズ34後のFull Attention/GDN gap closureをフェーズ35へ割り当て、
  cold loaderは未割当のまま維持する。
  この一覧は完了済みフェーズの範囲を拡張せず、個別作業の受入条件、
  実装順、対象targetは着手時の新しいprofileで固定する。
  一般論だけの候補をhard gateにしない。
- 実行時dispatch・同期:
  - 現行ネイティブ実行環境はsemantic opごとにcompletion owner、HIP completion event、timing event、registry handleを
    生成し、同一streamのsegment末尾で各ownerを個別queryする。フェーズ21でsegment単位completionを比較したが実時間改善が
    測定雑音内だったため通常の既定経路へ採用しなかった。event/completion pool、
    registry lock削減、parameter更新可能なnative command-listまたは実運用graph replayはフェーズ21へ含めず、
    未割当課題として維持する。requestごとの素朴なgraph instantiateは再導入しない。
  - decodeのtoken IDとpositionを別々の同期付きH2Dにせず、一つのstaging transferまたはdevice-side position生成へ
    まとめる。terminal argmax完了と4-byte token readbackも一つのstream boundaryへ含める。
  - full-attention層ごとのKV append ホスト待機は、受理済み状態だけを公開するtransaction契約を維持したまま、
    stream-ordered publicationまたはstaged stateで集約できるか測る。
- attention・KVハードウェア経路:
  - 現行汎用causal attentionはFP16 ID 2とpacked-KV ID 3を報告するが、実体はencoding引数で分岐する同一kernelであり、
    gfx1201 コードオブジェクトにもnative FP8変換、packed dot、WMMA/SWMMACがない。フェーズ16の正しさ・memoryの基準経路としては正しいが、
    RDNA4ハードウェア性能を評価するproviderではない。
  - フェーズ30でgfx1201 native FP8 codec、decode wave tile、prefill matrix attentionを別作業単位として比較した。native FP8読出しと
    wave32 reductionは採用し、gfx1030と`M=2..31`は基準経路を維持する。native append encodeとmatrix providerは不採用である。
    10,000超のモデル全体prefillは現行workspaceが利用可能VRAMを超えるため、chunked-prefill／資源作業をフェーズ31へ割り当てた。
  - フェーズ31後の10k+/16,385-token通常経路ではcausal attentionが支配的になったため、decodeのKV方向並列blockと固定combine、
    prefillのQ/K tile・GQA K/V共有、同じtile上のgfx1201 matrix innerをフェーズ33へ割り当てた。共通dispatch/scratch/softmaxを優先し、
    target/encoding/M/KV長範囲は独立採否する。
  - フェーズ35は`M>=128`の共通Q_TILE=4で4 query行 × GQA 4 headへK/Vを共有し、V620 Full Attentionを
    10.820秒から4.110秒へ62.02%短縮した。scratch/追加dispatch 0、4 KV encoding共通で採用し、`M<=127`はフェーズ33へ残す。
    固定llama.cpp 0.462秒に対してなお約8.9倍であり、次のattention workは残差4.11秒のbarrier、vector FP32 QK/PV、
    query/K tile organizationを新しいprofileから再分類する。vAttentionとKV formatは維持する。
  - Q/PのFP16/FP8化、softmax順序、accumulator変更は数値台帳のN0〜N3へ分類し、N2を性能だけで自動採用しない。
- GDN再帰経路:
  - フェーズ35はQwen shapeの状態列をwave32 x 4のworkgroupで所有し、`32 heads × 32 column groups=1,024 workgroups`へ
    広げた。V620 GDN familyは7.672秒から0.618秒となり固定llama.cpp 0.622秒と概ね同等、R9700 GDN-only E2Eも7.17%短縮した。
    token count 128未満はフェーズ28/29、state物理layoutとtransactionは既存のまま維持する。
  - token recurrenceのsequence-parallel scan、追加span分割、GDN layout再設計は比較対象 parity後の優先候補にしない。新しいprofileで
    GDNが再びcritical pathになった場合だけ別作業単位として再検討する。
- Dense BF16実行:
  - フェーズ27の新しいE1比較では、V620 projectionは比較対象より6.76%速く、R9700だけ12.53%遅かった。両target共通の
    projection provider gapではなく、全target非悪化かつ任意pattern 5%改善へ届く作業単位を固定できなかった。
  - フェーズ27のprojection除外粗い残差はprefill非projection workとR9700 MTP内部stepを含んでいたため、比較対象に対する
    3.80倍/3.54倍主張を撤回した。フェーズ28で確定出力step単位にfamilyを分解し、device処理だけを短縮する。
  - フェーズ33後の10,001-input profileではV620 `M>1` projection 248回のtiled16が66.561秒、R9700 hipBLASが0.642秒で、
    V620全体の73.89%を占めた。short `M=17`を根拠に全gfx1030 `M>8`へ適用したshape-insensitive selectorという前提変化を
    フェーズ34で再評価し、長行6 shapeだけ既存hipBLASへ送り、P23-O5のV620部分を完了した。
  - 実運用graph／command-list、gate/up+SiLU等のfamily融合、R9700限定decode projection providerは未割当課題として維持する。
    後者は安定した採用範囲の絶対/相対利益、確からしさ、数値/資源、分岐/保守費用を担当AIが総合して採用が妥当と判断でき、
    gfx1030等を基準経路へ確実に送れる共通registry keyがある場合に再検討する。
- FP8、NVFP4、MXFP4モデル経路:
  - Q/K/Vやgate/up等、同じBF16 activationを消費する複数linear間で動的FP8/NVFP4/MXFP4 activationとscaleを
    共有する。RMSNorm等の生成元からの直接量子化、quantize+matmul融合、M=1専用quantizerも比較する。
  - FP8 hipBLASLtは作業領域なし・単一heuristicに限定せず、有限workspace、複数solutionの実測、shape/target別
    algorithm cache、queue/stream別handleを比較する。gfx942 FNUZは旧activation quantizer固定を解消できるか実機で測る。
  - W4A4 quantizerはblock 16/32当たりのthread利用率、packed load/store、scale reductionを改善する。decodeとprefillで
    実質同じscalar bodyを使う経路を分離し、prefillのM/N/K tile共有、利用可能なtargetのnative行列経路、
    native pathを持たないtargetのpacked-dequant tiled GEMMを実装候補とする。
  - 低bit準備済みplanごとの`hipMalloc` workspaceを要求arenaへ集約し、同時実行しないlinear間で再利用する。
- sparse MoE実行:
  - 現行MXFP4 routed expertとBF16 shared expertのscalar K loopを、expertごとにtokenを実際にbatch化するgrouped GEMM、
    target別packed/native matrix providerへ置き換える。現行のexpert別groupingはblock順序を整えるだけであり、
    weight tileを複数tokenで共有するGEMMにはなっていない。
  - routed gate/up、SiLU、intermediate quantization、down、routing weight、共有expert combineの境界をprofileし、
    数値順序を保てる範囲でfusionとweight/activation tile再利用を行う。共有expertは既存BF16 providerを再利用する。
  - routerのstable top-8、softmax、expert groupingはthread-0/全pair再走査からwave-parallel reduction、stable selection、
    prefix-sum/compactionへ移す。router projection、status初期化、top-k/group metadata生成の融合も候補とする。
- 厳密一致MTP:
  - draft argmaxしか使わない経路の全語彙logits D2Hを除去し、target/MTP hidden stateをホスト`Vec`経由で
    D2H/H2Dせずdevice常駐のまま接続する。MTP prompt prefillのtoken-by-token ホストloopもdevice側でまとめる。
  - draft、serial-equivalent verify、逐次acceptをdevice-side orchestrationへ寄せ、reject時のaccepted-prefix replayを
    staged/COW state commitへ置き換えられるか検討する。通常逐次生成とのtoken、logits、KV、sampling結果の一致は維持する。
  - overhead削減後に承認率とtarget別profileからdraft幅を自動選択する。R9700の幅拡大、量子化path、sampling経路、
    V620再評価は同じ内部UXで行い、現行の遅い幅を無条件に有効化しない。
- sampling・フロントエンド・service:
  - non-greedy requestの全語彙BF16 logits D2H、CPU F32変換、全語彙penalty・sortをGPU samplingへ移し、
    ホストへは選択tokenだけを返す。CPU samplingを残す場合も候補buffer、token count、部分選択を再利用する。
  - schedulerの単一workerとgeneration全体を保持するbackend mutexを継続batchへ置き換え、decode batch、
    chunked prefillとのinterleave、per-sequence state、queue/stream別library handleを設計する。
  - bounded SSE event channelの`blocking_send`でGPU generationまで停止しないよう、boundedな内部ringとnetwork writerを
    分離する。disconnect cancellation、backpressure上限、visible output順序は維持する。
  - generationごとに全既出tokenを複製してprefix全体をdecodeし、全文snapshotを保持するホストO(n^2)経路を、
    byte-fallbackを保つincremental decoderと短いrollback windowへ置き換える。
- 要求状態・memory・長いcontext:
  - requestごとのgraph再構築、dynamic tensor単位のdevice allocation、KV/GDN state、prepared cacheの作り直しを、
    graph template cache、liveness arena、tensor alias、request owner/state pool、decode M=1 plan再利用へ移す。
  - prefix token列、モデル固定指紋、KV encodingをkeyにしたprefix/KV cacheとvAttention page共有/COWを検討する。
    KV、会話、モデルidentityの簡易永続化は再起動後の再prefill削減にも利用する。
  - フェーズ31ではchunked prefillによりprefill workspaceをselected chunkへboundedとし、同時liveでないrequest-owned
    intermediateだけをliveness arenaで再利用する。automatic defaultはtotal VRAM `<=16 GiB`で512、`>16 GiB`で
    16K/8K/4K/2Kを大きい順にfit判定する。vAttention型`virtual-contiguous` providerを実運用の既定として維持し、
    Paged Attentionはopaque KV state下の別physical-layout providerとして後続比較へ残す。
  - chunked prefillは長promptのlatency/peak memoryとrequest間fairnessを改善し、現行matmul一dispatchのM上限
    `65,536`を超える設定contextを実行可能にする境界として実装する。フェーズ31の直接の採用目的はまず10k+ モデル全体の
    memory成立性とlow-bit KV検証成立であり、5%速度改善を要求しない。
  - gfx942実機はVMM capabilityがtrueだったため、長い設定contextで全capacityを物理確保する
    `contiguous-resident`固定と、virtual-contiguousまたは増分commit providerを再比較する。
- モデル読込み・GGUF・vision:
  - shard／成果物検証後の再読込とtensor/chunkごとの同期uploadを、mmap、並列hash、disk read/CPU変換/H2Dの
    double buffering、複数transferの集約waitへ移す。検証済みidentity cacheは内容検証contractを弱めず利用する。
  - GGUF converterは単一container化だけでなく、runtimeのrow-major/transposed packed weight、scale plane、MoE layer blobを
    execution-readyに配置し、起動時repack、sidecar join、FNUZ等のtarget変換を減らせる余地を残す。ただしこの性能項目を
    フェーズ20の追加完了条件にはしない。
  - visionはlazy residentの初回起動、複数画像の逐次実行、vision embeddingのホストreadback/text graph再uploadを対象に、
    preload、image batch、device-to-device binding、image digest cache、greedy multimodalの不要logits readbackを検討する。
- context・target依存の条件付き候補:
  - 短contextではfull attentionは支配要因でない。長いcontextの新しいprofileで支配的になった場合だけ、full/sliding-window別
    tiled online softmax、FlashAttention系provider、quantized KVのvectorized unpack/scale共有、GQA head間KV tile共有を扱う。
  - gfx942はBF16で固定llama.cpp比の大きな差が残り、フェーズ36のGPU時間はGDN `73.95%`、Full Attention
    `25.12%`だった。フェーズ51でV620・R9700から移せる構造をwave64向けに適用し、GDN、Full Attention、
    wave64 MMVF、launch replay、GEMM solution、FNUZ quantizer、KV providerを新しい残差順に分離評価する。
    詳細は[フェーズ37以降の進行中計画](active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を正とし、
    単一MI300X VMの結果を別CDNA SKUへ一般化しない。
  - multi-GPU、expert/tensor/pipeline parallel、Infinity Fabric/RCCL/RDMAはcapacity・batch throughput候補とし、
    単一requestやPCIe構成では通信費を含む実測後に採否を決める。
- 現時点でそのまま再提案しない候補:
  - V620の全M/shapeを無条件にhipBLASへ切り替える案は、フェーズ9とフェーズ34の短M/small-N実測により再採用しない。
    フェーズ34で採用したexact production shapeとM thresholdだけを維持し、未知shapeへ一般化しない。
    R9700のtransposed GDN state、既存weight-only NVFP4 decodeの複数N列・scale broadcast、V620で現状のままMTP幅2を
    有効化する案は既存実測で改善しなかったため再採用しない。
  - 短contextでのfull attention/FA3-like最優先化とrequestごとの実運用HIP Graph生成も再採用しない。
    前提となるprofileまたは実装構造が変わった場合だけ、新しい候補として別に測定する。

## 現在の状態と次の作業

### 現在地

- 機能経路はフェーズ45まで完了している。現在の`main`には、構造化生成、状態再利用、追加推論API、
  Responses/Anthropic、汎用template、対話CLI、LoRA/control vector、動的モデル管理までが統合済みである。
- MI300Xの既存`gfx942`経路はフェーズ36で実機確認済みである。対象は99演算子、Qwen3.5-4B BF16/FNUZ FP8、
  4種KV、10,001入力／2出力、MTP、vision、OpenAI API、反復性能、固定llama.cpp比較、後始末である。
  詳細は[フェーズ36保存済み計画](archive/2026/08/11-20/phase36-mi300x-current-main-validation.md)を正本とする。
- フェーズ49はV620でGQA P32を限定採用し、long-prefill v2とHIP Graphを棄却して完了した。最終通常5行は5/5 PASSし、
  Phase 49開始時比でE2Eを24.24〜45.43%短縮した。固定llama.cppとの差は4行で+0.78〜+6.65%、10,001/2では
  sLLMが9.45%速かった。100k inputと20k outputの残差は後続へ持ち越し、全7行同等とは主張しない。
- フェーズ36後のR9700 10,001/2 E1比較は、sLLM `3.936429665`秒、固定llama.cpp `2.063845785`秒、
  比率`1.90733x`だった。詳細は[R9700 E2E履歴](../history/2026/08/21-31/r9700-sllm-llama-e2e-comparison.md)と
  [追跡済み要約](../../ci/matrix/r9700-sllm-llama-e2e-v1.json)を正本とする。
- フェーズ50はR9700 exact `gfx1201`、Code Object V6、wave32で6/7行PASS、1/7行FAIL（`100,000/2`のlayer 31 KV commit OOM）
  として完了した。PASS行のE2E中央値（sLLM／固定llama.cpp、ms）は、`17/17` `407.915/332.726`、`32/32` `759.729/604.069`、
  `1,024/128` `3,383.627/2,509.156`、`32/256` `5,959.860/4,712.364`、`10,001/2` `4,002.834/2,072.476`、
  `32/20,000` `532,486.026/377,632.768`だった。全PASSはHIP-only、fallbackなし、cleanup 0で、llama.cpp同等未達はhard gateにしない。
  追跡済み要約は[Phase 50 R9700 summary](../../ci/matrix/phase50-r9700-summary-v1.json)を正本とする。
- フェーズ50後のOOM分析で、従来の自動prefill selectorが16 GiB超の全GPUを16K候補から評価し、32 GiBのV620/R9700へ
  16K workspaceを許していた誤りを修正した。固定SGLang参照のcapacity tierを参考にし、ユーザー指定どおり24 GiB未満512、
  24〜35 GiB未満2K、35〜60 GiB未満4K、60〜160 GiB未満8K、160 GiB以上16Kを自動上限とする。各tierでは従来の
  exact graph memory見積りで下位bucketへ落とし、明示指定は上限を上書きできる。32 GiBのV620/R9700は2K開始、
  192 GiBのMI300Xは16K開始となる。selector境界のCPU testはPASSしたが、R9700 `100,000/2`の再実機結果は未取得である。
- フェーズ50ではexact `gfx1201`のresidual RMSNorm、GDN projection bundle、MLP gate-up-SiLU bundle、GQA4 P32（KV長4,096以上）を採用し、
  `gfx1030`限定経路、不採用経路、gfx942 wave64再設計を分類した。共通source変更後のV620 exact `gfx1030`通常5行は5/5 PASSで、
  フェーズ49 closeout比`-0.21〜+1.16%`だった。exact `gfx942` Cargo/probe/host selectorはPASSしたが、MI300X実機は未検証であり、
  次のフェーズ51で実施する。

### 次に進める独立経路

1. **フェーズ51・MI300X適用**: フェーズ49/50の成果と引継ぎ台帳をexact `gfx942`へwave64対応で適用し、同じ7行で検証する。
   旧フェーズ37〜38のGDN、Full Attention、FNUZ/GEMM、実行再生、KV残差もここへ統合する。
2. **フェーズ46〜48・機能経路**: ツール、承認制の組込みtool/MCP、WebUIの計画は保持するが、直近の既定優先順位では
   フェーズ49〜51の後に扱う。フェーズ47だけは引き続き明示承認を必要とする。

フェーズ37以降の作業単位、依存関係、受入条件、除外範囲は
[フェーズ37以降の進行中計画](active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を正本とする。
フェーズ49の3候補判定とフェーズ50のR9700採否、V620退行確認、gfx942引継ぎ準備を終えたため、次はフェーズ51を開始できる。
フェーズ50の詳細計画は[保存済み計画](archive/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)を正本とし、
既定実行順は50→51だが、フェーズ49または50の全7行llama.cpp同等達成を後続フェーズの開始条件にはしない。

### 継続方針

- README整備と人間による発表は番号を割り当てない将来タスクとし、製品フェーズの完了条件へ混ぜない。
- H3の必須化は引き続き観測事項であり、現時点では必須条件へ昇格しない。
- 現行の開発形態は`trusted-solo-development`である。下書き、統合、公開、文書のみの扱いは`AGENTS.md`を正本とし、
  過去フェーズ固有の検証手順を現在の一律条件へ読み替えない。

## 未解決事項

- AMDコンシューマーRDNA2を含む各gfx targetの厳密な実機検証範囲。
- ROCm 7.14.0とHWE kernel 6.17を組み合わせたV620/R9700 tupleについて、長時間安定性と正式な
  互換性状態を判断できるだけの実測が揃っていない。
- 追加op・shape・入力範囲の数値toleranceと、複数のO2/O3履歴run・分散・再現性が揃った後に定める
  性能回帰閾値。
- 資源条件の1 TOPS、16 GB、帯域の定義と例外承認基準。
- Infinity Fabric、他RDMA protocol、KV永続化の詳細設計。
- 量子化形式ごとのlayout、scale粒度、accumulator、fallback表。
- sudo以外の既存平文credentialの失効・rotationとsecret managerへの移行状況。
