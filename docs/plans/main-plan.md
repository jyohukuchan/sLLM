# sLLM メイン計画

## この文書の役割

- Git管理外の `sLLM.md` にある要件定義・開発方針・重要な決定を、開発に必要な範囲で追跡可能な形へ同期する。
- この文書には、重要な製品・アーキテクチャ・互換性上の決定、開発の順序と全体の流れ、現在地、未解決事項だけを置く。
  各Phaseの作業単位、測定値、試行錯誤、証拠は計画・履歴・`ci/matrix`を正本とし、ここではリンクと結論だけを残す。
  恒久的な実行手順も各正本文書へ置き、ここには重複させない。
- `sLLM.md` とこの文書に方針上の差異が生じた場合は、推測で統合せずユーザーへ確認する。
- 角括弧内の項目は、初期バージョンでは対応しない将来機能を表す。
- プロジェクト内の権限順は、現在の明示的なユーザー指示、`sLLM.md`、`AGENTS.md`、この文書の承認済み決定、進行中計画の作業固有条件、履歴に残す過去の事実、とする。下位文書と履歴は上位方針を上書きせず、新しい完了条件や阻害条件を作らない。
- 2026-09-17に全体の流れを追いやすくするため整理した。整理前の全文（Phase別の詳細経緯、性能残課題の詳細一覧、
  llama.cpp差分表など）は[整理前のスナップショット](../history/2026/09/11-20/main-plan-before-reorganization-2026-09-17.md)に保存した。
  整理による決定内容の変更はない。

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
- フロントエンド、モデル設定、tokenizer、スケジューラ、サンプリングの設定・契約、実行計画はRustで実装する。
  固定samplingのGPU上の候補選択・抽選は、GPU操作と同じC++/HIP層で実装する。
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

### サンプリングの当面の対応方針（決定済み）

- 2026-09-07のユーザー決定により、主要モデルをコーディングエージェントタスクで使うことを優先し、
  当面の公開APIとnon-greedy GPU samplingを以下の固定設定へ絞る。任意のsampler設定への対応拡大は
  後続とし、この固定設定への対応の完了条件には含めない。
  [Phase 81](archive/2026/09/1-10/phase81-fixed-gpu-sampling.md)で、この経路の高速化とAPI統合を完了した。

| 設定 | 固定値・動作 |
| --- | --- |
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `presence_penalty` | `0` |
| `frequency_penalty` | `0` |
| `repeat_penalty` | `1.0`（無効） |
| `repeat_last_n` | `0` |
| `min_p` | `0`（無効） |
| `typical_p` | `1`（無効） |
| DRY・XTC・Mirostat・dynamic temperature | 無効 |
| ユーザー指定の`logit_bias` | なし |
| `logprobs`／`top_logprobs` | 無効／`0` |
| `ignore_eos` | `false`（通常の終了tokenで停止） |

- `top_k`はモデル読込時に決まる固定値とする。Qwen3.8 thinkingは`20`、Gemma4は`64`を採用する。
  根拠は[Qwen3.8公式推奨](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)と
  [Gemma4公式推奨](https://huggingface.co/google/gemma-4-31B-it#best-practices)とする。
  他モデル・モードの値は採用時に決め、上記の値を無条件に流用しない。固定値は共通GPU samplerへ渡し、
  モデル名ごとの専用kernelを増やさない。top-k適用後の候補へtop-pを適用する順序を明確にし、
  top-k追加による出力分布の変更を、top-p単独経路と等価な性能最適化として扱わない。
- 2026-09-08の実装時profile解決: 既存Qwen3.5のcoding profileは
  [公式coding推奨](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices)を根拠に`top_k=20`とする。
  lockにgeneration_configがないことを、既存モデルを拒否する理由にしない。
  Ministral 3のlocked generation_config（SHA-256 `e0923390059f84a9180b00e5501778acc45ea9856cd7f2fd68208b360927c677`）には
  top_kがないため、同モデルの採用profileは`top_k=0`（無効）と明示する。これは未指定一般の解釈や公式推奨値ではなく、
  追加のtop-k制限を導入しない実装上の選択である。共通GPU samplerでtop-pを適用し、CPU fallbackや
  モデル全面拒否で代替しない。temperature／top_pはユーザー決定の`1.0`／`0.95`を維持する。
- `seed`、出力token上限、stopは要求ごとに変更可能な制御として残す。tool calling、JSON制約、
  reasoningの制御も維持し、それらに必要な内部token maskをユーザー指定の`logit_bias`と区別して
  共通GPU経路へ接続する。
- 公開APIは設定省略時に固定profileを適用し、固定値と一致する明示指定を受け付ける。
  対応外の明示指定は未対応エラーとし、黙って固定値へ置き換えたり、遅いCPU sampling経路へ切り替えたりしない。
  これは従来の可変sampler APIから対応範囲を狭める決定であり、実装時にAPI仕様とクライアント設定も同期する。
- GPU上で候補選択・抽選を行い、全語彙logitsのCPU転送・CPU前処理を除去して、ホストへは選択token等の
  必要最小限の結果を返す。作業bufferを再利用し、既存のHIP Graph・KV append/attention等のdecode最適化と
  API経路から併用できる実装にする。
- 固定設定は用途と対応範囲を絞る判断であり、全モデルの最適品質を保証するものではない。
  例えば[Qwen3.5公式](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices)は精密なcoding用途に
  `temperature=0.6`を推奨するが、当面の共通方針は`1.0`とする。penalty無効化で反復抑制を手放す点も含め、
  速度と代表的なcoding・tool呼び出しの実用動作を確認する。
- 固定値の決定と、実装・実機確認の証拠を区別する。全モデル・全形式の実機成功を一括して主張しない。
  Phase 81着手前の公開版はAPI既定値`temperature=1.0`／`top_p=1.0`と可変設定を持つ。
  固定profileへの切替はPhase 81で実装・実機確認・公開CI確認を完了した。性能・未対応範囲は
  [Phase 81履歴](../history/2026/09/1-10/phase81-fixed-gpu-sampling.md)へ記録する。

### モデルアーキテクチャ

- DeepSeek v4: MoE、DFlash。
- Qwen3.5: Dense、MoE、MTP。
- Qwen3.8: 27B Dense、MTP。最初の対象artifactを`unsloth/Qwen3.8-27B-NVFP4`へ固定し、文章生成を優先する。
  visionはこの性能laneをblockしない独立後続とする。
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
- MXFP8。
- FP16。
- Phase 53/54で評価した`kv-fp8-e4-block16`／`kv-fp8-e5-block16`は、2026-08-30のユーザー決定で廃止した。
  public parserからnative state-create ABIまでfail-closedに拒否し、既存ABI番号と履歴evidenceは予約値・監査履歴としてだけ残す。
- reviewed Qwen3.5-4B BF16 dense text／full attention／single GPU／head dim 256では、KV指定省略時を
  standard OCP `kv-mxfp8-e4`（E4M3FN value、block 32、E8M0 scale）とする（exact `gfx1030`、`gfx1201`、
  `gfx942:sramecc+:xnack-`）。明示`fp16`をrollbackとして残し、対象外model/laneは既存のfixed FP16 recipeを維持する。
  この既定変更は品質自動昇格ではなくユーザー明示決定であり、gfx1201のtop-1一致が旧閾値に未達だった事実はN2台帳へ記録済みである。
- `kv-mxfp8-e5`はexact `gfx1030`の明示比較形式として残し、既定にはしない。
- Qwen3.8専用APIの省略時KVはMXFP8 E4である。2026-09-08のユーザー決定により、static tensor FP8 KVは追加せず、
  MXFP8 E4の高速経路・API統合をPhase 83で行った（FP16は比較・明示rollback、常駐FP16 mirrorなし）。
  static FP8 KVは後続Phaseへ自動移設しない。
- Phase 85で、reviewed Qwen3.5-4BのMXFP8 W8A8／MXFP6 W6A6 GGUFと`kv-mxfp8-e4`の組合せを
  exact `gfx1030`／`gfx1201`の通常CLI/APIへ接続した。BF16比品質の同等性は認定していない。

### モデルの数値形式

- 2026-09-19のユーザー決定（`README.md`の方針）: 対応する重み／活性値の組は BF16/BF16、FP8/FP8、MXFP8/MXFP8、
  MXFP6/MXFP6、MXFP4/MXFP6、NVFP4/NVFP4 とする（EXL3は未定）。活性値だけBF16の低精度経路
  （NVFP4 W4A16、MXFP8 W8A16、MXFP6 W6A16）は廃止し、Phase 87で削除する。MXFP4の活性値はMXFP6（W4A6）とし、
  下記の2026-09-03の「W4A8のみ」決定を置き換える。agent的な用途を重視し、過度な量子化や遅いハードウェアには対応しない。
- 重み:
  - NVFP4。
  - [MXFP4（W4A6。ActivationはOCP MXFP6 E3M2、block 32、E8M0 scale。2026-09-19にW4A8から変更）。]
  - FP8。
  - MXFP8 E4M3（block 32、E8M0 scale、W8A8）。
  - MXFP6 E3M2（block 32、E8M0 scale、W6A6）。
  - BF16。
- 活性値:
  - FP8。
  - MXFP8 E4M3（動的block 32量子化）。
  - MXFP6 E3M2（動的block 32量子化）。
  - BF16。
- CDNA3では、e4m3fnモデルをVRAMへ読み込む際にe4m3fnuzへ変換する。
- 混乱を避けるため、テスト専用のe4m3fnuz量子化モデルは作成しない。
- NVFP4ではtensor scaleをtensor表現とkernel契約に含める。
- （2026-09-19にW4A6へ置き換え）2026-09-03のユーザー決定により、MXFP4の対応方針はW4A8だけに限定する。既存のW4A4 ABI、provider、
  model lock、実測記録は過去の実装事実として保持するが、今後の対応形式または新規model対応の根拠にはしない。
  W4A8の実装が完了するまで、MXFP4は方針とruntime実装に差分がある状態として扱い、既存W4A4をW4A8へ
  読み替えない。W4A8のActivationはOCP MXFP8 E4M3、K-axis block 32、E8M0 scaleへ固定する。
- 一般に「FP8 model」と呼ばれるartifactへの汎用対応は、2026-09-03のユーザー決定により保留する。現行の
  per-output-channel weight／dynamic per-token activation／F32 scaleの限定経路を汎用FP8対応とはみなさない。
  再開時は少なくともtensor/static、channel/token-dynamic、weight 128x128＋activation 1x128 block-dynamic、
  weight-only W8A16を別recipeとして扱い、`compressed-tensors`、`quant_method: fp8`、scaleとinverse-scale、
  F32／BF16／UE8M0 scaleをconverterでversioned内部契約へ正規化する。保留中は実装やprovider追加を開始しない。
- OCP MXFP8 W8A8／MXFP6 W6A6ではweightをvalue planeとblockごとのE8M0 scale planeとして常駐させ、
  BF16 activationを各matmulの前段で同じOCP block-32形式へ動的量子化する。積はFP32で累積し、graph境界のoutputは
  BF16 RNEとする。E8M0 NaN scaleはblock全体へNaN伝播し、Infはelement最大有限値へsaturationする。
  Kが32の倍数でないtensor、scale欠落、未対応targetはfallbackせず拒否する。
  初期実行targetはexact `gfx1030`／`gfx1201`に限定する。

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
- 2026-09-13のユーザー指示により、Git管理外・ignore対象のscratch copyや実験コードも、実際に流用した場合は記録する。取込みcommitがない場合は適用外とし、確認時hash・確認日と不明な取込み日時を区別する。コード自体をGitへ追加する必要はない。
- 詳細は `docs/provenance/README.md` を正とする。

## 開発・最適化の優先順位

### 最適化の共通化と既定採用の方針

Phase 79（共通化）、Phase 82（不採用最適化の削除・条件付き既定採用）、Phase 83.5後の追加共通化
（MTPをモデルアーキテクチャではなく投機的デコーディングの提案方式として整理し、model固有のhead・hidden・状態処理はadapterへ残す）を経て、以下を継続方針とする。

- 最適化は演算の意味、GPU能力、shape/layout、重み・KV encodingに基づいて適用する。model fingerprintによる成果物検証は維持し、最適化判定のモデル固定条件は必要な演算条件へ置き換える。
- prepared cache、same-stream segment owner、completion集約とGraph実行制御は共通execution層で再利用する。モデル固有graph、attention preprocess、GDN、model stateはadapter側に維持する。
- 正しく実行できる条件と性能上採用する条件を分け、既存selectorを整理する。選択した経路と不採用理由を観測可能にし、確認できた範囲から条件付きで既定採用する。強制選択は比較・切戻し用に残す。
- Graph制御の共通化とKV形式別kernel対応は別作業とする。FP16条件を単に外さず、量子化scale、配置、buffer寿命、同期条件を扱う。static tensor FP8とOCP MXFP8は別形式として扱う。
- 検証は影響する演算・境界・形式と小型の代表モデルを中心にする。数値誤差・logits/品質と性能を分け、生成列一致だけで量子化の採否を決めない。旧Phaseの長時間測定を一律に再実行しない。
- 今回の後続最適化ではモデル固有の追加速度探索を優先しない。全モデル・全KV形式対応や一律速度倍率を新しい必達条件にしない。
- 2026-09-08のユーザー決定: 理由があってどの経路でも既定採用されなかった候補は、共有処理・他target／shapeでの利用を確認して削除する。
  データ不足の候補は確認可能な範囲で追加検証し、条件付き既定採用を進める。現在のbaseline・採用経路に必要な処理は維持する。
- 削除する候補は、試した変更、比較条件、数値・性能結果、不採用理由、元のsource／証拠、削除範囲・commitを
  [Phase 82履歴](../history/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md)へ記録する。過去の測定履歴を保持し、
  観測事実と原因推定を分ける。今回判断できない候補は不足事項と再検討条件を残し、未確認を採用済みと扱わない。

具体的な対象と順序は[現行Phase 76〜88計画](active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)を正とし、
Phase 79の共通化内容は[Phase 79計画](archive/2026/09/1-10/phase79-common-optimization.md)に記録する。

### 既存の優先順位・採用基準

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

### 性能・品質の評価方針

- 代表性能条件は、template適用後の8,192入力token／実際に確定した128出力token、Qwen3.8 27B NVFP4、
  MXFP8 E4 KV、固定sampling、MTP有効、single GPU／batch=1とする。prefillにはMTPのprefix準備を、
  decodeにはdraft／verify／棄却・replay／samplingを含め、棄却tokenをthroughputへ加算しない。
  1 warmup＋3 measuredの中央値を用い、TTFT／end-to-end時間とばらつきも記録する。
- 2026-09-10ユーザー決定: Phase 83.5の速度目標（V620 200/20・R9700 500/25 tok/s）は比較用に残すが、
  新しい速度下限は設定しない。
- 2026-09-09ユーザー決定: MTP有無の出力token列の完全一致は必須としない。同一BF16参照に対する精度劣化が
  MTPなしと同程度であれば、MTPによる出力差を許容する。状態破損、要求履歴に依存する文章崩壊、sampling実装の不具合は含めない。
  単一の生成例やkernelのBF16出力一致だけでモデル品質の同等性を認定しない。
- 量子化MTP companionの採否は、実効速度・採用率・数値検証を合わせて判断する。BF16 companionが現在の既定である。
- MTPの採用率・実効速度の比較は[MTP採用率・実効速度ベンチマーク](../development/mtp-acceptance-benchmark.md)
  （`mtp-bench-v1`、coding agent 10言語＋翻訳、創作は参考値）を正本とする。主指標M1は、teacher forcingで保存した
  draft／target logitsから求める期待p/q受理率であり、prompt単位のクラスタ推論で判断する。
- 同一形式・同一入力でもGPU間でtarget hiddenが一致しないため、GPU間で採用率の絶対値を比較しない。比較はGPUごとの基準に対して行う。
- `SLLM_FORCE_BASELINE`は診断用の参照経路（T2）であり、本番のrollback手段ではない。本番の切戻しはbinary／commit単位で行う。
  詳細は[FORCE_BASELINE履歴](../history/2026/09/11-20/force-baseline-reference-oracle.md)を参照する。

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
  subagentへ積極的に委譲し、資源または依存上の理由がなければ利用可能な並列枠で同時実行する。subagentのmodelは使用中のCodex profileに従う。
  既定のOpenAI profileでは速度に優れるxhighのLunaを優先し、Terra/SolはLunaとmain agentで効率的に扱えない横断調査、
  反復失敗後の上位対応、または特に深い専門推論が必要な場合だけ使う。`codex -p deepseek-flash`等の別provider profileでは、
  OpenAI modelに固定されたLuna/Terra/Solを起動せず、そのprofileの既定subagentだけを使う（2026-09-19ユーザー指示）。main agentは編集確認、共有作業領域の競合解消、関連検査に責任を持ち、
  subagent利用や特定の`codex exec`実行方式を完了条件にしない。
- 各フェーズは受入条件、検証、計画・履歴の完了処理後に、そのフェーズだけを必要最小限のcommitへ整理して現在のGitHub branchへ
  pushする。次フェーズの変更を同じcommitへ混ぜず、共有済み履歴の書換えや強制pushを行わない。
- 作業単位は独立してreview・rollbackしやすい範囲とするが、細分化、不変identity、独立review、全matrix実行を各下書き時点の完了条件にしない。下書き、統合、release/push、文書のみの作業区分と実行手順は`AGENTS.md`を正本とする。
- AIが厳格な必須条件、独立review必須化、広範／GPU再実行、security境界、再利用制限、阻害段階、作業単位の追加分割、不変証拠の拡張を提案する場合、明示的なユーザー承認までは提案元・範囲・費用・期限を持つ非阻害提案として扱う。
- 受入条件は作業単位の開始時に固定する。実際の正しさ・security上の欠陥は阻害条件にできるが、review中に新しく作られた手続き上の要件は承認なしに遡及適用しない。
- ソース／build入力、toolchain、モデル固定、成果物digestから成る意味上のidentityをGit commit identityと区別する。文書だけの変更で意味上のidentityが変わらないことを確認できればコード／GPU証拠を再利用し、文書だけの完了処理や新しい独立reviewを行わない。
- 適用可能なservice／実行環境が対象にある場合だけ適用後smoke／healthを要求する。独立した適用先がないlibrary、tool、文書はpush可能である。
- 同じ単位の2回reject、review時間が実装時間超過、1時間以上の機能進捗停止、検証・文書が30%超、見積り1.5倍超、gate/受入条件変更のいずれかで、新規review・検証を停止し、ユーザーへ報告して計画を見直す。

### Phase完了時のcommit・pushとCI確認

2026-09-07のユーザー明示指示により、各Phaseの完了手順にcommit・GitHubへのpush、公開commitのCI結果確認、
失敗原因に応じた必要な修正を含める。この指示を継続的な公開・CI修復の許可として扱い、Phaseごとの再確認は求めない。
ユーザーによる個別の停止・保留指示を優先し、AGENTS.md／sLLM.md等の別途承認が必要な変更はその規則に従う。

- pushしたHEADに紐づくCI runを終了まで確認する。別commitの成功を流用せず、missing、cancel、timeout、想定外skipを成功と扱わない。
- 実行されたCIの失敗は、変更前からある不具合も含めて切り分けて修正し、最終公開HEADの対象CI成功まで繰り返す。
  成功させるためだけの検査削除、無条件skip、continue-on-error、数値基準緩和は行わない。docs-only変更だけでGPU実測を再実行しない。
- 外部障害、権限不足、広範な修復が必要な場合は、原因・run URL・残件を記録し「実装完了・CI確認待ち／修復中」として報告する。
- 完了報告には最終commit、CI run／結果、検証範囲と残件を記載する。

### 継続方針

- README整備と人間による発表は番号を割り当てない将来タスクとし、製品フェーズの完了条件へ混ぜない。
- H3の必須化は引き続き観測事項であり、現時点では必須条件へ昇格しない。
- 現行の開発形態は`trusted-solo-development`である。下書き、統合、公開、文書のみの扱いは`AGENTS.md`を正本とし、
  過去フェーズ固有の検証手順を現在の一律条件へ読み替えない。

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
| 完了 | 46 | 変換、量子化、imatrix、分割・結合、ベンチマーク、品質・デバッグ用ツールとKV default品質policy |
| 要承認 | 47 | 組込みtool/MCP実行。別worker/sandboxと信頼境界の承認前は開始しない |
| prototype完了・継続中 | 48 | 最小WebUI（GPU／throughput dashboard主画面、chat副画面）、server側model library、Hugging Face検索／download、source treeでの起動統合を実装。製品化項目は自動開始しない |
| 完了・限定採用 | 49 | V620 `gfx1030`でGQA P32を限定採用し、long-prefill v2とHIP Graphを棄却。通常5行の退行確認まで完了 |
| 完了・限定採用 | 50 | R9700 `gfx1201`でPhase 49変更を採否し、MI300X `gfx942`向けwave64引継ぎを準備 |
| 完了・target分離 | 51 | MI300X `gfx942`の7行とprofileをPASSし、GDN wave64候補を既定無効でtarget分離 |
| 完了 | 52 | R9700 `gfx1201`の長capacityをresident KVへ限定routeし、`10,001/2`と`100,000/2`の自動経路を再検証 |
| 完了・superseded | 53 | block16 descriptor v2をtarget別評価。当時は品質未達で`retain-fp16`、後のPhase 54決定で製品経路を廃止 |
| 完了・経路廃止 | 54 | exact gfx1030でblock16候補を評価したがMXFP8を上回らず、2026-08-30決定でblock16経路を廃止 |
| 完了 | 55 | Gemma 4 26B-A4B MoEをNVFP4 artifactから単一GPUのCLI/API/WebUIへ統合 |
| 完了 | 56 | Gemma 4 12B公式assistantによるMTPをtarget-only同値のCLI/API/WebUI経路へ統合 |
| 完了・foundation | 57 | DeepSeek V4 Flashの公式identity、圧縮attention、mHC、MoE、混合FP4／FP8、容量fail-closeを実装 |
| 完了・foundation | 58 | MiniMax M3の公式identity、MSA、MoE、MTP／multimodal metadata、manifest不整合／容量fail-closeを実装 |
| 完了・foundation | 59 | DiffusionGemmaの公式identity、causal encoder／bidirectional decoder、self-conditioning、block refinementを実装 |
| 完了 | 60 | Ministral 3 3Bの公式GGUF head permutationに合わせRoPEをadjacent-pairへ修正し、固定llama.cpp top-1、両RDNA実GPU、resident速度を確認 |
| 完了・実モデル評価済み | 61 | OCP MXFP8 E4M3 W8A8／MXFP6 E3M2 W6A6を統合。両RDNA operatorとgfx1030 Qwen3.5-4B品質／VRAM／速度を測定し、現providerは非defaultと判定 |
| 完了・共通採用／MMQ候補評価済み | 62 | 再利用可能scalar/block codecとtyped viewへMXFP/NVFP primitiveを分離。両RDNAでbit exact、W/A・KV・attention性能改善を確認し、llama.cpp由来multi-column構造をbenchmark-only評価 |
| 完了・v2限定採用 | 63 | contribution LDS削除とN64 tileをexact gfx1201へ限定採用。N=1,024拡張、K pipeline棄却、3+10 full-model再検証まで完了 |
| 完了・shape限定採用 | 64 | exact gfx1201 MXFP8 WMMAのweight direct-loadを対象shapeへ採用し、4B／9B prefillを改善 |
| 完了・shape限定採用 | 65 | activation／weight direct-loadのID36をmodel名非依存で採用し、4B／9B 2,048-token prefillをさらに改善 |
| 完了・ID37限定採用／attention棄却 | 66 | 共通prepared low-precision providerをMXFP8で実証し、ID37 N128をgfx1201の測定済みshapeへ採用。MXFP6、NVFP4／MXFP4、BF16 attentionまで実移植・採否を完了 |
| 完了・ID27 shape限定採用 | 67 | gfx1201のN方向再利用をgfx1030へ転用評価。ID38/39はbenchmark-only、既存ID27 col8を測定済みlarge projectionへ限定採用し、4B prefillを約2.88〜2.92倍へ改善 |
| 完了・内部MX fast path採用 | 68 | gfx1030 E4／scaleを分離測定。内部MX value planeのnormal common pathと安全なcombined-exponent block decodeを採用し、4B prefillをさらに約2.97〜4.35%改善 |
| 完了・ID41限定採用 | 69 | exact gfx1030 MXFP8 software-MMQで32-bit E4 ingressを既存ID27 scopeへ採用し、4B prefillを22〜24%改善。scale register化／combined候補はbenchmark-only |
| 完了・gfx1201 ID45 shape限定採用／ID46・gfx1030 ID43 benchmark-only | 70 | packed E3M2→E4M3 exact ingressでMXFP8 MMQ／WMMA骨格を再利用。P70-Fで4-value ingressのID45をID44比1.61〜1.69倍へ改善して既定化 |
| 完了・実モデル評価済み | 71 | Qwen3.5-27B MXFP6 reviewed model対応と両RDNAでのbounded-VRAM実測 |
| 完了・shape限定採用 | 72 | exact gfx1201 MXFP6 ID45 wide-N selectorをN<=32,768へ拡張 |
| 完了・ユーザー指定scope拡張 | 73 | exact gfx1201 MXFP8 ID31／34／36／37のN上限を32,768へ緩和。host／provider contractをPASSし、新規wide-N性能・数値GPU再検証は省略 |
| 完了・両target限定採用 | 74 | MXFP6 prefillを3反復で改善し、gfx1030 ID47／gfx1201 ID48を限定採用 |
| 完了 | 75 | gfx1030 MXFP8／MXFP6 decode共通half2経路を改善し、ID55／57を限定採用 |
| 完了・両target実モデルPASS（R9700 single-visible） | 76 | exact Unsloth Qwen3.8-27B混合NVFP4 artifactの統合、正しさ、baseline／profile |
| 完了・decode機能／dispatch PASS（速度残差はPhase 78へ統合） | 77 | 同artifactのsingle-request decode専用経路を成立。実用速度は未達のため、whole-model速度gateをPhase 78で閉じる |
| 完了・ユーザー承認による目標変更／未達受容 | 78 | r25を到達点として終了。V620 decode基準を実artifact帯域へ変更し、prefill等の旧目標未達・正式比較未実施を明記。モデル固有の追加最適化は要求しない |
| 完了・条件付き既定採用 | 79 | NVFP4/FP8 decode既定化、projection/実行制御共通化、prefill基準加算順復元 |
| 完了・公開CI成功 | 80 | CI修復、公開API／依存manifest同期、Rust資源設定、失敗診断と公開後CI確認 |
| 完了・公開CI成功 | 81 | 固定sampling profileの共通GPU実装・API統合。代表条件でprefill／decodeへの追加負担がほぼないことを確認 |
| 完了 | 82 | 不採用最適化の削除・試行と失敗理由の記録、データ不足候補の条件付き既定採用 |
| 完了・実装検証済み | 83 | MXFP8 E4 KV・固定sampling／MTP・CLI/API統合、両GPU長文・対話・lifecycleを確認。速度改善は83.5 |
| 完了・公開CI成功 | 83.5 | 共通演算とMTPを最適化。速度条件緩和を記録し、最終反復値は旧目標も達成。モデル方向の追加共通化も実装・検証完了 |
| 完了 | 84 | MTP sidecarと通常CLI/APIを接続。MXFP8のdecode退行によりBF16既定を維持。MXFP6も追加比較を完了し、BF16既定を維持 |
| 完了 | 84.5 | MTP接続・固定p/q・採否後の状態を限定照合。R9700のtarget差はattention演算順へ切り分け。本番既定は維持 |
| 完了・実装検証済み | 85 | 共通MXFP8／MXFP6 kernelをscope限定採用。両GPUの広範shape、本体・KV・MTP、chunk末尾の効果と数値を確認。BF16 MTP既定を維持 |
| 完了・既定採用せず | 86 | Qwen MTP catch-upを両GPU26条件で検証。期待p/q受理率では小さい正の効果があるが、分離catch-upは正味マイナスで既定不採用。ベンチマーク主指標を期待受理率へ切替 |
| 段階5完了・段階6計測中 | 87 | Qwen3.8 NVFP4の計測・棚卸しとread帯域計測を完了。WU1でV620 attentionのGQA共有を採用（MTPなし+8.09%／あり+6.88%）。WU1.1で両GPUのlong contextへsplit128を採用（単体TPOT比1.85%／2.22%短縮）。WU-C1で不要な切替を削除し、N0とCI hash連鎖の整合を確認。WU2でR9700 FP8のM1 dot4 GEMVを採用（単体TPOT比6.40%、モデルMTPなし+9.40%／あり+1.50%）。段階5でwhole graphと非同期readbackを接続し、両GPUのN0・停止・境界・profileを確認。後続はMTP companionのNVFP4化とMXFP6比較、W×A16の廃止等 |
| 計画済み・繰下げ | 88 | NVFP4のGPUリクエストバッチ処理を最適化（旧87、さらに前は旧86） |
| 完了 | X | llama.cpp HIPのQ5_1 Flash Attention構成を修正し、ローカルQwen補助エージェントへ反映 |
| 完了 | XA | host-required／通常H3／public-runtime H3 CIを修正し、Phase 52候補のpush後workflow完了まで確認 |

Phase 0〜75の各フェーズの詳細な経緯は、表からリンクされる計画・履歴と
[整理前のスナップショット](../history/2026/09/11-20/main-plan-before-reorganization-2026-09-17.md)を参照する。
Phase 37以降の割当と依存関係は[フェーズ37以降の進行中計画](active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)、
Phase 76以降は[Phase 76〜88計画](active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)を正本とする。

### 直近の流れ（Phase 76〜88）

Phase 76以降は、Qwen3.8 27B NVFP4を単一GPUで実用速度にすることを軸に進めている。

1. **Qwen3.8対応と速度基盤（76〜78）**: 混合NVFP4 artifactを統合し、decode専用経路を成立させた。
   Phase 78はユーザー承認により目標を変更して終了した（[履歴](../history/2026/09/1-10/phase76-78-qwen38-nvfp4.md)）。
2. **共通化と運用の立て直し（79〜82）**: モデル固定の最適化判定を演算条件へ置き換え（79）、CIを修復し（80）、
   固定samplingをGPU共通経路へ実装した（81）。不採用候補を削除し、確認済みの経路を条件付きで既定化した（82）。
3. **MXFP8 KV・固定sampling・MTPの統合と高速化（83、83.5）**: 正しい実装をPhase 83で完成させ、速度はPhase 83.5で
   llama.cppを参照して改善した。速度目標は2026-09-10に緩和したが、最終値は旧目標も上回った
   （MTPあり8192/128: V620 216.6/25.4、R9700 541.4/34.5 tok/s）。続く追加共通化でMTPを提案方式として分離した。
4. **MTP companionの量子化（84〜85）**: MXFP8 W8A8／MXFP6 W6A6のMTP companionは採用率を概ね維持したが、
   decodeがBF16より遅く、BF16既定を維持した（84）。限定診断で経路の不整合は見つからなかった（84.5）。
   Phase 85はMXFP8／MXFP6共通kernelを広範shapeで最適化し、scope限定で採用した。
   同Phase内の追加実験（M=1経路、ボトルネック診断、GPU別改善、A16化）は小幅な改善にとどまり、BF16既定は変わらなかった。
   ここまでで、量子化MTPの退行の大半はkernel時間ではなく採用率低下によるblock数増加であると分かった。
5. **採用率の測定基盤（Phase外の調査）**: 採用率を信頼して比較するため`mtp-bench-v1`を策定・凍結し、
   teacher forcingでの位置ごとの比較、GPU間差の調査、FORCE_BASELINEの参照経路化を行った（次節）。
6. **MTP catch-upの条件付け（86）**: BF16 companionでcatch-upの効果を両GPU・26条件で検証した。
   期待p/q受理率では小さい正の効果（V620で有意）があったが、分離実行では正味マイナスのため既定採用しなかった。
   融合catch-upは試していない。あわせてベンチマークの主指標を期待p/q受理率へ切り替えた
   （[履歴](../history/2026/09/11-20/phase86-mtp-catch-up-conditioning.md)）。
7. **次（87、88）**: Qwen3.8 NVFP4の単一要求decode（NVFP4 W4A4とFP8 W8A8）の最適化とW×A16の廃止、
   続いてNVFP4のGPUリクエストバッチ処理を行う。

### フェーズ外の調査

番号を持たない調査・基盤作業の結論と正本を示す。いずれも本番の既定選択は変更していない。

| 日付 | 調査 | 結論 | 正本 |
| --- | --- | --- | --- |
| 2026-09-11 | 言語・タスク別MTP採用率 | V620の採用率が一貫して低いとはいえない。GPU間の率一致を条件にしない | [履歴](../history/2026/09/11-20/mtp-language-task-acceptance.md) |
| 2026-09-14 | 最新llama.cppのMTP量子化比較 | llama.cppではMTP 8行列の量子化でもdecodeが約2.5〜4.4%改善。sLLMへは一般化しない | [履歴](../history/2026/09/11-20/llama-mtp-quantization-benchmark.md) |
| 2026-09-14〜17 | MTP採用率・実効速度ベンチマーク | `mtp-bench-v1`を凍結（Tier A 26条件、Tier B 3条件、256 token）。主指標を期待p/q受理率へ変更 | [仕様](../development/mtp-acceptance-benchmark.md) |
| 2026-09-15 | MXFP8のGPU間差 | GPU間ではtarget hiddenが一致せず、native codecと復号経路は原因から除外。WMMAは寄与要因の一つ。残りの帰属は未特定 | [履歴](../history/2026/09/11-20/mxfp8-gpu-divergence.md) |
| 2026-09-17 | FORCE_BASELINEの参照経路化 | 本番rollbackとしては直さず、host独立FP32 oracle（T1）とGPU上の帰属用参照経路（T2）に役割を限定。NVFP4 W4A4参照kernelのlaunch分割等で両GPUの実モデルを完走 | [履歴](../history/2026/09/11-20/force-baseline-reference-oracle.md) |
| 2026-09-17 | rocm_exl3調査 | 固定revisionをR9700で調査・局所修正。Qwen3.5-2BではQ4_K_Mがprefill、EXL3がdecodeで速く、9Bでは長いprefillだけEXL3が速い。sLLMへの形式採用とは分離 | [調査](../history/2026/09/11-20/rocm-exl3-investigation.md)、[9B比較](../history/2026/09/11-20/rocm-exl3-qwen35-9b-comparison.md) |
| 2026-09-18 | Qwen3.8-27B全語彙KLD | 初期8構成とKV・chunk・長文の40条件、64組のKLD比較を取得。非対応条件を区別し、NVFP4 E5はV620で取得。形式・GPU・実装の差を含む実測として記録 | [計画](archive/2026/09/11-20/qwen38-cross-engine-kld.md)、[履歴](../history/2026/09/11-20/qwen38-cross-engine-kld.md) |
| 2026-09-18 | Qwen3.8 MXFP8とvLLM FP8のKLD差 | 主因はMXFP8の重み・活性値のE8M0 scale選択（最大値の指数切り捨て）による飽和。両方を飽和回避scaleにすると平均KLD 0.0459→0.0178（vLLM FP8 0.0145）。活性値側の寄与が大きい。既定は未変更で、診断opt-inだけを追加 | [計画](archive/2026/09/11-20/qwen38-mxfp8-vllm-fp8-attribution.md)、[履歴](../history/2026/09/11-20/qwen38-mxfp8-vllm-fp8-attribution.md) |
| 2026-09-20 | vllm-mxfp4参照追加・実測 | R9700でAMD MXFP4＋FP8 activationを実行。既存2632位置のBF16比KLDは0.12529（BF16 KV）／0.13202（FP8 KV＋FP16 SSM）。速度・長文・比較条件は別記し、sLLM形式採用とは分離 | [履歴](../history/2026/09/11-20/vllm-mxfp4-investigation.md) |

### llama.cppとの機能差

- 2026-08-21に固定参照llama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`と公開CLI／HTTP機能を比較し、
  差分をフェーズ39〜48へ割り当てた。組込みtool/MCP実行（Phase 47、要承認）とWebUIの製品化（Phase 48の残件）、
  生成途中・wire sessionの再開を除き実装済みである。分類別の差分表は
  [整理前のスナップショット](../history/2026/09/11-20/main-plan-before-reorganization-2026-09-17.md)と
  [フェーズ37以降の進行中計画](active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を参照する。
  sLLMのAPI仕様は引き続き[OpenAI互換profile](../api/openai-compatibility.md)を正とする。
- モデルfamily、hardware／backend、並列・継続batch、複数GPU、性能provider探索は棚卸しの対象外とした。
  Vulkanと一般的なllama.cpp INT4/INT8+scale量子化は意図的な除外である。
- TurboQuantを含むその他の残りKV形式、残るモデルfamily、複数GPU／Infinity Fabric／RDMA、README整備、人間による発表、
  LMCache、RadixAttention、その他の将来MX形式には現時点でフェーズ番号を割り当てない。これらを初期versionの完了条件へ読み替えない。

### 性能最適化の残課題

実行時dispatch・同期、attention・KV、GDN、Dense BF16、FP8／NVFP4／MXFP4、MoE、MTP、sampling・service、
要求状態・memory、モデル読込みの未割当候補は、2026-08-18以降の棚卸しとして
[整理前のスナップショット](../history/2026/09/11-20/main-plan-before-reorganization-2026-09-17.md)の同名節に保存している。
一覧は完了済みフェーズの範囲を拡張せず、着手時の新しいprofileで受入条件・対象targetを固定する。一般論だけの候補をhard gateにしない。
このうち直近で扱う範囲はPhase 87（Qwen3.8 NVFP4の単一要求decode）とPhase 88（リクエストバッチ処理）である。

### 再提案しない候補

- V620の全M/shapeを無条件にhipBLASへ切り替える案は、フェーズ9とフェーズ34の短M/small-N実測により再採用しない。
  フェーズ34で採用したexact production shapeとM thresholdだけを維持し、未知shapeへ一般化しない。
  R9700のtransposed GDN state、既存weight-only NVFP4 decodeの複数N列・scale broadcast、V620で当時（2026-08-18時点）の実装のままMTP幅2を
  有効化する案は既存実測で改善しなかったため再採用しない。
- 短contextでのfull attention/FA3-like最優先化とrequestごとの実運用HIP Graph生成も再採用しない。
  前提となるprofileまたは実装構造が変わった場合だけ、新しい候補として別に測定する。

## 現在の状態と次の作業

- **完了**: Phase 86まで完了した。現在の既定はBF16 MTP companion、catch-up無効、Qwen3.8のMXFP8 E4 KVである。
- **完了（段階0）**: [Phase 87の計測と棚卸し](active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。
  現行は既にW4A4であると確認し、ユーザー指示でW4A16からの移行比較は省く。decode attentionも候補へ追加した。
  両GPU・MTP有無の時間／read-request量、形状別copy、R9700のFP16／MXFP8 E4 KVのKLDを取得した。
  [段階0の結果](../history/2026/09/11-20/phase87-stage0.md)に基づき、V620はMXFP8 E4 decode attention stage1を第一候補とする。
  [WU0のread帯域計測](../history/2026/09/11-20/phase87-wu0-read-bandwidth.md)も完了した。
  WU0は2026-09-20にクロック状態を補正して再計測し、WU1のV620 attentionは余地12.41／半分6.21 ms/token、WU2のR9700 FP8は正の形状別余地4.08／半分2.04 ms/token。
  read probeは物理上限ではなく、演算・復号の費用を含まない参照値である。
  [WU1](../history/2026/09/11-20/phase87-wu1-attention.md)は2026-09-20完了。
  V620のGQA共有を採用し、MTPなし14.295→15.451 tok/s、あり25.430→27.181 tok/s、生成列は一致した。
  R9700は候補が打ち切り線未達のため現行維持とした。2026-09-20に打ち切り線と採用基準を分け（TPOT 1%以上の短縮等）、
  [WU1.1](../history/2026/09/11-20/phase87-wu1-1-attention.md)も完了し、両GPUのKV長8192以上・M1〜3へsplit128をN1として採用した。
  単体では全round改善、TPOT比1.85%／2.22%短縮。モデルの生成列・MTP受理数は変わり、R9700 MTPありは−3.41%だった。
  新基準に従い単体の採否とモデル実測を分離して記録した。
  [WU-C1](../history/2026/09/11-20/phase87-wu-c1-cleanup.md)も完了。不要な8環境変数・CLI旧flags・実験経路を削除し、
  両GPUの削除前後token一致（N0）、host/GPU検証、CI hash参照連鎖と3種validatorsのPASSを確認した。[WU2](../history/2026/09/11-20/phase87-wu2-fp8.md)も完了し、R9700 FP8 W8A8のM1・8形状へnative dot4 GEMV（ID103）をN1として採用した。
  dot単体は3.269757 ms/token（通常TPOT比6.4039%）短縮、全AB/BA roundが正。R9700通常モデルはMTPなし+9.40%／あり+1.50%。
  V620は既存provider・生成列を維持し、速度差−0.03%／+0.15%。R9700 MTPなしのみtoken位置15から分岐する。
  両GPU単体・公開API／graph・モデル・host・CI検証はPASS。
  2026-09-20のWU2後の[GPU空白時間の再計測](../history/2026/09/11-20/phase87-idle-recheck.md)で、空白は2.6〜4.9 ms/token
  （割合は6.5〜9.1%）と段階0からほぼ変わらず、段階5の見込みも変わらないことを確認した。
  同時に、V620のM=1 NVFP4 decodeが直前のdecode attention kernelの種類で約10%遅くなることが分かった。
  [WU-D1の原因調査](../history/2026/09/11-20/phase87-wu-d1.md)も完了。KV読み出しだけのGQA型先行kernelで
  両NVFP4形状の約9〜10%の遅延を再現した。単独とstaged型先行は同程度。EA busy cycleの増加を確認したが、
  MALL／DRAMの内訳は未取得で物理機構は未特定。特定範囲と限界を記録し、production sourceと採用済みWU1／WU1.1／WU2を維持した。
  [WU-D2](../history/2026/09/11-20/phase87-wu-d2.md)も完了。V620は32 dispatch後も遅延が残り、軽い介在処理で回復しない。
  R9700は同じprobeで非再現、物理的な内訳は未取得。
  [WU-D3](../history/2026/09/11-20/phase87-wu-d3.md)は3候補とも不採用で完了。tile32／先読みはattentionが退行し、
  block配置変更は単体でほぼ中立、実モデルはMTPなし+0.344%／あり−0.093%で採用基準に未達だった。
  tokenは一致し、本番sourceは復元済み。2026-09-20のユーザー決定でD系統を打ち切り、物理機構は未特定のまま
  再開条件（MALL／DRAM内訳を取得できる計測手段、または実attentionのdata flowで罰則を再現するharness）だけを計画へ残した。
  影響はV620で約1.9 ms/token（TPOT比約3%）、R9700で約1.0 ms/tokenで、残る候補より小さい。
- **完了（段階5、2026-09-21）**: decode 1段全体のHIP graph、device上のtoken／hidden／状態引継ぎ、pinned非同期readbackを実装。
  両GPUの固定samplingでN0、停止・予算・context末尾、正常解放、次graph起動がreadbackを待たないことを確認した。
  初回完了時の通常8192/128のdecodeはV620 15.6762／29.4250、R9700 20.8622／34.6104 tok/s（MTPなし／あり）。
  MTPありは直前比+2.00%／+1.69%、MTPなしは-0.61%／-2.56%。[履歴・証拠](../history/2026/09/11-20/phase87-stage5.md)。
  同日の追加修正で、状態コピー・attentionの定数最適化・既知budget終端の不要replayを改善。
  MTP有無で全体経路を分けず、V620 16.1868／30.1174、R9700 21.4909／35.6390 tok/sとなり、
  MTPなしの低下を解消してMTPありも改善した。[追加修正・検証](../history/2026/09/21-30/phase87-mtp-off-regression.md)。
  [Phase 87変更のCI整合も修復](../history/2026/09/21-30/phase87-ci-repair.md)し、local H0／H1／H2と
  両targetのCI用直接compile/link・ELF検査がPASS。commit／pushと公開CIの再実行は未実施。
  段階5でgraph間隔は0.03 ms/tokenまで減ったが、残る空白の大半はgraph内のkernel間dispatch固定費
  （MTPなしでV620約8.0／R9700約6.6 ms/token、1,218 node/token）であり、host待ちではないと分かった。
  次は段階6（独立nodeを並列枝として表し、この固定費を重ねる。効かなければkernel融合へ切替）。
  後続はNVFP4 W4A4とFP8 W8A8等のdecode最適化、MTP companionのNVFP4化とMXFP6比較、W×A16の廃止等。
  続いてPhase 88（NVFP4リクエストバッチ処理）。
  詳細と受入条件は[Phase 76〜88計画](active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)に従う。
- **完了（番号なし）**: [低精度形式のscale選択の修正](archive/2026/09/11-20/low-precision-scale-selection.md)。
  2026-09-19のユーザー決定で、MXFP8重み・活性値とMXFP6重みのE8M0 scaleを飽和しない規則へ既定変更した
  （Qwen3.8 MXFP8の平均KLDは重み・活性値とも新規則で0.0459→0.0178）。NVFP4活性値の2候補選択は、不具合修正後に
  測り直すと悪化したため不採用。KV、MXFP6活性値、MXFP4は従来の規則のまま。
- **完了（番号なし）**: MXFP8／MXFP6／NVFP4／内部MXFP4の行列積カーネルを`native/lowp`へ切り出した
  [境界化計画](archive/2026/09/11-20/lowp-kernel-library-boundary.md)。両GPUの指定モデルでtoken列・logits一致、選択表4,700件、host/HIP検証を完了した。公開MXFP4 W4A8は契約定義のみで、実装はPhase 87へ残す。
- **承認待ち・自動開始しないもの**:
  - Phase 47（組込みtool/MCP実行）は承認制の独立laneとして維持する。
  - Phase 48はGPU／throughput dashboard主画面、chat副画面の最小WebUI、server側model library、Hugging Face検索／download、
    source treeでの起動統合まで完了した。static asset埋込み、versioned artifact配布等の製品化項目は自動開始しない
    （[Phase 48計画](active/2026/08/21-31/phase48-minimal-webui-prototype.md)、起動・認証の決定は[runtime文書](../architecture/runtime.md)）。
  - NVFP4の独立specialization（E2M1、block 16、E4M3 block scale、FP32 tensor scale）は未計画である。
  - gfx942のPhase 53相当実機証拠は、追加のMI300X検証項目がまとまった時点で一括実行する。

## 未解決事項

- AMDコンシューマーRDNA2を含む各gfx targetの厳密な実機検証範囲。
- ROCm 7.14.0とHWE kernel 6.17を組み合わせたV620/R9700 tupleについて、長時間安定性と正式な
  互換性状態を判断できるだけの実測が揃っていない。
- 追加op・shape・入力範囲の数値toleranceと、複数のO2/O3履歴run・分散・再現性が揃った後に定める
  性能回帰閾値。
- 資源条件の1 TOPS、16 GB、帯域の定義と例外承認基準。
- Infinity Fabric、他RDMA protocol、KV永続化の詳細設計。
- 量子化形式ごとのlayout、scale粒度、accumulator、fallback表。
- MXFP4 W4A4実装から、ActivationをOCP MXFP6 E3M2 block32／E8M0とするW4A6契約への移行範囲。
- sudo以外の既存平文credentialの失効・rotationとsecret managerへの移行状況。
