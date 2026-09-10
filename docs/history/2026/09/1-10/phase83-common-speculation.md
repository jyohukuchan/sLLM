# Phase83・83.5共通化の変更と検証

> 状態: 実装・host／実機検証完了。公開commitのCIはGitHub Actionsで確認する。

## 開始時点

- 基準commitは`865a9e9e2201df2b8996007a12e64c739a009399`。作業開始時のtreeはclean。
- Phase83.5は完了済みで、同commitのhost・basic H3・public-runtime H3は成功している。main-planのフェーズ一覧に残った旧状態・旧84/85番号を訂正する。
- GPU演算の形状条件と、coreのmodel/artifact固定条件は区別する。「共通kernelを使う」ことを「別モデルの通常入口でも選択される」証拠にしない。

## 変更一覧・検証

| Phase83・83.5の変更群 | 開始時の適用条件 | 今回の扱い |
| --- | --- | --- |
| 標準MXFP8 KV storage/append・resident配置・metadata query | encoding、layout、target、sliding条件。KV既定値のモデル別品質policyは別 | 共通backend実装を維持。既定KVの品質policy変更やGPU対応拡大へ読み替えない |
| MXFP8 attention（staged decode、prefill、wave-local KV接続修正） | head数/比率、head dim、M/長さ、encoding、target。model fingerprint参照なし | 既に演算条件で共有。kernel内の固定geometryを無根拠に緩和しない |
| NVFP4/FP8行列積、LDS padding、byte展開、small-M共有 | dtype/scale/layoutとM/K/N・target | 既に共通selector。model名を必要とする条件とshape条件を区別して保持 |
| Projection pack・入力量子化共有 | 共通pair契約に加え、Qwenのartifact/role数条件と専用shape provider | NVFP4のdynamic K/N契約と行数条件を共用し、HIP wrapperの旧固定K条件も除去。GemmaでM1の48 packを実確認。M2–4・M>=64は候補にはなるが測定Gemma shapeはnative採用条件外で通常経路へ戻る。FP8 GDNの固定geometryは保持 |
| Residual Add + RMSNorm | Qwen4BまたはQwen3.8 artifact、layer数、binding数、node label | semantic dataflowの共通passへ変更。QwenとMinistralの通常入口に接続。tensor ID・BF16中間出力・epsilon/scale・依存関係を保持し、backend非対応なら元経路 |
| MTP targetの複数row検証、terminal row保持、GDN checkpoint | model stateとrecurrent数値順、private M3のlayout/width | 共通の投機的制御とmodel adapterの状態責務を分離。GDNを他architectureのKVとして扱わない |
| companion state-only prefix準備 | Qwen MTP graphの未使用attention/O/MLPを省略 | head構造と必要stateに依存するadapter処理として保持 |
| BF16 row concatenation | rows/columns/contiguous/alias/同一session、同期完了 | model非依存の既存ExecutionSession/HIP演算を維持 |
| GPU support/p-q検証・小さいdecisionのreadback | GPU演算は共通。Qwen側parserは固定語彙数・Qwen型 | 共通`speculative_device`へ抽出し、vocabularyを引数化。Qwenの実経路を接続済み、model stateの失敗時cancelを維持 |
| MTP/external/ngramの提案・採否・公開 | 共通契約とmodel別制御が併存 | methodとprovider identityを分離。Qwen/Gemmaの通常検証を共通採否処理へ接続し、固定GPU経路のqueue・確定数・proposal RNG計算も共用 |

## 下書き検証

- `speculative_device`の2test成功。語彙数31/32/33/151936/262144の境界、draft幅1/2/3/7/8、全採用・先頭/途中棄却、不正幅0/9、NaN、accepted prefix不一致を検査した。hostのdecision検査でありGPU数値検証ではない。
- 既存coreの`fixed_k20`選択4test成功。旧Qwen公開型は共通decision型への互換aliasとして残した。device上のsupportやnative p/q演算式は変更していない。

- frontendの関連テストは101成功・1 ignored。通常MTPの共通検証、部分採用、確定数、provider識別、固定samplingのRNG計算を検査した。GPU p/qやモデル状態の実機検証の代わりにはしない。
- Gemma候補選択の16test成功。Qwenと異なるK=3840/N=15360、M=1/2/3/4/64/65、非共有入力、M=5/63と不整列Kを検査した。backend非対応時の検査は下記の最終mockと実機比較を参照する。
- 共通graphの編集中に実行した`shared_activation`検査は、削除済み旧identity helperを参照するgraph testによりcompile失敗した。実装失敗をPASS扱いにせず、graph側の更新後に再実行して成功した。

## 実機比較の基準

- Phase83.5のgfx1030 CLI（SHA-256 `28230effdf30880f2bb7676c2eecf5739f5b624a83335c6903cb7cf410467a51`）で、変更前のQwen3.5-4B BF16/MXFP8 KV・固定samplingの26 input/17 output、およびMinistral 3 3B BF16/FP16 KV・greedyの549 input/17 outputを取得した。いずれもHIP実行・fallbackなし。
- Ministralの最初の2回はCLIのgreedy専用条件（sampling指定およびseed指定）によりGPU投入前に拒否された。現行CLI実装に合わせてgreedy・seed指定なしへ訂正した。この拒否は数値検証の失敗や成功として数えない。
- 新候補gfx1030 CLI（SHA-256 `db7861b7bf23725b3295aafa95e0dbc481d1b7157d3c2c45f12871b195a22c32`）でMinistralの同一入力・17 outputが成功。input/generated/visible/decode-input token列、本文、finish/stop、usage、samplingが変更前と一致した。監査のkernel/submission数は394→343で、51組の融合と一致する。これはCLI結果と報告対象transitionの監査比較であり、全生成の総dispatch数や速度改善率へ読み替えない。
- 生成token列のcanonical JSON SHA-256は両者とも`4badd40cbf18526a2aae4d075e410e7ea5724e089e26410f4665b80bf5db6eb7`。比較記録は`.local-artifacts/phase83-common/ministral-cli-comparison.json`。
- Qwen4Bも変更前とtoken列・本文・finish/stop・usage・sampling・cleanupが一致し、kernel数は5940→5396だった。生成token列SHA-256は`1048aabdbf5473d8bc235ecbca64d5efd5d84447ed48548957362c62e3f2e6f7`。Gemmaと公開MTP経路は以下のcandidate-r2で比較した。

## 残す制限の意味

- Qwenのprojection loweringからモデルfingerprint・固定pack数・固定layer集合条件を外し、選ばれたmemberのrole/binding・接続・viewとrecipe digestの所有権を検査する。NVFP4契約のK/Nはweight viewから生成する。異なるfingerprintの1 pack部分集合、動的shape、recipe不一致・重複member・古い接続の拒否を検査し、graph関連25test成功・1 ignored。Qwen artifact専用のplan producerにはverified scale metadataの出所と成果物構成の検証が残る。別モデルの通常入口への実接続はGemma側のproducerで行い、Qwenのproducer自体が任意artifactに対応したとは説明しない。
- NVFP4共有演算のK/Nは既存ABIで可変であり、Qwenの5120/17408は全モデル共通の必須条件ではない。共有する入力・view、W4A4形式、正かつ有限で同一のinput-global scale、Kのblock16整列と既存行数・target条件は維持する。
- FP8 GDN packのK=5120、N=10240/6144、およびM3 checkpointのstate plane/layoutはnative実装の前提である。モデル名を外すだけで他形状に適用しない。
- Gemmaのgraphは構造を記述し、semantic descriptorとtensor IDは後のlayout構築で決まるため、今回のAdd/RMSNorm共通graph passを直接適用できない。Gemmaのprojection共有はdescriptorが揃うlayout段階で接続する。これらは別の最適化である。
- 固定K20のGPU p/q処理を共通化しても、GemmaのK64 profileや未実装の外部draft executorへ自動的に接続したことにはならない。モデルのhead/hidden/KV/GDN状態処理とprofile対応はadapterに残る。

## 統合確認・修正

- 共通residual passの初回focused reviewではblockerなしと報告されたが、rootの境界位置に関する指摘をfocused再確認した結果、Ministral最終残差のStatePublication境界を融合が越えるcorrectness defectを確認した。Add自身に境界がある候補と、非隣接Normの境界を前へ移す候補をskipするよう修正し、境界がAdd／中間node／Normにある場合と隣接Normの保持をtestへ追加した。Ministralは51組を融合し、最終Add→境界→Normの順序は元のまま保つ。初回reviewの結論を最終証拠として流用しない。
- Qwen4Bの融合数は32から64へ増えた。attention残差32個に加え、inter-layer MLP残差31個とfinal normが対象となる。実行テストの旧32個という期待値を64個へ訂正した。数を合わせるための適用制限は追加しない。
- Ministral graph 11test・execution 5testが成功。Qwenの融合非対応backendで通常requestが実行できる追加testも成功した。
- NVFP4 shared activationの6testが成功。異なるK/N、行数・role境界、support/prepare拒否の分解fallback、submit開始後にfallbackしないことを検査した。

- source編集確定後の`cargo test -p sllm-core --lib --no-fail-fast`は608成功・22 ignored。ログは`.local-artifacts/phase83-common/core-test-r2.log`。編集中のGemma mockを読んだ直前のcompile失敗は成功結果として使わず、確定後の全core結果へ置き換える。
- Gemma fallbackのfocused mockはsupport拒否・prepare拒否で分解要求を返し、submit失敗では分解要求を返さないことを確認した。最終mockは単一packのsubmit helperを検査しており、モデル全体での2本のMatmul実行・監査値の確認は実機比較と合わせて評価する。
- Clippyが新規`committed_input_rows`のconst Vec::lenにMSRV 1.85との不一致を報告したため、通常のfnへ訂正した。動作・数値式の変更はない。

- Gemma実機候補r1はHIP wrapperの旧K=5120固定チェックで`InvalidMatmulDescriptor`（status 280）となり失敗した。nativeとcoreのdynamic shape対応だけでは通常経路が成立していないことを確認した。`crates/sllm-hip/src/qwen38_projection_pack.rs`のNVFP4検査を既存contractのK/Nへ合わせ、FP8 GDNの固定条件を保持した。hostのdynamic shape/行数境界2test成功、candidate-r2を再buildして実機比較を完了した。
- 失敗直後のsysfsは解放途中だったが、process終了後の再確認でV620 VRAMは17,215,488 bytes、GPU busy 0へ復帰した。失敗をGPU PASSやcleanup即時成功と扱わない。

## Gemma実機比較

- candidate-r2（gfx1030、`sllm-phase79-gemma-logits` SHA-256 `19814baa941bc4210d202101702ab8dd12c9f586c89472b859c95e2c48491548`）で、input長2/3/4/5/63/64/65と各1回のteacher-forced decodeを、共有既定と`SLLM_PREPARED_PROJECTION_SHARING=0`の別processで比較した。
- 全14位置の全logitが完全一致（max absolute difference 0）。これは同じNVFP4モデルの既存GPU演算との回帰比較であり、BF16 full-model品質同等性を証明したものではない。
- 既定側は各caseのdecodeで48 packを実行し、prefill packは0だった。nativeの`qwen38_projection_pack_nvfp4_adopted_variant`がM>1で選択済みsmall-M／compensated variantとshapeを要求するため、測定Gemmaのprefill候補はprepare Unsupportedから通常gate/upへ分解される。行数候補の拡張をprefillでの実採用や速度改善と説明しない。
- 両processとも実HIP・fallbackなし・非zero dispatch、runtimeのcurrent/retryable/durable cleanupは0。ここでfallbackなしとはCPU fallbackなしを指し、上記のGPU Matmulへの分解とは区別する。
- 比較記録は`.local-artifacts/phase83-common/gemma-logits-comparison-r2.json`、raw logitsのSHA-256は同記録へ保存した。

## 公開MTP経路・最終検査

- candidate-r2のserver SHA-256は`9a642c9e1cd123d1c1fce3576d91bb755462f4b1b5ba0460f1bc0a0fbb4ffa40`。V620 gfx1030でQwen3.8 NVFP4、MXFP8 E4 KV、固定T=1/P=.95/K20、MTP幅2を確認した。
- 通常JSON、SSE、3 content event後のcancel、同serverでのrecovery、8192 input/128 outputの5要求は全てHTTP 200。長文のshutdown auditはinput8192/output128、HIPのみ、CPU fallbackなし、MTP提案106・採用74・棄却32だった。旧速度目標の再評価やBF16品質判定は行っていない。
- server終了時のcurrent/request-state/workspace/retryable/durableは全て0。証拠は`.local-artifacts/phase83/common-v620-api-r2/execution.json`。一時serverを終了し、既存R9700 serviceは変更していない。
- core/frontendとHIP crateのaffected Clippyは`-D warnings`で成功。全workspaceのrustfmtとMarkdown local linksも確認した。最終公開commitのCIはpush後に確認する。
- 次Phase84の比較基準を今回の共通化後candidateへ更新し、共通化による差をMTP量子化の差へ混ぜない。MTP方式の共通制御とmodel adapterの境界をPhase84計画にも明記した。

- MTP付きCLIもcandidate-r2で成功。本文・finish reason・usageがPhase83.5 closeoutと一致した。このCLIはtoken ID列を返さないため、token完全一致を主張しない。本文SHA-256は両者とも`5419469336797d0d5453a923647a56b9d5254bb7b76620ea0e701f3dc05e7e66`。提案12・採用10・棄却2、実HIP・CPU fallbackなしだった。
- 最終candidate-r2のCLI SHA-256は`dda6262c0eeec01e443272a0a03d13e324feffbf33225343e7b88ae8b004e2e5`。build前後の481 source hashは一致し、実機確認後にもsource driftなしを確認した。r1からr2の変更はGemmaファイルの整形とNVFP4 pack用HIP wrapperのshape検査のみで、先に確認したBF16 Qwen4B／Ministral経路の演算と制御は変えていない。
- 境界修正とHIP wrapper修正のfocused再reviewはいずれも追加blockerなし。main-plan・runtime・次Phase84と同期し、同じ目的の変更を1 commitへまとめる。公開commitのhost・basic H3・public-runtime H3は[GitHub Actions](https://github.com/jyohukuchan/sLLM/actions)の該当commitで確認する。

作業計画: [Phase83・83.5共通化](../../../../plans/archive/2026/09/1-10/phase83-common-speculation.md)。
