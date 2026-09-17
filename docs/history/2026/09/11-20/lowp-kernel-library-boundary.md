# 低精度カーネルのライブラリ境界化

## 状態と範囲

完了。2026-09-17のユーザー決定に従い、Phase 87より先に`native/lowp`を分離した。
移動前の基準は`ba640680d4bb82bc9ed2ae7a0658c6a2618c6232`。今回の変更は未コミットのlocal draftとして検証する。
kernelの数値・監査ID・採用範囲を変えないN0の構造変更であり、MXFP4 W4A8の実装は含めない。

## 実装

- codec、typed view、provider契約、低精度kernelとlauncher、variant選択を`native/lowp`へ移した。
  gfx1030 software FP8 outerも含む。BF16、native FP8、BLAS、KV/attention本体は`native/hip`に残した。
- 公開C APIに版・target・形式契約、plan、activation量子化、launchを設けた。
  workspace・staging・streamはcaller所有とし、library内のallocation・同期・CPU fallbackを追加しない。
- 通常matmulとprojection packをC APIへ接続した。packは量子化を1回共有し、2つのplanを起動する。
  既存MXFP4 W4A4は内部APIを使い、公開APIは新しいW4A8 v1契約だけを定義して未対応として拒否する。
- CMake単体build、親からのリンク、Rust build入力、CI inventory・依存境界検査を更新した。
  host CIの既存H0行へ検査を組み込み、単体HIP CI行は増やしていない。
- 来歴は既存のllama.cpp由来コードの新pathへ対応付けた。今回の外部コード新規importはない。

## 検証中に修正した問題

1. 移動後のprojection packで旧adoption gateを保持し、M64 FORCE_BASELINE等の従来の拒否条件を復元した。
2. selectorの`supported/enabled/adopted`は最適化採用の監査値であり、端数Kを実行できるbaselineの
   運用上の可否とは異なる。C APIの実行可能性と監査値を分離し、NVFP4 `M=1,N=17,K=15`も旧動作を維持した。
3. CMakeのarchive出力先変更前のファイルが親build directoryへ残り、Rustが古い`libsllm_lowp*.a`を
   優先リンクしていた。新しい`lowp/`を先に検索するよう修正した。同一入力に対して旧archiveは801、
   現archiveは0を返すことを直接確認し、修正後の実runtime NVFP4 oracleもPASSした。
4. 既存benchmarkのClippy指摘には理由付きの局所allow、型alias、同義の文字区切り指定を適用した。
   演算経路は変更していない。整形はrepositoryのclang-format 18を使用した。
5. H3の直接compile/link行はCMakeと独立してsourceを列挙するため、lowpのpublic include、host plan/launch、
   BF16側に残るmatmulと移動後のlowp kernelをすべて明示するよう修正した。
   strict wrapperはclean/pinned-container公開CI用なので、dirty localでは同じmatrixのargvを直接実行し、
   公開CIのimmutable identityを名乗らずlocal-draftのcompile-only証拠として記録する。
6. build.rsのlowp入力登録はliteral配列loopへまとめたため、既存のH3入力解析testもその構文へ対応した。
   登録漏れ・重複・別pathへの置換を拒否する負例を維持し、runtime/build条件は追加変更していない。

## 現時点の証拠

- 境界表: 4,700ケース。旧sourceから生成し、provider・variant・tile・activation footprintを照合した。
- host: 既存公開runtime/C ABIを含む7テスト、単体のC API/fixture各2テスト、C11 callerのcompile/link/実行がPASS。
- kernel object: 両targetとも移動前115 symbolと移動後のBF16/lowp objectのunionが一致し、欠落・追加なし。
  生の命令列digestはgfx1030で75/115、gfx1201で72/115一致。PC-relative addressと末尾paddingを
  除いた命令列は両targetの115/115で一致した。数値同等性は別途GPU出力で照合する。
- 移動前の両GPUモデル測定は取得済み。V620の最初のrunnerはquality workerの空stdoutをJSONとして
  読んで停止したが、workerはexit 0でcaptureを保存していた。captureを直接検証し、残り3行を
  `baseline-tail-gfx1030`で実行した。失敗記録を成功へ書き換えず、2つの記録を対応付ける。
- H0は628件すべてPASS。V620の移動後GPU測定も完了し、operator出力、Qwen3.8の128 tokenと
  target/draft logits、Qwen3.5のMXFP8/MXFP6 logitsと短いtoken列が移動前と一致した。
- R9700も同じ比較がPASSし、両GPUともHIP-only、fallback 0、cleanup 0を確認した。
  Qwen3.5のlogitsは既存quality fixtureの10ケース（prefill/decode計20行）とrepeatを照合し、
  同fixtureのtoken digestも一致した。通常CLIは別の固定17入力／7出力のtoken列を照合した。
- 既存H3のcompile/link argvと静的検査がPASS。public ABI 117 symbolとlowp C API 8 symbol、
  exact target bundle、Code Object V6/wave32を確認した。RMSNorm専用H3も両targetの4 argvを実行した。


## 性能（8,192入力／128出力、1 warmup＋3 measured）

単位はtok/s。token数、固定sampling、BF16 MTP companion、MXFP8 E4 KVを前後で揃えた。

| GPU | 区間 | 移動前中央値 | 移動後中央値 | 差 | 移動前範囲 | 移動後範囲 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| gfx1030 | prefill | 217.042 | 216.077 | -0.445% | 216.454–218.830 | 215.488–218.271 |
| gfx1030 | decode | 25.478 | 25.361 | -0.459% | 25.402–25.479 | 25.281–25.463 |
| gfx1201 | prefill | 542.247 | 542.512 | +0.049% | 540.972–545.729 | 540.387–545.655 |
| gfx1201 | decode | 35.007 | 35.133 | +0.360% | 34.620–35.135 | 34.525–35.203 |

全4区間で前後の測定範囲が重なる。中央値の差は最大約0.46%で、今回の反復幅を超える退行は観測しなかった。
3反復の観測であり、一般的な性能同等性の保証へ拡張しない。kernel命令列の非配置部分が一致すること、
固定token列・logitsが一致することも合わせてN0の構造変更として受け入れた。
R9700のservice状態・設定hash、両GPUのperformance levelは実行前後で一致した。

## 検証の範囲

hostはH0 628件とnative CMake 7件、独立C11 caller、両単体buildのhost契約を確認した。
HIP compile-onlyは既存H3のmatrix argvとELF/device inspectionをlocal draftとして確認した。
公開workflowを起動・公開したという意味ではなく、commit/pushは今回行っていない。
H0後のH3 command/input/schema修正は公開H3の75テスト・692サブテスト（除外0）と、RMSNorm H3の対象テスト・validatorで再確認した。

## 証拠の保存先

生成binary、raw report、disassemblyはGitに加えず、`.local-artifacts/lowp-boundary/`へ保存する。
結果・hash・比較値は[集約記録](../../../../../ci/matrix/lowp-kernel-library-boundary-v1.json)へ保存した。


## 追記: 境界の残件修正（2026-09-17）

完了確認で見つかった3件を、ユーザー指示により修正した。数値・選択・監査IDは変えていない。

1. **lowp側に残っていたsLLM固有のIDと名前を移した。** BF16、hipBLAS、FP8 nativeのkernel ID文字列と
   `KernelVariant`の値（1、2、3、4、5、7、12、13、16、17、91）をsLLMの`matmul_kernel_internal.hpp`の
   `HostKernelVariant`へ移した。lowpの`KernelVariant`は低精度providerの値だけを持ち、値1を
   `Unspecialized`（低精度の特化経路なし）とし、他の値は統合側の予約値として再利用しない。
   lowpの`logical_kernel_id`等は`lowp_`付きの名前にして未知のIDに`nullptr`を返し、sLLM側の同名関数が
   自分のIDを解決してから委譲する。lowpのFP8 software selectorに残っていた到達不能なnative FP8分岐は削除した。
   `switch`で統合側の値を扱う1か所（projection packのgrid計算）は、`switch`前の`if`へ移した。
2. **sLLMがlowpの内部ヘッダを相対パスで読む形をやめた。** `native/lowp/src/internal/`を
   `native/lowp/include/lowp/detail/`（同一repository内向けのC++連携ヘッダ。安定性は約束しない）へ移し、
   sLLMの本番ソースは`<lowp/detail/...>`でincludeする。CMakeのG1 flagにもlowpのinclude pathを加えた。
   過去の研究用probe（`native/hip/tests/phase78_*`）は記録済みのbuild手順を壊さないようfile相対のままにし、
   kernel本体の`.inc`を読む3件も変更していない。
3. **古い来歴コメントを削除した。** `native/hip/src/matmul_kernel.hip.cpp`冒頭のNVFP4 byte permutation由来の記述を消した。
   該当コードはlowpへ移っており、`native/lowp/src/lowp_kernel.hip.cpp`とTHIRD_PARTY_NOTICESの記録は既に移動後を指している。

CI資産（H3/RMSNorm H3/G2/P0の入力台帳、artifact schema、path-to-suite、build.rs）のpathとhashを更新した。
H3の直接compile入力はpath名でsortする契約のため、移動後の順序へ並べ直した。

### 検証

- 対応表: variant値0〜127×3 target×384 shapeの`logical_kernel_id`、`device_symbol`、target別symbol、
  `grid_size_x`、`workgroup_size_x`（147,968行）が変更前と完全一致した。
- host: 公開runtime/C ABIを含む7テスト（4,700ケースの選択表を含む）、既存selector単体テスト2件、
  lowp include境界検査、clang-format 18、rustfmtがPASS。
  `phase83_5_nvfp4_small_m_vgpr_reuse_selector_test`は境界化前のHEADでも同じくFAILする既存の問題で、今回は変更していない。
- device code: 両targetで本番buildを作り直し、BF16側のdevice objectはbyte一致、lowp側は105関数の命令列が
  すべて一致した（device ELFの差はpath等のmetadataだけ）。
- GPU（両GPU、HIP-only、fallbackなし）: 演算子evidence（MXFP、NVFP4、内部MXFP4 W4A4）、Qwen3.8の128 tokenと
  target/draft logits、Qwen3.5 MXFP8/MXFP6の品質captureと短い生成が移動前の基準と一致した。
  性能は同条件でV620 prefill/decode 217.65/25.39、R9700 541.96/35.06 tok/sで、前回の範囲内だった。
  R9700 serviceは実行前から停止しており、状態・設定hashとperformance levelは前後で変わらない。
- 既存H3の直接compile/link argvを両targetで実行しPASS。公開workflowは起動していない。

raw reportと比較結果は`.local-artifacts/lowp-boundary/fixup-*`と`.local-artifacts/lowp-fixup/`に保存した。

[完了計画](../../../../plans/archive/2026/09/11-20/lowp-kernel-library-boundary.md)
