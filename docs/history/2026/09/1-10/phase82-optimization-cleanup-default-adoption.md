# Phase 82: 最適化の採否・削除理由の記録

> 2026-09-08: 実装・検証完了。公開HEADのCI確認はPhase完了時の公開手順に従う。

## 記録の役割

削除後も「何を試し、なぜ採用しなかったか」を調べられる履歴とする。
既存archiveを保持し、ここから候補ごとの比較条件・結果・元の実装と削除commitへ辿れるようにする。
実装で整理した24のmatmul候補は[候補別の試行・不採用理由](phase82-retired-matmul-candidates.md)へまとめた。
削除前source、各フラグ、比較結果、維持する共有処理、再検討条件から元の実装と測定履歴へ辿れる。
既定採用した範囲と保留候補は[最終selector一覧](phase82-default-adoption-scope.md)を参照する。
候補IDは詰め直さず、残存するlogical identityと過去の監査結果の対応を保つ。

## 削除範囲

matmul ID32、33、35、38、39、40、42、43、46、49、50、51、52、53、54、56、65、69、70、80、81、83、85、86の
候補固有実装、selector、workspace／launch、Graph／projection packの許可一覧、Rust evidence identityと専用probeを整理した。
採用済みID41／55／57等が使う共有templateやcodec、比較・切戻し用の基準経路は保持した。
ID72は末尾fallback修正前の退行値だけで削除せず、数値分類N2のopt-inとして残した。
ID85はstandaloneの約1.30倍だけで有望と判定せず、その後のproduction退行と既定解除の記録に従って削除した。
個々の数値、GPU／shape、未測定事項、再検討条件は候補別記録を正とする。

## 通常設定の実モデル比較

旧Phase81 r3とPhase82の既定設定を、同一GPU／モデル／固定入力replayで比較した。
Qwen3.8はFP16 KV、temperature 1.0、top_p 0.95、top_k 20、seed 123、1 warmup＋3 measured。
Gemma4 Dense 12B NVFP4は同じ固定samplingでtop_k 64、17入力／17出力（decode継続16回）、1 warmup＋3 measured。
表は測定中央値。反復値・MAD・token hash・正確なbinary／モデル固定情報は[測定記録](phase82-optimization-evidence.json)へ残す。
GPU間や異なる設定の結果を一つの速度比較へ混ぜず、合成変更を候補単独の寄与率にしない。

| モデル／GPU | 入力／出力 | 旧prefill ms | 新prefill ms | 旧decode tok/s | 新decode tok/s | 生成token列 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8／V620 PCI03 | 17／17 | 2,487.44 | 2,475.18 | 12.0148 | 16.0148 | 一致 |
| Qwen3.8／V620 PCI03 | 512／32 | 57,788.35 | 57,847.86 | 10.7489 | 14.1049 | 一致 |
| Qwen3.8／R9700 | 17／17 | 1,874.67 | 1,862.51 | 8.1012 | 19.8708 | 一致 |
| Qwen3.8／R9700 | 512／32 | 44,813.19 | 44,776.24 | 7.5759 | 17.6912 | 一致 |
| Gemma4 Dense／V620 PCI43 | 17／17 | 1,553.22 | 1,546.14 | 14.6764 | 15.6017 | 一致 |
| Gemma4 Dense／R9700 | 17／17 | 1,082.37 | 1,084.27 | 11.3384 | 18.1205 | 差あり、N1 |

Qwenのprefillは概ね不変で、decodeはV620で約31〜33%、R9700で約134〜145%改善した。
短文request setupはGraph準備等でV620の約9 msから約18 msへ増えたが、TTFTはほぼ不変、e2eは改善した。
Gemma V620のgreedy controlは16.3516→15.3098 tok/sと遅くなった一方、要求されたGPU固定profileは上表の改善だった。
host固定profileも11.2000→11.1302 tok/sであり、全sampling modeの一律改善とは主張しない。
Gemma R9700は全3 modeでtoken hashが異なる。NV67の同じ式・scale・丸めを保ったreduction treeへの変更を
Phase79と同じN1誤差非増加の根拠で採用するもので、token一致や品質同等性の主張ではない。
全実行はHIP、fallbackなし、cleanup後の残存割当／retryable cleanup／quarantine 0。

旧binaryで既存の手動最適化presetも比較した。V620 512／32ではprefill 1,587.75 ms、TPOT 68.96 ms、
R9700ではprefill 425.31 ms、TPOT 56.26 ms、V620 9,435／128ではprefill 29,906.80 ms、TPOT 64.18 msだった。
このpresetにはID62／ID72やP64／rocBLAS F32など今回保留した候補が混在する。通常設定とのtoken差もあり、
これらをPhase82通常設定の達成値や各candidateの個別効果へ読み替えない。

R9700 APIは最適化opt-inなしのFP16 KV設定で14／14 probeが成功した。
固定profileの省略／明示、拒否すべき設定、JSON制約、reasoning、SSE、生成content確認後の切断・回復とcleanupを検査した。
Qwen3.8専用APIはgfx1201だけを検証し、V620 API対応の追加を主張しない。測定用に停止した既存serviceは復帰した。

## 検証と修正した不具合

- core Qwen実行64件、共有prepared実行15件、HIP library150件、MXFP evidence16件、native public-runtime host fault testを確認した。
  既定／明示ON／OFF／未知値／対象外shape、dispatch identity、resource所有権とcleanupを含む。
- gfx1030／gfx1201のrelease buildは成功。最終整形後のr7はsource変更なしで完了し、r5で実行した4 binaryとtargetごとにSHA-256が一致する。
  API r4のserverとGemma runnerも同一性を確認して再利用し、probeは実際にリンクしたarchive hashを別に記録する。
- wave8量子化は両GPUで26条件×5設定を検証した。M1／3／17、K1／15／16／17／127／128／129／5120、
  有限BF16のzero／subnormal／極値／tie、global scaleの正値／0／負値を含み、独立oracle、書込みcanary、cleanupが成功した。
- public projection-pack／Graph probeは両GPUで成功し、ID84の既定選択を2つのproduction tupleで確認した。
  constant／alternatingの構成入力による独立integer oracle、baseline一致、repeat、cleanupの検証であり、全入力のモデル品質検査ではない。
- H3 public-runtime契約31件、Rustfmt、Clippy（対象3crate、all-targets、no-default-features、warnings禁止）、
  C++ format、Python compile／staticを確認した。削除symbolとsource inventoryを現行実装へ同期し、matrix revisionを13にした。
  host契約fixtureのcopyからローカルbuild／reference／cacheを除外し、検査対象sourceを保持したまま不要な大量copyを解消した。
- 初回のQwen GPU実行はGDN row32のnative既定変更にRust metadata validatorが追従しておらず失敗した。
  Rust側も同じ省略時・切戻し・shape条件へ修正し、host9件と修正後full-modelで確認した。失敗実行をGPU PASSへ数えない。
- 初回API probeはQwen3.8文章専用profileへ未対応のtool成功を要求して失敗した。既存capabilityに対応した
  明示`--tools unsupported`をprobeへ追加し、HTTP400／`unsupported_parameter`を確認した。従来のtool対応profileの成功検査は保持する。
- 量子化oracleの初回失敗は、極大有限BF16に対する浮動小数点の距離比較が全codeで同じ値になったことによるoracle側の誤判定だった。
  最大有限codebook端点448への飽和を先に扱い、端点内は独立nearest／tie-even比較へ修正した。修正後に両targetの全条件を再実行した。
- 初期buildのROCm論理root設定と削除後の共有K上限名の残存参照は修正済み。H3の負例テストが出力する意図的FAILと、suite全体のPASSを区別する。

## 保留した高速経路と性能の限界

NVFP4 prefill ID62はcorrected ID59と同じ実数式でも加算順が異なり、長いKで逐次和の誤差上限が増えるためN2として保留した。
K5120ではID62が319回のblock加算、ID59は概ね28段、K17408では1087対76となる。
`phase82_prefill_id62_oracle.py`の9 shapeのhost構造比較は、長いKのsampleで最大絶対差0.75を観測した。
これはGPU PASSでも品質不合格判定でもなく、N1による自動既定化をしない根拠の補助である。
ID64一般WMMA、ID72 staging、GQA6 P64／P128、rocBLAS F32も既定採用へ広げない。
P64 LDS表現のN0とpartition自体のN2を分け、Phase33の別providerに対する承認を流用しない。

今回の通常設定は高速prefillの手動presetを全て既定化したものではない。9,435入力／128出力のV620通常設定は
prefill 1,129,664.29 ms、TPOT 121.68 ms、decode 8.2182 tok/s、cleanup 0で終了した。
0 warmup＋1 measuredの長文動作確認であり、MAD 0を安定性能の証拠にはしない。
旧opt-in長文はN2候補を含むため、この差を同一経路の性能退行比較や採用候補単独の寄与率には使わない。
全モデル／全KV形式／MI300Xへの拡大、常駐serviceへの新binary再配置は実施していない。

## attention／GDNの最終判定

以下はPhase82の実装終了時点の判定である。再監査の結果、Phase33 C1のユーザー承認をGQA6 P64へ自動流用しない。C1の原典はwave8 providerのexact `gfx1030`／`gfx1201`、M=1、KV長1,024以上、head dim 256、4 encodingであり、GQA6 P64は6-wave・64-partitionの別providerと専用symbolを持つ。P64 partition／mergeはN2のまま明示opt-inで保留し、P64 FP16 LDS表現だけをN0として分離して記録する。rocBLAS F32もこの承認範囲に含めない。

### 削除した専用経路

Phase66 typed prefillとgfx1030 long-prefill-v2は、単に採用証拠が不足していたことを理由に削除したものではない。元のsource point（`0b2f0a45378375311d208effcdb32ad02dcf9349`）には、typed prefillが`native/hip/src/causal_attention_kernel_internal.hpp`／`causal_attention_kernel.hip.cpp`、selectorとworkspace／launchが`native/hip/src/causal_attention_runtime.inc`、Rust側の対応が`crates/sllm-hip/src/kv_state.rs`にあった。

- Phase66 typed prefill（`SLLM_CAUSAL_ATTENTION_PHASE66_TILED_PREFILL`、Q4K4／Q4K8／Q8K8）は、FP16／MXFP8 E4 KVのbit一致とoracleを確認した一方、primary同期行で既存経路より4.3〜27.3%遅く、Phase66履歴がproduction不採用と記録している。[Phase66履歴](phase66-gfx1201-reusable-low-precision-attention-transfer.md)の「完了結果」と[Phase79 selector inventory](phase79-selector-inventory.md)のscopeを根拠とする。
- gfx1030 long-prefill-v2は、operatorのM=1,024／4,096／10,001では52.96%／56.36%／58.60%短縮した。しかし`100,000/2` full-modelの単一warmupが約33分となり、current controlの1 warmup＋3 measured合計約20分を超えたため、Phase49で不採用・明示opt-in隔離となった。[Phase49完了履歴](../../08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を原典とする。

Phase82の削除範囲は、この棄却済み候補固有のkernel、selector、workspace／launch分岐、専用runner／probeに限る。Phase66／long-prefill-v2の履歴、archive、独立oracle、共通baselineは保持し、現行productionの`native/hip/src/causal_attention_kernel.hip.cpp`、`native/hip/src/causal_attention_runtime.inc`、`crates/sllm-hip/src/kv_state.rs`に残る共通baseline・採用済みproviderと、`sllm-full-attention-g1-evidence`共通probeの一般検査は削除理由へ含めない。再検討する場合は、元の棄却条件を変える対象GPU／shapeの数値・性能証拠を候補固有に取得する。

### GQA6 decode P64 — 明示opt-in／HOLD（partitionはN2、LDSはN0）

- 旧flagは`SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64`、selectorは`native/hip/src/causal_attention_runtime.inc`の`use_decode_gqa6_split_p64`と`crates/sllm-hip/src/kv_state.rs`の`decode_gqa6_split_p64_enabled`だった。Phase79棚卸しではgfx1030／gfx1201とも明示opt-inであり、Phase82で試した既定ONは撤回した。現在は`=1`だけを有効とし、未設定／`0`／未知値、および`SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`をbaselineへ戻す。
- 適用可能なshapeはquery_count=1、q_heads=24、KV heads=4、head_dim=256、`FP16_V1`、gfx1030のKV長`>=8192`またはgfx1201のKV長`>=4096`で、P128が明示要求された場合はP128を優先する。このshape契約はsupport範囲であり、既定採用の承認ではない。gfx1201についてもP64固有のユーザー承認は確認できない。
- P64 LDS変更の分類は`OUT-2026-09-05-P78-P64-LDS`のN0（FP16 bitsをLDSへ置き利用時にFP32へ変換、partition／QK／softmax／V／merge／BF16丸め順は不変）である。P64 partition／mergeそのものは別providerのN2であり、`OUT-2026-08-20-P33-C1`のwave8承認やN0のLDS証拠をP64採用の承認へ読み替えない。
- V620（AMD Radeon Pro V620、gfx1030、UUID `GPU-08b2ddcbd6e6b36c`、ROCm `/opt/rocm/core-7.14`、LLVM23、wave32、Code Object V6）の新しいP64/P128 probeは、L=8191/8192/8193/9435、seed=0/7919の8条件でFP64 stable-softmax oracle、repeat、nonfinite、cleanupを確認した。P64は表示した全出力でoracleのBF16丸めと一致し、P64 weighted medianは`9067.158 us`、P128は`2609.570 us`であった。P128には最大1 ULP差があるためP128保留は維持する。このgfx1030のbounded evidenceは、gfx1201を含む既定採用の承認へは拡張しない。証拠は[追跡済み検証要約](phase82-optimization-evidence.json)（ローカル原本: `.local-artifacts/phase82/runs/gfx1030-v620b-attention-r1/verification-report.md`）にあり、P64/P128 executable hashは`88dfe916e056894e15992c7b702d3a0f34aa27b86b949e1cd7fed5b2b71cd140`、P64/P128 object hashは`d7d4b6f6bc230fd2c787c453245da9ae538f24cabaae58e6325a391c57f7efd6`である。
- 最終source hashはcausal kernel `a5dc9a116fb8a862b8051c19b60495d5c302bf2192acf42465db2e97be696f45`、internal header `98334d8bae28970e5de53b2ff5c0e59a90d2a9713b76a2ca094b732e45e0cdf4`、native selector `a97f32d26c3b1738e3d2975d5a235c6a0920fec437cbc9c9f9039351181eed07`、Rust selector `20a7952430e96d677c072aeefc222929bd30deb1316d133b1b1fbcee8381aa8c`である。P64 kernel evidenceはselector差分とは独立するため、report内の旧selector hashを最終source identityとして再利用しない。

### GQA6 rocBLAS F32 — 既定ON変更を撤回、HOLD

- 旧flagは`SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32`および`SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F32`で、selectorは同じnative runtimeと`kv_state.rs`の`use_gqa6_rocblas_f32`／`use_gfx1201_gqa6_rocblas_f32`だった。Phase79記録はgfx1030／gfx1201の明示opt-inであり、既定adoptionではない。既存履歴からこのproviderについてのユーザー承認は確認できなかった。
- このproviderはQK/PVをrocBLAS F32へ切り替え、reduction／providerの順序を変更するためN2候補である。worst-case誤差の非増加証明はなく、直接kernel probeのbounded evidenceだけでは既定採用の根拠にならない。したがって未設定／`0`／未知値はbaseline、`=1`だけが明示opt-in、force-baselineは従来通りとし、既定ONにしたnative/Rust selector変更だけを撤回した。新しいworst-case gateは追加していない。
- V620のfresh production launcherはgfx1030、q_heads=24／KV heads=4／head_dim=256、FP16 KV、tail、context 4096/4097/8192/9435、seed 0/7919の8条件で、FP64 stable-softmax oracle、repeat、actual nonfinite 0、cleanupをPASSした。最大BF16 ULPは1、bit mismatchは0〜8だったが、これはN2候補の限定証拠であり、selectorの既定ONを承認する証拠ではない。executable hashは`4f7335550d0391a7ff7b71c11a8ec676df2b225b3caec947b9d154b497920e7e`、object hashは`a98a16c26c60229e6ef3f64aa918cba1d35311465e08f6f33f2914a0b76334fe`、最終native selector hashは上記`a97f32d2...`、Rust selector hashは上記`20a79524...`である。
- N2のままHOLDを解除するには、対象scopeの速度・誤差・出力影響を提示してユーザーが採否を決める。N0／N1へ再分類する場合はその解析根拠を別途記録する。それまでは共有rollback経路（qtile4／K32）を維持し、rocBLAS専用実装・ABI・履歴probeは保持する。

### gfx1030 GDN row32 LDS — N0として既定ON

- 旧flagは`SLLM_LINEAR_ATTENTION_GFX1030_ROW32_LDS`、selectorは`native/hip/src/linear_attention_runtime.inc`の`row32_lds_state_opt_in`である。未設定または`1`を有効、`0`／未知値、および`SLLM_GDN_FORCE_BASELINE=1`をrollbackとした。
- 適用範囲はgfx1030、token_count=1、qk_heads=16、value_heads=48、head_dim=128のQwen3.8 GDN decodeだけである。generic provider、他GPU、他shapeへの既定化は行わない。row32の式・出力・stateはbaselineと同じで、変更はLDS layout／launch routeに限定されるためN0とした。
- fresh production probeはcandidate／baseline双方のresource、determinism、独立oracle、finite、cleanupを確認し、baselineとcandidateの出力差0、max ULP0、state差0を得た。candidateは`1.47348983 ms`、baselineは`3.72327542 ms`、speedup`2.52684161`だった。証拠は[追跡済み検証要約](phase82-optimization-evidence.json)（ローカル原本: `.local-artifacts/phase82/runs/gfx1030-v620b-attention-r1/verification-report.md`）にあり、production linear object hashは`e557d89974272d8b67fa096dc7de4918dcae934aac1b155f14cd3342e576dab1`、GDN probe executable hashは`b4b8041253e9606fada37c68bded86d72a9469f26142caa2f31f5197b0caa593`、probe object hashは`5fd9ad70ddee5cbdc5117aeea1b049865ea8dd6d77a08c37bb640ed24f7713c6`、runtime source hashは`e095d376cf3ed5eca781f3d755e972132ceae8c3c882cbf46fd274c9fc1b2516`である。
- 他GPU／他shapeの採用は保留し、同じN0条件（式・state・境界契約不変）を確認できる新証拠がある場合だけ再検討する。共通baselineとforce-baseline rollbackは保持する。

### 最終確認範囲

上記GPU証拠はV620の既存承認targetと記載したshapeに限る。GPU実行をしていないtarget／shape、全モデルの性能、rocBLAS selectorの既定経路については主張しない。source point、source hash、binary hash、実行条件、oracle、repeat、finite、cleanupは同reportに対応付けた。

## commitと証拠の対応

実装・削除・検査コードのcommitは`fff63c574f1ffa7541efbef5c02e91b856db4a7d`
（`perf(hip): retire rejected candidates and adopt scoped defaults`）。
削除前は`0b2f0a45378375311d208effcdb32ad02dcf9349`であり、両commitの差分と候補別履歴から削除実装へ辿れる。
最終r7 source manifest SHA-256は`254acab626f75e9723e1ab7f5bcac8eadb0c149e7316418f928eb2b77c8dacec`。
ビルド時は未commitだった同一sourceを実装commitへ対応付け、後続の文書commitは実行sourceを変更しない。
最終公開HEADの`host-required`、通常H3、public-runtime H3の結果は、そのHEADに紐づくGitHub Actionsを正とし、
失敗があれば修正・再push・再確認までを公開手順に含める。ローカルGPU証拠をCPU CIのGPU PASSとして扱わない。

[対応計画](../../../../plans/archive/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md) ·
[メイン計画](../../../../plans/main-plan.md)
