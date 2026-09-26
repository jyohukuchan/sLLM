# 未計画の課題一覧（backlog）

認識済みの最適化余地やその他の課題のうち、**対応する具体的な計画がまだないもの**をまとめる。
計画（`docs/plans/active`の計画、またはPhaseの段階・作業単位）へ組み込んだ項目はここから外し、移した先へのリンクを残す。
この一覧は採否の条件やhard gateではない。着手するときは新しいprofileで上限と受入条件を決め、
[main-planの変更の採否ルール](main-plan.md#変更の採否ルール2026-09-24ユーザー決定)で判断する。

2026-08-18〜09-17の棚卸し（実行時dispatch、attention・KV、GDN、Dense BF16、MoE、sampling・serviceなど）は
[整理前のmain-plan](../history/2026/09/11-20/main-plan-before-reorganization-2026-09-17.md)の「性能最適化の残課題」節にあり、
この一覧では繰り返さない。

各項目には、内容、見込みや根拠、次の一手、出典を書く。見込みは記録時点の概算である。

## 性能

| ID | 内容 | 見込み・根拠 | 次の一手 | 出典 |
| --- | --- | --- | --- | --- |
| P1 | MXFP6のM=1 kernel `sllm_mxfp6_w6a6_m1_col2_v1` が読み出し帯域の2〜3割（95〜161 GB/s）しか出ていない。MTP companionのMXFP6 sidecarはWU-3Sで廃止したが、MXFP6本体モデルのdecodeでも同じkernelを使う | BF16 kernelの1.1〜1.7倍の時間。E3M2の展開と積和が律速と推定 | MXFP6本体モデルを扱うときに、展開算術を共有helperへ置いて最適化する | [段階3の原因調査](../history/2026/09/21-30/phase87-stage3.md) |
| P2 | NVFP4 MTP companionの小さな無駄: k/vの行列積（N1024）が143／203 GB/sで1 stepに3回起動。MTP層のRMSNormがNVFP4出力に未対応で、量子化kernelが1 stepに6〜7個別nodeで走る | 合計で1 blockあたり0.2〜0.3 ms（0.5%未満） | NVFP4 companionの速度を再検討する際に、段階7と同じproducer融合とk/vのpack化を検討する | 同上 |
| P3 | 段階7 C1の残り: `linear_attention_state`→GDN outの48 node（head別blockで全headのamaxを求められない）、`final_rmsnorm`→lm_headの1 node（最終行aliasでscale planeを参照できない）、LinearAttention層の`input_rmsnorm`の48 node（BF16のb/aも同じ出力を読むため2本目の出力が要る） | node削減。GDN outはkernel設計の変更が必要 | GDN kernelの再設計や2出力bindingを行うときに合わせて検討する | [段階7 C1](../history/2026/09/21-30/phase87-stage7-c1.md) |
| P4 | MXFP8 KV前処理のproducer融合（段階7 C3） | 上限V620 0.159／R9700 0.119 ms/token。新kernelを伴う「重い変更」のためend-to-end 1%に届かない見込み | 軽い変更で実現できる形が見つかった場合だけ再検討する | [Phase 87 段階7の上限](archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階7の着手時上限と候補2026-09-22) |
| P5 | R9700のverify lm_head（M=3、hipBLASLt 2.347 ms）がM=1のdot4（2.013 ms）より遅い | 確定tokenあたり約0.13 ms（約0.5%） | M=3のdot4変種を試す | [縮小語彙headの探索](../history/2026/09/21-30/phase87-mtp-proposal-shortlist.md) |
| P7 | MTP draft headの低bit化＋厳密rerank（vllm-mxfp4のP1） | 段階9の縮小語彙headと同じ費用が対象 | 段階3の結果とdraft headの寄与をprofileしてから判断する | [vllm-mxfp4の分析](../history/2026/09/21-30/vllm-mxfp4-optimization-analysis.md) |
| P9 | V620のM=1 NVFP4が、直前のGQA型KV読み出しで約10%遅くなる近傍依存（D系統、打ち切り済み） | V620約1.9 ms/token | 再開条件（MALL／DRAMの内訳を取れる計測手段、または実attentionのdata flowで再現するharness）が満たされたら | [段階6](../history/2026/09/21-30/phase87-stage6.md)、[WU-D3](../history/2026/09/11-20/phase87-wu-d3.md) |
| P10 | graph内のkernel間のdispatch固定費（1 nodeあたり約5 µs、段階7後も約970 node/token） | MTPなしでV620約6.1／R9700約5.2 ms/token（段階6後） | attention前処理やGDNの小kernelなど、node数をさらに減らす融合候補を洗い出す | [Phase 87 今後の順序](archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#今後の順序2026-09-22整理) |
| P11 | 段階3のprofileで、R9700のMTPありのwall時間がkernel時間より1 blockあたり約20 ms長かった（V620では差がほぼない） | profiler由来かどうか未確認。段階6のprofileではgraph内の隙間は約2.4 ms/token | profilerなしのmarker計測、または`ROCPROFILER_QUEUE_INTERPOSITION=0`で取り直して確かめる | `.local-artifacts/phase87/stage3-speed-cause/` |

P8（GDNのconv＋recurrentの追加融合）とP13（gate/upとGDN qkv/zのdual-output bundle）は、2026-09-26に
[Phase 88](active/2026/09/21-30/phase88-qwen38-llama-kernel-parity.md)の段階5と段階3へ移した。

P12は[WU-3P](archive/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)へ移して2026-09-25に完了した。
現行`row8_tiled256`からexact MTP prefix形状だけ既存DP4A／WMMAへ変更し、MTP prefix時間を
V620 7.527→0.287秒、R9700 6.018→0.201秒、通常prefillを43.515→36.198秒／
20.746→14.916秒へ短縮した（[記録](../history/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)）。
2026-09-24のNVFP4既定化だけに適用されたprefill／TTFT例外と共通ルール不適用は、WU-3Pには転用していない。

## 保守・品質

| ID | 内容 | 次の一手 | 出典 |
| --- | --- | --- | --- |
| M1 | 段階7の融合kernelで、block単位の量子化処理（amax、block scale）がrmsnormとelementwiseのkernelに重複している。要素単位の符号化はlowpの共有helperを使っている | AGENTS.mdの方針どおり、共有`__device__` helperへまとめる | [段階7 C1](../history/2026/09/21-30/phase87-stage7-c1.md) |
| M2 | 時間に敏感なhost test（短い打ち切り時間や、processの起動を待つもの）が、local fast modeの負荷下で失敗しうる。`test_rmsnorm_p0_builder.py`の1件は補強済み | fast modeだけで失敗したら`--serial`で確かめ、該当testの待ち時間を補強する | [testing.md](../development/testing.md#local-fast-mode-automatic) |
| M3 | h0のlocal fast modeで、MSRVの列の依存検証（`validate_rust_dependencies.py`）が、直前のMSRV checkとcargo環境が違うため再検査になり約48秒かかる | 両者のcargo環境を揃えてfingerprintを共有する | 同上 |
| M4 | 別セッションのクラウド比較作業（Cinference、SGLang、llama.cppのcloud build tool、関連文書、`THIRD_PARTY_NOTICES.md`の2行）が未レビュー・未commit | 内容を確認し、Phase 87とは別のcommitにする | 作業ツリー |
| M5 | Ministral 3の対応の削除（2026-09-25ユーザー提案。Mistral系の利用者が少なく、main-planの対応予定アーキテクチャにも入っていない）。Ministral専用はRust約1.2万行（`ministral3*.rs`、YaRN、frontend、evidence binary 2本）、専用のYaRN RoPE kernelと公開ABI（`SLLM_HIP_MINISTRAL3_YARN_*`）、CLI／server／model library／GGUF読み込みの分岐。他モデルからの利用はない | 1つの整理の作業単位として削除する。Phase 81の固定sampling GPUテストの題材をQwen3.5／Gemmaへ置き換え、head dim 128など他に使われなくなる経路も確認して整理する。公開ABIの番号とstatus codeは欠番として残し、Phase 60の履歴と証拠は残す。種類D／F（対応範囲の削除）として、他モデルの出力不変とCIを確認する | [Phase 60](../history/2026/08/21-31/phase60-ministral3-3b-production.md)、`crates/sllm-core/src/ministral3*.rs` |

## 環境・外部依存

| ID | 内容 | 現在の対処 | 出典 |
| --- | --- | --- | --- |
| E1 | Codexから起動するprocessのfd soft limitが1024。ROCm 7.14のVMM（`hipMemCreate`ごとにfdを消費）と組み合わさると、V620の長いcontextで確保に失敗しうる | 段階10（Paged KV本移行）でVMMを使わなくなる。それまではGPU runnerでsoft limitを上げることを検討する | [paged移行の検討記録](../history/2026/09/21-30/kv-paged-migration-decision.md) |
| E2 | HIP 7.14のsegmented graphでsignalがunderflowする | V620の計測runnerだけ`DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` | [software互換性](../compatibility/software.md) |
| E3 | gfx1201で、map完了後もGPUにmappingが反映されない（ROCm/rocm-systems #11693、未解決） | 段階10でVMMを使わなくなる | [paged移行の検討記録](../history/2026/09/21-30/kv-paged-migration-decision.md) |
| E4 | rocprofiler-sdk 1.3.2で、並列枝を持つgraphのprofileがcompletion signal待ちで止まる | profiling時だけ`ROCPROFILER_QUEUE_INTERPOSITION=0` | [段階6](../history/2026/09/21-30/phase87-stage6.md) |

## 利用者・再現性

| ID | 内容 | 次の一手 | 出典 |
| --- | --- | --- | --- |
| U1 | 段階9の縮小語彙draft headは、model側に生成物（`.sllm/mtp-draft-vocab-98304.u32`）を置いたときだけ有効になる。生成に使うSWE-chatはgated datasetで、再現にはHugging Faceでの同意が要る | 利用者が効果を得られる配布方法（生成手順の案内、生成物の配布の可否）を決める | [段階9](../history/2026/09/21-30/phase87-stage9.md) |
| U2 | MTP有効時の既定はNVFP4 companion sidecar（`<artifact_root>/.sllm/mtp-nvfp4-v1/`）で、無ければfallbackせずエラーになる（BF16 companionは実行時に選べない）。sidecarはCPUの変換toolで作れるが、NVFP4の活性値scale manifestは追跡対象外（`.local-artifacts/phase87/stage3/`）にしかなく、別環境ではGPUでの較正をやり直す必要がある | 較正済みscale（5値）とそのhashを生成手順と一緒に追跡するか、配布方法を決める | [段階3](../history/2026/09/21-30/phase87-stage3.md)、[companion文書](../development/mtp-companion-quantization.md) |

## 計測上の注意

| ID | 内容 | 次の一手 | 出典 |
| --- | --- | --- | --- |
| B1 | 2026-09-24にMTP companionの既定をNVFP4へ変えた。代表条件（coding8192／128、MTPあり、単一prompt）では、R9700で受理数が75/105→66/123に下がり、同一processのAB/BAでTPOTが約13.3%遅い（V620は約0.5%遅い）。26条件の自由生成の平均では受理の低下は見えない | 切替前後のMTPありの代表値を直接比べない。段階10・11などMTPありを測る作業では、切替後の値を基準に取り直し、可能なら複数promptやM4も併記する | [段階3](../history/2026/09/21-30/phase87-stage3.md) |

計画: [main-plan](main-plan.md)
