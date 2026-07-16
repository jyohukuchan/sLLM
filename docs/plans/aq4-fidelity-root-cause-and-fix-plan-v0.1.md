# AQ4 fidelity root cause and fix plan v0.1

Status: Phase 1完了、Phase 2未着手。P2 fidelity calibrationはNo-Go凍結中、GPU実行は未承認。

## 前回の要点(初版時点)

- `docs/plans/aq4-production-prefill-decode-optimization-plan-v0.1.md` のP2（baseline凍結とprofile）はsource oracle/path oracle比較を要求しており、その過程でAQ4 pathとBF16 sourceのgreedy/logit不一致が判明した。
- weight provenance監査（`tools/audit_aq4_weight_provenance.py`）は量子化済み256テンソル全件でexport時metricsと実測がほぼ完全一致（最大差3e-8オーダー）することを確認した。ディスク上のAQ4重みとexport/packaging過程は健全と判断できる。
- GPU差分trace（attempt3 production run）は3行すべてで`decoder_layer:0`が最初の不一致点であることを示した。greedyは`source=220`に対し`path=41330`。source oracleとactive AQ4 pathの最終hidden/logitの相対L2は約0.545/0.615だった（この値はモデル全体の最終出力の相対L2であり、layer0単体の相対L2ではない点に注意）。
- layer0の重みfamily（QKV/Z/A/B）を個別にBF16 sourceと比較したmatvec単体の相対L2は2.0〜2.9%で、量子化誤差として妥当な範囲に収まる。単体の誤差はモデル最終出力で観測される大規模な不一致を説明できない。
- 07/15の正式fidelity calibrationは、独立holdout 19ケース全件でlogits相対L2が凍結policyのceiling 1.0を超過（最大1.2494246455220739）し、No-Goが確定した。holdoutの再実行は禁止されている。
- `docs/proposals/aq4-p2-correctness-fidelity-amendment-v0.1.md`は、BF16 source比較用のholdout契約とactive AQ4 path比較用のbehavioral oracle契約を分離する改定案を提示しているが、未承認・未bindのままである。
- 07/16はP3プロファイリング用harness（GPU実行権限・ロック・rocprof連携）の連続失敗（v7〜v19試行）に費やされ、`journal/2026/07/16/p3-paused-current-state-after-actual-v16.md`でユーザー指示により一時停止した。recovery codeはcommit済みだが未QA・未実行。

## 今回の変更点(この版での改定)

Phase 1（layer0 hybrid diagnosticの実装）を`gpt-5.6-terra`（`max`）へ委任し、CPU-onlyで完了した（commit `95357e89bdf3c6bb7afa7bc31a01c692163d74ec`、journal: [aq4-layer0-hybrid-diagnostic-v0.1.md](../../journal/2026/07/17/aq4-layer0-hybrid-diagnostic-v0.1.md)）。

結果は初版の想定を裏切るものだった。layer0の実forward式（Conv state→recurrent state→attention residual→post norm→SwiGLU MLP residual、recurrent state計算は実runtime関数`runtime_host_linear_attn_recurrent_f32`を直接利用）をCPU上でBF16 sourceと比較したところ、**layer0 output hiddenの相対L2は0.042451**であり、これは07/15の重みfamily単体誤差（2〜3%）と同じ桁に収まった。つまり、**CPU参照実装で再現したlayer0の数式そのものは、量子化誤差として妥当な範囲を超えていない**。

これは初版のWorking HypothesesであるH1〜H4（runtime合成ロジックの欠落、dequant/row-scale適用ミス、state初期化バグ、レイアウト/RoPEバグ）の**前提を大きく揺るがす**。これらは「layer0全体で相対L2が1.0を超える大きな不一致がある」ことを起点にしていたが、Phase 1はCPU参照実装ではその大きな不一致を再現できなかった。

この版では、Phase 1の実測結果を踏まえてWorking Hypothesesを全面的に見直し、Phase 2以降を「CPU参照実装では説明できない残差はどこから来るか」を切り分ける構成に再設計する。既存のPhase 2〜7は破棄せず、新しい調査結果に基づき優先順位を入れ替えたPhase 2〜8として再構成する。

## Goal

（変更なし）AQ4 runtime実行経路のうち、BF16 sourceとの最終出力不一致を引き起こしている演算・構成・実行系統を特定し、恒久的な修正を実装し、凍結済みP2 fidelity gate（`docs/specs/aq4-p2-calibration-evidence-binding-v0.1.md`基準）へ独立holdoutで合格させる。

## Success Criteria

解決済みとみなす条件:

1. ~~`one_at_a_time_hybrid`診断を実装し、layer0の実際のforward式を段階ごとにBF16 sourceと比較できる。~~ **Phase 1で完了。**
2. layer0単体のCPU参照実装では説明できない残差（0.042 vs 最終出力0.615相当）の出所を、次のいずれかに分類する: (a) 深さ方向の誤差蓄積、(b) GPU kernel実装固有の乖離、(c) 診断harnessと実productionの構成差、(d) その他未特定要因。
3. 分類した出所について、最初に許容誤差を超える具体的な演算・層・境界を特定する。「原因不明のまま最終出力だけが悪い」という粒度で終わらせない。
4. 修正はruntime hard-codeの局所パッチではなく、共有backend registry/kernel経路に対する再現可能な実装修正として表現する（manifest override不可、モデル名分岐の直書き不可）。ただし原因が量子化近似誤差の深さ方向蓄積だと判明した場合は、Non-Goalsに従いquantizer policy改定へ切り出す判断も許容する。
5. 修正後、CPU oracleで対象箇所のgreedy/hidden/logit不一致が解消することを、修正前と同一fixture・同一座標で確認する。
6. GPU差分traceを1回の承認済みwindowで再取得し、モデル最終出力の不一致が解消していることを確認する。
7. 独立holdout（tuning caseと非重複）による正式P2 fidelity calibrationで、凍結policy（logits相対L2 ceiling 1.0、その他5指標）に対し全ケース合格する。
8. 既存のAQ4 release gate（stop/soak/cancel/failure-restart等）と`same_artifact_all_m1` behavioral oracleに新規回帰がない。

## Non-Goals

（変更なし）

- AQ4量子化フォーマット（AQ4_0のgroup構成、codebook設計）そのものの再設計は行わない。原因が量子化近似誤差の理論限界だと判明した場合のみ、別計画（quantizer policy改定）へ切り出す。
- prefill/decode性能最適化（P3〜P7の候補kernel実装）は本計画の範囲外。`aq4-production-prefill-decode-optimization-plan-v0.1.md`側の担当のまま凍結する。
- 凍結済みP2 fidelity policyのceiling値を緩和・再交渉しない。閾値変更が必要と判断した場合は、根拠を`docs/proposals/aq4-p2-correctness-fidelity-amendment-v0.1.md`の手続きに従って別途提案する。
- 07/16に一時停止したP3プロファイリングharness（resident launcher/operator/finalizer cascade）の復旧作業は行わない。それは独立したrecovery track（`p3-paused-current-state-after-actual-v16.md`の再開手順）に委ねる。
- SQ8_0側の変更は行わない。共有runtime componentを触る場合はSQ8のrelease gateへの影響を都度確認するに留める。

## Working Hypotheses

### Phase 1後の再評価: H1〜H4はCPU参照実装レベルでは棄却、GPU kernel固有の可能性として保留

初版のH1（runtime合成ロジックの欠落）、H2（dequant/row-scale適用ミス）、H3（state初期化バグ）、H4（レイアウト/RoPEバグ）は、いずれも「layer0全体で相対L2が1.0を大きく超える」という前提のもとに立てた仮説だった。Phase 1はこの前提を裏切り、CPU参照実装（実runtime関数`runtime_host_linear_attn_recurrent_f32`を含む）でlayer0を再現すると相対L2は0.042に収まった。

ただし重要な注意点として、**production hot pathはGPU上のHIPRTCカーネル実装（`runtime/src/ullm_runtime_hiprtc_sources.inc`等）を使い、これはCPU参照実装（Phase 1の診断や既存oracleが使う`production standalone AQ4 matvec`）とは別のコードパスである。** Phase 1が検証したのはCPU参照実装だけであり、GPU kernel実装そのものは未検証のまま残っている。したがって、H1〜H4と同種のバグ（欠落した演算、scale適用ミス、state初期化ミス、レイアウト/RoPEのズレ）が、CPU参照実装ではなくGPU kernel実装の側に固有に存在する可能性は排除できていない。これをH5として独立に扱う。

### H5: GPU kernel実装（HIPRTC、CPU参照実装とは別コードパス）固有のバグ

CPU参照実装は量子化誤差相当（0.042）に収まる一方、GPU実測（production run）ではモデル最終出力の相対L2が0.615に達する。両者が別々の実装であることを踏まえると、この乖離を最も直接的に説明できるのはGPU kernel実装固有の欠落・誤適用である。現時点で最有力の仮説。

### H6: 診断harnessと実productionの構成差（session/chunk/position/warm state）

Phase 1の診断はlayer0を単独で、position 0からのcold state（ゼロ初期化recurrent/conv state）で計算している。実productionのresident driverは、paged KV、chunk境界、warm/継続sessionなど、より複雑な状態管理を経由してlayer0を実行している可能性がある。診断harnessの構成が実productionの構成と数式レベルではなく実行経路レベルで異なっていれば、Phase 1が「layer0の数式は正しい」と示しても、実productionのlayer0呼び出し自体が異なる入力・異なる境界条件で実行されている可能性が残る。

### H7: post-norm epsilon不一致（副次的候補）

Phase 1の観測で、AQ4 runtime側のpost-norm epsilonが`1e-5`、BF16 source config側が`1e-6`であることが判明した。attention residual（相対L2 0.033045）からpost norm（0.178438）への最初の明確なジャンプと一致するタイミングだが、この規模の差（1桁のepsilon差）だけでモデル最終出力の相対L2 0.615という規模を説明するには不十分である可能性が高い。安価に検証できるため、Phase 2と並行して切り分ける価値はあるが、単独の主要因とはみなさない。

### H8: 量子化誤差の深さ方向蓄積（単一バグではなく複利的増幅）

layer0単体の相対L2が0.042（量子化誤差相当）であっても、これが32層を通じて線形または非線形に蓄積すれば、最終的に0.615相当まで増幅される可能性がある。この場合、原因は特定の実装バグではなく、**AQ4量子化そのものの近似誤差が、hybrid 32層という深さでは許容範囲を超えて複利的に拡大する**という、より根本的な量子化フォーマットの限界を意味する。この仮説はH5/H6よりも検証コストが低く（CPU-onlyで層をchainするだけ）、かつ最も倹約的（Occamのカミソリ）な説明でもあるため、次の調査で最優先に検証する。

## Strategy

1. まずH8（深さ方向蓄積）をCPU-onlyの最小コストで検証する。Phase 1の診断harnessを複数層にchainし、相対L2の層ごとの成長曲線を測定する（Phase 2）。
2. H8だけでモデル最終出力の相対L2規模を説明できるかを判定する。
   - 説明できる場合: 量子化近似誤差の複利的蓄積が根本原因と確定し、層ごとの寄与を分析して「量子化フォーマットの理論的限界」として扱うか「一部の層/テンソルだけ精度を上げる」scoped fixで対応可能かを判断する（Phase 3a）。
   - 説明できない、または一部しか説明できない場合: 残差をH6（構成差、CPU-onlyの監査で検証可能）とH5（GPU kernel固有、GPU windowが必要）で切り分ける（Phase 3b、Phase 3c）。CPU-onlyで検証できるH6を先に試し、GPU windowの消費を最小化する。
3. H7（epsilon不一致）は、Phase 2の作業と並行してCPU-onlyの安価なcontrol比較で切り分ける。
4. 原因を分類し、対応するfix pathを選ぶ（Phase 5）。
5. CPU oracleで修正を検証してから、単発承認GPU windowで実機確認する（Phase 6-7）。
6. 正式P2 fidelity gateを独立holdoutで再実行し、Go判定を得てから性能最適化計画（P3以降）へハンドバックする（Phase 8）。

## Phase 1: Layer0 hybrid diagnosticの実装 — 完了

Status: **完了**（commit `95357e89bdf3c6bb7afa7bc31a01c692163d74ec`、実行: `gpt-5.6-terra`/`max`）

### 実施内容

- `crates/ullm-engine/src/bin/ullm-aq4-layer0-family-isolation.rs`の`one_at_a_time_hybrid`を実装。layer0の全forward（Conv state→recurrent state[実runtime関数を使用]→attention residual→post norm→SwiGLU MLP residual）をAQ4復号値で再現した。RoPEはlinear-attention層のため`not_applicable`として明示記録。
- BF16 sourceとの逐次比較器（`tools/compare-aq4-layer0-hybrid.py`）、実context fixture生成器（`tools/prepare-aq4-layer0-hybrid-input.py`）を追加。full tensorはメモリに保持せず、段階ごとに比較後破棄。
- 専用test 6件が合成fixtureでhybrid経路の正しさを検証。
- 成果物: `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-hybrid-diagnostic-v0.1/`

### 結果

| stage | relative L2 | max abs | cosine |
| --- | ---: | ---: | ---: |
| QKV dequant+row-scale | 0.025647 | 0.936214 | 0.999687 |
| Z dequant+row-scale | 0.029672 | 0.586211 | 0.999571 |
| recurrent state after | 0.038704 | 0.623254 | 0.999283 |
| attention residual | 0.033045 | 0.070312 | 0.999518 |
| post norm | 0.178438 | 0.166057 | 0.984072 |
| MLP up projection | 0.171038 | 0.100061 | 0.985317 |
| **layer0 output hidden** | **0.042451** | 0.069627 | 0.999107 |
| diagnostic LM-head readout (34 rows) | 0.026799 | 0.022404 | 0.999674 |

layer0 output全体の相対L2（0.042451）は量子化誤差として妥当な範囲に収まり、モデル最終出力で観測された相対L2 0.615を単独では説明しない。post norm境界での明確なジャンプ（0.033→0.178）とepsilon不一致（AQ4 `1e-5` vs source `1e-6`）を副次的候補として記録した（H7）。

### 元のExit Criteriaとの差分

初版のExit Criteriaは「layer0の不一致規模がGPU差分traceの規模と整合する」ことを想定していたが、実測はそれを満たさなかった。これは失敗ではなく、**「layer0の数式自体は疑わしくない」という有益な絞り込み結果**として扱う。この結果を受けて、以降のPhaseを全面的に再設計した。

## Phase 2: 深さ方向の誤差蓄積測定(H8) — 次の最優先タスク

並列性: CPU-onlyのため単独で実行可能。GPU/serviceは不要。

### Tasks

1. Phase 1のhybrid診断harnessを拡張し、単層ではなく連続する複数層（例: layer 0〜7、hybrid archのlinear-attention/self-attention両方を含む範囲を選ぶ）をchainしてBF16 sourceと逐次比較できるようにする。各層の出力hiddenを次層の入力へそのまま渡し、cold state（layer0開始時点のみゼロ初期化）を維持する。
2. 層ごとの相対L2/cosine/max absを記録し、成長曲線（layer index vs relative L2）を作成する。
3. 対象範囲にself-attention層を最低1層含める。hybrid archの層タイプ配列（linear-attention 24層+self-attention 8層の具体的な並び）を`crates/ullm-engine/`のモデル定義から確認し、self-attention層のindexを特定してから範囲を選ぶ。
4. 成長曲線の形状（線形/劣線形/超線形/途中で跳ねる）を分類する。全32層を通した推定曲線と、モデル最終出力の実測相対L2（0.615）を比較する。
5. 全hidden/stateをメモリに保持しない。層ごとの集計値と固定座標サンプルだけを保存する（OOM回避、AGENTS.md方針）。

### Deliverables

- 拡張したchaining診断ツール
- 層ごとの相対L2/cosine/max abs表、成長曲線
- `benchmarks/results/YYYY-MM-DD/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-multilayer-accumulation-v0.1/`

### Exit Criteria

- 測定した層数の範囲で、成長曲線の形状が確認できている。
- 全32層への外挿によって最終相対L2 0.615がどの程度説明可能かの定量的な見積もりが出ている（「説明できる」「部分的に説明できる」「説明できない」のいずれかに分類）。

## Phase 2b: post-norm epsilon control比較(H7、Phase 2と並行可)

### Tasks

1. Phase 1のharnessに、AQ4 runtime側のpost-norm epsilonをsource側の値（`1e-6`）へ一時的に揃えたcontrol modeを追加する。production runtimeの実際のepsilon値は変更しない（読み取り専用control、診断専用フラグの追加のみ）。
2. 同一fixtureでpost norm以降の相対L2の変化を測定する。
3. epsilon差だけで説明できる誤差量を定量化する。

### Exit Criteria

- epsilon差の寄与が定量化されている。Phase 2の結論に対して無視できる規模か、修正すべき規模かが判定できている。

## Phase 3: 残差の切り分け(Phase 2の結果に応じて分岐)

### Phase 3a: H8で説明できる場合 — 層別寄与分析

Tasks:

1. 成長曲線から、誤差増幅が特に大きい層またはテンソルfamilyを特定する。
2. 特定した層が特定のtensor family（linear-attention固有か、MLP固有か等）に偏っているかを確認する。
3. 「AQ4フォーマット全体の理論的限界」として扱うか、「特定層/テンソルだけ精度を上げるscoped fix」で対応可能かを判断する。後者の場合はNon-Goalsの量子化フォーマット再設計には該当しない（既存フォーマット内でのgroup/codebook適用範囲の調整に留める場合）。

Exit Criteria: 対応方針（quantizer policy改定へ切り出す／scoped fixで対応する）が確定している。

### Phase 3b: H6が疑わしい場合 — 診断harnessと実productionの構成差監査(CPU-only)

Tasks:

1. 実production driver（resident driver、session/chunk管理コード）が実際にlayer0を呼び出す経路を、Phase 1/2の診断harnessの構成と1行単位で突き合わせる。
2. position/chunk境界、warm state引き継ぎ、paged KVとの相互作用など、診断harnessが省略している要素を洗い出す。
3. 差異が見つかった場合、その差異を再現する追加のCPU-only診断を作成し、同じ層/同じfixtureで再測定する。

Exit Criteria: 構成差が実測誤差に寄与するかどうかが判定されている。寄与する場合は差異の具体的な箇所が特定されている。

### Phase 3c: H5が疑わしい場合 — AQ4 CPU参照 vs AQ4 GPU kernelの段階別差分(単発承認GPU window)

H8/H6で説明が尽きない残差がある場合にのみ実施する。BF16 sourceとの比較ではなく、**同じAQ4量子化のCPU参照実装とGPU kernel実装を直接比較**することで、量子化近似誤差の問題とGPU実装バグの問題を完全に分離する。

Tasks:

1. Phase 1のCPU stage report（QKV dequant、Z dequant、recurrent state、attention residual、post norm、MLP、layer output）と同じ境界で、GPU kernel実行時の中間値を取得できるよう、07/14の`ullm-aq4-differential-trace`を拡張する。
2. R9700排他lock・`HIP_VISIBLE_DEVICES`固定・active service非変更の専用windowで、同一fixture・同一座標のGPU中間traceを1回だけ取得する。再試行は行わない。
3. AQ4 CPU参照とAQ4 GPU kernelの段階別差分を比較する。BF16との比較ではないため、量子化誤差そのものは両者で理論上ほぼ一致するはずであり、有意な差があればGPU kernel実装固有のバグと確定できる。
4. 07/16に一時停止したP3 harnessとは別の実行系列として扱い、そのlock/serviceの再開手順とは独立に、通常のGPU承認手続きに従う。

Deliverables:

- AQ4 CPU vs AQ4 GPU 段階別差分report

Exit Criteria:

- 段階別差分が量子化演算の理論上の丸め誤差（ほぼゼロに近いはず）に収まるか、有意な乖離がある場合はその段階が特定されている。

## Phase 4: 原因分類とfix設計

Phase 2〜3の結果に応じて、次のいずれかのfix pathを選ぶ。

### Path A: runtime合成ロジックの実装漏れ（H1、GPU kernel固有として再検証された場合）

- Phase 3cでGPU kernel固有の欠落演算が見つかった場合、該当箇所をGPU kernel source（`runtime/src/ullm_runtime_hiprtc_sources.inc`等）に修正として実装する。

### Path B: dequant/row-scale適用のバグ（H2、GPU kernel固有として再検証された場合）

- GPU kernel内でのscale展開・group境界処理を、CPU参照実装（Phase 1で健全性確認済み）と突き合わせて修正する。

### Path C: recurrent/conv state初期化のバグ（H3、GPU kernel固有として再検証された場合）

- GPU kernel側のstate初期化コードをCPU参照実装と突き合わせて修正する。

### Path D: AQ4レイアウト/RoPEのバグ（H4、GPU kernel固有として再検証された場合）

- GPU kernel側のQKV分割・転置箇所のstride/axis解釈をCPU参照実装と突き合わせて修正する。

### Path E: 診断harnessが発見した実production構成差の修正（H6）

- Phase 3bで発見した構成差（chunk境界、warm state、position handling等）を、production driver側の実装修正として反映する。

### Path F: 量子化フォーマットの深さ方向蓄積が理論的限界（H8）

- Non-Goalsに従い、この計画を凍結し、quantizer policy改定を別計画として起案する。scoped fix（特定層/テンソルの精度向上）で対応可能と判断した場合は、Phase 3aの結論に基づき最小限の量子化policy変更を本計画の範囲内で実装する。

複数のPathにまたがる原因が判明した場合は、原因ごとに個別のfix commitとして分離し、各fixの寄与を独立に検証する。

### Exit Criteria

- 修正方針が1つのPath（または明示された組み合わせ）に確定している。
- 修正がruntime hard-codeの局所パッチではなく、共有経路またはquantizer policyへの再現可能な実装修正として設計されている。

## Phase 5: CPU側fix実装と回帰テスト

### Tasks

1. 選定したPathの修正を実装する。
2. Phase 1-3で使った診断harnessを同一fixtureで再実行し、対象箇所の誤差が解消することを確認する。
3. 既存のCPU oracle回帰test（`cpu_reference_executor.rs`関連、`ullm-quant`関連、07/05 hidden3994関連test）を実行し、新規回帰がないことを確認する。
4. 修正がlinear-attention/self-attention双方で妥当かを、Phase 2で使った複数層fixtureで再確認する。

### Deliverables

- fix commit（runtime/engine/quantizer policy側）
- 修正前後の段階別・層別誤差比較report
- 既存回帰testの実行結果

### Exit Criteria

- 対象箇所の誤差が、量子化誤差として妥当な範囲まで縮小している。
- 既存CPU回帰testに新規失敗がない。
- GPU/serviceは未使用。

## Phase 6: GPU差分trace再確認（単発承認window）

### Tasks

1. 07/14の`ullm-aq4-differential-trace`専用binaryを、修正後sourceで再ビルドする。
2. R9700排他lock・`HIP_VISIBLE_DEVICES`固定・active service非変更の専用windowで、修正前と同一fixture・同一座標のGPU差分traceを1回だけ取得する。再試行は行わない。
3. モデル最終出力の不一致が解消しているか、他層で新規不一致が出ていないかを確認する。
4. 07/16に一時停止したP3 harnessとは別の実行系列として扱う。

### Deliverables

- GPU差分trace（修正後）
- 修正前traceとのside-by-side比較report

### Exit Criteria

- モデル最終出力のBF16 sourceとの不一致が、量子化誤差として説明可能な範囲に収まっている。
- サービス状態・healthz・lock ownerに異常がない。

## Phase 7: 正式P2 fidelity gate再実行と計画へのハンドバック

### Tasks

1. `docs/specs/aq4-p2-calibration-evidence-binding-v0.1.md`の凍結policyに対し、独立holdout（tuning caseと非重複であることをhashで確認）で正式calibrationを再実行する。
2. 全ケースが相対L2 ceiling 1.0を含む5指標policyに合格することを確認する。
3. 既存のAQ4 release gate（stop/soak/cancel/failure-restart）と`same_artifact_all_m1` behavioral oracleに回帰がないことを確認する。
4. Go判定が得られたら、`aq4-production-prefill-decode-optimization-plan-v0.1.md`のP2へ結果をハンドバックし、P3以降の性能最適化候補選抜を再開可能にする。
5. `docs/proposals/aq4-p2-correctness-fidelity-amendment-v0.1.md`の承認要否を判断し、必要なら正式binding specへ反映する。

### Exit Criteria

- P2 fidelity gateがGoで確定している。
- 親計画（性能最適化計画）が本計画の完了を前提に再開できる状態になっている。

## Decision Tree

1. Phase 2の成長曲線が全32層外挿でモデル最終出力の相対L2 0.615をおおむね説明する場合:
   - Phase 3aへ進み、層別寄与を分析してPath F（quantizer policy改定またはscoped fix）を検討する。
2. Phase 2の成長曲線が0.615を説明しない、または部分的にしか説明しない場合:
   - 残差についてまずPhase 3b（構成差監査、CPU-only）を実施する。
   - 構成差が見つかり残差を説明する場合: Path Eで修正する。
   - 構成差で説明が尽きない場合: Phase 3c（GPU kernel段階別差分、単発GPU window）へ進む。
3. Phase 3cでGPU kernel固有の乖離が見つかった場合:
   - 乖離した段階の性質に応じてPath A〜Dのいずれかで修正する。
4. Phase 2bのepsilon control比較で無視できない寄与が確認された場合:
   - 他の原因と独立に、epsilon不一致の修正もPath A相当として実装する。
5. Phase 2〜3で単一原因に絞り込めず、複数要因が絡む場合:
   - 要因ごとに独立commitへ分割し、各commit適用後に層別誤差を再測定して寄与を切り分ける。
6. Phase 5のCPU修正後も誤差が量子化誤差の妥当範囲まで縮小しない場合:
   - 原因特定が誤っている可能性が高いとみなし、Phase 2〜3へ戻る。

## Risks

- **層別測定のOOM/実行時間肥大化**: Phase 2で層数を増やすとメモリ・実行時間が線形以上に増える可能性がある。
  - Mitigation: 全hidden/stateを保持せず、層ごとの集計値のみ保存する。まず4〜8層程度の小さい範囲から始め、必要に応じて拡張する。
- **GPU承認手続きのオーバーヘッドが07/16と同様に肥大化する**: 厳格なsingle-use契約下でのGPU実行は、07/15-07/16で v1→v19 規模の連続失敗を招いた実績がある。
  - Mitigation: Phase 2〜5をCPUだけで完結させ、GPUが必要なPhase 3c/6はH8/H6で説明が尽きた場合にのみ実施する。GPU window前にCPU側で候補を十分絞り込んでから申請する。
- **07/16に一時停止したP3 harness復旧作業との競合**: 同じGPU/lock/serviceを扱うため、誤って復旧作業のroot/lock/artifactへ干渉する危険がある。
  - Mitigation: Phase 3c/6は`p3-paused-current-state-after-actual-v16.md`のroot/lock/artifactを一切参照・変更せず、独立した新規GPU windowとして申請する。
- **H8が正しかった場合、修正手段が本計画のNon-Goalsに抵触する**: 量子化フォーマットの再設計が必要と判明した場合、本計画の範囲では対応できない。
  - Mitigation: Phase 3aでscoped fix（範囲を限定した精度改善）とquantizer policy改定（別計画）を明確に分岐させ、後者に該当する場合は速やかに切り出す。
- **修正がSQ8_0など共有runtime経路に影響する**: AQ4とSQ8_0が共有するmodel forwardコード（norm、活性化関数等）を触る場合、SQ8のrelease gateへ影響しうる。
  - Mitigation: 修正箇所がAQ4専用kernelか共有経路かをPhase 4で明示し、共有経路の場合はSQ8側回帰testも実行する。
- **holdoutとtuning caseの重複によるfalse-go**: 原因特定・fix検証に使ったfixtureをそのままP2正式gateのholdoutに流用すると、過適合したfixで見かけ上合格してしまう。
  - Mitigation: Phase 1-6で使う診断用fixtureと、Phase 7の正式holdoutを明確に分離し、hashで非重複を確認する（`docs/proposals/aq4-p2-fidelity-holdout-protocol-v0.1.md`の方針に従う）。

## Next Actions

1. Phase 2: hybrid診断harnessを複数層chainへ拡張し、self-attention層を最低1層含む範囲（例: layer 0〜7相当）で層ごとの相対L2成長曲線を測定する。
2. Phase 2b: post-norm epsilon control比較を並行して実施する。
3. 成長曲線がモデル最終出力の相対L2 0.615をどの程度説明するかを判定し、Decision Treeに従ってPhase 3a/3b/3cのいずれかへ進む。
4. GPU window（Phase 3c/6）は、CPU-onlyの調査（Phase 2〜3b）で候補を絞り込んだ後にのみ申請する。
