# 汎用推論最適化 P1-A

## 前回の要点

- P0でModelGraph/StateSchema、BackendOpRegistry、production trace、prefill validationの契約を固定した。
- 本番経路のtokenwise prefillとcontext依存decodeが、実測性能低下の主因である。

## 今回の変更点

- backend/model名に依存しない`model_graph`、`state_schema`、`execution_batch`を追加した。
- graph topology、weight binding、演算別shape/state cardinality、state kind/layout/ownership/transactionをGPU確保前に検証する。
- ExecutionBatchは初期の矩形batch、保守的な上限、workspace headroom、commit nonce、request間state handle非共有を検証する。
- 相互レビューで見つかったstate alias、巨大・疎packing、shape未検証、Graph/State種別不整合を修正した。
- `ullm-engine` lib test 453件、cargo check/fmt/doc、whitespace checkを実行した。

## 次の行動

- P1-B1で決定的なCPU reference executorのstateless subsetを追加した。Embedding、Linear、FusedLinearGroup、Silu/Relu、GatedMlp、Residual、LmHeadに対応する。
- 実行前に対応node、format/layout、総要素数、一時領域、計算量を検証し、未対応やOOM/CPU DoS相当をfail-closedにした。
- 相互レビュー後、layoutの暗黙誤解釈、GELU近似の曖昧さ、aggregate memory未集計、出力clone、work budget不足を修正した。
- P1-B2でNorm/GELU/stateful nodeの数値契約、typed failure trace、Qwen3.5 hybridとQwen3 denseのadapter graph fixtureを追加する。
- P1-B2の最初の単位として、NormをRMS/Layer、scale/unit-offset/scale+bias、last-axisの型付き契約にした。
- CPU referenceに分類付き失敗、失敗node、最大4096 nodeの完了trace、上限1KiBの診断を追加した。token IDは診断へ保持しない。
- F32 last-axis RMSNormのScale、UnitOffsetScale、ScaleAndBiasをCPU referenceへ接続した。幅1を含むwork budgetとF32中間overflowもpreflight/typed Numericalでfail-closedにした。
- CPU referenceを含む`ullm-engine` lib testは483件に増え、fmt/check/doc/diffとともに成功した。
- 次はRoPEとattention/stateful semanticを独立oracleとtransaction APIに基づいて追加する。
- RoPEを`[values, positions]`の明示的2入力にし、head geometry、SplitHalf/Interleaved、角度・符号・headごとのfrequency resetを型付き契約にした。
- RowMajor/TokensHidden/PackedRaggedをgraph semanticで扱え、position shape、整数format、output metadata、geometry overflowをGPU確保前に検証する。
- CPU referenceでF32 RoPEのSplitHalf/Interleaved、headごとのindex reset、partial suffix、rank3/U64、PackedRaggedを実行できるようにした。
- F32で正確に表せないpositionはCPU capability不足としてallocation前に`Unsupported/position_precision`で拒否し、実値は診断へ保持しない。
- 固定literal goldenで回転符号とhead resetを固定し、`ullm-engine` lib test 497件が成功した。
- projection/norm/RoPE/gate/O projectionを含まない`CausalGqaAttentionCore`を追加し、stateなしF32のcausal/GQA/stable softmaxをT² score matrixなしでCPU実行できるようにした。
- `state_transaction`moduleでowner epoch、batch nonce、handle lease、committed generationを組み合わせ、read-only snapshotとconsumeされるprepared deltaの境界を定義した。
- metadata検証はfallibleな参照Vecとsort/binary searchを使い、OOM時はResourceでfail-closedにする。統合lib testは519件が成功した。
- ヘッド単位の正規化を特定モデル名へ結び付けず、`GroupedLast { groups, group_width }`として汎用化した。各連続groupは独立してRMSを計算し、`[group_width]`のscale/biasを全groupで共有する。
- CPU referenceはF32 RowMajor/TokensHiddenでGroupedLast、UnitOffsetScale、ScaleAndBiasを実行し、checked work budgetと非有限値の型付き失敗を維持する。従来のLast軸は1 groupとして同じ経路へ還元する。
- Luna実装後にTerraの読み取り専用監査を行い所見0件、統合`ullm-engine` lib testは527件が成功した。
- 次は汎用GroupedLastSplitとsigmoid gated multiplyを追加し、その後にQwen3/Qwen3.5 adapter fixtureへ接続する。
- `GroupedLastSplit`でgroup内部の複数segmentをsegment別outputへchecked gatherし、`GatedMultiply(Sigmoid)`で数値的に安定なpointwise gateを表現できるようにした。いずれもモデル名を含まず、graphではF32/BF16/FP16とPackedRaggedを含むtoken-local semanticsを定義した。
- CPU referenceはF32 RowMajor/TokensHiddenに限定し、aggregate output reservation、work budget、非有限値、typed preflight/runtime failureを検証する。負の極端なsigmoidが正の有限subnormalを保つ回帰も固定した。
- Terra監査の軽微なテスト穴2件を修正後、統合`ullm-engine` lib testは539件が成功した。
- 次はQwen3.5のlinear/recurrent attentionに必要な汎用stateful operatorを設計し、dense/hybrid adapter fixtureへ接続する。
- 連続末尾segment用`LastAxisSplit`、group単位L2正規化、固定正scaleを追加し、RMSのmean-square意味とは分離した。CPU参照はF32 RowMajor/TokensHiddenで固定期待値を検証する。
- linear/recurrent attentionを`CausalDepthwiseConv1d`、`GatedDecayParameters`、`GatedDeltaRuleScan`へ分解し、任意chunkを一括scanできる汎用semanticを定義した。
- 旧`Recurrent` stateを維持したまま複数instance用`RecurrentBank`を追加し、conv historyはcanonical `[channel,age]`、oldest→newest順とした。物理backend配置が異なる場合はsnapshot/import/exportで変換する。
- layer/state一意性とsource weight canonicalizationはproduction adapter admissionのblocking gateとして明記した。CPUは未実装nodeをpayload前にtyped Unsupportedで拒否し、統合lib testは557件が成功した。
- 次は3演算のstate-free CPU oracleとchunk同値goldenを実装し、その後にprivate working stateからPreparedStateDeltaを返すstateful入口へ進む。
- 3演算のstate-free F32 CPU oracleを追加し、zero-prefix conv、stable decay transform、private state bank delta-rule scanを実装した。複数key/value head、非ゼロ初期state、部分更新率の手計算goldenで更新順とhead mappingを固定した。
- `adapter_admission`moduleでRequestLayer stateのnode/layer exact mapping、一意参照、全weight occurrenceのslot/logical ID、汎用shape transform recipeを構造検証する。
- 構造tokenは検証したgraph/bindings/states/spec実体をborrowし、別instanceへのreplayを防ぐ。同logical weightのrecipe一致と、既存validator前の4096 record/edge上限も固定した。
- このtokenはpayload hash、transform実行、upload identityを証明せずproduction capabilityではない。後続のverified package/canonical evidence gateが必要である。
- 統合lib testは571件が成功した。次はSiLU gateを汎用pointwise演算へ追加し、Qwen3/Qwen3.5 fixture構築へ進む。
- `GatedMultiply`をSigmoid/SiLU対応へ拡張し、stable SiLUとactivation別workをCPU referenceへ追加した。
- Qwen3-14Bの40 dense attention層と、Qwen3.5の`[linear,linear,linear,self]×8` attention-stack fixtureを汎用nodeだけで構築した。Qwen3.5 linear層は`conv→SiLU→split`、self層Q/K epsilonは1e-5をruntime evidenceに合わせた。
- `CausalGqaAttentionCore`のPaged/Sliding KV stateをStateSchemaへ接続し、geometry/format/ownership/transactionを検証する。fixtureはattention-only F32 structural contractであり、payload evidence、q-gate/reorder、MLP等を含まない。
- stateful CPU入口を追加し、single-request denseのConvHistory snapshotから私有final historyを作り、全graph成功後だけPreparedStateDeltaを返す。ColdPrefill Zeroed、exact mapping、2-state atomic prepare、失敗時snapshot不変、public 2chunk同値を固定した。
- 新stateful admission metadataは4096上限後のfallible sorted Vecを使う。既存stateless共通execute_coreのBTreeMap allocation負債は残る。
- 統合lib testは583件が成功した。次はGatedDeltaRuleScan stateを同じdeltaへ接続する。
- RecurrentBank付きGatedDeltaRuleScanをstateful CPU経路へ接続し、非ゼロbank、2chunk継承、Conv+Scan混在delta、後段overflow時のsnapshot不変、2x2 canonical strideを手計算goldenで固定した。
- `execute_stateful_with_owner_traced`でpre-begin admission、owner snapshot、prepare、atomic commitを接続し、commit成功後だけoutputsを返す。Admission/Begin/Execution/Commitのtyped failureとtraceを公開した。
- fake ownerは全metadataとpayloadをtemporary rootへstage後に1 swapし、generation conflict/second-entry fence/commit failureでroot不変を検証する。未参照ModelShared stateはpre-begin resourceへ数えず、binding対象だけをruntime snapshotと同じ集合で計上する。
- 統合lib testは597件が成功した。次はstructural adapter admissionをverified package payload/canonical transform evidenceへ昇格する。
- F32 RowMajor Identity passthroughを外部trust manifest digest、source/canonical/recipe/binding digest、structural token instanceへ結ぶverified evidence receiptを追加した。symlink/path escape/TOCTOU/JSON heap amplificationをfail-closeにした。resident upload capabilityではない。
- typed backend operation registryをproduction resident AQ4へ接続した。model名ではなくphase/format/layout/geometry/width/device featureでQKV prepare、delta scan、plain/fused KV writer、paged GQA readをload時resolveする。
- capabilityは実HIP propertyのgcnArchName、ABI、env policy、scratch ABI probe成功から作る。R9700隔離でgfx1201を確認し、probe失敗時cache非公開、InPlace失敗後resetまでpoisonを固定した。
- 全モデルworkspaceをselected package metadataからupload前に集計し、全32 layerのoperation resolutionとrequest execution coverageをstructured auditにした。production layerの対象direct ABI callは0。
- 製品package resident smokeは1-token/8-tokenの2 request、reset、sibling engine 0、coverage completeで成功。8-token prefill 80.47 tok/s、表示decode94.94 tok/s。promotion比較はlegacy側の既存SQ8 execution failureで停止した。
- 統合lib testは636件成功、R9700専用probe testも手動成功した。次はM<=128 chunk prefillを同じregistry familyへ追加する。
