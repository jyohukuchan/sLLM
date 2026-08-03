# Phase 2 H3・G0・model-free GPU path計画

## 状態

- 作成日: 2026-08-03
- 状態: active
- 対象期間: Phase 2前半
- 上位計画: [main plan](../../../../main-plan.md)
- CI正本: [CI・テスト方針](ci-test-strategy.md)

実装進捗:

- 作業単位0と1のstatic contractを完了した。
- 作業単位2のCMake/build接続、exact 2 target direct compile/link、artifact検査、fail-closed集約、non-required workflowをcommit `03f90be1ad85145e3abee86e67615c1e17f552b4`として公開した。GitHubの2 compile rowはPASSし、aggregateはrun identityのcontainer伝播漏れをfail-closedに検出したため、その修正は次candidateへ含める。
- 作業単位4のcanonical tuple、identity-only native HIP observer、read-only health/process observer、exact H3 artifact binding、2 row aggregateとnegative contractを完了した。commit `e91ff35caac8247fc056eb14a1d6cee2a2319cc5`（tree `75b229791cd3cf7c6ed38c25264b0cd09a9cde33`）でH0〜H3とcanonical `gfx1030`/`gfx1201`のG0・aggregateがPASSした。
- 次のrollback境界は作業単位5のmodel-free diagnostic pathとし、7日を待たずG1実装へ進む。

## 目的

Phase 1のhost-only skeletonを維持したまま、ROCm 7.14.0で再現可能なHIP compile evidenceと、専用local hostで同一immutable candidateを検証するGPU evidence経路を追加する。到達点は、`Cargo -> ullm-hip -> versioned C ABI -> native HIP -> GPU`を通るmodel-freeの最小実行である。

この計画が完了しても、数値op、model load・推論、性能、一般的なGPU互換性は完成扱いにしない。

## 範囲

含むもの:

- ROCm 7.14.0固定toolchainと同一root検証。
- exact `gfx1030`/`gfx1201`の独立したH3 compile-only row。
- HIP artifact metadata、report、sidecar、集約のfail-closed contract。
- H3 required昇格用の非同期観測開始。
- trusted local execution、完全SHA指定、canonical GPU直列実行、G0 preflight。
- model-free diagnostic kernelによるallocation、copy、dispatch、completion、copy-back、resource解放。
- 実行前後のprocess・device health確認と、同一candidateのG0/G1 evidence。

含まないもの:

- Qwen3.5その他のmodel、tokenizer、weight、model slice、G2/G3。
- GEMM、attention、RMSNorm、sampling等のsemantic numerical op。
- 数値tolerance、性能threshold、P0/P1、対応GPUの`experimental`からの昇格。
- generic processor、fat binary配布、runtime JIT、public fork PRのGPU実行。
- GitHub self-hosted runnerの完成。初期実行はreview済みproject commitだけを扱う専用local host経路とする。

## 正本上の決定

### H3観測は開発を停止しない

H3はnon-requiredで導入し、20回以上かつ7日以上の観測は`host-required`へ昇格する条件にだけ使う。観測開始後、G0とmodel-free GPU pathを直ちに並行して進める。この計画の完了条件にH3 required昇格は含めない。

### bootstrap gateを先に解消する

現行CI正本はnative buildやtarget変更へ、まだ存在しないG0/G1/G2/G4/P0を一括要求しており、H3自身を導入できない循環がある。実装前にCI正本を次の適用範囲へ変更し、変更後の規則をこの計画より優先する。

| 変更scope | merge前に必要な同一candidate evidence |
| --- | --- |
| 文書、schema、matrix、runnerの非実行contract | H0〜H2、negative self-test。GPU/runtime behaviorを変更する場合は下位行も適用 |
| H3 toolchain、compile-only、artifact metadata | H0〜H3。H3はexact target別にPASSし、GPU実行・対応実績とは表記しない |
| trusted local runner、G0 preflight | H0〜H3、runnerのhost negative test、canonical deviceの同一SHA G0、実行前後health |
| model-free native HIP実行 | H0〜H3、canonical `gfx1030`/`gfx1201`の同一SHA G0/G1、CPU fallbackなし、artifact一致 |

G2はmodel path、G4は互換性昇格、P0は性能または実運用dispatchへ影響する変更から要求する。未実装機能のevidenceをbootstrap作業へ形式的に要求せず、実際に変更したscopeのevidenceは省略しない。

### identityとGPU対象

- `reviewed_sha`、`tested_sha`、`workflow_sha`は同じ40桁commit SHAとし、Git tree OIDも記録する。
- H3 artifactはexact `gfx1030`とexact `gfx1201`を別々に生成し、host自動検出、別target、generic targetへfallbackしない。
- 初期G0/G1は専用local hostのV620 `gfx1030` 1台とR9700 `gfx1201` 1台をcanonical rowとし、直列実行する。UUID/BDFはG0実装時にversioned tupleへ固定する。
- 2台目のV620はspareまたは再現確認用であり、同一changeの必須rowへ追加しない。
- compile-onlyと実機evidenceを混ぜず、H3 reportへ実行成功を記録しない。

### model-free最小経路

- public inference opを追加するためのprobeにしない。diagnostic kernelとその呼び出しはtest/evidence用途として明示し、`ullm-core::Backend::execute`の数値対応を主張しない。
- host stubは既定のCPU CI経路として残し、HIPを暗黙有効化しない。HIP buildは明示optionと検証済みROCm root/targetを要求する。
- diagnostic kernelは小さな整数bufferをGPU上で決定的に更新するだけとし、入力・出力をbyte exactで比較する。caseには1要素だけでなく、3、17、境界前後など非2冪・非整列値を含める。
- CPU実装、GPU kernel emulation、別backend fallbackを禁止し、`selected_backend=hip`、dispatch 1件以上、`fallback_used=false`をreportで検証する。
- completionまでqueue、buffer、eventを所有し、完了後に全resourceを解放する。timeout、途中失敗、caller側の早期dropでもuse-after-freeまたはleakを許さない。

## 作業単位

### 0. 正本とbootstrap gateの同期

変更対象:

- `docs/plans/main-plan.md`
- `docs/plans/active/2026/08/1-10/ci-test-strategy.md`
- 必要なら[GPU互換性方針](../../../../../compatibility/gpu.md)の古いevidence記述
- 対応history

実施内容:

1. 上記のscope別gateをCI正本へ反映する。
2. H3観測がrequired昇格だけの条件で、G0以降と並行することを明記する。
3. `gpu.md`の「実機検証結果なし」と、限定smokeを記録する[AMD GPU方針](../../../../../compatibility/amd-gpu.md)・[software方針](../../../../../compatibility/software.md)の表現を、検証scopeを保って整合させる。
4. H3、G0、G1、model-free probeの証明範囲を分離する。

受入条件:

- H3導入自身が未構築G2/G4/P0を要求する循環がない。
- GPU evidence hard gateを弱めず、変更scopeごとの必要evidenceが一意に決まる。
- 正本文書間に実機smokeの有無や証明範囲の矛盾がない。

### 1. ROCm toolchainとartifact contract

主な変更候補:

- `ci/toolchains/rocm-7.14.0.json`
- `ci/schema/hip-artifact-metadata-v1.schema.json`
- `ci/matrix/hip-compile-v1.json`
- toolchain検査・artifact metadata生成/検証script
- CMakeと`ullm-hip-sys/build.rs`の明示HIP configure path

contractに固定する項目:

- toolchain imageのregistry、immutable digest、対象platform。
- canonical `ROCM_PATH`、ROCm release、`amdclang++`のabsolute path/version、LLVM major。
- HIP headers、device libraries、CMake packageの解決済みpathと同一root判定。
- source commit SHA、tree OID、build option、CMake generator、build type。
- exact target、code object version、`xnack`/`sramecc`/wavefront等のcodegen feature。
- artifact path、size、SHA-256、ELF target metadata、必要なsymbol/section検査結果。

fail-closed条件:

- imageがtagだけ、digest不一致、ROCm root不在、別root混在、LLVM major不一致。
- target未指定、複数/別/generic target、feature欠落または正規化不能。
- metadata欠落、schema不正、artifact/sidecar hash不一致、stale identity。
- source treeまたは共有build directoryへの生成物出力。

受入evidence:

- exact targetごとに同じsourceから再buildし、metadata contractを満たす。
- target、root、hash、metadataを意図的に壊したnegative testが全てfailureになる。
- Phase 1 host-only buildとH0〜H2が引き続きPASSする。

### 2. H3 compile-only row

実施内容:

1. ROCmをjob中にinstallせず、digest固定imageを使う。
2. job冒頭でtoolchain同一root検査を行う。
3. exact `gfx1030`と`gfx1201`を独立rowでcompileする。
4. 同じROCm rootのLLVM toolでELF/code object metadataを検査する。
5. `test-result-v1`とHIP artifact metadataをsidecar hash付きで出力し、2 rowをfail-closed集約する。
6. H3はnon-requiredの独立checkとして開始し、`host-required`にはまだ含めない。

H3が証明するもの:

- 指定ROCm/toolchainでsourceがcompile/linkできる。
- artifactが指定exact target/codegen metadataを持つ。
- reportとartifactが同じimmutable candidateに属する。

H3が証明しないもの:

- GPUでloadまたは実行できること。
- 数値正しさ、性能、SKU/OS tupleの対応。

受入条件:

- 両rowが時間上限内にPASSし、missing/duplicate/unknown/stale/cancelを集約が拒否する。
- 一方のtarget artifactを他方へ差し替えるnegative testがfailureになる。
- artifactを再利用する後続G0/G1が同じcontent SHA-256を要求できる。

### 3. H3 required昇格観測

H3導入と同時に、run ID/attempt、row、state、duration、artifact digest、infra error、missing/cancelを機械集計する。観測はG0以降と並行し、開発者の待機taskにしない。

required昇格reviewの条件は、20回以上かつ7日以上、全期待row `PASS`、他state/cancel/schema error 0、artifact hash一致、p95 12分以下、最大15分以下、unexpected `INFRA_ERROR` 0、missing result 0とする。条件未達でもH3はnon-requiredのまま維持し、G0/model-free pathの合否を上書きしない。

この計画の完了時点で7日を満たしていなければ、観測を継続するfollow-upとして残し、model-free path完成を待たせない。

### 4. trusted local GPU evidenceとG0

主な変更候補:

- `ci/matrix/gpu-runtime-v1.json`
- local GPU runner、G0 preflight、result集約script
- GPU result/tuple schemaの必要な拡張とnegative self-test
- 実行・診断手順を置く開発文書

実行境界:

- maintainerが完全にreviewしたproject commitだけを40桁SHAで指定する。
- dirty worktree、branch/tag、異なるreviewed/tested SHA、外部binaryを拒否する。
- canonical GPUをUUID/BDFで1台だけ選択し、visibility環境変数だけをsecurity boundaryにしない。
- `gfx1030`と`gfx1201`はhost lockを使って直列実行する。lock取得失敗や競合を成功扱いにしない。
- jobへsecret、controller credential、Docker socketを渡さず、実行codeへ`sudo`やGPU resetを許可しない。

G0で記録・検証する項目:

- OS point release、kernel、amdgpu driver、amdhsa execution ABI。
- ROCm build/runtime release、compiler、HIP/ROCr libraryの解決済みabsolute path。
- GPU product、UUID、BDF、runtimeのexact `gcnArchName`、wave size、device-local memory。
- artifact target/codegen/code objectと実GPUの一致。
- 実行前のdevice health、既存process、温度等の診断可能なraw fact。
- reviewed/tested/workflow SHA、tree OID、tuple digest、artifact/report hash。

G0はresource gateの未確定閾値からsupport可否を決めず、raw factとbinary load eligibilityだけを記録する。

終了処理:

- 成否にかかわらず子process残留、device health、artifact/report hashを確認する。
- 異常時はnon-PASSとし、新規GPU作業を停止してhostをquarantine対象として報告する。
- reset/rebootが必要ならjob外のhost controllerだけが行う。

受入条件:

- canonical `gfx1030`/`gfx1201`の同一candidate G0が直列にPASSする。
- wrong GPU、wrong artifact、別ROCm root、dirty/mismatched SHA、process残留をnegative testが拒否する。
- G0 reportだけをkernel実行または数値正しさの証拠に昇格させない。

### 5. model-free最小GPU実行経路

実装順序:

1. 明示HIP buildでnative contextを生成し、runtime/library/targetをartifact metadataと照合する。
2. queue、device buffer、event/completionのnative ownershipを実装し、安全なRust wrapperへ閉じ込める。
3. host-to-device copy、test/evidence専用diagnostic kernel dispatch、completion待機、device-to-host copyを接続する。
4. Rust側がcompletionまでqueue/buffer/eventを強参照し、完了後に決定的に解放する。
5. `ullm-cli`または専用test binaryから全経路を一回のdocumented commandで実行する。

C ABI方針:

- 既存ABI v1の関数とstruct layoutを破壊しない。additive APIで済まない場合はABI versionを更新し、header、checked-in bindings、layout probe、C/C++ compile testを同一changeで更新する。
- caller-owned descriptor、文字列、一時arrayへのpointerをreturn後に保持しない。
- null、struct size、ABI version、reserved field、double destroy、invalid handle、timeout、HIP errorを明示statusとerror sinkへ変換する。
- diagnostic kernelの入口はtest/evidence用途と分かる境界に置き、将来のsemantic op dispatch ABIを固定しない。

G1 case:

- exact targetごとに1、3、17要素と、設定したcopy/dispatch境界の前後を実GPUで実行する。
- 入力は固定seedで生成し、期待する単純変換をbyte exactで比較する。
- selected backend、dispatch ID/count、fallback、artifact hash、GPU UUID/BDF、開始/終了時刻をreportへ記録する。
- public eventを早期dropするcase、途中失敗、timeout後のcleanup、複数buffer access modeを検査する。

受入条件:

- 同一candidate SHA/treeと同一artifactに対し、canonical `gfx1030`/`gfx1201`のG0/G1がPASSする。
- `selected_backend=hip`、dispatch count 1以上、`fallback_allowed=false`、`fallback_used=false`である。
- 全caseの出力がbyte exactで、0件選択、skip、quarantine、timeout、crashをPASSにしない。
- 実行後にprocess残留、resource leak、device health悪化がない。
- host-only環境では明示的なHIP unavailableを維持し、CPU fallbackでGPU testを代替しない。

### 6. 適用・文書同期・終了判定

1. 同一candidateのH0〜H3、G0/G1、schema/negative test、build/lintを成功させる。
2. 検証済みcandidateを専用local hostへ適用し、canonical 2 rowでmodel-free smokeと終了後health checkを行う。
3. artifact/report digestとcommit/tree identityをhistoryへ記録する。
4. [runtime architecture](../../../../../architecture/runtime.md)、compatibility文書、CI正本、main planを実装事実へ同期する。
5. H3観測が7日未満ならrequired昇格を未完了follow-upとして残すが、この計画のmodel-free実装完了を妨げない。

## 全体の完了条件

- bootstrap gateがCI正本へ反映され、計画とhard gateに循環がない。
- ROCm 7.14.0 image/toolchain/artifact contractがversioned fileとして固定されている。
- exact `gfx1030`/`gfx1201` H3がnon-requiredでfail-closedに動き、昇格観測が自動継続している。
- trusted local経路が同一immutable SHA以外を拒否し、canonical 2 GPUのG0を直列実行できる。
- model-free最小GPU pathが両GPUでG1 PASSし、CPU fallback、model、semantic numerical opを使っていない。
- 実行前後health、process残留、artifact/report hashが確認され、異常を成功扱いしない。
- 対応範囲の誇張がなく、compatibility lifecycleは実装scopeに応じて`experimental`のまま記録される。

## rollback境界

- toolchain/artifact contract、H3、G0 runner、model-free runtimeを独立したreview・rollback可能な作業単位にする。
- host-only stubを常に維持し、HIP path失敗時にCPU CIを偽のGPU成功へ切り替えない。
- candidateのtest、local適用、適用後healthのどれかが失敗したらpushせず、最後の検証済みrevisionへ戻す。
- device health異常、process残留、rollback不能時は追加GPU実行を停止し、host状態と未適用範囲を報告する。

## 未確定事項

- local runner lockとhost controllerの恒久配置。
- diagnostic kernel用C ABIをadditive public ABI、private test ABI、専用test binaryのどれに置くか。実装前にarchitecture reviewで決める。

[対応する履歴](../../../../../history/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)
