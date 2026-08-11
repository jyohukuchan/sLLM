# Phase 3 Stage A model lock・RMSNorm・G2計画

## 状態

- 作成日: 2026-08-04
- 状態: complete
- 対象期間: Phase 3 Stage A
- 上位計画: [Phase 3 Qwen3.5-4B BF16 text生成計画](../../../../archive/2026/08/1-10/phase3-qwen35-4b-bf16.md)
- CI正本: [CI・テスト方針](../../../../active/2026/08/1-10/ci-test-strategy.md)
- model固定正本: [model lock](../../../../../models/model-lock.md)

## 目的

Phase 2で完成したmodel-free G1を土台に、Qwen/Qwen3.5-4Bの固定revisionへ結び付いた最初のBF16数値経路を実装する。到達点は、完全なmodel lockから実weight sliceを抽出し、Rustのsemantic opからpublic HIP backendを通してRMSNormを実行し、独立したNumPy oracle、canonical `gfx1030`/`gfx1201`のsemantic G1とG2、短いP0 smokeで正しさと実行経路を証明することである。これは[Phase 3全体計画](../../../../archive/2026/08/1-10/phase3-qwen35-4b-bf16.md)のStage Aであり、Phase 3自体の完了点ではない。

この計画は、モデル全体の文章生成を完成させる計画ではない。attention、MLP、KV/state、prefill/decode、tokenizer実行、CLI生成、G3は、最初のmodel-bound数値経路が安定した後に別計画で追加する。

## 現在の進捗

2026-08-08時点の現行dirty worktreeに対する状態を次とする。ここでいうhost PASSは`local-development`であり、immutable candidateまたはGPU evidenceへ昇格しない。

host capsuleの独立review指摘を修復中に中断した`ci/tools/execution_capsule.py`の部分変更（現行file SHA-256 `44801a0832756f0e6966cb7b23bd25653d6cfba91d6816ededf7f9fe63239ac9`）は検証済みcandidateではなく、host/GPU evidenceに使用しない。2026-08-08のユーザー判断により、直前版SHA-256 `a1464bcf5ae1407aaa91b984a6782af0a06d574447afcd052864440f7faedbba`へのbyte-for-byte復元は打ち切り、A0 security hardeningの部分変更を放棄する。Stage Aのsemantic実装はdirect testと標準containerによる`local-development`確認で再開できる。immutable host evidenceが必要になる前に、現行部分変更を継承しない最小のtrusted-development baselineを別作業単位として作成し、通常回帰とreviewを通した新identityを固定する。

## 工程別時間予測と中断契約

2026-08-08のStage A再開では次を初期見積りとする。各中断上限は予測上端に1時間を加えたwall-clockであり、到達時は安全なrollback可能点で一旦停止して報告する。ユーザーが明示的に中断した時間は除外し、上限到達後に見積りを後付けで延長しない。

| 工程 | 予測時間 | 中断上限 | 完了境界 |
| --- | ---: | ---: | --- |
| A1. semantic G1 fresh独立review・direct host回帰 | 1〜2時間 | 3時間 | trusted-development境界でのfindings、実行test、次の修正単位を固定 |
| A2. review指摘修正・host再検証 | 2〜4時間 | 5時間 | 対象findingsを閉じ、関連direct regressionがPASS |
| A3a. G2 host contract・実行経路 | 2〜4時間 | 5時間 | real-weight sliceのschema、runner、negative test、host contractがPASS |
| A3b. P0 host contract・実行経路 | 2〜4時間 | 5時間 | P0 case-set、schema、runner、negative test、host contractがPASS |
| A4. immutable evidence用の最小baseline | 2〜4時間 | 5時間 | 中断A0を継承しないbaselineをreviewし、候補identityを固定可能 |
| A5. canonical 2 GPU evidence | 3〜6時間 | 7時間 | 同一candidateのrequired rowとaggregate、前後healthがPASS |
| A6. 適用後確認・正本/history同期 | 1〜2時間 | 3時間 | smoke、health、適用状態、文書が同一identityへ同期 |

工程内に独立した修正が複数見つかった場合は、同一file ownershipが衝突しない範囲でさらに分割し、分割単位ごとに開始時刻と中断上限を記録する。A1開始時点をStage A再開時刻とし、A1が3時間へ到達した場合はreviewが未完了でも停止する。

旧A3のG2とP0は独立した受入境界を持つため、再開時点でA3a/A3bへ分割する。各工程の開始時にwall-clock開始時刻とhard中断時刻を記録し、A3aの未完了時間をA3bへ移し替えない。

### 再開後の実績

- A1は2026-08-08 15:29:39 JSTに開始し、15:50:49 JSTに完了した（約21分、見積り内）。direct host回帰はsemantic G1 86件、reference 26件、RMSNorm H3 19件、Rust workspace 116件、focused evidence binary 7件、public-runtime host 1件を含めてPASSした。fresh reviewは、(1) canonical semantic G1がnonfinite activationだけを生成しnonfinite raw scaleを実行していないこと、(2) bounded raw responseをupload前に削除するため保存証拠から数値比較を独立再計算できないこと、の2件をblocking findingとして`FAIL`とした。その他のreview10修復、locked dtype/epsilon/shape、fallback禁止、root artifact hygieneは成立した。
- A2は2026-08-08 15:51:54 JSTに開始した。上記2件だけを修復対象とし、`execution_capsule.py`、custom capsule、同一UID adversarial hardeningへscopeを広げない。予測2〜4時間、hard中断時刻は同日20:51:54 JSTとする。
- A2の最初の`workspace-write`実装実行は、repository command開始前に`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`で停止し、file変更はなかった。AGENTS.mdのfallback規則に従い、開始時刻と中断上限を維持したまま、同じ禁止事項・file scopeの`danger-full-access`実行へ切り替えた。
- A2実装は2026-08-08 16:18:23 JSTに完了した。canonical caseを15件へ拡張してN=2560のraw-scale NaN/+Inf/-Infを追加し、2 row x 15 caseのbounded raw responseとSHA-256 sidecarを明示upload対象へ加え、report/aggregate/candidate/row/case/orderへ結合した。implementer direct host検証はsemantic G1 87件、reference RMSNorm 26件、contract/matrix/manifest、Python、C++、Rust、`git diff --check`がPASSした。実GPU evidenceはA5まで未実行である。
- A2を閉じる前のfresh独立reviewを2026-08-08 16:18 JSTに開始した。既知のbubblewrap起動不能を踏まえ、非変更・network/GPU/model禁止の`danger-full-access` transport fallbackとし、A2の開始時刻、予測、中断時刻20:51:54 JSTはリセットしない。
- A2 fresh独立reviewは2026-08-08 16:34 JSTに`FAIL`で完了した。A1の2 blockerそのもの（raw-scale非有限値とbounded raw-response再計算）は直接回帰を含めて閉じたが、report/artifact/aggregate schemaのnested object計23箇所がschema単体で未知keyを拒否せずclosed/exactではないこと、focused semantic G1回帰がCMake compiler-broker client認証で78 PASS・1 FAILとなることを新たなblockerとした。A2の開始時刻と20:51:54 JST中断上限を維持して、この2件だけを修復・再reviewする。
- 上記2件のA2継続修復は2026-08-08 16:33:46 JSTに`workspace-write` sandboxで開始し、今回はbubblewrapを含めrepository readまで正常に起動した。fallbackは使用せず、A2全体のhard中断時刻20:51:54 JSTを維持する。
- ただし同sessionは最初のread後、複数回連続してbubblewrap `RTM_NEWADDR`でrepo actionを開始できず、編集・test前に16:36 JSTで中断した。16:36:27 JSTから同一scope・禁止事項の`danger-full-access` fallbackへ切り替え、A2開始時刻と20:51:54 JST中断上限はリセットしない。
- A2継続修復は2026-08-08 17:05 JSTに完了した。report/artifact/aggregate schemaの全object境界を閉じ、stdlib validatorへ`patternProperties`と`uniqueItems`検証を追加し、Draft 2020-12とstdlibの双方で未知keyを拒否する再帰回帰を追加した。implementer報告ではsemantic G1 80件・subtest 70件、reference RMSNorm 26件、contract/matrix/manifest、Python、C++、Rust、runtime closure、`git diff --check`がPASSした。CMake回帰は実client SHA-256を用い、test側でbroker fdを30以上へ複製して通しているため、productionのfd認証regexが有効な10〜29を拒否する可能性を未解決論点として残す。
- 同17:05 JSTからA2再reviewを開始した。予測30分〜1時間とし、schema閉鎖、実emitted documentとの整合、全回帰に加え、fd=30固定がproductionの3以上の任意の有効な継承fd契約を隠していないかを独立判定する。A2全体のhard中断時刻20:51:54 JSTは維持する。
- 同再reviewの`read-only` sandboxはmain planの初回read後にbubblewrap `RTM_NEWADDR`で停止し、検証開始前に中断した。2026-08-08 17:06:46 JSTから非変更・network/GPU/model/container禁止の`danger-full-access` transport fallbackで継続し、review予測とA2 hard中断時刻はリセットしない。
- A2再reviewは2026-08-08 17:22 JSTに`FAIL`で完了した。A1由来のraw-scale非有限値、bounded raw-response保持・identity結合・offline再計算、schema全object境界の閉鎖、semantic G1 88件、reference 26件、Python/C++/Rust/各validator回帰はPASSした。一方、(1) production CMakeのfd regexが有効な継承fd 10〜29を拒否し、testのfd=30固定がこれを隠すこと、(2) stdlib validatorが明示propertyと`patternProperties`の重複制約を同時適用せず、`prefixItems: [false]`で例外になること、をblocking findingとした。
- 同17:22 JSTからこの2件だけのA2継続修復を開始する。修復予測30分〜1時間、その後のfresh再review予測30分〜1時間とし、A2全体のhard中断時刻20:51:54 JSTは維持する。
- A2継続修復の`workspace-write` sessionは必要なreadを完了したが、2回の`apply_patch`がbubblewrap `RTM_NEWADDR`で編集前に失敗したため中断した。2026-08-08 17:26:42 JSTから同一scope・禁止事項の`danger-full-access` transport fallbackへ切り替え、修復予測とA2 hard中断時刻はリセットしない。
- A2最終修復は2026-08-08 17:41 JSTに完了した。production CMakeをcanonical decimalの全fd>=3受理へ修正してfd 3/9/10/29/30/300と予約・不正・非canonical値の非実行境界testを追加し、stdlib validatorのproperty/pattern重複適用とboolean schemaをDraft 2020-12へ一致させた。CMake source identity変更に伴うRMSNorm H3 manifest digestも更新した。implementer direct host検証はsemantic G1 90件、reference 26件、Rust 116件、Python/C++/全validator、`git diff --check`がPASSした。
- 同17:41 JSTからA2最終fresh独立reviewを開始する。予測30分〜1時間とし、fd全境界、validator differential、H3 source identity、A1由来evidence、全回帰を再確認する。A2 hard中断時刻20:51:54 JSTは維持する。
- 同最終reviewの`read-only` sandboxは初回read後にbubblewrap `RTM_NEWADDR`が再発し、検証開始前に中断した。2026-08-08 17:43:33 JSTから同じ非変更・禁止事項の`danger-full-access` transport fallbackで継続し、review予測とA2 hard中断時刻はリセットしない。
- A2最終reviewは2026-08-08 17:58 JSTに機能gate 1〜5をすべてPASSしたが、scope gateだけを`FAIL`とした。指摘対象は17:43:46 JSTにmain agentがこのactive planとmain planへ同期したreview開始・sandbox fallbackのstatus行であり、17:41:18 JST完了のrepair implementer変更ではない。AGENTS.mdがmain agentへ計画編集を割り当てているため、code scope違反と区別する必要がある。
- 同17:58 JSTから上記provenance・ownershipだけの独立scope再判定を開始する。予測10〜20分とし、機能gateは再実行済みPASS evidenceを保持する。A2 hard中断時刻20:51:54 JSTは維持する。
- A2 scope再判定は2026-08-08 18:01 JSTに`PASS`で完了した。repair artifactは17:41:18 JST、main/Stage A plan statusは17:43:46 JST以後であり、AGENTS.mdがmain agentへ割り当てる計画同期と確認された。sole P1を撤回し、最終reviewの機能gate 1〜5 PASSと合わせてA2を完了とする。A2は15:51:54開始から約2時間9分で、2〜4時間予測内かつ20:51:54 hard中断時刻前に完了した。
- A3a G2 host contract・実行経路は2026-08-08 18:01:23 JSTに開始した。予測2〜4時間、hard中断時刻は23:01:23 JSTとする。schema/matrix/case-set/negative test、slice extractor、dedicated G2 binary/runner/aggregateのhost-only境界を対象とし、canonical GPU G2実行はA5まで行わない。
- A3aの`workspace-write` sessionは正本read後にbubblewrap `RTM_NEWADDR`が連続し、編集前に中断した。2026-08-08 18:04:49 JSTから同一scope・禁止事項の`danger-full-access` transport fallbackへ切り替え、A3a開始時刻、予測、23:01:23 JST hard中断時刻はリセットしない。
- A3a実装は2026-08-08 18:40 JSTに完了した。closed G2 schema 6種、exact `gfx1030`/`gfx1201` matrixとpre-registered tolerance、synthetic-only model-slice extractor、dedicated public RMSNorm evidence binary、artifact builder、runner、aggregate、host suite登録、negative testを追加した。candidate/prerequisite hashとreport↔artifact identityをcanonicalに結合し、非GPU hostのrelease binaryは`HIP unavailable`で失敗してCPU fallbackによるG2 PASSを生成しない。implementer direct host検証はG2 14件、semantic G1 90件、H3 19件、model-lock 21件、reference RMSNorm 26件、Rust workspace 116件、Python/C++/matrix/manifest/contract、MSRV、`git diff --check`がPASSした。GPU、実model/cache、raw slice、network、P0は未実行である。
- 同18:40 JSTからA3a fresh独立reviewを開始する。予測30分〜1時間とし、closed schema、slice offset/size、canonical case-set、dedicated binaryのpublic RMSNorm経路、host stub fail-closed、candidate/artifact/prerequisite binding、runner/aggregateの負条件、scopeと回帰を独立確認する。A3a全体の23:01:23 JST hard中断時刻は維持する。
- 同reviewの`read-only` sandboxは正本read後、shell監査がbubblewrap `RTM_NEWADDR`で失敗したため検証前に中断した。2026-08-08 18:42:22 JSTから同じ非変更・network/GPU/model/container禁止の`danger-full-access` transport fallbackで継続し、review予測、A3a開始時刻、23:01:23 JST hard中断時刻はリセットしない。
- A3a fresh独立reviewは2026-08-08 19:03 JSTに`FAIL`で完了した。exact matrix/case-set、slice recipe/offset、dedicated public RMSNorm binary、host stub非zero、host-only登録、no-G2-workflow、focused/broad回帰はPASSした。一方、(1) tolerance schemaの`type: ["string", "null"]`をrepository stdlib validatorが正しく制約しないこと、(2) runnerが実際の5120-byte sliceをslice recordのSHAへ結合しないこと、(3) artifactのbinary/source/sidecar identityが実fileへ結合されず宣言値で成立すること、(4) candidate SHA相互一致、case seed/hash、prerequisite row、health target、有限error、nonzero evidence hash等を検証せず偽造PASSを受理すること、をblockerとした。関連H3 102件中1件のbuild-script parser失敗はA3a以前のmtimeであり別follow-upとする。
- 同19:03:46 JSTから上記4 blockerだけのA3a修復を開始する。修復予測1〜2時間、続くfresh再review予測30分〜1時間とし、A3a全体の23:01:23 JST hard中断時刻は延長しない。GPU/model/cache/raw real slice/network/P0/deferred capsuleへscopeを広げない。
- 同修復の`workspace-write` sandboxはread前にbubblewrap `RTM_NEWADDR`で停止し、変更はなかった。2026-08-08 19:06:10 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a blocker修復は2026-08-08 19:36 JSTに完了した。nullable SHAのDraft/stdlib parity、synthetic safetensorsから同一FDで読み取る実5120-byte sliceのsize/SHA結合、実binary・canonical sidecar・dedicated Rust source・実build source-setのidentity、candidate/prerequisite/report/aggregate/health/case/nonzero hashのfail-closed検証を追加し、A5 parser/oracle以前の数値PASS昇格を禁止した。implementerのG2 focused 22件、semantic G1 90件、model-lock 21件、reference 26件、G1 29件、H3関連28件、Rust/Python/C++/matrix/manifest回帰と`git diff --check`はPASSした。既知の`test_fail_closed` capsule stderr問題とA3a以前のH3 parser失敗は範囲外として記録する。
- 同19:36 JSTからA3a fresh独立再reviewを開始する。予測30分〜1時間とし、元の4 blockerの再現probe、source-setの完全性、実file/slice/sidecar結合、strict Git candidate結合、aggregateの実file hash再計算、A5以前のPASS不能、focused/broad回帰を独立確認する。A3a全体の23:01:23 JST hard中断時刻は維持する。
- 同再reviewの`read-only` sandboxはbubblewrap `RTM_NEWADDR`によりrepository access前に停止し、file read・test・変更はなかった。2026-08-08 19:40 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a fresh独立再reviewは2026-08-08 19:56 JSTに`FAIL`で完了した。nullable schema parity、実5120-byte slice結合、focused G2 22件、strict Git CLI、A5以前のPASS禁止、semantic G1/H3/model-lock/reference/Rust/Python/C++回帰はPASSした。一方、(1) G1 binaryをcanonical G2名へ改名して整合sidecarを付けるとartifact builderが受理すること、(2) build source-setとG2 path registrationが実Cargo/native入力11件を漏らすこと、をblockerとして再現した。同19:56 JSTからこの2件だけを1〜2時間で修復し、その後30分〜1時間のfresh再reviewを行う。A3a全体の23:01:23 JST hard中断時刻は延長しない。
- 同修復の`workspace-write` sandboxはrepository access前にbubblewrap `RTM_NEWADDR`で停止し、変更はなかった。2026-08-08 20:00 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- 2026-08-08 20:00 JSTに開始した最初の`danger-full-access` processは自processを別subagentと誤認して待機へ入り、repositoryを編集していないことを確認して20:05 JSTに中断した。同20:05:46 JSTからnested processを禁止したdirect implementerで再開し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3aの2 blocker修復は2026-08-08 20:37 JSTに完了した。43 fileのcanonical build-input manifestとbuild-time生成identityを追加し、builder・validator・runnerがsource-setを独立再計算する。専用G2 binaryはHIP/model/cacheへ触れないidentity queryを提供し、G1/H3/任意実行file、symlink、非regular file、identity不一致を拒否する。G2 focused 26件、locked offline Cargo build/check、format、matrix/contracts、実binary identity、1行query、5120-byte memfd、host HIP unavailable nonzero、G1回帰、`git diff --check`はPASSした。H3 3件は共有`build.rs`に対する既存dirty stateのhash/parser期待値不整合として残り、独立再reviewで帰属を再確認する。同20:38 JSTから30分〜1時間予測のfresh独立再reviewを開始し、A3a全体の23:01:23 JST hard中断時刻は維持する。
- 同fresh再reviewの`read-only` sandboxは`pwd`を含む全commandがbubblewrap `RTM_NEWADDR`でprocess開始前に停止し、file read・test・変更はなかった。2026-08-08 20:40 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a fresh独立再reviewは2026-08-08 21:04 JSTに`FAIL`で完了した。改名した実G1拒否、43/43 source closure・Cargo rebuild登録・path ownership、実debug G2のexact 1-line identity queryとhost `HIP unavailable` nonzero、G2 26件、G1 134件、model-lock 21件、reference 32件、Rust workspace、C++ host staticはPASSした。一方、(1) canonical marker/identity/query/sidecarを偽造した任意Python executableと最小C ELFをbuilder・validator・runnerがidentityとして受理すること、(2) query helperが前後空白または末尾改行なしを受理すること、(3) A3a変更後の`build.rs` hash 2件とH3 rerun parser 1件がstale/failであることをblockerとした。H3合同回帰が内部で`rocm-smi` health照会を開始したため直ちに中断し、GPU kernel/model実行は行っていない。既存release G2 binaryは再build前でidentity query未対応として区別する。同21:05 JSTから上記3系統だけを45〜75分で修復し、20〜40分のfresh再reviewを行う。23:01:23 JST hard中断時刻は延長しない。
- 同修復の`workspace-write` sandboxは正本とdirty baselineをread後、`echo`や`/tmp`を含む全commandがbubblewrap `RTM_NEWADDR`でprocess開始前に停止した。file編集・testはなかった。2026-08-08 21:13 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a最終blocker修復は2026-08-08 21:30 JSTに完了した。fixed locked/offline Cargo buildとcanonical builder outputのbyte identityへstaged artifactを結合し、exact 1-line queryを共通化した。任意Python executable、最小C ELF、改名G1、symlink/nonregular/nonexecutable/wrong-name/stale sidecar/duplicate marker/source-set/query format負例をbuilder・validator・runnerで拒否し、H3 `build.rs` hash/source-setとrerun parser互換を同期した。実装者のG2 29件、G2 contracts、safe H3 static/hash/parser、Rust 116件、reference 26件、model-lock 21件、G1 23件、formatと`git diff --check`はPASSした。fresh debug G2 SHA-256は`5f1c1f37cb64b24362c79889010e897936cf2ccc155e1981ad6c9affff1350f3`、host通常実行は`HIP unavailable`でexit 1だった。同21:31 JSTから20〜40分予測のfresh独立再reviewを開始し、23:01:23 JST hard中断時刻は維持する。
- 同fresh再reviewの`read-only` sandboxは全commandがbubblewrap `RTM_NEWADDR`でprocess開始前に停止し、file read・test・変更はなかった。2026-08-08 21:32 JSTから同一非変更scopeの`danger-full-access` transport fallbackで継続し、review予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a最終fresh独立再reviewは2026-08-08 21:47 JSTに`FAIL`で完了した。実fixed offline build、G2 29件、safe H3 static 27件、Rust 116件、C++ host static、43/43 source closure、matrix/path登録、exact queryと8種のquery負例、Python/C ELF・改名G1等の偽造拒否、host `HIP unavailable` nonzero、A5以前の数値PASS禁止はPASSした。一方、public `build_artifact()`直接呼び出しがowned buildを実行せず既存canonical binaryの完全copyを受理することと、ambient `CARGO_TARGET_DIR`でCargoが別directoryへ出力してもbuilderがstaleな固定`target/debug`を返すことをblockerとした。同21:47 JSTからこの2点だけを20〜35分で修復し、10〜20分のfocused再reviewを行う。A3aの23:01:23 JST hard中断時刻は延長しない。
- 同修復の`workspace-write` sandboxは正本をreadした後もbubblewrap `RTM_NEWADDR`が再発し、対象code確認・編集前に2026-08-08 21:49 JSTに中断した。同21:49:49 JSTから同一scopeの`danger-full-access` transport fallbackで継続し、修復予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a builder ownership修復は2026-08-08 21:58 JSTに完了した。public `build_artifact()`がfresh owned Cargo buildを必ず実行し、CLIは同じowned resultをprivate helperへ一度だけ渡す。Cargo子processの`CARGO_TARGET_DIR`をrepo-local `target`へ固定し、caller-supplied copyとambient target redirectを拒否・無効化した。実装者のG2 runner 16件、schema/slice/aggregate 16件、contracts/matrix、実ambient-target probe、Cargo build/query、host fail-closed、Rust fmt/check、Python 81 file compile/static、diff checkはPASSした。同21:59 JSTから10〜20分予測のfresh独立再reviewを開始し、23:01:23 JST hard中断時刻は維持する。
- A3a focused独立再reviewは2026-08-08 22:06 JSTに`FAIL`で完了した。public APIのowned buildとambient `CARGO_TARGET_DIR`固定は静的に成立したが、module-level `_build_artifact_from_owned_binary()`が通常のPython callで直接利用でき、underscore名だけではowned-build capability境界を閉じないことをsole blockerとした。read-only transportのbubblewrap再発によりfocused testの独立再実行は未完了である。同22:06 JSTからhelper完全除去だけを5〜10分で修正し、10〜15分のfocused再reviewを行う。23:01:23 JST hard中断時刻は延長しない。
- 同修正の`workspace-write` sandboxはmain plan read後にbubblewrap `RTM_NEWADDR`が再発し、編集前に2026-08-08 22:07 JSTに中断した。同時刻から同一scopeの`danger-full-access` transport fallbackで継続し、予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a helper完全除去修正は2026-08-08 22:13 JSTに完了した。module-level manifest helperを削除し、public `build_artifact()`内でowned buildを一度だけ実行してmanifest/validatorへ渡す。CLIはpublic APIだけを一度呼び、caller copyを拒否する。実装者のG2 focused 32件、contracts、matrix登録、実ambient Cargo build/query、host fail-closed、Rust MSRV/format、Python compile、diff checksはPASSした。同22:13 JSTから10〜15分予測の最終focused独立reviewを開始し、23:01:23 JST hard中断時刻は維持する。
- 同最終reviewの`read-only` sandboxはmain/active plan read後にbubblewrap `RTM_NEWADDR`が再発し、code/test確認前に2026-08-08 22:15 JSTに中断した。同時刻から同一非変更scopeの`danger-full-access` transport fallbackで継続し、予測と23:01:23 JST hard中断時刻はリセットしない。
- A3a最終focused独立reviewは2026-08-08 22:22:15 JSTに`PASS`で完了した。module-level bypass/helper不在、public `build_artifact()`とCLIのone-build境界、copied binary拒否、ambient `CARGO_TARGET_DIR`固定、G2 focused 32件、contracts/matrix、実Cargo build/query、host `HIP unavailable` fail-closed、Python/Rust/diff checksを独立確認した。A3aは18:01:23開始から約4時間21分で、予測上端4時間を約21分超えたが23:01:23 hard中断時刻前に完了した。canonical GPU数値実行はA5まで未実行である。
- A3b P0 host contract・実行経路は2026-08-08 22:23:00 JSTに開始する。予測2〜4時間、hard中断時刻は2026-08-09 03:23:00 JSTとする。P0 case-set、closed schema、runner、aggregate、versioned review disposition、host-only negative testを対象とし、GPU/model/cache/network/container、`rocm-smi`、deferred capsule、canonical P0数値実行は対象外とする。
- A3bの`workspace-write` sessionは必須文書read前にbubblewrap `RTM_NEWADDR`が連続し、file変更・testなしで2026-08-08 22:24:48 JSTに安全停止した。同22:25:04 JSTから同一scope・禁止事項の`danger-full-access` transport fallbackで継続し、A3b開始時刻、予測、2026-08-09 03:23:00 JST hard中断時刻はリセットしない。
- A3b実装は2026-08-08 23:06:36 JSTに完了した。P0 versioned matrix/review policy、7 closed schema、validator/runner/2-row aggregate、focused negative test、host suite/path/manifest登録を追加した。case-setは非整列`3x37`、locked hidden size 2560、B=256の255/256/257、5 warmup・21 measured iteration、kernel/wall median・MADを固定し、public RMSNorm source-set、BF16/F32/offset-one、candidate/artifact/model/dispatch/health/process identityへ結合する。P0 focused 18件、隣接H1/G1/G2 contract 20件、matrix/manifest、Python 87 file compile/static、diff checksはPASSした。A5 producer/parser、immutable artifact、canonical 2 GPU、実health/process、review dispositionは未実装・未実行で、それ以前のnumeric PASSは拒否する。同23:08 JSTから30分〜1時間予測のfresh独立reviewを開始し、A3b全体の2026-08-09 03:23:00 JST hard中断時刻は維持する。
- 同fresh reviewの`read-only` sandboxは初回一括readだけ成功したが表示出力が途中切断され、以後9回連続でbubblewrap `RTM_NEWADDR`により分割read・差分監査・testを開始できなかった。reviewerはcode不合格ではなく監査基盤障害として2026-08-08 23:12:26 JSTに`FAIL`で安全停止し、実装者の自己申告からPASSを推定していない。同23:13 JSTから同一非変更scope・禁止事項の`danger-full-access` transport fallbackでfresh reviewを再実行する。review予測30分〜1時間とA3b hard中断時刻2026-08-09 03:23:00 JSTはリセットしない。
- A3b fresh独立reviewは2026-08-08 23:23:51 JSTに`FAIL`で完了した。P0 focused 18件、隣接H1/G1/G2 static 28件、validator 3件、独立negative probe 15件、Python AST 8件、diff checksはPASSし、case/dtype/timing、median/MAD再計算、2-row exactness、A5以前のreport/aggregate PASS拒否、threshold・claim禁止、A5 producer/parser延期自体は成立した。一方、canonical `source_set_sha256`とP0 path ownershipが12 fileだけを対象とし、宣言するpublic pathのSemanticOp/Backend/Rust bridge/ABI bindings/CMake等少なくとも34 build inputを漏らすため、`crates/sllm-core/src/op.rs`変更後もdigestが変わらないことをsole P1 blockerとした。同23:24 JSTからこのsource closureだけを30〜60分で修復し、その後20〜40分のfresh再reviewを行う。A5 producer/parser、GPU/model/cache/network/container/capsuleへscopeを広げず、A3b hard中断時刻2026-08-09 03:23:00 JSTは延長しない。
- 同修復の`workspace-write` sandboxは必須文書read前にbubblewrap `RTM_NEWADDR`で全shell commandが停止し、file変更・testなしで2026-08-08 23:26 JSTに安全停止した。同23:27 JSTから同一scope・禁止事項の`danger-full-access` transport fallbackで継続する。修復予測30〜60分とA3b hard中断時刻2026-08-09 03:23:00 JSTはリセットしない。
- source closure修復は2026-08-08 23:43 JSTまでに45 source pathのversioned manifest、exact source-order/bytes digest、P0 path ownership、omission/reorder/path mutation/代表source変更のnegative testを実装した。focused P0 21件、隣接G1/G2 26件、manifest/Python static/diff checksは実装中にPASSした。一方、担当processが明示的な禁止範囲だった`rocm-smi`を起動した形跡を15分監視で検出したため、追加変更を許さず同23:43 JSTにprocessを中断した。`rocm-smi`は既に終了し、GPU kernel/model実行や成果物生成は確認していない。未完了の最終command自己申告には依存せず、現在のworktreeを20〜40分予測のfresh独立reviewで再検証する。A3b hard中断時刻2026-08-09 03:23:00 JSTは延長しない。
- fresh read-only reviewは2026-08-08 23:43〜23:51 JSTに現worktreeを直接監査し、修復process中断直前にmanifest/validatorがG2専用binary・G2 build manifestを加えた47 path案へ遷移した一方、focused testが45 pathかつ`src/bin`不在を要求する旧案のままである自己矛盾をblockerとして確定した。test起動前にbubblewrap `RTM_NEWADDR`が連続し、同時刻の監視でscope外の`rocm-smi` processも検出したためreviewerを停止した。同23:52 JSTからP0 host contract + existing public pathの45 pathへ戻してG1/G2専用producerを除外する整合修正だけを予測20〜40分で行い、substep hard中断時刻を2026-08-09 01:32 JSTとする。A3b全体の03:23 JST上限も維持する。
- 同整合修正の`workspace-write` sandboxは正本read途中からbubblewrap `RTM_NEWADDR`が連続し、編集・test前に2026-08-08 23:54 JSTで停止した。同23:54 JSTから同一file scope・禁止事項の`danger-full-access` transport fallbackへ切り替える。20〜40分予測、substep 01:32 JST、A3b 03:23 JSTの中断時刻はいずれもリセットしない。
- 45 path整合修正は2026-08-09 00:00:06 JSTに完了した。変更はP0 public-path manifest、P0 validator、P0 artifact schemaの3 fileに限定し、移行途中fieldとG2専用binary/build manifestをP0 identityから除外、45 pathのcanonical order digestとschema上限を一致させた。P0 focused 21件、隣接G1/G2 static 26件、P0/matrix/JSON validator、Python compile/static各87 file、diff checks、G1/G2 ownership分離probeはPASSした。同00:01 JSTから15〜30分予測のfresh独立reviewを開始し、review hard中断時刻を01:31 JST、A3b全体を03:23 JSTのままとする。
- fresh reviewの`read-only` sandboxは全指定fileを読了したが、test起動がbubblewrap `RTM_NEWADDR`で全件連続失敗し、出力更新も約2分停止したため2026-08-09 00:06 JSTに監査基盤`FAIL`として中断した。同00:07 JSTから同一非変更scope・禁止事項の`danger-full-access` transport fallbackでreviewを再実行する。15〜30分予測、review 01:31 JST、A3b 03:23 JSTの中断時刻はリセットしない。
- 訂正: 23:43および23:51監視で検出して担当process/reviewerによるscope違反と記録した`rocm-smi`は、Phase 3 processの子ではなく、作業開始前から別terminal配下で稼働するPID 16827の`watch -n 0.25 rocm-smi`が生成したprocessだった。従って両subagentが`rocm-smi`を起動したという判定を撤回する。23:43の実装停止自体は中断時点の47/45 path不整合をfresh reviewで確認済みであり、その後の45 path修正・再reviewを継続する。既存watchは本作業の所有外なので変更しない。
- A3b final fresh独立reviewは2026-08-09 00:15 JSTにblockerなしの`PASS`で完了した。P0 focused 21件、隣接G1/G2 static 26件、P0/matrix/JSON validator、Python compile/static各87 file、45 pathと5代表source mutationを含む独立negative/temp-copy probe、`git diff --check`がPASSした。GPU/HIP runtime、model/cache/raw slice、network/container、deferred capsule、broad host suite、Rust/native build、commit/pushは実行していない。A3bは22:23:00開始から約1時間52分で、2〜4時間予測内かつ03:23:00 hard中断時刻前に完了した。
- A4 immutable evidence用の最小baselineは2026-08-09 00:16:00 JSTに開始する。予測2〜4時間、hard中断時刻は2026-08-09 05:16:00 JSTとする。中断A0の未検証`execution_capsule.py`を継承・実行せず、trusted-development期間に必要な最小のhost evidence経路とcandidate identity境界をreviewして固定可能にする。GPU/model/cache/raw slice/network、canonical evidence、commit/pushはA4の対象外とする。
- A4 read-only調査は必須文書と一部worktreeを読んだ後、bubblewrap `RTM_NEWADDR`の反復と単純probe停止により変更なしで中断した。同一非変更scopeの`danger-full-access` transport fallbackは2026-08-09 00:29:10 JSTに完了し、現行runnerがHEAD/indexにない未追跡`execution_capsule.py`と`process_containment.py`をimportし、network guardもcapsule markerを要求するため、このままではA5へ進めないと判定した。最小修正は`ci/tools/run_host_suite.py`、`ci/tools/network_guard.py`、`ci/tests/test_fail_closed.py`のA0由来部分だけを外し、review済みpre-A0 direct runnerのregistered-command、network namespace、timeout/output/RSS/count、process-group cleanup、identity/aggregate境界を再利用する。実装45〜90分、fresh review 30〜60分とし、A4 hard中断時刻05:16 JSTは維持する。
- A4 direct baseline実装は2026-08-09 00:50:34 JSTに開始から34分34秒で自己検証PASSした。変更は`ci/tools/run_host_suite.py`、`ci/tools/network_guard.py`、`ci/tests/test_fail_closed.py`だけで、focused test 14/14、runner wrapper count 14/14、`self_test.py`、Python compile/static各87 file、matrix/JSON validator、diff check、禁止参照scanがPASSし、未検証`execution_capsule.py`と`process_containment.py`のSHA-256も不変だった。実装担当のPASSだけではA4を完了扱いにせず、30〜60分見込みのfresh独立reviewを継続する。A4 hard中断時刻05:16 JSTはリセットしない。
- A4 fresh独立reviewのread-only transportは、repository access前にbubblewrap `RTM_NEWADDR`で全commandが失敗し、2026-08-09 00:52:41 JSTに変更なし・未判定で終了した。これはcode findingではない。同じ非変更・host-only・offline範囲を`danger-full-access` transport fallbackで再実行し、review見込み30〜60分とA4 hard中断時刻05:16 JSTは維持する。
- A4 fresh独立reviewのfallbackは2026-08-09 01:04:21 JSTに開始から10分19秒で`FAIL`を確定した。capsule/containment参照除去、禁止2 fileのhash不変、focused 14/14、wrapper 14/14、self-test、matrix/JSON、Python compile/static各87、negative probe 10/10、diff checkはPASSした。一方、(1) 全`.py` commandをunittest候補とするため登録済み29 command中13 validatorを拒否する、(2) row output上限ちょうどで未完了のbreach flagがresult validatorと不整合、(3) network-isolation setupがcommand timeout外、(4) malformed route回帰caseを削り過ぎ、の4件をblockerとした。この4件だけを30〜60分で修復し、30〜60分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 上記4件のA4修復は2026-08-09 01:05:17 JSTに`workspace-write`で開始したが、readの一部だけが通った後、`apply_patch`がbubblewrap `RTM_NEWADDR`で起動拒否された。source変更前に停止し、同一2 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修復見込み30〜60分とA4 hard中断時刻05:16 JSTはリセットしない。
- A4 review blocker修復のfallbackは2026-08-09 01:19:17 JSTに開始から14分01秒で自己検証PASSした。変更は`run_host_suite.py`と`test_fail_closed.py`だけで、focused/wrapper 17/17、登録29 commandのunittest wrapper 13/direct 16（validator 13 direct）、self-test、matrix/JSON、Python compile/static各87、diff check、禁止参照0、禁止2 file hash不変がPASSした。実装担当のPASSだけでは閉じず、30〜60分見込みのfresh独立再reviewを続ける。A4 hard中断時刻05:16 JSTはリセットしない。
- A4 fresh独立再reviewは2026-08-09 01:30:12 JSTに9分25秒で`FAIL`を確定した。登録29 command分類、focused/wrapper 17/17、self-test、各validator、禁止参照/hash、独立negative 95件は成立したが、(1) HEADにあったIPv4 9項目・IPv6 8項目のsemantic route mutation回帰がtestへ復元されていない、(2) `verify_parent_restored()`がdeadlineを先に検査し、期限切れとparent namespace不一致が同時発生すると復旧失敗をmaskする、の2件が残った。この2件だけを15〜30分で修復し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- A4最終2 blocker修復は2026-08-09 01:37:20 JSTに開始から5分38秒で自己検証PASSした。IPv4 9項目・IPv6 8項目のsemantic route mutation、counter-onlyとmalformed route回帰を復元し、parent namespace不一致をdeadline切れより先に報告する復旧検査順序と両組合せのtestを追加した。変更は`ci/tools/network_guard.py`と`ci/tests/test_fail_closed.py`だけで、focused/wrapper 19/19、独立probe、self-test、matrix/JSON、Python compile/static各87、禁止参照0、禁止2 file hash不変、diff checkがPASSした。実装担当のPASSだけではA4を閉じず、同01:37 JSTから20〜40分見込みのfresh独立再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- A4最終fresh reviewの`read-only` transportは最初のsafe commandがshell起動前のbubblewrap `RTM_NEWADDR`で失敗し、文書・code・testを読めない監査基盤`FAIL`として2026-08-09 01:39:32 JSTに変更なしで停止した。これはcode findingではない。同一非変更scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- A4最終fresh reviewのfallbackは2026-08-09 01:49:16 JSTに約10分で`FAIL`を確定した。focused/wrapper 19/19、self-test、matrix/JSON、Python compile/static各87、登録29 command分類、route全semantic field、row境界、process-group cleanup、禁止参照/hashはPASSした。一方、(1) isolation内部probeとchild namespace検査の完了後にabsolute deadlineを再確認せず期限後の成功を許す、(2) registryにない`python -m unittest ci.tests.test_fail_closed`別名を受理する、の2系統をblockerとした。この2系統だけを15〜30分で修復し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2系統修復の`workspace-write` sessionは正本と対象codeを読了した後、最初の`apply_patch`がbubblewrap `RTM_NEWADDR`で対象fileを開けず、2026-08-09 01:55 JSTに変更・testなしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替え、修復15〜30分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2系統修復のfallbackは2026-08-09 02:10:19 JSTに開始から約15分19秒で自己検証PASSした。未登録module aliasをexact registry identityから除外し、isolation probe・child検査・外部接続検査・command wrap後にabsolute deadlineを再確認して期限後の成功とprocess launchを防止した。変更は`ci/tools/run_host_suite.py`、`ci/tools/network_guard.py`、`ci/tests/test_fail_closed.py`だけで、focused/wrapper 24/24、登録29 command（unittest 13/direct 16、validator 13 direct）、delayed deadline・route・row境界・process cleanupの独立probe、self-test、matrix/JSON、Python compile/static各87、禁止参照0、禁止2 file hash不変、diff checkがPASSした。実装担当のPASSだけではA4を閉じず、20〜40分見込みのfresh独立再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- A4 fresh独立再reviewの`read-only`実行は2026-08-09 02:20:30 JSTに`FAIL`を確定した。bubblewrap `RTM_NEWADDR`により実行testと保護file hashの独立再確認は不能だったが、静的監査で、(1) `run_bounded_process()`内の`Popen()`直前にabsolute deadlineを再確認せず期限後のchild起動を許す、(2) `child_main()`の最終deadline検査が`NetworkIsolationError`をcleanに捕捉しない、(3) registry外のdirect `python -m pytest`や`cargo`等を`execution_argv()`が受理する、の3件をblockerとした。この3件だけを20〜40分で修復し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同3 blocker修復の`workspace-write` sessionは対象確認後、全workspace commandがbubblewrap `RTM_NEWADDR`で起動不能となり、2026-08-09 02:24 JSTに編集・test・hash再確認なしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替え、修復20〜40分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 最初の同fallbackは自身をmain役と再解釈してread-only調査、続いて実装用`codex exec`を再帰起動し、対象sourceを編集せず入れ子化したため、2026-08-09 02:30 JSTにmainが中断した。3対象fileのhashは02:10版から不変である。実装担当自身が再委譲せず直接修正するよう役割を明示して同fallbackを再起動し、修復20〜40分とA4 hard中断時刻05:16 JSTはリセットしない。
- 直接実装担当による同3 blocker修復は2026-08-09 02:42:57 JSTに自己検証PASSした。変更は3対象fileだけで、host commandを登録29件の完全argv一致allowlistへ閉じ、既存absolute deadlineをbounded runnerへ渡して環境構築後・`Popen()`直前に期限切れをFAIL/timed-outへ変換し、child環境構築から`execvpe()`までのdeadline例外をcleanなexit 2へ変換した。`test_fail_closed.py` 27/27、登録29/unittest 13/direct 16/validator 13、未登録variant 57拒否、deadline/child独立probe、self-test、matrix/JSON、対象3 file compile、保護2 fileを除外したPython compile/static 85 file、禁止参照0、diff check、保護hash不変がPASSした。途中の通常Python validator呼出しは保護2 fileもAST read対象にした可能性があるためevidenceから除外し、最終束は明示除外して再実行した。実装担当のPASSだけではA4を閉じず、20〜40分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportはmain plan初回read後、続くcommandがbubblewrap `RTM_NEWADDR`で起動不能となり、2026-08-09 02:47 JSTに変更なし・code判定なしで停止した。同一非変更scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 02:55:24 JSTに約8分で`FAIL`を確定した。focused/wrapper 27/27、登録29 commandと未登録variant 136拒否、self-test、matrix/JSON、保護対象を除外したPython compile/static 85 file、child期限切れclean exit、route・row/resource境界、禁止参照0、保護2 file hash不変はPASSした。一方、`run_bounded_process()`の起動期限切れ例外で`verify_parent_restored()`が0回となり、parent namespace不一致が未確認・deadline FAILにmaskされることを独立再現した。またfork descendantがrunner return直後に一時観測され、process cleanupも完全には証明できていない。この2点を15〜30分で修正・決定的test化し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2点修正の`workspace-write` sessionは計画読取の一部だけ成功した後、bubblewrap `RTM_NEWADDR`が反復してsourceへ到達できず、2026-08-09 03:00 JSTに編集・testなしで停止した。同一2 file中心・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正15〜30分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2点修正のfallbackは2026-08-09 03:08:24 JSTに自己検証PASSした。`run_host_suite.py`と`test_fail_closed.py`だけを変更し、isolation plan成立後の親namespace復元確認を必ず1回実行して不一致を起動期限切れより優先し、process-group leaderをreapせず保持したまま残存group memberへTERM/KILL、残存時はfail-closedとした。focused/wrapper 29/29、self-test、matrix/JSON、保護対象を除外したPython compile/static各93、禁止参照0、diff check、保護hash不変がPASSした。実装担当のPASSだけでは閉じず、20〜40分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは全commandがshell起動前のbubblewrap `RTM_NEWADDR`で失敗し、2026-08-09 03:10:22 JSTに変更なし・code判定なしで停止した。同一非変更scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 03:23:01 JSTに10分13秒で`FAIL`を確定した。direct/wrapper 29/29、self-test、matrix/JSON、保護対象除外Python 85 file、allowlist未登録247拒否、route・schema境界、fork cleanup 12/12、leader未reap、保護hashはPASSした。一方、(1) `/proc`列挙・stat読取不能を空集合としてcleanup成功扱いする、(2) `Popen()`後selector初期化前の例外でchildを残す、(3) 同一PGIDのreparent済みdouble-fork descendant RSSを計上しない、(4) 通常execution timeoutとparent namespace不一致の同時発生で`INFRA_ERROR`を`FAIL`へ上書きする、の4 blockerを独立再現した。この4点を20〜40分で修正・negative test化し、20〜40分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同4点修正の`workspace-write` sessionは計画読取途中からbubblewrap `RTM_NEWADDR`が反復し、2026-08-09 03:26 JSTに編集・testなしで停止した。同一2 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正20〜40分、再review20〜40分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同4点修正のfallbackは2026-08-09 03:35:44 JSTに自己検証PASSした。`run_host_suite.py`と`test_fail_closed.py`だけを変更し、`/proc`列挙・stat異常をfail-closed、個別PIDの`ENOENT`/`ESRCH`だけを一時消失扱い、同一PGID全memberのRSS/stateをsnapshot集計、selector構築・登録失敗を含む`Popen()`後のcleanupとleader reap、parent namespace復元不一致の`INFRA_ERROR`優先を実装・回帰化した。direct/wrapper 35/35、登録29 command分類、self-test、matrix/JSON、保護対象を除外したPython compile/static各85 file、diff check、禁止参照0、保護2 file hash不変がPASSした。実装担当のPASSだけではA4を閉じず、03:36 JSTから20〜40分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは正本の初回read後、全shell再起動がbubblewrap `RTM_NEWADDR`で失敗しcode/test監査へ進めず、2026-08-09 03:38:57 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、20〜40分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 03:52 JSTに`FAIL`を確定した。前回4 blocker、direct/wrapper 35/35、self-test、matrix/JSON、保護対象除外Python compile/static各85、allowlist未登録535拒否、実`/proc` scan 200回、alternate double-fork RSS、output境界、禁止参照0、保護hashはPASSした。一方、(1) stdout/stderrの非一時的`EIO`をEOF扱いして出力と上限超過を失いexit 0にできる、(2) 空のIPv4 route入力をmissing headerとして拒否せず空snapshotとして受理する、の2 blockerを独立再現した。この2点だけを10〜20分で修正・negative test化し、15〜30分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2 blocker修正の`workspace-write` sessionは初回read前にbubblewrap loopback設定で失敗し、再試行も進捗なく2026-08-09 03:53:40 JSTに編集・testなしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正10〜20分、再review15〜30分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2 blocker修正のfallbackは2026-08-09 03:59:18 JSTに自己検証PASSした。非一時的pipe `OSError`を再送出し、`EAGAIN/EWOULDBLOCK`だけを一時状態として継続、IPv4空入力をmissing headerとして拒否し正しいheaderのみの空tableと区別した。EIO cleanup/reapと空routeのnegative testを追加し、direct/wrapper 37/37、独立repro各1、self-test、matrix/JSON、保護対象除外Python compile/static各85、allowlist 29/13/16/13、diff check、禁止参照0、保護hashがPASSした。実装担当のPASSだけではA4を閉じず、04:00 JSTから15〜30分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは正本初回read後のsandbox初期化が3回連続でloopback設定に失敗し、2026-08-09 04:01:03 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、15〜30分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 04:13 JSTに`FAIL`を確定した。既知6 blocker、direct/wrapper 37/37、allowlist 29/13/16/13、pipe EIO/EINTR/EAGAIN、procfs fail-closed、Popen後cleanup、double-fork RSS、route semantic field、restoration mismatch優先、self-test、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、保護hash不変はPASSした。一方、(1) sudo fallbackの新規network namespaceではloopback初期化前の`/proc/net/route`が正当に空であり、空入力拒否により実hostのnetwork isolation self-testを失敗させる、(2) 通常execution timeout後も期限切れdeadlineをparent restoration検査へ渡し、正常復元でも`FAIL`を`INFRA_ERROR`へ誤分類する、の2 blockerを独立再現した。この2点だけを5〜15分で修正・回帰化し、10〜20分のfresh再reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同2 blocker修正の`workspace-write` sessionはrepository読取前からbubblewrap loopback初期化に失敗し、代替読取確認も進捗しないため2026-08-09 04:15:53 JSTに編集・testなしで停止した。同一3 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正5〜15分、再review10〜20分、A4 hard中断時刻05:16 JSTはいずれもリセットしない。
- 同2 blocker修正のfallbackは2026-08-09 04:21:45 JSTに自己検証PASSした。sudo fallbackで固定system toolによりloopbackをupにしてから既存`setpriv`権限drop・capability除去・no-new-privilegesへ移行し、通常execution timeout後のparent namespace復元検査を期限非依存で必ず実行するようにした。正常復元は`FAIL/timed_out=true`、実際の不一致併発は`INFRA_ERROR`を維持する。変更は3対象fileだけで、focused direct/wrapper各39/39、実host network guard self-test、`self_test.py`、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、diff check、保護hash不変がPASSした。実装担当のPASSだけではA4を閉じず、10〜20分見込みのfresh独立reviewを行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportは正本初回read後、`true`を含む全shell起動がbubblewrap loopback初期化に失敗し、2026-08-09 04:24:24 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、10〜20分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 04:32:12 JSTに`FAIL`を確定した。通常timeoutの実host probeは`FAIL/timed_out=true/復元検査1回`、不一致併発の`INFRA_ERROR`優先、direct/wrapper 39/39、実host network guard self-test、self-test、matrix/JSON、allowlist変異132/132拒否、既知blocker回帰、保護対象除外Python compile/static各85、禁止参照0、保護hash不変がPASSした。一方、sudo fallbackのroot側`unshare/sh/ip/setpriv`をambient `PATH`の`shutil.which()`で選び、repository-controlled pathをroot実行prefixへ受理できる1 blockerを独立再現した。固定absolute system pathとownership/permissionをfail-closedに検査する最小修正を5〜10分、fresh再reviewを10〜15分で行う。A4 hard中断時刻05:16 JSTはリセットしない。
- 同1 blocker修正の`workspace-write` sessionは正本読取の一部とstatus確認だけ成功した後、bubblewrap loopback初期化が連続失敗し、2026-08-09 04:34:22 JSTに編集・testなしで停止した。同一2 file・同一禁止事項の`danger-full-access` transport fallbackへ切り替える。修正5〜10分、再review10〜15分、A4 hard中断時刻05:16 JSTはリセットしない。
- 同1 blocker修正のfallbackは2026-08-09 04:44:44 JSTに自己検証PASSした。sudo fallbackを固定absolute候補へ限定し、canonical実体、regular/executable、root所有、group/world非writable、trusted親directoryを検査し、symlink解決不能と異なるinodeへの候補分岐を拒否、同一inodeのsystem aliasだけを許可した。PATH改変、全5 tool missing、非root所有、group/world writable、symlink、非regular、inode曖昧性を回帰化した。変更は`network_guard.py`と`test_fail_closed.py`だけで、direct/wrapper 46/46、実host network self-test、self-test、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、diff check、保護hash不変がPASSした。10〜15分見込みのfresh独立reviewを行い、A4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewの`read-only` transportはmain plan初回read後の次commandがbubblewrap `RTM_NEWADDR`で失敗し、2026-08-09 04:46:04 JSTに変更なし・code判定なしで停止した。同一非変更・offline host-only scopeを`danger-full-access` transport fallbackで再実行し、10〜15分見込みとA4 hard中断時刻05:16 JSTはリセットしない。
- 同fresh独立reviewのfallbackは2026-08-09 04:52:52 JSTにblocker 0の`PASS`で完了した。固定absolute system tool、canonical実体、root ownership、permission、trusted parent chain、same-inode alias許可、異inode・missing・symlink異常・非regular・非executable拒否、adversarial `PATH`からroot prefixへのrepository path非混入を独立確認した。direct/wrapper各46/46、実host sudo network-isolation self-test、通常timeoutの`FAIL/timed_out=true`とparent復元検査1回、self-test、matrix/JSON、保護対象除外Python compile/static各85、禁止参照0、保護hash不変、diff checkがPASSした。A4は00:16:00開始から4時間36分52秒で、予測上端4時間を36分52秒超えたが、hard中断時刻05:16:00 JST以内に完了した。これにより中断A0を実行経路から外したtrusted-development baselineはreview済みとなり、候補identityを固定可能である。A5は同一immutable candidate SHAを必要とするため、local checkpoint commit等のidentity固定をユーザーが許可するまで開始しない。
- ユーザー指示によりA5 canonical 2 GPU evidenceを2026-08-09 13:43:40 JSTに開始した。予測3〜6時間、hard中断時刻は同日20:43:40 JSTで、再試行してもリセットしない。工程内をA5.0 candidate scope監査・local checkpoint固定（30〜60分、hard 15:43:40）、A5.1同一SHAのhost/H3/preflight（30〜60分、開始から2時間で中断）、A5.2 `gfx1030` evidence（45〜90分、開始から2時間30分で中断）、A5.3 `gfx1201` evidence（45〜90分、開始から2時間30分で中断）、A5.4 aggregate・前後health・独立review（30〜60分、開始から2時間で中断）へ分割する。各子工程のhard時刻とA5全体hard時刻の早い方で停止する。中断A0の未追跡`execution_capsule.py`と`process_containment.py`はcandidateへ含めず、読取・実行もしない。
- A5.0 candidate-scope reviewの`read-only` transportはbubblewrap loopback初期化に連続失敗し、2026-08-09 13:46 JSTに文書・diff未読、変更なしで停止した。同じ非変更scopeを`danger-full-access` transport fallbackで再実行し、A5.0 hard中断時刻15:43:40 JSTはリセットしない。
- A5.0 candidate-scope review fallbackは2026-08-09 13:53:56 JSTに、現行差分から`.gitignore`と中断A0の保護2 fileを除く155 pathをcheckpoint候補として`PASS`判定した。candidateはmodified 38、新規117、最大blob 173852 bytes、新規content 2699460 bytes、全candidate content 3523675 bytesでhygiene上限内、diff check、plan/history相互link、禁止path・artifact・credential scanがPASSした。`.gitignore`には`/passwords.txt` ignore削除、Phase 3外の`/.agents/skills/update/`追加、既存`.local-artifacts`規則再編が混在し、既存行変更の許可を確認できないためcandidateから除外する。元worktreeの`.gitignore`と保護2 fileを変更・削除せず保持し、155 pathだけをlocal checkpoint commitへ固定した後、そのSHAからclean linked worktreeを作成してstrict evidenceを取得する。A5.0 hard中断時刻15:43:40 JSTは維持する。
- A5の最終runtime candidateはcommit `ac2baa3a0734d0894353ba180259d979da5a831e`、tree `4e43a9c42c9aa2dfa6a6d438610fa54c4e482d10`として固定した。P0 Cargo buildへ900秒timeout、combined 4 MiB output上限、private session/process group、TERM・2秒grace・KILL、bounded leader reap、同一session/process-group消滅確認、resource単位の独立closeを追加した。required CPython 3.12.10を含むfocused 31件と、SIGKILL失敗時に主timeoutを保持しつつ残留memberを診断する回帰をPASSし、focused独立再reviewはaccepted scopeのhigh/medium 0件で`PASS`した。
- 同candidateのfresh host evidenceはH0 305/305、H1 151/151、H2 35/35で`PASS`した。固定ROCm containerのbase H3とRMSNorm H3はいずれもcanonical `gfx1030`/`gfx1201`の2 rowを`PASS`し、pre-GPU G0、private G1、controller-owned semantic G1、real-weight G2、P0、post-GPU G0も同じSHA/tree・canonical順で`PASS`した。G2はread-only 13-file cacheを再hashしてlocked 5120-byte slice SHA-256 `8104f6b0c777fd9bc60925f81a7179cfb7bf9621b4abf26a4d0f98b6e9a9bfe9`を使用し、各target 6 case・6 HIP dispatch・fallbackなし・health OK・process cleanを記録した。P0は各target 5 case・130 HIP dispatch・fallbackなし・health OK・process cleanで、threshold、最適化、他engine比較を主張しない`review_required` dispositionを`PASS`した。
- A5 review 9の最初のread-only transportはbubblewrap `RTM_NEWADDR`で全command実行前に停止したため判定へ使わず、同一非変更scopeのfresh `danger-full-access` transport fallbackを実行した。fallback reviewerはfull `986c8b86..ac2baa3a` 5-file差分、H0〜H3と全GPU aggregate、57 sidecar、G2/P0 validator、P0 cleanup、focused 15 test、`git diff --check`を独立確認し、2026-08-09 23:16 JSTにhigh/medium/low 0件の`PASS`を確定した。linked worktreeのremote/branch local Git configではsemantic G1 live authority再計算を意図どおり拒否したが、正式semantic G1 evidenceは当該configを除いたclean独立cloneでsealed controllerから生成され、embedded candidate、row、digestは一致した。
- 以上によりPhase 3 Stage Aを完了とする。適用・rollback境界は上記`ac2baa3a` runtime candidateとし、Phase 3全体は完了扱いにしない。手作業で再構成したlocal A5 commandはcontainer mount path、target別build root、numeric workflow run ID、short UNIX socket root、canonical JSON newline、builder-owned outputの現行contractとずれて複数回fail-closedになった。次のGPU evidence refresh前に、workflow/controllerを正本としてcommandを導出するtracked orchestrationまたはdry-run preflightを2〜4時間の独立作業単位で整備し、その後にStage BのRust model I/O・text frontendへ進む。`/tmp`の手書きrunbookは再利用しない。

| 作業単位 | 現在の状態 | 残る受入証拠 |
| --- | --- | --- |
| 0. 正本・schema・完了境界 | 完了 | 最終candidateのidentityと結果をhistoryへ同期済み |
| 1. 完全model lock | 完了 | verified read-only 13-file cache、locked slice、最終candidate bindingがPASS |
| 2. reader記録・model contract | 固定llama.cpp/vLLMのreader記録と採否を作成済み | 後続op追加時の差分追記 |
| 3. config・safetensors host基盤 | 完了 | hash済みFD、tiny negative、実cache/slice bindingがPASS |
| 4. RMSNorm contract・oracle | 完了 | BF16/FP32 offset-oneとcanonical GPU照合がPASS |
| 5. public HIP実行基盤 | 完了 | 同一candidateのH3/G0/private G1回帰がPASS |
| 6. baseline RMSNorm kernel・semantic G1 | 完了 | exact 2 targetのH3、sealed-controller G1、aggregateがPASS |
| 7. real-weight G2 | 完了 | dedicated binary、実slice、2 GPU 12 case、aggregateがPASS |
| 8. RMSNorm P0 | 完了 | 2 GPU 10 case・260 dispatch、review disposition、aggregateがPASS |
| 9. 適用・終了判定 | 完了 | post-GPU health/process、独立review 9、正本同期がPASS |

本計画はhistoryとともにarchiveした。次の順序は、(1) GPU evidence commandのtracked orchestration/dry-run preflightを独立作業単位で整備、(2) Phase 3全体計画のStage B Rust model I/O・text frontendへ進む、とする。Stage A runtime byteを変更する場合は`ac2baa3a`のevidenceを流用せず、新identityに対して影響gateを最初からやり直す。

## 範囲

含むもの:

- `Qwen/Qwen3.5-4B`の解決済み完全commit SHAと完全model lock。
- config、safetensors index、全weight shard、tokenizer、chat template、generation config、license/model card等、将来の初期縦切りが消費する全ファイルの固定。
- 固定したllama.cppとvLLMからのreader記録と、実装者へ渡すコード表現を含まない技術要点。
- model configとsafetensors metadataのRust parser、hash・dtype・tensor集合のfail-closed検証。
- 最初のsemantic opとしてのRMSNorm contract、NumPy oracle、BF16 baseline HIP kernel。
- public inference側のcontext、buffer、queue/completion、submitに必要な最小C ABIと安全なRust wrapper。
- synthetic tiny fixtureによるH1/H2と、固定model cacheから抽出するreal-weight sliceによるG2。
- private diagnostic G1とは別のsemantic RMSNorm G1 schema/runner/aggregate。
- exact `gfx1030`/`gfx1201`のH3、G0、private diagnostic G1、semantic RMSNorm G1、G2、RMSNorm P0、同一candidate集約、実行後health確認。

含まないもの:

- full modelのload・forward・文章生成、G3、Chat Completions API。
- tokenization/chat templateのruntime実行。これらのファイル固定と構造検証だけを行う。
- embedding、Matmul、RoPE、attention、MLP、logits、sampling。
- KV cache、linear/recurrent state、prefill/decode、scheduler。
- vision、MTP、MoE、speculative decode、複数request、複数GPU。
- optimized RMSNorm、kernel自動選択の性能tuning、P1、llama.cpp性能比較。P0は短いRMSNorm実行経路smokeだけを含む。
- Qwen3.5-2B/9B、他model、他backend、generic target、対応GPUのlifecycle昇格。
- full weight、raw slice、trace、binaryをGitまたはGitHub Actions artifactへ保存すること。

## Phase 3で固定する決定

### 完了境界

Phase 3 Stage Aの完了点を「完全model lock + public RMSNorm + semantic G1 + real-weight G2 + P0 smoke」とする。G2/P0の成功は一つのmodel由来opの数値正しさと短い実行経路観測だけを証明し、full model推論、文章生成、性能最適化、一般的なGPU対応を証明しない。

このmain plan上のPhase 3 Stage Aは、CI正本の実装段階`Phase 3: GPU runner基盤`とは別の作業区分である。CI側のrunner基盤は完了済みであり、この計画はCI実装段階Phase 4/5に相当するruntime・kernel・model sliceを進める。

H3 required昇格の20回以上・7日以上の観測は並行follow-upとし、この計画の開始条件・完了条件に含めない。

### 当面の実行trust boundary

2026-08-08から今後数週間は単独maintainerのtrusted development期間とし、local/GPU実行をmaintainerが確認したcodeと明示commandに限定する。外部PR、fork由来code、未review script、第三者binaryは専用hostで実行しない。この限定期間では、悪意ある同一UID processやhostile persistent runnerに耐えるcustom capsuleの完成をStage Aの前提から外す。

secret・Docker socketを渡さないこと、可能な範囲のcontainer/network隔離、timeout・resource上限、process cleanup、実行前後GPU health、immutable candidate/artifact identityは維持する。外部codeを実行する前、または複数のtrust boundaryへ移る前には、ephemeral VM/JIT runnerまたはjob後reimageをhard gateとして導入し、CI正本とPhase 3全体計画を更新する。

中断されたcustom capsule hardeningは放棄し、過去SHAへの完全一致復元を要求しない。local-developmentでは当該部分変更を実行経路から外してdirect testと標準containerを使い、immutable evidence取得前にcleanな最小baselineを新規作成してreviewする。

### 最初のsemantic op

最初のopはRMSNormとする。理由は、Qwen系の実weightを使い、reduction、scale、BF16入出力、非整列shapeを含みながら、GEMMやattentionより小さな独立単位としてcontractとGPU経路を検証できるためである。

baseline contractは次とする。

- input activation、raw scale weight、outputはBF16。
- accumulationと平方和はFP32。
- Qwen3.5 HF weight semanticsはoffset-one variantとし、実効scaleをFP32で`1 + raw_weight`として適用する。raw checkpoint weightを通常scaleとして直接乗算せず、事前変換も行わない。
- epsilonはlocked configから取得し、暗黙の既定値へfallbackしない。
- 初期layoutはrow-majorで最終次元が連続。rank、stride、alignment、alias可否をdescriptorで明示する。
- outputはinputと同shape。scaleは最終次元と同じ要素数。
- epsilon、NaN/Inf、zero-length、overflow、alias、unsupported stride/dtypeの扱いをcontract testで固定する。
- in-place対応は初期範囲に含めず、入力・出力aliasは明示的に拒否する。
- exactな隠れ次元とtensor名はmodel lock/config確定後に埋め、推測で固定しない。
- rank 1〜8の連続した最終次元を`N`、全leading dimensionの積をrow数`R`としてflattenする。Phase 3 model pathのbatch=1制約を公開RMSNorm opのshape制約へ流用しない。
- prepared planは再利用可能とするが、同一planのin-flightは1件だけ許可する。completionがterminalになる前の再executeとplan releaseは`PUBLIC_BUSY`で拒否し、pointerや所有権を消費しない。
- nonfinite activation/raw scaleはhost scan、reject、sanitizeせずIEEE演算へ渡す。NaN payload bitの一致は要求せず、NaN/Inf classificationだけを検証する。epsilonは引き続きfiniteかつpositiveを必須とする。

固定revisionの実configがこのRMSNorm contractと整合しない場合は、別opへ黙って差し替えず、この計画とmain planをreviewして更新する。

### 採用model revision

- `repo_id`: `Qwen/Qwen3.5-4B`
- `requested_revision`: `main`
- `resolved_revision`: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 解決日: 2026-08-04
- 解決元: Hugging Face公式model APIと同じ完全SHAのimmutable tree/download endpoint。

このrevisionを、後で動くbranch名へ追従させずPhase 3のmodel identityとして固定する。公式metadata上はApache-2.0、base modelは`Qwen/Qwen3.5-4B-Base`、top-level architectureはmultimodalな`Qwen3_5ForConditionalGeneration`である。Phase 3はtext-onlyなので、top-level configと全shardをlockしたうえで`text_config`とtext RMSNorm weightだけを実行対象にし、vision config/tensorは既知だが未消費の集合としてmodel contractへ列挙する。未知tensorとして黙って無視せず、vision対応済みとも表記しない。

このrevisionには`generation_config.json`が存在しないため、placeholderや別revisionのfileを追加しない。Phase 3で実行しないtokenizer、chat template、image/video processorもrevision identityと将来の初期縦切り入力としてlockするが、runtime対応を意味しない。

### model lockとmodel slice

- branch/tagだけで固定せず、解決済み40桁commit SHAを使う。
- lock対象は実際に消費する全ファイルとし、各ファイルにsize、実bytesのSHA-256、Git blob ID、LFS OID、immutable download locatorを記録する。
- license、model card、base model、変換系列のevidenceもlockへ含める。
- lock fingerprintはRFC 8785 JCSで正規化したschema versionとmodel本体から計算する。versioned generated-token停止policyを含む現行Qwen lock fingerprintは`sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935`とする。
- 既存CI helperの通常のkey sort JSON serializationをJCSとして流用しない。JCS実装または独立validatorを追加し、既知fixtureで相互検証する。
- upstream dtype、architecture、layer schedule、RMSNorm epsilon、weight tying、vision/MTP/MoE/custom code有無は実configとtensor indexから確定する。
- 未知architecture、未review custom code、missing・duplicate・unexpected・unconsumed tensor、hash/dtype不一致を黙って無視しない。
- G2 sliceはGPU hostのread-only model cacheから実行時に抽出する。repositoryには抽出recipe、source lock fingerprint、tensor名、offset・shape、tool/script hash、引数、出力size/SHA-256だけを保存する。
- activationは固定seedの独立生成fixtureとし、model outputや他engineの出力を期待値として流用しない。

### public実行経路

private G1 diagnostic ABIをsemantic opへ昇格させない。RMSNormは`SemanticOpDescriptor -> Backend -> sllm-hip -> versioned public C ABI -> native HIP registry -> baseline kernel`を通す。synthetic RMSNormの数値結果は専用semantic G1へ記録し、private G1 reportへ継ぎ足さない。

- Rustがmodel/config、tensor view、semantic op、backend選択、completion lifetimeを管理する。
- native側はcontext、allocation、queue/event、copy、dispatch、kernel registryを管理する。
- public C ABIはopaque handle、固定幅整数、status、caller-owned error sink、`struct_size`/versionを維持する。
- 既存ABI v1を壊さない。additive変更で済まない場合はversionを上げ、header、checked-in bindings、layout probe、C/C++ testを同じcandidateで更新する。
- capability queryまたはprepareでdtype、layout、shape、targetを検証し、実行時の別backend、CPU、generic kernelへのfallbackを禁止する。
- completion前のbuffer/queue/event解放を禁止し、timeout・early drop・error時はPhase 2のbounded reaper/circuit breaker契約を維持する。
- executeはadditiveな`sllm_rmsnorm_execute`とversion 1のdispatch infoを追加し、既存のgeneric `sllm_completion_query` / `wait` / `release`を再利用する。RMSNorm専用completion queryは追加しない。
- dispatchはcontext単位のnon-zero monotonic ID、count 1、kernel ID 1、`rmsnorm.baseline.wave32.v1`、device symbol `sllm_rmsnorm_baseline_wave32_v1`、workgroup 256、fallback allowed/used falseを報告する。
- baseline capabilityは`N <= 4096`、`R <= UINT32_MAX`、wave32、`grid.x=R`、`block.x=256`とする。`N=4097`またはlaunch上限外はcompletionを作らず`UNSUPPORTED`とする。

### semantic G1 compiler exact-action input boundary

- compiler clientの認証環境と、最終`amdclang++`環境は別の対象である。clientのsocket/session/token/FD認証値はparentが観測・認証して発行とexecuteを同じclient observationへ結び付けるが、compiler action manifestへは入れない。
- action manifestは最終sealed compilerへ`execveat`する完全な環境、argv、cwd、出力と入力を保持し、spawn時に環境を削除・追加・変換しない。結果を受領・検証したclientはHMAC認証済みACKを返し、action eventはACK frame hashとacknowledged stateを保持する。ACKのない結果はcomplete evidenceではない。
- exact input closureはsealed compilerの同一環境・compile flagでの`-M` preprocessor discoveryが返す全header、sealed driverが返すresource directory配下の全regular file（builtin header/configを含む）、AMDGPU device bitcode全体、compiler driverの再帰dynamic-loader closureから作る。compiler環境からHOME/XDG/Clang config/resource/include/executable/library search overrideを排除するため、これ以外のmutable config選択は許可しない。各fileはmanifestへhash/device/inode付きで入れ、最終spawn直前にlive再検証する。resource/dependency discoveryが不正、closureが空、入力が消失/置換/symlink化した場合はbuildをfail-closedにする。

### 数値正しさ

- Python+NumPyの独立oracleを正とし、PyTorchを使わない。
- oracleはFP32 accumulationを明示し、BF16のdecode/roundingを独立実装または既存の小さな検証済みfixtureで確認する。
- exact metadataとstatusはexact match、数値値はop・shape・入力範囲・accumulation・出力dtype別のtoleranceで比較する。
- 初期acceptance budgetはGPU結果を見る前に`tolerance_id=rmsnorm-bf16-f32-output-v1`、`atol=0.0078125`、`rtol=0.015625`として固定する。finite BF16 outputをFP32へdecodeして`abs(actual-reference) <= atol + rtol*abs(reference)`で比較し、NaN/Infはclassificationを比較する。calibrationで誤差分布を記録するが、同一candidateの結果に合わせた事後的な拡大は禁止する。
- caseには1行だけ、複数行、非2冪・非整列の最終次元、dispatch境界`B-1/B/B+1`、小値・大値・zeroを含める。locked modelの実隠れ次元も必須caseとする。

## 作業単位

### 0. 正本・schema・完了境界の同期

主な変更対象:

- `docs/plans/main-plan.md`
- この計画と対応history
- `docs/plans/active/2026/08/1-10/ci-test-strategy.md`
- 必要な場合だけruntime、model lock、compatibility正本

実施内容:

1. Phase 3 Stage Aの完了点と非対象を正本へ反映する。
2. G2をmodel path変更のhard gateとして具体化し、public HIP runtime/kernelに必要なP0 smokeを残しつつ、G3/G4/P1を誤って要求しないscope別gateを確認する。
3. `weight/activation=BF16`と、後続MVPの`KV=FP16`を混同しない。
4. G2/P0だけでfull model、性能最適化済み、または対応GPUの表記を昇格させない規則を固定する。
5. private diagnostic G1とsemantic RMSNorm G1のschema/evidenceを分離する。

受入条件:

- main plan、CI正本、runtime、model lockのPhase 3境界に矛盾がない。
- 未実装のG2/P0をそれ自身のbootstrap candidateへ循環要求せず、RMSNorm/model pathの最終candidateからsemantic G1/G2/P0が省略されない。
- G2/P0 schema/runnerの非実行contractだけを初めて構築するcandidateはH0〜H2とhost negative self-testを必須とする。初回enablement candidateはH0〜H2、同一candidateのH3 PASS evidence、canonical G0/private diagnostic G1/semantic RMSNorm G1/G2/P0を必須とし、canonical aggregate確立後のmodel path・schema・runner・tolerance変更からsemantic G1/G2/P0をhard gateにする。
- H3は最終candidateの必須PASS evidenceとするが、branch protection上のrequired check昇格は20回以上・7日以上の観測条件に従う。
- model lock fingerprint、reviewed SHA/tree、artifact/report digest、tuple digestを別identityとして保持し、一つのaggregate run graphへ結び付ける。

### 1. Qwen3.5-4B revision解決と完全model lock

実施順序:

1. upstreamのmodel page、license、base model、採用済みrevision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`を人間がreviewできる資料として取得する。
2. requested revision `main`が解決日に上記40桁commit SHAへ解決されたevidenceを保存し、以後の取得は完全SHAだけを使う。
3. config、generation config、tokenizer、chat template、safetensors index、全shard、license/model card、実際に参照する追加fileを列挙する。
4. immutable locatorから取得した実bytesのsize/SHA-256、Git blob ID、LFS OIDを記録する。
5. JCS fingerprintを生成し、alias `qwen3.5-4b-bf16`をそのfingerprintへ結ぶ。
6. offlineでlock、cache、全hashを再検証できるvalidatorを作る。

lock確定時に明示する事項:

- exact architecture/model typeとlayer schedule。
- hidden size、layer数、attention/KV head数、intermediate size、RMSNorm epsilon。
- upstream weight dtypeとsafetensors tensor集合。
- vision、MTP、MoE、custom code、unused/optional tensorの有無と拒否方針。
- tokenizer/chat template/special token/EOS等はlockするが、この計画ではruntime対応しないこと。

受入条件:

- floating revision、placeholder、illustrative shard名が残っていない。
- file欠落、byte改変、wrong commit、wrong LFS OID、fingerprint不一致、未知fileをnegative testが拒否する。
- full modelをGit管理せず、CPU CIがdownload/loadしない。
- lockのlicense/base-model evidenceがreview可能である。

rollback境界:

- model lockとvalidatorを独立candidateにする。後続実装開始後にrevisionを変更する場合は別modelとして新fingerprintを作り、旧lockを書き換えない。

### 2. 固定参照実装のreader記録とmodel contract

reader担当は、固定manifestのllama.cppとvLLMから次だけを調査する。

- config validation、tensor name mapping、safetensors shard解決。
- RMSNormのsemantic contract、epsilon、accumulation、layout、境界case。
- Qwen3.5固有のlayer schedule、未知tensor、optional componentの扱い。
- model slice testとhardware別testの分割方法。

利用規則:

- vLLM等はreaderとimplementerを分離し、コードのcopy/adapt/portをしない。implementerへはコード表現を含まない要点だけを渡す。
- llama.cppから直接流用する場合は実装前にreuse modeを決め、`docs/provenance/README.md`に従ってnotice、source blob/hash、local path、変更、import commitを記録する。
- 固定checkoutにQwen3.5対応がない、または実modelと矛盾する場合は、その事実を記録して一次sourceのconfig/model cardを優先する。

受入条件:

- 採用・不採用判断と根拠が再確認できる。
- vLLM等のコード表現がimplementerの成果物へ混入していない。
- model contractがlock済みconfig/tensor indexと一致する。

### 3. model lock・config・safetensors host基盤

主な責務候補:

- `sllm-core`: model lock型、config型、tensor catalog、model alias/fingerprint、backend非依存error。
- model I/O用の小さなRust moduleまたはcrate: safetensors metadata/index解決、mmap/read-only byte range、hash検証。
- `sllm-cli`: offline verifyとslice抽出の入口。文章生成CLIとは分ける。

実施内容:

1. model lockを厳密にparse・validateし、unknown schema/versionを拒否する。
2. configからRMSNormに必要な値を型付きで取得し、missing/unknown/out-of-rangeを拒否する。
3. safetensors indexとshard metadataを照合し、tensor名、dtype、shape、offset、byte rangeを検証する。
4. 必要なRMSNorm weightだけをread-onlyで取得し、full shardの複製やCPU CIでのfull loadを行わない。
5. architecture外のtensorを黙って無視せず、expected/optional/rejectedを明示する。

H1/H2 fixture:

- repository内には数KiB級のsynthetic config、lock、safetensorsだけを置く。
- missing shard、overlap/out-of-range offset、wrong dtype/shape/hash、duplicate/unconsumed tensor、unknown architectureをnegative testに含める。
- full Qwen modelや実weight sliceをfixtureにしない。

受入条件:

- offline H1/H2が時間・size予算内でPASSする。
- parserがnetwork、custom code実行、CPU fallbackを行わない。
- model I/Oとsemantic op/backendの責務が分離され、後続model/opを追加するためにQwen固有分岐をGPU kernelへ埋め込まない。

### 4. RMSNorm semantic contractと独立oracle

実施内容:

1. `SemanticOpKind`とdescriptorへRMSNormを追加し、入力2、出力1、epsilon、layout、alias contractを表現する。
2. capability query、prepare、executeが同じvalidation規則を使う。
3. fake backendはmetadata/control-planeだけを検証し、数値成功を返さない。
4. NumPy oracleとcase generatorを追加し、BF16 encode/decode、FP32 accumulation、output比較を独立に実装する。
5. locked model shapeを含むcase manifestと、非2冪・非整列・境界前後のtiny caseを作る。

受入条件:

- arity、shape、dtype、stride、epsilon、alias、zero/overflow、unsupported encodingのnegative testがある。
- H2はtiny oracleだけを実行し、GPU kernelをCPUで模倣しない。
- tolerance候補、測定方法、NaN/Inf比較規則がmachine-readableである。

rollback境界:

- model parserと切り離したsemantic contract candidateとする。descriptor変更で既存Copy/Add/Matmul contractを破壊しない。

### 5. public HIP実行基盤

実施順序:

1. public context、buffer、queue/completionに必要なC ABIを設計し、ABI compatibility判断を記録する。
2. native allocation/copy/submit/completionをpublic inference pathへ実装する。
3. Rust wrapperがopaque handleと使用bufferをcompletionまで強参照する。
4. `Backend::supports/materialize/execute`をHIP backendへ接続し、host stubは明示`HIP unavailable`を維持する。
5. private G1は診断用として残し、public opと別々のevidenceとして継続検証する。

必須negative test:

- null、wrong ABI/struct size、reserved非zero、invalid/double destroy handle。
- wrong device/target/artifact、unsupported dtype/layout/shape、zero dispatch。
- timeout、early drop、copy/dispatch failure、error sink truncation。
- completion前解放、reaper上限、circuit breaker、process残留。

受入条件:

- C header、checked-in Rust bindings、layout probe、C/C++ compile testが一致する。
- CPU-only build/testはHIP unavailableを返し、GPU成功を装わない。
- G1の既存contractとcleanup/health evidenceが回帰しない。

### 6. baseline BF16 RMSNorm HIP kernel

実施内容:

1. semantic RMSNormをnative op registryとHIP kernel registryへ登録する。
2. correctness優先のbaseline kernelをexact `gfx1030`/`gfx1201`向けにbuildする。
3. FP32 reduction、epsilon、BF16 rounding、tail処理をcontractどおり実装する。
4. target、shape、alignment、layoutに適合しない場合は明示unsupportedとし、別kernelやCPUへfallbackしない。
5. dispatch ID、kernel ID、dispatch count、artifact hashをreport可能にする。

受入条件:

- exact 2 targetのH3 artifact検査がPASSする。
- tiny synthetic caseを両GPUでoracleと比較し、locked model shapeと`B-1/B/B+1`を含む。
- 数値結果をprivate diagnostic G1へ混ぜず、semantic RMSNorm G1の2 row aggregateまでPASSさせる。
- unsupported条件、wrong target、zero dispatch、artifact差し替えが非PASSになる。
- このbaselineをoptimizedまたは高性能と表記しない。

rollback境界:

- public runtime基盤とkernelを可能な限り別candidateにし、kernel無効化でmodel-free G1とhost pathを維持できるようにする。

### 7. real-weight sliceとG2 evidence

主な変更候補:

- model lock、model slice、G2 runtime artifact、G2 report、G2 aggregate、op別toleranceのversioned schema。
- canonical 2 rowだけを列挙する`ci/matrix/g2-runtime-v1.json`と、順序付きcase-set。
- slice準備、G2 artifact build、contract検査、trusted local実行、2 row集約の各tool。
- suite registry、path-to-suite mapping、schema/matrix/runner/aggregateのhost-only negative test。

private diagnostic G1 reportへmodel provenanceと数値項目を継ぎ足さず、semantic G1とG2の専用reportを正本とする。既存private G1は同一candidateで再実行する前提evidenceとしてdigestを参照する。

G2入力:

- source: 固定model lock fingerprintに一致するread-only model cache。
- weight: `model.language_model.layers.0.input_layernorm.weight` 1件。index上のsource shardは`model.safetensors-00002-of-00002.safetensors`、shapeは`[2560]`、dtypeはBF16、epsilonは`text_config.rms_norm_eps=1e-6`、scale modeはoffset-oneとする。safetensors header lengthは79064 bytes、tensor data offsetsは`[15360, 20480]`、file absolute byte rangeは`[94432, 99552)`であり、5120 bytesと一致しなければ実行しない。
- activation: 固定seed、shape・値域が明示された独立synthetic BF16 input。
- oracle: 抽出raw weightと同じactivationを使い、FP32で`1 + raw_weight`を適用するNumPy FP32 accumulation RMSNorm。

G2 reportに追加する項目:

- source model lock fingerprint、resolved model commit SHA。
- tensor名、source shard、offset、shape、dtype、抽出recipe/tool/script hash、全引数。
- slice pathではなくslice size/SHA-256。raw sliceはlocal artifactとする。
- epsilon、input seed/generator/version、input/output hash。
- oracle version、case ID、比較metric、tolerance ID、max/mean error、NaN/Inf count。
- tolerance policyのversion、content SHA-256、校正candidate、承認者・日時・根拠。G2結果を見て同じcandidateの閾値を拡大しない。
- reviewed/tested/workflow SHA、tree OID、tuple digest、GPU UUID/BDF、exact target。
- selected backend、op/kernel/dispatch IDとcount、artifact hash、fallback allowed/used。
- G2 runtime binary/embedded code object/sidecarのSHA-256、H3と共通のsource candidate・toolchain・exact target。H3/G1 binaryのhashをG2 runtime artifactとして代用しない。
- `model_used=true`、`full_model_used=false`、`tokenizer_used=false`、`generation_used=false`。
- 実行前後health、開始/終了時刻、cleanup/process残留。

fail-closed条件:

- lock/cache/slice/artifact/candidate hashの不一致。
- tensor missing・wrong dtype/shape、抽出範囲外、unknown tolerance。
- `selected_backend != hip`、dispatch 0、fallback使用、skip、0件収集。
- oracle mismatch、NaNによる比較回避、timeout、crash、GPU/child process残留、health悪化。
- canonical 2 rowのmissing/duplicate/stale/別candidate混在。

受入条件:

- 同一immutable candidateとmodel lockに対し、canonical `gfx1030`/`gfx1201`のG2が直列にPASSする。
- raw model/sliceをGit、CI artifact、reportへ埋め込まない。
- G2 aggregateがG0、private diagnostic G1、semantic RMSNorm G1、H3 artifact identity、model lock fingerprint、RMSNorm resultを同一identityへ結び付ける。

導入順序:

1. schema、matrix、case-set、negative testをhost-onlyで確立する。
2. dedicated G2 binaryをbuildし、H3/G1 binaryの差し替えを拒否する。
3. canonical 2 GPUでnon-required calibrationを行い、誤差分布と境界caseを確認する。
4. tolerance policyをversioned fileとして固定する。candidate固有に閾値を緩めてPASSへ変えない。
5. 同一candidateのG0/private diagnostic G1/semantic RMSNorm G1を再実行し、canonical G2 2 rowとaggregateをPASSさせる。

### 8. RMSNorm P0 smoke

主な変更候補:

- RMSNorm専用P0 case-set、report、2 row aggregate、versioned review disposition。
- kernel/wall latency、warmup、反復回数、median、robust spread、dispatch/kernel ID、artifact hashを記録するrunner。

実施内容:

1. semantic G1/G2と同じpublic RMSNorm pathを使い、synthetic非整列shape、locked model hidden size、合法なdispatch境界`B-1/B/B+1`を測定する。
2. canonical `gfx1030`/`gfx1201`を直列実行し、同じreviewed/tested/workflow SHA、tuple、artifact、model lock、dtype、case-setへ固定する。
3. 承認済みperformance thresholdがない間は`review_required`とし、reviewer、理由、日時、三点の完全な実測値なしにPASS集約しない。
4. P0結果を性能最適化済み、llama.cppより高速、またはperformance hard gate確立済みという主張へ使わない。

受入条件:

- 欠落、重複、stale、別candidate、非GPU、dispatch 0、fallback使用、不正な時間、artifact不一致、health悪化を拒否する。
- canonical 2 rowのP0とreview dispositionが同一immutable candidateでaggregateまでPASSする。

### 9. 適用・文書同期・終了判定

1. 各作業単位を独立candidateとしてreviewし、必要なhost evidenceを得てから次へ進む。
2. 最終candidateのH0〜H3、canonical 2 GPUのG0、private diagnostic G1、semantic RMSNorm G1、G2、P0、schema/negative test、build/lintを全て成功させる。
3. 同じcandidateを専用local hostへ適用し、G2/P0 smokeと実行後health/process確認を行う。
4. commit SHA、tree OID、model lock fingerprint、artifact/report digestをhistoryへ記録する。
5. runtime、compatibility、model lock、CI正本、main planを実装事実と検証scopeへ同期する。
6. 完了後、attention・MLP・KV/state・prefill/decode・G3の次計画を作る。

## 推奨candidate分割

1. Phase 3計画・model lock schema/validator。
2. 確定Qwen3.5-4B lockとreader/model contract。
3. config・safetensors host基盤とtiny fixture。
4. RMSNorm semantic contract・NumPy oracle。
5. public HIP runtime ABI/wrapper。
6. baseline RMSNorm kernel・synthetic実GPU数値test。
7. G2 slice extractor・schema/runner/aggregate。
8. RMSNorm P0 schema/runner/aggregate。
9. 同一identityのcanonical semantic G1/G2/P0、文書同期、完了記録。

candidateを整理・squashしてidentityが変わった場合は、影響範囲のtestとGPU evidenceを新identityに対してやり直す。

## 全体の完了条件

- Qwen/Qwen3.5-4Bのplaceholderなし完全model lockとfingerprintが存在する。
- 固定参照実装のreader記録、採用判断、provenance境界がreviewできる。
- config/safetensors parserがoffline・fail-closedにRMSNorm weightを解決できる。
- RMSNormのsemantic contract、capability、NumPy oracle、boundary case、toleranceが固定されている。
- public Rust/C ABI/native HIP経路でbaseline RMSNormを実行でき、private G1を流用していない。
- exact `gfx1030`/`gfx1201`の同一candidate H3、G0、private diagnostic G1、semantic RMSNorm G1、G2、P0がfail-closed aggregateまでPASSする。
- G2がreal weight、BF16 activation、FP32 accumulation、fallbackなし、dispatch 1件以上を証明する。
- P0がRMSNormの短い境界観測、dispatch/artifact identity、versioned review dispositionを証明し、性能最適化済みとは主張していない。
- 実行後healthとprocess cleanupが正常で、raw model/slice/traceがGit管理されていない。
- compatibility表記は限定scopeの`experimental`を維持し、full model推論・文章生成・性能を主張していない。

## rollback・fail-stop

- model lock、host loader、semantic contract、public runtime、kernel、semantic G1、G2、P0 runnerを独立してreview・rollbackできる単位にする。
- test、GPU適用、適用後healthのいずれかが失敗したcandidateはpushせず、直前の検証済みrevisionを維持する。
- host pathまたはmodel-free G1を偽のG2 fallbackに使わない。G2失敗時はRMSNorm/model対応を未完了とする。
- device health異常、process残留、resource回収不能、rollback不能時は追加GPU実行を停止し、host状態、最後の正常revision、未適用範囲を報告する。
- upstream revisionを変更する場合は新lock fingerprintとし、旧evidenceへ混ぜない。

## 未確定事項

- Stage Bでlock済み738 tensorをmain text required、vision/MTP known-unconsumed、config-conditional、rejectedへ分けるexactなmachine-readable分類。
- Stage B以降のpublic C ABI追加をv1 additiveにできるか、version更新が必要か。
- 次回GPU evidence refresh前のtracked orchestration/dry-run preflight。既存workflow/controllerのcontractを複製せず導出する。
- 外部code実行を再開する前のephemeral VM/JIT runnerまたはjob後reimageによるsecurity boundary。

[対応する履歴](../../../../../history/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)
