# CI・テスト方針策定履歴

## 2026-08-02

- 前回の開発で発生した、GPU処理をCPUで実行して長時間化した問題と、テスト導入が遅れて細かな不具合が蓄積した問題を設計上の入力とした。
- ローカル`reference/`の配置状態を確認した。各参照directoryは空であり、固定source配置後の再調査が必要と判断した。
- workspace外の読み取り専用mirrorからvLLMとTensorRT-LLMのtest tier、hardware metadata、代表model、quarantine、性能回帰の分離方針を調査した。
- GitHub ActionsとROCmの公式資料から、公開fork PRとself-hosted runnerの安全境界、ephemeral runner、timeout、matrix、GPU isolationを調査した。
- CPU CIで許可する処理と禁止するGPU-scale処理を分離した。
- H0〜H3、G0〜G4、P0〜P1のテスト階層、初期時間予算、result state、shape方針、数値比較方針を草案化した。
- PR、信頼済みdispatch、main push、daily、weekly、releaseの実行matrixを草案化した。
- repository skeletonからGPU runner、kernel、model slice、end-to-end、performanceへ進むPhase 0〜6を定義した。
- file変更を伴わない4つの調査用Codex sessionは全て終了コード0で完了した。
- 読み取り専用reviewで、GPU runnerのcommit/cache信頼境界、CPU testの累積resource制限、main-planのCI導入順序、result集約、AMD binary key、KV dtypeの不足を検出して草案へ反映した。
- 再出発レビューを受け、`255/256/257`を含むperformance-cliff sanity、同一reviewed SHAのGPU merge gate、`always()`を含むfail-closed集約条件を追加した。
- repository hygieneとcredential方針をPhase 0/1のgovernance baselineへ追加した。
- push前reviewを受け、H3 required昇格条件、P1 weekly/release予算、build・ROCm・target・codegen変更のH3/G4 gate表現を統一した。
- source-lockの7件を固定完全SHAの一次sourceとして4 reader sessionで再調査し、全sessionが終了コード0で完了した。外部codeのcopy、adapt、portは行っていない。
- 段階化、明示登録、決定的sharding、per-test timeout、GPU preflight、immutable artifact再利用、isolated test、warmup/metricを採用し、暗黙skip、0件成功、soft-fail、可変外部data、root/privileged runnerを不採用とした。
- H0/H1/H2の並列required row、`host-required`集約、8/10/8分のhard timeout、workflow p95 10分・hard上限15分を確定した。
- tier/属性marker、versioned suite registry、path-to-suite manifest、test-result/compatibility-tuple schemaの正本pathと必須概念を確定した。
- H3のPR compile rowを`gfx1030`/`gfx1201`とし、20回以上・7日以上に加えて全期待rowのPASS、他state/cancel/schema errorなし、artifact hash一致、時間・infra条件をrequired昇格条件にした。`gfx1200`はnightly/release compile-onlyに残した。
- 初期GPU evidenceを専用local hostの`gfx1030` 1台と`gfx1201` 1台の直列実行とし、2台目の`gfx1030`をspare/nightly用とした。暫定local経路はtrusted project commitだけに限定し、public fork PRからはGPU runnerを直接使わない。

## 2026-08-03

- Phase 1のrepository skeletonとしてRust workspace、`sllm-core`、`sllm-hip-sys`、`sllm-hip`、`sllm-cli`を追加した。
- Cargo `build.rs`からCMake C++17 static host stubを`OUT_DIR`配下へbuild・linkし、versioned C ABI、caller-owned error sink、reserved-field検証、checked-in bindingsを追加した。
- host stubはHIP backend/contextを明示的なunavailableとして返し、GPU成功またはCPU fallbackとして扱わないcontractをRust H1で確認した。
- `test-result-v1`、compatibility tuple、hygiene allowlist schema、suite registry、host matrix、path mapping、共通runnerとaggregatorを追加した。
- H0へRust format/clippy/MSRV、C++ format/static host build、Python、Markdown/link、schema/workflow、license/provenance、tracked tree、registration、negative self-testを登録した。
- H1へRust workspace test、Python contract/API test、CI contract test、H2へ固定seedのtiny NumPy boundary/KV/sampling oracleを登録した。
- H2 subprocessへ4 GiB address-space上限、pytestへsocket禁止を適用し、model、GPU、network、CPU fallbackをrequired host evidenceから除外した。
- missing/duplicate/unknown/stale report、schema/hash/identity不一致、non-success needs、0件収集、意図的format/test failure、禁止tracked pathがfail closedになることを確認した。
- GitHub-hosted CPUだけでH0〜H2を並列実行し、`host-required`へ集約するworkflowを追加した。official Actionsは完全commit SHAへ固定し、H3とGPU runnerは含めていない。
- Phase 1完了監査で不足していたopaque handle/access/completion ownership、C/Rust ABI parity、TensorView/NVFP4境界、semantic op arity、error-sink truncation、CLI error exitを実装し、Rust testを22件へ拡張した。
- test harnessをactual collected/selected/outcome count、strict clean SHA identity、network namespace、row/command resource limit、bounded outputへ拡張し、異常をschema-validな非PASSとして扱うcontractを追加した。
- Python host dependencyをtransitive dependencyとartifact SHA-256までlockし、checkout credentialを保持せず、Rust 1.97.1/MSRV 1.85.0をcommandごとに明示した。
- fixture consumer mappingをH1/H2へ同期し、hash-locked isolated venvでH0/H1/H2とlocal-development aggregateがPASSすることを確認した。immutable evidenceはcommit後のclean checkoutとGitHub required workflowで別途固定する。
- Phase 2 bootstrapで未構築のG0/G1/G2/G4/P0をH3自身へ循環的に要求しないよう、GPU hard gateを変更scope別へ分割した。
- H3 toolchain/artifactはH0〜H3、G0 runnerはH0〜H3とcanonical G0、model-free runtimeはH0〜H3とcanonical G0/G1を同一candidateへ要求する。G2/G4/P0はmodel、互換性昇格、性能または実運用dispatchへ実際に影響するときから追加する。
- H3の20回以上・7日以上の観測はrequired昇格だけの条件とし、G0、GPU runner、model-free runtimeを並行して進めると明記した。
- H0 network guardが`/proc/net/route`と`/proc/net/ipv6_route`の動的counterまで完全一致比較してparent network破壊を誤検出していたため、routeの意味属性だけを厳格に正規化するよう修正し、counter変化、意味属性変化、不正形式、fresh namespaceの空IPv4 routeを回帰testへ追加した。
- H0 network guardが親namespace identityに加えてroute/interface topologyの同値性まで復元判定へ使い、hosted runnerの背景変化をchildの破壊と誤帰属していたため、復元判定を親netns identityのfail-closed比較へ限定し、child側のloopback-only、default-routeなし、権限drop検証を回帰testへ追加した。

## 2026-08-10

- ユーザー確認を受けて開発policyをresetした。H/G/P各tierとGPU resultのfail-closed判定は維持しつつ、全draft checkpointへのimmutable SHA、全host/GPU matrix、fresh独立review、docs-only closeoutの強制を廃止した。
- 以後はdraft development、integration candidate、release/push candidate、docs-onlyのlaneへ分ける。draftはfocused test、integrationは影響testと1回のreview、releaseはclean immutable identityと最終matrix、docs-onlyはMarkdown/link/正本整合だけを要求する。
- 過去の同一SHA gateとcloseout実績は当時の事実として保持するが、現在の完了条件にはしない。意味上のsource/build identityが変わらないdocs-only変更では、対応関係を確認して既存GPU evidenceを再利用できる。

[対応する計画](../../../../plans/active/2026/08/1-10/ci-test-strategy.md)
