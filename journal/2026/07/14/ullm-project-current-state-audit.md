# uLLM-project 現状監査

Date: 2026-07-14 JST

## 前回の要点

- uLLMは、低ビットLLM向けの`.ullm`モデル形式、AQ4_0/SQ8_0量子化、Rust制御面、C++20/HIP実行層、検証・配備基盤を一体で開発している。
- 2026-07-12時点で、source-correctなQwen3-14B SQ8_0の単一worker/OpenAI Gateway/OpenWebUI製品経路はrelease evidenceまで完了していた。
- その後、Qwen3.5-9B AQ4_0 resident経路、長文decode最適化、reasoning/thinking budget対応とv2 release契約が追加された。

## 今回の変更点

- repository、Git履歴、設計文書、Rust/C++/Python実装、最終release evidence、active manifest、systemd、Docker、GPU processを読み取り中心で照合した。
- Gitは`main == origin/main == ecbb36b20050f3405b96e0a166ac2736bdfb922f`で、worktreeはclean、tagはない。
- 現在のactive productはQwen3.5-9B AQ4_0 reasoningである。manifest SHAは`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`、promotion sourceは`ae8b2bb7c2735f4dc761773957bf45f470dd5a8c`、worker SHAは`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`。
- live状態は`ullm-openai.service=active/running`、`NRestarts=0`、llama.cpp serviceはinactive、OpenWebUIはrunning/healthy。R9700上では`ullm-aq4-worker`のみが約7.35GB VRAMを使用していた。
- final release bundleとPhase 0 evidenceを現checkoutのvalidatorで再検証し、いずれも`structurally_valid=true`、`gate_eligible=true`になった。
- retained evidenceでは、OpenWebUI 100-chat soakが100/100、500 lifecycle records、restart 0。旧v2と現candidateのHTTP/SSEは各100ケースでcorrect/reset 100/100、budget overshoot 0、全modeのp95差分が計画閾値内である。
- 実装は、`ullm-quant`、`ullm-runtime-sys`、`ullm-engine`、Python OpenAI Gateway、systemd/OpenWebUI配備、tools/tests/benchmarksに分かれる。Rustがpackage・scheduler・session・worker lifecycle、C++20/HIPがdevice bufferとkernel、PythonがOpenAI API・tokenizer・release orchestrationを担当する。
- 共通`ModelGraph`、`StateSchema`、`ExecutionBatch`、CPU reference executor、state transaction、typed backend registryは実装が進んでいる。一方、現行AQ4/SQ8 resident sessionはまだ完全なgeneric executor経路へ移行していない。Qwen3/Qwen3.5 fixtureもattention stack中心で、embedding、MLP、final norm、LM head、verified production payload adapterは未接続である。
- SQ8については、2026-07-09/10の旧sidecar結果だけが2D source scale未適用で無効である。後続のcanonical artifact v0.2と2026-07-12製品releaseはsource-correctな別経路であり、旧結果の不正を現SQ8製品全体へ一般化しない。
- README、deploy README、Gateway README、served-model profile説明にはSQ8既定や「AQ4 reasoning未activation」などの古い記述が残り、現active AQ4 reasoningと一致しない。
- active workerのpromotion sourceはrepository HEADより前だが、その後の差分はrelease gate tool、tests、docs、evidenceで、Rust worker/Gateway本体の変更ではない。
- final bundleに記録されたrollback用environment SHA `601f3757…`と、現行`/etc/ullm/openai-gateway-manifest.env` SHA `68dd3a02…`は一致しない。active manifestとsystemd unitのSHAは一致する。現サービスの妥当性を直ちに否定する差分ではないが、次回activation/rollback前にbundleを現行environmentへ再結合する必要がある。

## 次の行動

1. OpenWebUI managed modelのmanifest markerと、UIからの厳密な`thinking_budget_tokens`指定を現active manifestへ合わせて証明する。
2. generic inference planの次のgateとして、Qwen3/Qwen3.5 production adapterをembeddingからLM headまで完成させ、resident AQ4経路をtyped graph/registry/executor/traceへ段階的に移す。
3. prefillは約120〜130 tok/s、長文decodeはsplit paged attention適用後およそ66 tok/sが現在の実測目安であり、prefill chunkingとproduction executor統合を次の性能課題として扱う。
4. READMEとdeploy/Gateway/profile文書を、現行AQ4 reasoning product、rollback用SQ8 product、legacy modeに分けて更新する。
5. 次回のbundle-bound activationまたはrollback前に、現行environment hashの変更理由を監査し、release bundleを再生成する。
6. single active request、待機queueなし、context 4096、text-only Chat Completionsというv0.1制約を維持し、request batchingや機能拡張は別gateで進める。

## 2026-07-14 18:36 JST 追補

### 前回の要点

- 現稼働製品はR9700上のQwen3.5-9B AQ4_0 reasoningであり、GatewayとOpenWebUIまでrelease gateが完了している。
- 汎用ModelGraph、StateSchema、CPU reference executor、typed backend registryは実装中だが、現行resident sessionはまだQwen3.5/AQ4専用実行経路である。
- 次の重点はprefill/decode最適化とproduction traceの実行幅証明である。

### 今回の変更点

- 調査中にmainが更新され、18:36 JST時点のHEADは`d1f86a06108082ee929b763b7167dd18e68d9ff2` (`Capture Qwen3.5 BF16 source oracle`)。`origin/main`より22 commit先行、behind 0である。P2 workload/toolの追跡済み未commit変更と、P0/P1 evidenceの未追跡directoryが残る。
- 稼働中サービスは変更されていない。`ullm-openai.service` はactive/running、`NRestarts=0`、active manifest SHA-256は`feb3190d...cb44`、worker SHA-256は`177f3106...48d`。OpenWebUIは`sha256:ef5ae4...b409`でhealthyである。
- P0 identity/rollbackとP1 schema/mechanics gateは完了し、現在はP2 baseline/profile準備である。AQ4 sessionは要求M=1/8/16/32/64/128と物理実行から解決したM、prepare/commit/discard/cancel/resetを監査し、sanitized terminal factsをworkerの構造化stderrへ出せる。
- Qwen3.5-9B BF16 CPU source oracleはP2 `source-oracle-v1`として保存され、3 bounded rows、実行約15.7秒、`status=available`である。一方、同一AQ4 artifactのall-M=1 path oracleとsource/path linkは未作成で、P2 candidateはpromotion不可である。
- 現行性能の履歴目安はprefill約117〜129 tok/s、context 1339の長文decode約66.53 tok/s。ただし旧identityの値なので、P2でactive identityのbaselineを取り直す。P1 mechanics smokeの13 ms前後は`python3 -c 'print(...)'`であり、推論性能ではない。
- 2026-07-09/10の旧SQ8 same-model sidecarはsource 2D scale未適用で無効。ただし、2026-07-12のcanonical artifact v0.2製品releaseはsource-correctな別経路であり、旧結果と混同しない。
- 対象CPUテストは現checkoutで`qwen35_aq4_session` 36件、`session_worker_backend` 2件が成功し、`git diff --check`も成功した。GPU/live workerの新HEAD検証は行っていない。

### 次の行動

1. AQ4 all-M=1 path oracleとsource/path linkを同一case/identityで作成し、独立validatorを通す。
2. active identity、bound policy、power、production traceを同一run rootへhash-bindingし、CPU→R9700の順でP2 baselineを取る。R9700は常に1件ずつ実行する。
3. profile結果から最初の一つのbottleneck familyを選び、P3 prefill candidateへ進む。
4. README/deploy/Gateway/profileの古いSQ8既定・AQ4未activation表記を、現行AQ4 product、SQ8 rollback product、legacy pathに分けて更新する。

## 主要根拠

- `uLLM-project/docs/concepts/ullm-concept-v0.1.md`
- `uLLM-project/docs/plans/generic-production-inference-optimization-plan-v0.1.md`
- `uLLM-project/docs/plans/generic-reasoning-thinking-budget-production-plan-v0.1.md`
- `uLLM-project/docs/specs/aq4-reasoning-openwebui-release-v0.1.md`
- `uLLM-project/benchmarks/results/2026-07-13/qwen35-9b-aq4-reasoning-v0.1/release-bundle-ae8b2bb-20260714-final.json`
- `uLLM-project/benchmarks/results/2026-07-13/qwen35-9b-aq4-reasoning-v0.1/http-identity-matched-p95-analysis-20260714.md`
- `journal/2026/07/12/current-project-development-environment-audit.md`
- `journal/2026/07/13/paged-decode-split-final-deployment.md`
