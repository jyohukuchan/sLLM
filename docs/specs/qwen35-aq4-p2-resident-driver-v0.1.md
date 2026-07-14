# Qwen3.5 AQ4 P2 resident driver v0.1

## 前回の要点

`cf11261`と`b6b21cb`のresident batch v1は、1 child processへ84 case、各2 warmupと10 measuredを送るCPU fake用の最小protocolだった。ready identityはpackage manifestとdeviceの一部だけであり、case/result/order/release、OOM後のprocess再利用可否、実AQ4 runtimeとの接続は未定義だった。

## 今回の変更点

`ullm.aq4_p2_resident_driver.v2`は、`ullm-aq4-p2-resident-driver`がserved modelとpackageを起動時に1回だけloadし、そのmodel runtimeを全caseで再利用するbounded NDJSON protocolである。R9700 lockは外側runnerだけが所有し、driverはlockを取得せず、指定されたruntime deviceがserved `gfx1201` identityと一致することを確認する。

readyはmodel load後に1回だけ発行し、resident session ID、driver binary、build commit、manifest worker binary、package manifest/content、served manifest、model/revision、format/implementation、runtime device、required guard setのSHA-256をexact fieldsで返す。runnerはP2 identityの`resident_driver_identity`およびhash bindingと完全一致させ、drift時は最初のcaseを送らない。

`case_begin`はexpanded case binding、identity、preflight、policy、fixtureの各absolute pathとfile SHA-256を明示し、case self-hash、identity self-hash、expanded binding、preflight exact fields、greedy sampling、AQ4 target control、prompt/context/generated count、requested/resolved M、deviceをdriver側で再検証する。CLI固定値による不足fieldの暗黙補完は禁止する。

各caseはrun index 0–1をwarmup、2–11をmeasuredとしてexact orderで実行する。runtimeはresolved M単位でpromptをdispatchし、decodeがある場合はresident LM headのgreedy tokenを次stepへcommitする。各runはtiming、operation audit digest、state digest、requested/resolved/actual width、lifecycle、reset、resource、terminal factsを返す。M fallback、case swap、duplicate/reuse、unknown field、run order違反はfail-closeする。

run完了後はGPU同期と全request state resetを必須とし、case終了時にcommitまたはdiscard、reset、baseline restorationをrelease factsとして返す。cancelはactive caseをdiscard/resetする。OOM、HIP fault、reset failure、stdout publication failureではprocessを再利用しない。OOM/HIP faultの`run_complete.terminal.reuse_forbidden=true`をflushした後、driverはnonzero exitし、runnerは残りcaseを実行しない。driverはlockを二重取得しない。

v2 runnerが発行するcase rawはresident session/model-load identityと12 runのraw-v2互換timing/audit/state/lifecycleを保持する。GPU/live captureはこの実装作業には含めない。

## 次の行動

R9700で実行する前に、実binary SHA、build commit、package content、served manifest、guard setをP2 identityへbindする。その後、外側runnerが1回だけdevice lockを取り、v2 driverを起動して2 warmup＋10 measuredの84 caseを順番に実行する。
