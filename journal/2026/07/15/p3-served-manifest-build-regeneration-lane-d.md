# P3 served-manifest build/regeneration lane D

## 前回の要点

- profile actual v3 は runner の pinned FD-mapがserved manifestを`/proc/self/fd/<descriptor>`へargv置換し、driverのno-symlink path検査が`/proc/self`を拒否したため、ready前にexit 1した。
- source commits `065b82f95948ad6a269233619539a40c084e35bf`、`c4e413ee93f87f86bd4a81d869953ae86c1c10dc`、`1e65fd5c99845c7a64e707df5bf140ca6d62ff82`、`81ceebb13518f590b5dbf439cd00b35e508c1c3f` は、logical argvを維持したinherited FD読取、legacy ready制限、Python互換canonical self-hashを順に導入した。

## 今回の変更点

### clean buildとpin

- clean worktree `/tmp/ullm-served-manifest-lane-d` をHEAD `81ceebb13518f590b5dbf439cd00b35e508c1c3f`、tree `5e98c3812f9eebdaed3e6085ab2e13521e249521` で作成した。既存main dirtyと共有targetを使用しなかった。
- initially absent `/tmp/ullm-served-manifest-lane-d-target` で `CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver` を実行した。GPUとserviceは使用していない。
- source Git blobは `7e37119cc8b66dc0e0f7abcf49b896fcdad8315f`、raw SHA-256は `0acb46d1ab8730267edf40b505224ff157760ec19aa40a07ee1b389860ec54bf`。
- release binaryは3,506,904 bytes、SHA-256 `458b8603d6823a1c20ea93e7c0d757c8910f3c36c9a2a34ab536853c0c9e7d34`、ELF Build ID `e7313ba6f51feac74f14b5ffd100333265362e1e`。
- toolchainはcargo `1.96.0 (30a34c682 2026-05-25)`、rustc `1.96.0 (ac68faa20 2026-05-25)`、host `x86_64-unknown-linux-gnu`、LLVM `22.1.2`。
- generatorはsource commit/tree/blob/raw、binary SHA/bytes/Build ID、toolchain、jobs、incremental、locked、profile、exact commandをprepared trust-roots/bundle/launch-commandとbinding manifestへ固定した。source/test commitsは `1f5b12803759e6596021dfd8c5e1455f2635f586` と `a44074278d4bbd5e243153ab8c5be272489e23a2`。

### CPU-only verification

- Rust resident-driver unit tests: 22 passed。Python canonical Unicode golden、FD-map self-hash、pinned served manifest path swap、negative schemasを含む。
- Rust `served_model::tests`: 3 passed。
- 実binary integrationは `ULLM_TEST_AQ4_P2_RESIDENT_DRIVER` で必ずunskipし、capture main → fake rocprof → actual runner → detached real Rust driverをexecした。非ASCII logical path `served-モデル/active.json` をFD-mapへ含め、served FDのparse/validate後、production guard未設定で意図的に停止した。
- integrationは `required environment ... must equal 1` を確認し、`pinned FD map self-hash differs`、旧`served-model manifest traverses a symlink`、`runtime device query failed`がないこと、driver exit 1、reap、process-group cleanup、capture failure evidenceを確認した。1 passed / 28 deselected。
- prepared/binding/runner testsは107 passed。captureのレーンD対象は26 passed / 3 deselected。
- 全3 Python filesの一括実行は133 passed / 3 failedだった。3 failuresは再生成されたbindingに対して既存profile launcherが旧B root identityをpinしているための`stage=constants / B root identity differs`であり、profile/ready/maintenance artifactを更新しない今回の境界どおりである。

### fresh official regeneration

- main canonical pathでofficial `resident-one-case-smoke-prepared-v1/` と `resident-one-case-smoke-binding-v4/` だけを削除後にfresh生成した。profile/ready/maintenance artifactには触れていない。
- prepared bootstrap runnerはhistorical control memberのcommit `3dc4aa612b6cfd87675d0bd9fe506426f43e64f9`、SHA-256 `e7dae31c64b3844a09fbba7ef36bbae7834e21d5d217bad679dd50bdf314ff02` を維持した。
- binding actual runnerはcommit `81ceebb13518f590b5dbf439cd00b35e508c1c3f`、blob `b7a3af27b17bd9dfae926c320eda04f7c3afae4e`、SHA-256 `5d4cf385a83961f8aedc37d36c3e4625d783ec7ddd6b17de4f93648516d42354`。`same_runner=false`で役割境界を維持した。
- trusted validatorはcommit `a44074278d4bbd5e243153ab8c5be272489e23a2`、tree `2bb2036e44c4e328c9f61f0400b462db5db11a85`、blob `392706210060499d1384ceb9bcf58d324e9cca05`、raw SHA-256 `f11394f84cdf8b858634bab20a48ba24d19cd51a0f0c95783dfe329f33e1e976`。
- preparedはrunner subprocess 1、fake handshake PASS、model/GPU/service 0。bindingはactual runner 1、trusted validator 1、fake driver 1、model/GPU/service 0。
- `validate`、`validate-binding`、両`SHA256SUMS` readbackはPASS。全memberはsingle-link regular、resident-driver 0555、その他は0444。

### artifact SHA-256

- prepared `bundle.json`: `fe9ad93acb21dc98f2cb8c9a442fe4373ca44022b675da2cb61008bf9ff3811c`
- prepared `trust-roots.json`: `3185ad06940143a91aa6bb46456c82084421ce21d4f762150db3894651278b81`
- prepared `launch-command.json`: `714010fc43ec335caf22e8383f771d0aaead09d95a43cc2f8b7a723732439553`
- prepared `runner-dry-run-evidence.json`: `0050e78174487722a1a21a1b0fc59fac8217f9a4dcfa5c227849144c6c3480cf`
- prepared `SHA256SUMS`: `12e72ded1804ca075fde19f7ceca4d02cde9df2558489288e8ff850caf1a2b2b`
- binding `binding-manifest.json`: `da0d7fe01a091a5c42b23435c9734c23f84af64460e5713b8aadca78225ab187`
- binding `runner-subprocess-evidence.json`: `0e60bd3b6924be604a269b447e6dbea6ec9f5fbd8c2fd134334cd73991f36895`
- binding `validator-report.json`: `a6af7c425935971d1ec8be878888922c319222f3b900afad5a1a9421216f84d2`
- binding `SHA256SUMS`: `6aa93fbb4cb709906c46ea2e497e052360f01ae9e3ae8d08b4d45323eea412fb`

## 次の行動

- downstream profile launcher/ready/maintenanceは新しいprepared/binding rootを独立に再pinし、旧B root identityを解消する。
- その再pinとCPU-only検証が完了するまでactual GPU/service実行は行わない。
