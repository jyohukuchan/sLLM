# P2 AQ4 layer-0 QKV GPU probe gate

## 前回の要点

CPU integration evidenceは、固定input/package/report/outputのSHA chainを検証済みである。
GPU実行、service停止、holdout観測、数値Go/No-Go、promotionは未実施である。

## 今回の変更点

- 固定commit `2bcef0d897d43ea1ff397dc558f7e0e179d8a904`から、`CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe`を実行し、nlink=1のimmutable probe copy、build receipt、SHA256SUMSを作成した。
- `run-gpu-probe-gate.sh`を追加した。既定はread-only preflightで、mock/preflight modesはsystemctl、ROCm SMI、GPU probeへ到達しない。明示的な`EXECUTE_GPU_PROBE=1`だけがservice stop/restore、RuntimeDirectory lock、observer、standalone probeを許可する。
- gateはinput/package/active/profile/worker/probe build identity、`HIP_VISIBLE_DEVICES=1`→filtered HIP ordinal 0→global runtime device 1→`gfx1201`を固定検証する。成功reportもstandalone、`fused=false`、`unclassified`、`promotion_eligible=false`を要求する。
- 数値閾値、Go/No-Go、holdout観測、promotion処理は実装していない。docsにもdiagnostic-only契約を明記した。
- shell syntax、mock preflight、相互排他、改変input fail-closedのテストを追加した。

## 次の行動

君の明示的なGPU/service実行許可までは、mock/preflightとbuild receiptの検証だけを行う。許可後もstandalone diagnostic reportの収集に限定し、数値判断やpromotionは別承認に委ねる。
