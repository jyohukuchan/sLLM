# Qwen3.5 AQ4 P2 resident driver

## 前回の要点

resident batch v1 runnerは84 caseを1 fake childへ送る計画とready identityだけを持ち、実Rust AQ4 driver、case/run binding、release、terminal reuse禁止は未実装だった。

## 今回の変更点

- protocolを`ullm.aq4_p2_resident_driver.v2`へ更新し、readyへdriver/worker/package manifest/package content/served model/model/format/implementation/device/guard/build identityを追加した。
- case_beginへexpanded case、identity、preflight、policy、fixtureのpath+SHAとexecution/sampling/controlを明示し、暗黙補完を禁止した。
- 新binary `ullm-aq4-p2-resident-driver`は`Qwen35Aq4ModelRuntime`を1回だけloadし、caseごとのresolved Mで実prompt/decodeをdispatchする。外側runnerのR9700 lockは二重取得しない。
- 各run後に同期resetし、timing/audit/state/lifecycle/resource/terminalをexact resultへ保存する。case_endはcommit/discard/reset/baseline releaseを返す。
- OOM、HIP fault、reset failure、protocol/publication failureはfail-closeし、reuse禁止eventをflushした後にprocessを終了する。cancelはactive caseをdiscard/resetする。
- CPU mockでcase swap、order、reuse、unknown/duplicate、M fallback、state leak、ready drift、release failure、OOM terminal、cancelを検証した。Python fake runnerではready drift、case swap、result order、release drift、OOM、reset failureを検証した。
- `CARGO_BUILD_JOBS=1`でdriver unit test 6件、`cargo check`、`cargo build`、engine lib test 717件（1件ignored）を通した。Python runner test 6件と`py_compile`、担当ファイルの`rustfmt`と`git diff --check`も通した。
- commit `0fd7993`（`Add AQ4 P2 resident batch driver`）へ保存した。
- 独立QA follow-upで、Cargo hardlinkではなく`nlink=1`のdetached driver copyをnormative artifactにした。runnerはspawn前にabsolute/non-symlink/single-link fileのSHAを独立計算してbound identityと照合し、ready self SHAとも再照合する。
- runnerへ既定`/run/ullm/r9700.lock`のnonblocking exclusive lockを追加した。owner metadataはlock file、専用JSON、raw、summaryへ保存し、contention、ready拒否、OOMを含む全経路で解放する。
- fixture index内pathはabsolute、`..`なし、親を含むsymlinkなし、strict resolve済みだけを許可し、driverもserved manifestと全protocol linkをabsolute-onlyにした。
- QA negativeとしてCargo-style hardlink拒否/detached copy受理、lock contention/例外cleanup、fixture relative/親symlink拒否、ready self SHA swap拒否を追加した。
- follow-up検証ではRust driver 7件、Python runner 9件、`py_compile`、owned `rustfmt`/`git diff --check`が成功した。`CARGO_BUILD_JOBS=1`のdriver check/buildとengine lib test 717件（1件ignored）も成功した。
- QA follow-upをcommit `319d618`（`Harden AQ4 resident driver launch boundary`）へ保存した。
- GPU/liveは実行していない。

## 次の行動

detached driverとpackage/content/served/guard identityをP2 identityへbindしてR9700 runを行う。
