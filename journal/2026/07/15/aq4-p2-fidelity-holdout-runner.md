# AQ4 P2 holdout runner実装記録

## 前回の要点

`generate-aq4-p2-fidelity-holdout.py` と既存Rust captureはcalibration 24行を固定していた。freeze receipt後にholdout 24行を一度だけ評価する実行境界は未実装だった。

## 今回の変更点

- Rust `ullm-aq4-fidelity-capture`へ既定動作を変えない`--subset holdout`、`--cases-file`、holdout SHA拘束を追加した。
- `prepare-aq4-p2-fidelity-holdout-cases.py`でholdout 24行だけをsource casesへ変換する。
- `run-aq4-p2-fidelity-holdout.py`でCPU preflight、one-shot execute、attempt marker、failure/result receiptを実装した。
- failureはtimeout/OOM/nonzero/partial/validator違反をimmutableに封じ、success/No-Goはremainingを0にする。freeze policyは再導出しない。
- source/active identity、split/policy/cases/freeze/actual receipt SHA、finite/shape/greedy/top-k/relative-L2/stateを検証する。

独立監査receipt `803bb5108a6decd89a1d410dc59a698b4a8a952dba4b11de4a36805eaf35c256` のNO-GO（B1〜B8）を受け、次を追加した。

- 共有served-model実装のSQ8 promotion validatorを再利用し、actual receiptのexact schema、prepared/maintenance/executor lineage、profile/served/package/worker/overlayを再検証する。
- source-holdout、guard、capture buildをformal receipt化した。buildはsource commit/tree、clean worktree、Cargo.lock、log、実行binaryまでlive bindする。package全memberは1 MiB streamingでRustと同じcontent hashをpreflight/execute直前に再計算する。
- preflight commandを各固定値から再構成し、execute側のdevice index指定を廃止した。device index/backend/name/architecture/ID、ROCR/HIP/CUDA可視device、guard環境をplanへ固定した。
- attempt発行後かつspawn直前に全frozen inputとsource artifact全memberをpath/SHA/size/mode/nlink/symlink条件で再検証する。
- `/proc`のchild PID/PGID/session/exe/command、process-group全PID、package配下open fd/memory mapによるmodel-load、`/dev/kfd`/render node fdによるGPU process censusを追加した。
- Rust artifactへ24行のstate/reset/direct-scheduler evidenceを追加し、regular member 0444/nlink=1、directory 0555、file/directory/parent fsync、create-new manifest/SHA256SUMS、rename no-replaceを実装した。
- attempt後のopen/spawn/census/timeout/OOM/nonzero/partial/publication failureをstage/errno付きimmutable failureにする。retryとpartial adoptionは常にfalseである。

## 検証

- `pytest -q tests/test_aq4_p2_fidelity_holdout_protocol.py tests/test_aq4_p2_fidelity_holdout_runner.py tests/test_aq4_p2_fidelity_holdout_cases.py tests/test_qwen35_aq4_fidelity_capture.py`（27 passed）
- `cargo check -p ullm-engine --bin ullm-aq4-fidelity-capture`
- `cargo test -p ullm-engine --bin ullm-aq4-fidelity-capture -- --test-threads=1`（8 passed）
- `python3 -m py_compile tools/run-aq4-p2-fidelity-holdout.py tools/prepare-aq4-p2-fidelity-holdout-cases.py`
- `uvx ruff check ...` / `uvx ruff format --check ...`（pass）
- shared promotion/served-modelを含む広めの61件は60 passed。1件はisolated worktreeでもfixture内の本repo絶対pathを期待する既存テスト`test_aq4_reasoning_candidate_binds_v2_worker_separately_from_active_v1`だけがpath差で失敗し、今回の変更範囲外である。

`cargo fmt --all --check` と対象Rust fileの直接checkは基点由来の既存未整形（file冒頭のserde_json import、既存の長いjson!/一行関数群）で失敗する。変更前commitをrustfmtしてから変更後をrustfmtしたread-only差分では、差分はrecursion limit、subset/cases SHA分岐、runtime identity fields、usageだけであり、既存コードの整形差分をcommitへ混入させていない。

監査修正後も同じread-only比較を行った。`cargo fmt --all --check`の既存差分は2011行、変更前/変更後を別copyでrustfmtしたsemantic差分は196行で、今回追加したseal helper、holdout device env、state/reset evidence、manifest binding、封印publication、単体テストだけである。

追補監査receipt `350ac7855e2f295b9c75551aefbc23e482466f1bc404710649c558df15c815f6` の残件B2/B3/B5/B8に対し、次を修正した。

- Python runnerとholdout cases adapterは、`O_NOFOLLOW`で開いた同一fdからhash対象bytesとJSON/JSONL parse対象bytesを取得し、前後fdと最終pathの`dev/ino/mode/uid/gid/nlink/size/mtime_ns/ctime_ns`を照合する。mtimeを復元した競合もctime差で拒否する。
- Rust captureも同じfingerprintで安定読み取りし、served-modelは同一fdから得たbytesを`load_served_model_bytes`へ渡す。runnerは全frozen inputの検証済みfdをspawn完了まで保持し、capture binary自体は`/proc/self/fd/N`からexecする。
- runtime device IDはRust evidenceと同じinteger型に固定した。子環境は明示allowlist、guard、device可視性だけから構成し、完全な環境objectとSHA-256をplanへ凍結する。`os.environ.copy()`は実行経路から除去した。
- rows/vectorはchmod 0444後にfile fsyncを行ってからdirectory chmod/fsync、rename、parent fsyncへ進む。chmod/post-chmod-fsync/closeの注入失敗を単体テストで確認する。
- attempt消費前に独立したhidden rescue failure receiptをcreate-new、write、file fsync、chmod、再fsync、parent fsyncし、fdを保持する。attempt markerのlink/fsync/close後例外、通常failure/result publication例外、finally例外は最外層でこの封印済みfailureへ収束し、retryを許可しない。

追補検証はholdout focused pytest 34件、Rust capture test 9件、Cargo check/test、py_compile、`uvx ruff check`、`uvx ruff format --check`が成功した。全pytestはgateway依存（fastapi/package）不足など11件のcollection errorで開始不能だった。`PYTHONPATH=.`の広域testsは233 passed/1 skippedまで進み、isolated worktreeにない既存benchmark fixtureと別系統のmaintenance bindingなど69件が失敗した時点で中断した。GPU、service、sudoは使用していない。全workspaceの`cargo fmt --all --check`は今回の対象外Rust fileを含む基点由来の既存未整形が残るため失敗し、対象capture fileも基点時点からrustfmt未適用である。

GPU実行は行っていない。実機での残課題は、凍結済みsource artifactをholdout casesで生成し、指定Rust binaryを一回だけ起動してresult receiptを作ることだ。
