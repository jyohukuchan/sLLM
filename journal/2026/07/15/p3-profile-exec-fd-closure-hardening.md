# P3 profile 実行境界の FD 固定

## 前回の要点

- profile は validator/gates の後に capture → rocprof → resident runner の順で実行する境界まで分離した。
- target manifest は論理 argv、base environment、入力ハッシュ、ROCTx、出力先を固定していた。

## 今回の変更点

- target manifest の全 `input_files` と symlinked runtime file を open 済み FD で保持し、rocprof の target argv だけを `/proc/self/fd/N` に置換した。論理 argv と manifest self-hash は変更しない。
- interpreter、runner script、JSON 入力、validator、live preflight、resident driver、served manifest、ROCTx library の FD を rocprof と target child に明示継承する。
- capture の Python helper 閉包を producer、selector、profile classifier の 3 本に限定した。path、7-field identity、SHA-256 を target/ready contract と evidence に固定し、検証済み bytes から module を生成する。
- producer の通常 CLI も selector と profile classifier を SHA-256 検証済み bytes から読み込む。capture 経路では検証済み module を注入し、同じ helper を path から再読込しない。
- callback を `on_rocprof_started` と `on_runner_completed` に分離した。rocprof の生成成功と runner の完了証明を同義にしない。
- launcher の `profile_capture` を exact schema にし、未知 field、runner/rocprof、timeout、cleanup、children の矛盾を拒否する。復旧用の詳細は strict `profile_diagnostics` に分離した。
- 成功 `capture-artifact.json` と失敗 evidence はともに mode `0444` とする。
- verify→spawn 間の swap/restore でも差し替え runner が実行されず、FD に固定した bytes だけが実行される回帰を追加した。transitive helper 3 本にも同じ差し替え回帰を追加した。

## 次の行動

- maintenance adapter を新 callback と exact outcome schema に接続する。
- launcher/capture/maintenance の確定 commit・blob・raw SHA と exact test count を QA attestation に再固定する。
- execute-binding と base/profile ready artifact を公式 generator で再生成し、canonical readback と dry-run を通す。
- actual、GPU、service、HTTP は実行しない。
