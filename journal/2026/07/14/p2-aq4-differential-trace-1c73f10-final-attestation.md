# P2 AQ4 differential trace final attestation — `1c73f10`

実施日: 2026-07-14  
対象: `uLLM-project` 固定 HEAD `1c73f10` (`Close AQ4 trace identity TOCTOU`)、後続の build identity 監査対象 `1295512` および現 HEAD `28ec343`

## 前回の要点

`863157b` までに、診断トレースの入力、ケース／リプレイ、実行バイナリ、モデル・マニフェスト、ビルドコミットおよび生成段階の結合を確認した。残課題は、ツール／アクティブ／パッケージの SHA-256 計算におけるパス再オープンと、同一 FD 上の変更検出だった。

## 今回の変更点・確認結果

`1c73f10` の `required_regular_sha256` を再監査した。

- `symlink_metadata` で regular file、`nlink == 1`、256 MiB 以下を fail-closed で要求する。
- `File::open` 後に FD のメタデータを事前値と比較し、1 MiB チャンクで同じ FD をストリーミングして SHA-256 を計算する。
- 読み取り後は同じ FD の `metadata()` を比較し、`dev/ino/size/mode/mtime/ctime/nlink` の変化を拒否する。
- ツールバイナリ、アクティブ served-model manifest、パッケージ manifest はすべて上記ヘルパーを使い、入力パスの再オープンはない。`sha256_file` の残る呼び出しは、生成済み一時出力の `SHA256SUMS` 作成だけである。
- 生成 manifest の `"cases_path"` は一意（ソース内のリテラル出現数 1）。
- 既存の opt-in、入力上限、重複／欠落／余分ケース・リプレイ、3-row・full-context hash、scratch/output 上限、symlink replacement、identity guard、bounded row contract のテストを再実行した。

## 検証

```
ULLM_BUILD_GIT_COMMIT=$(git rev-parse HEAD) cargo test -p ullm-engine --bin ullm-aq4-differential-trace  # 9 passed
python3 -m unittest -v tests/test_qwen35_aq4_differential.py tests/test_aq4_p2_input_controls.py  # 6 passed
rustfmt --edition 2024 --check crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs crates/ullm-engine/src/qwen35_aq4_model_runtime.rs  # passed
cargo check -p ullm-engine --lib  # passed
git diff --check 863157b..1c73f10  # clean
```

source/path differential v2、source/path oracle v2 の既存 `SHA256SUMS` は全対象ファイルと一致した。source differential v2 は 3 行（prompt 3 行 + prompt 2 行の全 3 rows）で、step-1 の context hash は prompt plus replay prefix 契約に従う。生成 runtime は診断モード・model_loads 1 であり、本番 worker から `visit_intermediate_trace` の呼び出しはない。

### `1295512` / 現 HEAD の build identity 再監査

`1295512` の専用バイナリのコード blob は `73bbaf50eb04b9c3dc4ac934b02e3dcf79bab8ca` で、現 HEAD `28ec343ac59e6d22e710035d7874df9fbd8f890f` でも同一だった。`f081b3f` 以降の `28ec343` は AQ4 family-exclusive GPU profiler の 4 ファイルだけを追加し、専用バイナリ、入力、既存 P2 evidence 本体を変更していない。

`ULLM_BUILD_GIT_COMMIT=$(git rev-parse HEAD)`（現 HEAD 28ec343）で Rust 9 テストを再実行し、9 passed。Python 6、rustfmt、lib check も pass した。ビルド済みバイナリには 40 桁の現 HEAD SHA が埋め込まれていることを `strings` で確認した。

未設定挙動も確認した。build 環境変数を未設定にしたバイナリは、valid input 後に `ULLM_BUILD_GIT_COMMIT is missing from the binary build` で終了する。一方、未設定環境での Rust test harness は 9 tests を通過する（テストはモデル実行を呼ばず、manifest 作成経路を通らないため）。埋め込み済みバイナリに異なる 40 桁 runtime env を与えた場合は `runtime ULLM_BUILD_GIT_COMMIT differs from embedded build commit` で fail-closed となる。manifest は埋め込み値を `identity.build_git_commit` に記録する。

## 判定

この固定 HEAD について、コード安全性・入力／出力境界・プロヴェナンス・同一 FD アイデンティティ検証・既存測定証跡の条件はすべて満たす。R9700 GPU Gate は **Go** と判定する。ただし、実機 GPU の常駐ラン・OOM／温度・性能値の取得は未実施であり、次の実行段階の責任範囲とする。

## 次の行動

親エージェントはこの判定と検証ログを統合し、実機での R9700 GPU Gate 実行へ進める。作業中の親エージェント変更（P0/P1/P2 bundle 等）は本監査では変更していない。以降、この固定 HEAD の監査では新規 commit を作成しない。

### 現 HEAD `10ebe856` に対する旧 detached artifact 実行案

現 HEAD と監査済み `28ec343` の差分は resident smoke trust-root 境界の 14 paths に限られ、専用 bin/model runtime、source tool、cases、replay、source/path evidence 本体には差分がない。専用 trace source blob は `73bbaf50eb04b9c3dc4ac934b02e3dcf79bab8ca` のままである。

監査済み release binary は SHA-256 `356d131fc578debea418f6c67d7b89272bfb02700495775be98471c44e3bd0b7`、埋め込み build commit は `28ec343ac59e6d22e710035d7874df9fbd8f890f`。cases raw SHA は `15fed90dd2e16a5b68d4498c8632257d80ac94c56ed614696b0884c65f4836f2`、detached replay raw SHA は `1ee0b9228e1bc3a0ae9175e5693bf3770f9b89e872349554562dbd4b6b4747dc`。実行時 helper は cases/replay の同一バイト SHA、active manifest SHA `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`、package manifest SHA `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`、binary SHA を manifest に束縛する。

ただし、提示された `run-gpu-gate.sh` は `HEAD=$(git rev-parse HEAD)` で現 `10ebe856` を runtime `ULLM_BUILD_GIT_COMMIT` に渡すため、埋め込み `28ec343` と mismatch fail-closed になる。さらに cargo release binary は `nlink=2`（target/release と deps の hardlink）で、`required_regular_sha256` の `nlink == 1` guard に拒否される。実際に GPU を起動せず同一引数で実行し、`trace binary must be a regular file` を確認した。nlink 1 のコピーにすると binary／active／package identity を通過して package manifest missing まで進むため、artifact 自体の結合は成立する。

また、script の全 `rocm-smi` 呼び出しは裸コマンドで、実体 `/opt/rocm/bin/rocm-smi` を絶対指定していない。PATH 依存のため gate 安全性条件を満たさない。

source-differential-trace-v2、path-differential-endpoint-v2、path-differential-analysis-v2、source-oracle-v2、path-oracle-v2 の全 `SHA256SUMS` は再検証で全件 `OK` だった。なお `/opt/rocm/bin/rocm-smi` 自体は `../libexec/rocm_smi/rocm_smi.py` への symlink であり、絶対パス指定に加えて実体の固定 SHA を記録する余地がある。

したがって、現状の旧 artifact 実行案は **No-go**。`28ec343` を明示的に pin、nlink 1 の artifact copy を固定 SHA で使用し、`/opt/rocm/bin/rocm-smi` を絶対指定すれば detached artifact 自体は Go 候補となる。

### 最終 gate script `6c760727...`

最終 `run-gpu-gate.sh` の SHA-256 は `6c760727046429e11ca21477067332ae012fa62d54a45216bd95dcf8161dd62c`。`bash -n` は pass。前版からの変更は `pgrep -f` を `pgrep -x ullm-aq4-worker` に限定する修正として確認した。固定 commit／binary／cases／replay SHA、nlink 1 detached copy、ROCm realpath／実体 SHA、固定 `/usr/bin/python3.12` の SHA・version、30 個の required-environment、lock、PREFLIGHT_ONLY の stop 前分岐、post health／stat／SHA／NRestarts 検証を静的に確認した。

ただし、`systemctl stop` の直後に `STOP_MARKER` を touch してから親 trap が restore 判定するため、stop が非ゼロ終了・割り込み、または marker 作成失敗の窓では、停止済みサービスを復旧できない。observer marker も pre-stop `kill -0` と stop の間に確認されず、競合窓が残る。さらに PREFLIGHT_ONLY は stop しないものの gate log を作り、通常実行の「兄弟ログ不存在」guardにより同じ script の再実行を阻害する。

このため、最終 script も **No-go**（サービス復旧の fail-closed 条件を厳密に適用した場合）と判定する。GPU/service は起動・停止していない。

### FINAL_FREEZE `4ec5f688...` 再監査（最終判定）

途中版を凍結後、最終 script SHA `4ec5f68884fed32a0078dc79e1ec45c274a16d84fdae196e5e87c249b610d5f3` を再監査した。`bash -n` は pass。`PREFLIGHT_ONLY=1` を read-only で実行し、rc=0、サービス active、`output`／detached binary／gate・monitor・run log／全 marker の残留なし、stdout-only の `preflight_stop_run=skipped` を確認した。

最終版では、worker の `nlink=2` と固定 SHA/stat、active/package の固定 SHA/stat、durable stop marker を `RESTORE_NEEDED=1` の後・service stop の前に作成する順序、observer の pre-stop 5 秒 freshness／post-stop 新 sample handshake、lock、`RUN_STARTED_MARKER` 後の候補実行、trap による restore と early cleanup が実装されている。30 件の worker required environment、ROCm realpath／実体 SHA、固定 Python direct 実行、post active/package/worker stat・SHA、NRestarts、container health も静的に pass した。

前版の stop 復旧窓、observer race、PREFLIGHT log residue は最終版で解消された。oracle lane の mock/order 証跡と合わせて、本番 R9700 GPU Gate は **Go** と最終判定する。GPU 実行自体はこの監査では行っていない。

### FINAL_FREEZE `c02fc8f1...` lock-order review

最終 freeze 候補の script SHA-256 は `c02fc8f1a752200ed9b6ef01e7884bdbfb073f59ccf671e2c393da14ea4c4774`。`bash -n` は pass。通常の `PREFLIGHT_ONLY=1` は GPU／service を変更せず rc=0 で、固定 commit／binary／cases／replay SHA、ROCm／Python SHA、active/package/worker SHA・stat、service MainPID／lock owner、`NRestarts=0`、health／models、および残留なしを確認した。

ただし `PREFLIGHT_LOCKED_ONLY=1` は read-only 条件を満たさない。script は `RESTORE_NEEDED=1` と durable stop marker を先に設定し、locked 分岐が observer 停止後に return 0 する。その後 EXIT trap の `cleanup` が `restore_service` を呼び、サービスを停止していないにもかかわらず `systemctl start ullm-openai.service` を実行する（該当箇所は 238--244、282--286、221--236、179--219 行）。これは locked preflight の service action 禁止に反するため、この SHA の判定は **No-go**。修正後に `RESTORE_NEEDED=0` または marker/trap arm 前で locked branch を終えること、通常／locked 両 preflight の rc=0・無残留・サービス状態再確認を再監査する必要がある。GPU／service stop は行っていない。

### 最終 freeze 候補 `f1ec7dbd...` 再確認

上記 No-go 指摘を反映した現行 script の SHA-256 は `f1ec7dbd5ce37a96b6b9f77f03d7739eee86dfc86de4ca41b7bda52144031c89`（git blob `d54f7366e4b1d3df14769c7bb3e52d421c9632ca`）。`bash -n` は pass。通常 `PREFLIGHT_ONLY=1` と `PREFLIGHT_LOCKED_ONLY=1` をそれぞれ read-only で実行し、両方 rc=0、候補／output／gate・monitor・run log／全 marker の残留なしを確認した。locked 分岐は marker/trap arm 前に独立し、`lock_owner_preflight=expected_service_mainpid`、`locked_preflight_systemctl_mutations=0`、observer cleanup 完了を出力する。

両実行後もサービスは `ActiveState=active`、`SubState=running`、`MainPID=2087869`、`NRestarts=0`、R9700 lock owner は同じ `2087869` で、ROCm／health／models と SHA・stat 検証も pass。GPU 実行、service stop/start、commit は行っていない。このスナップショットは旧 `c02fc8f1...` No-go を置き換える **Go 候補**だが、親エージェントがさらに編集する場合は新 SHA を再監査する。

### 最終 freeze 候補 `65bbef4b...` 再確認

output manifest 検証の Python 呼び出しも固定 `/usr/bin/python3.12` にした現行 script の SHA-256 は `65bbef4b05c75728e49824714fdd52313e78c9dfb37185872377390b0dd21d09`。`bash -n`、通常 `PREFLIGHT_ONLY=1`、`PREFLIGHT_LOCKED_ONLY=1` はすべて rc=0。両 preflight で health／models、ROCm、SHA・stat、lock owner を確認し、候補／output／各ログ・marker および `/tmp/ullm-aq4-lock-owner.*` の残留はなかった。終了後も `ActiveState=active`、`SubState=running`、`MainPID=2087869`、`NRestarts=0`、lock owner `2087869` を再確認した。service stop/start、GPU gate本実行、commit は行っていない。

### 現行微修正 `4400263a...` の再確認

mode 同時指定を明示的に拒否する guard（`PREFLIGHT_ONLY=1` と `PREFLIGHT_LOCKED_ONLY=1` の同時指定は rc=64）を加えた現行 script の SHA-256 は `4400263a6f0c1f4705ffb19ec87cd526dd46eab2c64de8856f86d53035a88a73`。`bash -n`、単独通常 preflight、単独 LOCKED preflight はそれぞれ rc=0。同時指定は rc=64 で service/GPU 操作なし。単独実行後の residue なし、サービス active／running、MainPID・NRestarts・lock owner は従前と同じで、65bb の Go 候補条件に回帰はない。

### freeze復元確認 `65bbef4b...`

親エージェントの指定により freeze対象を `65bbef4b05c75728e49824714fdd52313e78c9dfb37185872377390b0dd21d09` へ復元。SHA-256一致と `bash -n` rc=0をread-onlyで再確認した。上記同一SHAの通常／LOCKED preflight証跡に基づき、両mode環境変数をunsetする本番呼出しの最終判定は **Go**。GPU/serviceの本番実行、編集、commitは行っていない。
