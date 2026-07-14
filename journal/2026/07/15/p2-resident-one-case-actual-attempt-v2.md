# P2 resident one-case actual attempt v2

## 前回の要点

- v1 actualはservice stop直後の単発AMD owner gateで失敗したため、v2 harnessは30秒のabsolute deadline内で旧ownerの解放をpollし、stable 2回を要求する契約へ更新された。
- commit `4205c7e9f90f6a2e208d868b1f0035208f7bdabc`のoperator manifest SHA-256 `737489b25f326a140c57f6efe7da5150e2179706289192ad2be0c8c7802fdf1f`を唯一のargv源とした。

## 今回の変更点

- manifest/SHA256SUMS/permissions、HEADと全input hash、3 fresh outputのABSENT、formal health `9/6/6`、service/worker/`NRestarts=0`/GPU/KFD/lock、RAM約79.6GB available、disk約2.58TB freeを確認した。
- 同一PTYでsudoをprimeし、manifestのcwdと9要素argvを`subprocess.run(..., shell=False)`へそのまま渡して1回だけ実行した。開始は`1784059046164349739` unix ns、終了は`1784059097627767992` unix ns、elapsed `51,463,418,253 ns`、return code `1`。再試行とprofile実行はしていない。
- service stop後、poll attempt 0は両service inactive、旧worker pgrep return code 1、AMD-SMI process command return code 0/stderr emptyまで観測したが、その直後にobserver `HarnessError`となった。decisionは`terminal_failure`、reasonは`stopped observation failed`で、stable 2へ進まなかった。
- AMD stdoutはrawを保存しない契約で、保存値はSHA-256 `c623fc11440b2bf81199ddefe42cadc330fa31ecde1cd268ff0ab930889e09ca`だけである。このため失敗時JSONのtop-level type、root keys、process entry fieldsは事後確定不能。poll evidenceも内側のHarnessError messageを`error_type=HarnessError`へ潰している。
- code path上はAMD process command後かつAMD VRAM command前なので、exact causeは新observerのAMD process JSON parse/schema/probe validationのいずれかに限定される。旧formal parserはrootがexact `[{'gpu': 2, 'process_list': []}]`でなければ一括して`target GPU compute owners are not zero`とする。一方、新observerは各entryの`process_info.pid`を先に整数化するため、消滅中processの非標準entryやroot差をowner pendingとして扱わずterminal schema/probe errorにする。rawを保存していないため、この候補間をさらに特定する証拠はない。
- launcher/model load/warmup/measuredは未開始で`0/0`、runner raw/summaryとlauncher outputはABSENT。outer restoreはattempted/passedし、新service main PID `4101742`、worker PID `4101820`、`NRestarts=0`で復帰した。post formal healthは`9/6/6`、全endpoint 200、GPU/KFD ownerは新workerだけ、lockとproduction hashesも正常で、actual関連childは残っていない。
- immutable evidence `resident-one-case-smoke-maintenance-evidence-v2/`はdirectory `0555`、files `0444`、SHA256SUMS PASS。SHA-256はlauncher evidence `5a532a37cd48d688f7f54808431035597552b2e2f6744f29d218266b051b7d36`、poll `dda93fe8c9e8b1101fe986ba635f49a01f452de93036558acf85be59c3d4b67c`、marker `0d5826adccff776fea1f00d10c5efe73c9f9fb7400563a0127c332214f2da034`、SUMS `35cbfcd675dadcfe280ed51518e5de4d88f4538f1cfd990a4f50c6ec4b516151`。
- 追加read-only診断としてproduction active状態で同じAMD-SMI argvを1回実行した。raw SHA-256は`247105d398f2b1087a29330e4ce1085d7c277da8d2ddf47ba02bf6b2f5b4bc3f`、631 bytes、return code 0、stderr empty。top-levelはlist length 1、root keysは`gpu/process_list`、entry keyは`process_info`、process_info fieldsは`cu_occupancy/evicted_time/mem_usage/name/pid`、pidは正のintだった。新observer parserとの差分はなくacceptされる形であるため、恒常的なAMD-SMI version/schema差ではなくstop直後の一時的な出力形状がv2 failureの対象だと絞り込めた。rawと構造要約は`resident-one-case-amd-smi-active-diagnostic-v1/`へsecret-free、`0555`/`0444`で保存した。

## 次の行動

- v2のsingle-use outputは再利用せず、actualも再試行しない。
- 次版ではAMD process rawをサイズ制限・secret検査後にimmutable保存するか、少なくともJSON top-level type/root keys/entry fieldsと内側exception messageを保存する。
- 消滅中processの非標準entryはforeign ownerと区別し、PIDを安全に抽出できない場合も同一attemptのterminal failureにせず、deadline内の次pollへ進める契約を検討する。
