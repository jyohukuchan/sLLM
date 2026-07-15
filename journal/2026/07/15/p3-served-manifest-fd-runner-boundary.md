# P3 served-manifest FD runner boundary

## 前回の要点

P3 actual rc1 は resident driver の ready 前に失敗した。A4 runner が論理 served-model manifest パスを `/proc/self/fd/N` へ置換し、Rust の通常パス loader が magic symlink を拒否したためである。rc1 は warmup、measurement、ROCTx marker がすべて 0 の診断証拠として保存済みである。

## 今回の変更点

- runner は resident driver の論理 7-argv を維持し、argv[2] の served-model manifest を `/proc/self/fd/N` へ置換しない。
- argv[0] の resident driver executable は従来どおり pinned FD から実行する。
- 既存の sealed `ULLM_AQ4_PINNED_FD_MAP`、served-manifest FD、その他の binding FD は `pass_fds` と環境を通して子へ継承する。新しい環境変数は追加していない。
- ready event の top-level `served_model_binding` を exact schema で検証する。profile execution では `pinned_fd`、`inherited_sealed_fd`、`inherited_fd_map`、`control_input/read`、map の logical path・identity・SHA-256、single read、logical path 未 open を要求する。
- historical prepared fake-ready に binding がない場合の互換は、offline one-case dry-run validation だけに限定した。runtime ready では binding が必須である。
- この互換フラグは `args.dry_run` からだけ設定する。historical fixture を同じ one-case non-dry 契約へ渡した場合は、live preflight や driver spawn より前に binding 欠落として fail-closed にする。
- driver-process evidence を v2 に更新し、論理 argv と canonical SHA-256、argv index 0/2 の semantic FD binding、FD-map digest と closure、ready の served-model binding を記録する。ephemeral descriptor 番号は記録しない。

## 検証

- `pytest -q tests/test_run_aq4_p2_resident_batch.py`: 42 passed
- `pytest -q tests/test_capture_aq4_p3_diagnostic_profile.py tests/test_launch_aq4_p2_resident_smoke.py`: 35 passed
- `python3 -m py_compile` で runner と変更した Python tests を検証した。
- actual rc1 failure を回帰 fixture として確認した。
- logical manifest を pinned FD 作成後に replacement へ差し替え、fake driver が sealed map の元 FD バイト列だけを読み、論理 argv[2] を維持することを確認した。
- fake rocprof、実 runner、CPU-only fake driver を通す live integration で ready 到達、84 cases 完了、driver-process v2 evidence を確認した。
- missing/closed FD map を fail-closed で確認した。
- historical fake-ready は dry-run でのみ通り、同じ fixture の non-dry 検証は失敗することを確認した。通常の live ready も top-level binding 欠落を拒否する。

GPU、service、actual capture の実行はしていない。既存 actual rc1 artifact と Rust・maintenance artifact は変更していない。

## 次の行動

Rust resident driver の同じ top-level `served_model_binding` 契約と統合し、detached binary と runner source pin を更新する。その後に prepared/binding/profile-ready/quiet-window/operator artifact を再生成し、rc1 を保持したまま rc2 を新規実行する。
