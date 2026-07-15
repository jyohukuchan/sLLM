# P3 historical synthetic READY fixture boundary

## 前回の要点

- profile diagnostic actual v6では、pre-spawn bundle検証が旧5項目の`fake-ready.json`をlive READY用のexact 6項目検証へ渡し、resident driverをspawnする前に停止した。
- 失敗候補のcanonical raw SHA-256、driver identity SHA-256、session SHA-256は、historical synthetic fixtureと完全に一致した。

## 今回の変更点

- historical synthetic `fake-ready.json`の検査を`validate_historical_synthetic_ready_fixture`へ隔離した。
- fixture検査はdry-runとnon-dryの両方で旧5項目を受理するが、`stage=pre_spawn_fixture_only`、`runtime_proof=false`、`ready_proof=false`、`model_load_proof=false`として記録する。
- live `validate_ready`から互換フラグを削除し、`served_model_binding`を含むexact 6項目を常に必須にした。
- spawn後の`_recv(..., "ready")`は、candidate audit付きのstrict `validate_ready`だけを呼び、fixture専用関数を呼ばない。
- fake-ready echo subprocess countは従来どおり1回である。
- actual実行、GPU command、model load、service操作は行っていない。

## Verification

- fixture/live境界のtargeted tests: `7 passed in 15.95s`
- resident batch runner tests: `54 passed in 23.90s`
- profile capture tests: `48 passed, 1 skipped in 13.15s`
- Python compileと`git diff --check`: PASS

## 次の行動

- runner sourceを新しいbinding artifactへ取り込み、trusted source SHAとcommitを更新する。
- 次のactual前にはnon-dry fixture検査がlive preflightまで進み、live READYの5項目候補が引き続き拒否されることをreadbackする。
