# 固定HEAD 4e210bc AQ4 differential trace 再監査

## 前回の要点

- 旧 `0ec8534` には step>0 の context prefix 欠落があり、GPU Gate を保留していた。
- `6ddc9d1` の Vec/u8 scratch alias 修正と、`71d4c08` 以降の bounded/provenance hardening を再確認対象とした。

## 今回の変更点

- 固定HEAD `4e210bc`（祖先 `71d4c08`, `144ae9b` を含む）を読み取り監査した。共有コード・GPU/service は変更していない。
- source v2 payload、superseded marker、Rust専用 binary、model visitor、低メモリ unit tests を照合した。

## 確認結果

- source v2 は3 rowsで、context は prompt + replay prefix と一致した。
  - fixture-prompt-0 step0: length 3 / hash `42ea...`。
  - fixture-prompt-0 step1: length 4 / hash `6af160...`（`[11,12,13,220]`）。
  - fixture-prompt-1 step0: length 2 / hash `3bca...`。
- v1 は `SUPERSEDED.json` で `invalid_superseded` と明示され、v2を replacement として束縛している。v2の `SHA256SUMS` 3ファイルは実体と一致した。
- `validate_inputs` は `run` の唯一経路であり、期待2 case/合計3 rows、prompt/replay IDs・prompt hash・replay/source sequence hash、重複・欠落・余分な case、1 MiB input capを fail-closed する。Rust negative testsを含む専用6 testsが通過した。
- `finish_record` は1行32 KiB、writerはpayload累積96 KiB、最終出力全体も96 KiBで検査する。`MAX_OUTPUT_BYTES` はwriterとfinal total checkへ接続済み。
- `read_intermediate_trace_row` は別 `Vec<u8>` にD2Hし、`chunks_exact(4)`、trailing byte拒否、finite検査、little-endian decodeを行う。旧 `&mut` aliasは残っていない。visitorはscratch 32 KiB上限、embedding + 32 layerの出力を同一stream copy/synchronizeで採取する。
- production workerからvisitor/model_mutへの参照はなく、専用binaryは明示flagが無い場合にmodel load前に拒否する。

## 未解決とGate判定

- 出力manifestの `input_binding` は raw cases/replay SHA、expected cases、actual replay sequenceを保持するが、実際にparseした prompt token列の `actual_prompt_ids_sha256` は個別に保持しない（expected prompt hashは保持）。
- `identity.build_git_commit` は `ULLM_BUILD_GIT_COMMIT` 未設定時に文字列 `unknown`、active manifest SHAはファイル不在時に `None` となる。要求された「tool binary/build/active/package/device/guard-set identityを実値で保持」を常に満たすfail-closed契約ではない。さらに raw bytesをparseした後に同じパスを再openしてSHAを計算するため、入力差替え時にparse内容とmanifest SHAが乖離し得る。

したがって、コード経路とbounded trace自体は改善済みだが、identity/provenanceの実値要求を厳密に採る場合は GPU Gate **No-go**。`build_git_commit` と active manifest を必須化し、parse済みraw bytesのSHAをそのままmanifestへ保存し、actual prompt bindingを追加した後に再監査する。

## 検証

- `cargo test -p ullm-engine --bin ullm-aq4-differential-trace` — 6 passed、GPU/service未使用。
- `python3 -m unittest -v tests/test_qwen35_aq4_differential.py tests/test_aq4_p2_input_controls.py` — 6 passed、低メモリ。
- `rustfmt --edition 2024 --check`（専用bin/model runtime）— passed。
- live GPU、worker、gateway、systemd は未実施。

## 次の行動

- provenanceをfail-closed化（parse bytes SHA、actual prompt hash、必須build/active identity）して専用negative testsを追加する。
- 修正後、同じ固定HEAD形式で再監査し、その後のみ承認済みGPU窓を開く。
