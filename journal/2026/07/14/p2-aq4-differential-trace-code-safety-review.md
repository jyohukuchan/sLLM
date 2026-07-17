# P2 AQ4 differential trace 専用 code safety review

## 前回の要点

- `52b19ef` は、AQ4 intermediate differential trace を production worker/gateway から分離した専用 binary と、model runtime の opt-in visitor、session の診断用 `model_mut` を追加した。
- live GPU、worker、gateway、systemd は今回も起動していない。

## 今回の変更点

- `52b19ef` の専用 binary、Cargo `[[bin]]` 宣言、`Qwen35Aq4IntermediateTraceObserver`、`visit_intermediate_trace`、session `model_mut` を読み取り監査した。
- 初回確認時の HEAD は `5817534` で、先行した入力制御監査の 4 ファイルは `e1bbe0c` と byte-level で同一だった。その後の追跡 HEAD は `0ec8534` で、履歴の reset/rebase/checkout/force 操作は実施していない。
- 実行結果と残課題をこの journal に記録した。共有 production code は変更していない。

## 独立確認

### 既定動作と到達可能性

- `run` は `enabled=false` の場合、output の存在確認、cases/replay の読み込み、model load の前に明示的なエラーを返す。`--enable-intermediate-trace` は専用 binary の引数にだけ作用する。
- `rg` で `model_mut(` を検索した結果、呼び出しは専用 `ullm-aq4-differential-trace` binary の 1 箇所だけだった。production worker/gateway から visitor へ到達する参照はない。
- Cargo の専用 `[[bin]]` は既定 worker と別 target で、既存 worker target の呼び出し経路へ変更を加えていない。

### 採取位置、同期、形状、座標

- `visit_intermediate_trace` は最新 dispatch 後の resident embedding output と各 `layers[position].output_buffer()` を、hidden 要素数の再利用 host scratch へ `copy_to_host(0, bytes, Some(stream))` し、各コピー直後に同じ stream を `synchronize` する。`RuntimeBuffer` 側の範囲検査も通る。
- 専用 binary は `with_prefill_chunk_tokens(1)` を固定するため、sequence output の先頭 row を読む visitor と dispatch の width が一致する。layer は `dispatch_layer_stack` の前段 output を次段へ渡すため、enumeration 順の出力は各 decoder layer の post-layer row である。
- embedding/layer は `[0,1,1024,2048,4095]` を含む hidden row を finite 検査し、final norm は同じ座標、LM head は `[0..32)` の logit 座標を固定採取する。`begin` と `finish_record` が hidden/logit shape と embedding+32 layer の 33 stage を検査する。
- final norm/logit は `observe_prepared_calibration` の generation epoch 検査後に採取され、LM head を再実行せず既存 logits を D2H する。`context_token_ids_sha256` は prompt に replay token の prefix を連結した列を JSON compact + terminal newline で hash し、source replay の hash-bound sequence と整合する。
- output は `*.incomplete-$PID` に書き、manifest/payload/runtime/SHA256SUMS の作成後に未使用 output へ rename する。既存 output は上書きしない。row size は 32 KiB 以下で fail-closed する。

## 検出した課題（GPU gate 前に直す候補）

1. 固定 3-row guard が無い。`run` は `cases.cases` の非空と各 `step_count <= 128` だけを検査し、総 row 数を 3 に制限しない。`MAX_ROW_BYTES` は 1 行の上限であり、任意 case 数・step 数による総出力、処理時間、メモリを抑えない。P2計画は 2 case/step 2+1 の 3 rows を固定するため、`MAX_CASES`、checked sum の `MAX_ROWS=3`、期待 total == 3 を追加する。
2. replay の重複 `case_id` が `BTreeMap` の `collect` で後勝ち上書きされる。重複 ID を明示拒否し、cases と replay の集合一致（余分な replay も拒否）を検査する。
3. `load_json` は `fs::read` で cases/replay を無制限に一括保持する。固定診断入力の byte 上限を設け、上限超過を parse 前に拒否する。prompt token 数、token ID の vocabulary/context 範囲も bin 側で早期検査すると fail-closed 境界が明確になる。
4. 出力 manifest は cases/replay のパスを記録するが、入力ファイル SHA-256 や case ごとの prompt hash/sequence hash を束縛しない。payload の context hash は正しいが、後から同じパスが差し替えられた場合の provenance を manifest 単体で再検証できない。入力 SHA-256、case IDs、prompt/sequence hash を manifest に保存する。
5. `read_intermediate_trace_row` は `values` と同じアドレスを `from_raw_parts_mut` で `raw` として借用し、その `raw` を参照しながら `values.iter_mut()` へ書き戻す。これは同一領域への同時 `&mut` alias で、safe API の borrow checker が防げない未定義動作リスクである。別の `Vec<u8>` を host copy の受け皿にし、copy 完了後に `chunks_exact(4)` を values へ decode する実装へ変更する。
6. 追跡中の HEAD `0ec8534` は step>0 の `context_tokens` を `vec![replay_tokens[step - 1]]` としており、prompt prefix を落としている。正しい context は `prompt_token_ids + replay_tokens[..step]` であり、step1 の fixture-prompt-0 は長さ4、fixture-prompt-1 は長さ2のままである。context hash/length の証跡を修正するまで trace は利用不可である。

上記は production worker の既定挙動を変える不具合ではないが、diagnostic trace の「bounded / fixed 3 rows」という証跡契約には未充足である。特に 5 の `&mut` alias は専用診断 path 内の未定義動作リスクなので、コード安全面の gate は live GPU 実行へ進める前に alias 修正、固定 row/input guard、provenance 束縛を条件とする（現状 No-go）。

## 検証

- `cargo test -p ullm-engine --bin ullm-aq4-differential-trace` — 2 tests passed（GPU/service 未使用）。
- `rg -n "model_mut\\(" crates/ullm-engine/src --glob '*.rs'` — session accessor 定義と専用 binary の呼び出し以外なし。
- commit 52b19ef の読み取り確認 — Cargo target、visitor、stream copy/synchronize、shape/coordinate checks、epoch/context hash、atomic output を確認。
- live GPU、worker、gateway、systemd は未実施。

## 次の行動

- 専用 binary に duplicate/extra ID、input file byte cap、固定 3-row guard を追加し、negative unit tests を通す。
- guard 修正後にのみ、承認済み GPU 窓で専用 binary を実行し、中間 stage trace と source oracle の差を確認する。
