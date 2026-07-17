# 固定HEAD 863157b 最終独立再監査

## 前回の要点

- `4e210bc` 時点の主な残件は、manifestのactual prompt binding不足、build/active identityの任意値、入力parse後のidentity再openによるTOCTOUだった。

## 今回の変更点

- `863157b`（祖先 `71d4c08`, `144ae9b`, `4e210bc`, `839e546`）を読み取り監査した。コード、GPU/service、共有runtimeは変更していない。
- source/path v2 evidenceのcommit差分を確認し、source v2 payload/manifest/SHA256SUMSの不変性を再確認した。

## 確認結果

- Rust binaryの`load_json_with_sha`は、同一bounded bytesをparseし、そのbytesのSHAをmanifestへ保持する。`read_bounded_file`はregular file、open後identity、読み取り後identityを比較し、symlink replacementを拒否する。
- `validate_inputs`はrun唯一経路で、期待2 cases/3 rows、prompt IDs/count/hash/step、replay IDs/source hash、duplicate/missing/extra、1 MiB capをfail-closedする。
- `actual_cases`にprompt token IDs/count/hash/step、`actual_replay_sequences`にreplay hash/source sequence hashを保持する。`build_git_commit`は40桁hex必須、active/package manifestはregular fileかつSHA必須、device name/backend/memoryとguard-set hashも実値検査する。
- row 32 KiB、payload/output total 96 KiBのwriter接続、Vec<u8> safe decode（chunks_exact、trailing、finite、little-endian）、full-context 3 rows、v1 superseded、production隔離/明示flag OFFは回帰なし。
- `source-differential-trace-v2` のmanifest/payload/runtime SHA256SUMSは実体と一致し、839e546以降にsource-v2 evidenceの改変はない。

## 残る問題とGate判定

- `required_regular_sha256`（active/package/tool identity）は、`symlink_metadata`でregular fileを確認した後、別の`sha256_file(path)`でpathを再openする。cases/replayのような同一fd identity前後検査がなく、metadata確認後から再open中のrename/replacementで、manifest SHAが実際に確認したidentityと乖離し得る。
- `manifest` JSON objectに`cases_path`キーが重複している。serde_jsonでは最終値が有効になるが、証跡フォーマットとして不要な重複であり、除去した方が安全である。

上記のmetadata identity TOCTOUを厳密なprovenance要件の欠落と判定し、R9700 GPU Gateは **No-go** とする。active/package/toolについても、同一fd（またはidentity前後検査付き）でSHAを計算してmanifest identityを固定した後に再監査する。

## 検証

- `cargo test -p ullm-engine --bin ullm-aq4-differential-trace` — 9 passed、GPU/service未使用。
- `python3 -m unittest -v tests/test_qwen35_aq4_differential.py tests/test_aq4_p2_input_controls.py` — 6 passed、低メモリ。
- `rustfmt --edition 2024 --check`（専用bin/model runtime）— passed。
- source v2 evidence SHA256SUMS 3/3一致、v1 superseded marker確認。
- live GPU、worker、gateway、systemdは未実施。

## 次の行動

- `required_regular_sha256`をfd固定hash＋identity前後検査へ統一し、重複manifest keyを削除する。
- 修正後に同じ9 Rust/6 Python/rustfmtとevidence SHA検証を再実行し、Gate判定を更新する。
