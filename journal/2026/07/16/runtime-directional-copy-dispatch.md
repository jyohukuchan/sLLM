# HIP runtime の方向別非同期コピー

## 前回の要点

- 中央の `HipRuntime::copy_async` は、方向を表す `kind` を受けながら、すべてを汎用の `hipMemcpyAsync` に渡していた。
- public C ABI と Rust FFI は、すでに host-to-device、device-to-host、device-to-device を別々の操作として公開している。

## 今回の変更点

- `kind = 1 / 2 / 3` を、それぞれ `hipMemcpyHtoDAsync`、`hipMemcpyDtoHAsync`、`hipMemcpyDtoDAsync` に厳密に割り当てた。
- 不明な `kind` は、ゼロバイトの場合も含めて失敗させる。汎用の `hipMemcpyAsync` は、この実行経路から削除した。
- ROCm 7.2.1 のヘッダーで `hipDeviceptr_t` が `void *` であることと、3関数の引数型を確認した。HIP ヘッダーが利用できるビルドでは、`decltype` と `static_assert` によって関数ポインター型を照合する。
- CPU の公開操作で host-to-device、device-to-device、device-to-host の連鎖を確認する単体テストと、動的解決シンボル、方向割り当て、不明値拒否を固定するソース契約テストを追加した。
- public C ABI と Rust FFI の宣言は変更していない。

検証結果:

- `python3 -m unittest tests.test_ullm_runtime_directional_copy`: 3件成功
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-runtime-sys cpu_directional_copy_contract_covers_all_three_runtime_copy_paths -- --test-threads=1`: 1件成功
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-runtime-sys cpu_buffer_ -- --test-threads=1`: 4件成功
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-runtime-sys zero_byte_buffer_copy_accepts_end_offset -- --test-threads=1`: 1件成功
- `cargo fmt --check -p ullm-runtime-sys`: 成功
- `python3 -m py_compile tests/test_ullm_runtime_directional_copy.py`: 成功
- `git diff --check`（担当ファイル）: 成功
- `/opt/rocm/lib/libamdhip64.so` の読み取り専用シンボル確認: 3関数を `hip_4.2` の global export として確認
- GPU、サービス、actual capture は実行していない。

ワークスペース全体の `cargo fmt --all --check` は、担当外で既存の未整形変更があるため失敗した。担当 crate に限定した確認は成功している。

## 次の行動

- この変更を組み込んだ将来の許可済み profile で、HIP API trace が方向別の3関数として記録されることを確認する。今回の作業では GPU 実行や capture は行わない。
