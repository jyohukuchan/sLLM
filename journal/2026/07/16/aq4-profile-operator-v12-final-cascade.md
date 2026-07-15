# AQ4 profile operator-v12 final cascade

## 前回の要点

- phase-aware finalizer の authority は commit `370ab8cff2fc745d85657260329a80fab21b0acb` である。
- operator-v11 の actual は、pre-stop snapshot で returncode 1 となり、サービス停止、launcher、capture、rocprof を一度も開始せずに終了した。
- その失敗証跡は commit `854e5a348bd3c0f442f2371a0d3619308bce3b95` に封印されている。

## 今回の変更点

- operator cascade を ready-v14、quiet-v17、maintenance-v10、command/result/audit-v12 へ更新した。execute binding、runtime、execute evidence、capture は v9 を維持した。
- ready-v14 を commit `39af01d5dca7c76eb53fdbffc59dc976a2d24e6c`、tree `78a457d681ee23c43f87e0094ae60331635704c3`、ready binding SHA-256 `6664abaafdf76adcc40565652dbbaa6ab0dbb1f131d1a4b011d66007fd059891`、SHA256SUMS SHA-256 `803046262d5b0d106ccecccb2979b3d8ff5d7d8bf4eece5b3a49f377f9c5b00d` に固定した。
- previous command-v11 の sealed manifest を commit/tree/raw/semantic/SHA256SUMS と exact argv で検証するようにした。manifest に埋め込まれた operator-v10 の未実行状態と historical actual-v9 の封印状態も維持した。
- actual-v11 を `pre_stop_failed_sealed` として厳格に読み戻すようにした。maintenance/result/audit の8ファイルだけが commit `854e5a34...` に存在し、runtime/execute/capture が不在であることを確認する。
- actual-v11 は returncode 1、invocation 1/1、retryなし、サービス/GPU未接触、pre-stop no-op restore、package full hash 1回、finalizer `370ab8cf...` による読み取り専用 recovery、runtime/execute/capture inventory が null であることを要求する。
- fresh operator-v12 は9パスすべてが不在でなければ準備できない。actual 実行、GPUコマンド、サービス操作は行っていない。
- `tests/test_prepare_aq4_p3_profile_operator.py` は v12 名前空間、actual-v11 final state、partial/mixed 拒否、exact-one、rc=0/17 finalizer、pre-stop no-op を検証する。

検証結果:

- `python3 -m py_compile tools/prepare-aq4-p3-profile-operator.py`
- operator test: 34 passed
- 関連回帰: 255 passed, 1 skipped
- `git diff --check`: pass

## 次の行動

- source/tests/journal だけを commit し、commit 後の source blob/raw SHA-256 と trusted-source snapshot を再確認する。
- quiet-v17 と operator-command-v12 の成果物生成は、この source authority を基準に別作業として行う。
