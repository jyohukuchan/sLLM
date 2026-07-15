# P3 A4 hybrid FD-map foundation

## 前回の要点

A3 は target argv の全runtime pathを `/proc/self/fd/N` に置換したが、実trusted runnerの非symlink path契約と両立せず、bundle root、lock、validator、live preflight、driver、served manifest、ROCTxの意味上のpathも変えていた。

## 今回の変更点

- logical argvと証跡は元のabsolute pathを維持し、Python/runnerのbootstrap execだけを保持FD pathへ投影した。
- captureがexact target manifestからcode/control/lock/rootを用途別にopenし、canonical JSONをsealed memfdへ格納する `ullm.aq4_p3_inherited_fd_map.v1` を追加した。
- closureを次の4分類へ分離した。
  - `code_execution_closure=pinned_fd`
  - `control_input_closure=pinned_fd`
  - `device_lock_closure=pinned_fd`
  - `data_integrity=trusted_pre_post_guarded`
- runnerはcode/controlを保持FDからexec/pread/dlopenし、lockは継承O_RDWR FDへ直接flockする。bundle rootはlogical pathと保持directory FDの開始・終了identity guardを行う。
- child validatorとresident driverにはlogical commandとeffective FD commandを分離し、FD-mapと全descriptorを継承する。
- ROCTxはlogical invocation path、actual resolved path、保持FD digestを別々に維持する。
- source validator pathのswap試験は共有正本を変更せず、`tools/` 配下のprivate copyだけをswapする。pytest finalizerでprivate copy/backup/replacementを全削除し、正本SHA不変を検査する。

## 検証

- `/usr/bin/python3.12 -m py_compile` 対象3 source: pass
- runner/capture/launcher 3 test群: `134 passed, 1 deselected`
- deselectはgenerated execute-bindingが新launcher source SHAへ未追随のため。generated pin更新後に再有効化する。
- 実 `tools/run-aq4-p2-resident-batch.py` をfake rocprof経由で起動し、canonical one-case bundleのCPU-only dry-runとvalidator subprocessを完走した。
- validator logical path swap/restoreではreplacement markerが生成されず、保持済み旧validator FDだけが実行された。
- control fileとlock logical path swapでは、保持済みcontrol bytesと元lock inodeだけを使用した。
- ROCTx symlink swapでは、mapのresolved digestがnullにならず、保持済み旧bytesと一致した。
- 実GPU、service操作、HTTPアクセスは行っていない。

## インシデントと是正

初期のvalidator swap testが約8秒だけ共有正本をrenameしてstubへ置換し、別レーンが一時状態を観測した。test終了時に元SHA `0b3341d3e9d6e3dde8cff05eb8dd43fe2ec8b176a8a913183dbee638dd25c175` へ復元済み。以後はprivate copyのみをswapし、repo内backup/stub/marker残留0件と正本SHA不変を確認した。

## 次の行動

prepare/validator generatorを同FD-map schemaへ対応させ、prepared-v1とbinding-v4を再生成する。その確定SHAをlauncher constants、maintenance、ready artifacts、QA attestationへ反映する。
