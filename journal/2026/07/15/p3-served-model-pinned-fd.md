# P3 served-model pinned FD boundary

## 前回の要点

- profile runner は `ULLM_AQ4_PINNED_FD_MAP` v1 と `served_manifest` binding を resident driver へ継承していた。
- driver argv の served-model manifest は logical path を `/proc/self/fd/N` へ置換していたため、driver 自身はFD mapのschema、seal、identity、SHAを検証せず、manifest pathを再openしていた。

## 今回の変更点

- resident driver が既存FD-map envをdevice queryより前にexact parseするようにした。
- FD-mapのsealed memfd、strict JSON、self-hash、schema/status、closure contract、全bindingのfield set、role/path/descriptor uniqueness、method/closure、FD identityを検証する。
- `served_manifest` はlogical argv path、`regular_file`、`control_input/read`、single link、安全mode、bounded size、map SHAと一致することを要求する。
- served manifest bytesは継承FDをdupし、bounded `pread`で一度だけ取得する。同じbytesをSHA-256とstrict manifest parseへ渡し、logical manifest pathは再openしない。
- manifest-relative resourceのbaseとREADY evidenceにはargvのlogical pathを維持する。
- READYへ `ullm.aq4_p2_served_model_binding.v2` を追加した。raw FD番号は記録せず、`descriptor_transport=inherited_fd_map`というsemantic bindingだけを記録する。
- FD-map envがない非profile経路は従来のpath loaderを維持した。`/proc/self/fd/N`をlogical manifest argvとして渡す旧経路は明示的に拒否する。

## 検証

- resident driver unit tests: `20 passed`
- served-model loader unit tests: `3 passed`
- logical pathを無効なreplacement markerへ交換した後も、継承FDの旧bytesから読み込み、replacementをopenしない回帰testが成功した。
- map/child FD closed、role、method、closure、path、SHA、identity、seal、schema、unknown field、nonregular、unsafe mode、multi-linkの負例がfail-closedした。
- map absent path compatibility、READYの12-field exact binding、旧`/proc/self/fd` logical path拒否が成功した。
- `cargo check -p ullm-engine --bin ullm-aq4-p2-resident-driver`、担当2ファイルの`rustfmt --check`、`git diff --check`が成功した。
- actual、GPU workload、service操作は行っていない。

## 次の行動

- P3 runnerはdriver argvのlogical manifest pathを維持し、既存FD mapとchild FDsをそのまま継承する。
- runner側READY validationとprocess evidenceは、新しいsemantic `served_model_binding`をexact検証し、ephemeral FD番号をdurable evidenceへ保存しない。
