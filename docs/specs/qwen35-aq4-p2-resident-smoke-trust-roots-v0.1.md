# Qwen3.5 AQ4 P2 resident smoke trust roots v0.1

## 前回の要点

初版のoffline bundle validatorは、bundle内のcase、identity、file hashを相互照合していた。この方法では、攻撃者が意味とhashを同時に再束縛した場合に、外部の正しい値との差を判定できない。

## 今回の変更点

v3 validatorの正しさはbundle内の値から導出しない。次の独立trust rootからpre-run payloadを再構築し、JSONの全階層とexact bytesを比較する。

- current source/runner Git commit `3dc4aa612b6cfd87675d0bd9fe506426f43e64f9` とtree、runner blob、expander blob、fixture generator blob
- normative driver Git commit `319d6187b29e877536aa5dfe80c02bde0c77ed7a` とtree、resident source blob。current `3dc4aa6`でもresident source blobがbyte不変であることをGit blobで検証する
- detached clean worktree buildで確定したresident binary SHA-256
- `/etc/ullm/served-models/active.json`、active worker、package manifest、1045-file package tree、required guard set
- official P2 case manifestと、trusted expanderが生成するofficial case

R9700 host bindingはofficial caseを上書きして隠さず、source device、bound `gfx1201` device index 1、visible device 1を独立したruntime-binding objectとして記録する。fixture、identity、synthetic preflight、policy、fake-readyは、この明示的なbindingとtrust rootsから決定的に再生成する。

`dry-run.json`は手作りしない。bundleに同梱したcurrent `3dc4aa6`のtrusted runnerを、prepare時に`--one-case-smoke --dry-run`でsubprocessとしてexactly once実行する。runner自身がbundle v3、case/fixture/identity binding、synthetic fake-readyをvalidate-only handshakeで検証して生成したplanだけを採用する。subprocessのexact argv、exit code、stdout、stderr、それぞれのSHA-256、plan SHA-256を`runner-dry-run-evidence.json`へ固定する。

validatorはplanのexact schemaと、1 case、12 transactions、warmup 2、measured 10、`execution_mode=one_case_smoke`、`smoke_only=true`、`promotion_eligible=false`、`validation.mode=validate_only`、fake-ready handshake passedを独立検査する。通常profileは84 casesとして別経路に固定し、one-case成果物を通常84-case成果物へ昇格させない。

bundle memberはsingle-link regular file、exact mode、固定SHA-256、exact directory coverageを要求する。JSON duplicate、全階層のunknown/semantic drift、symlink component、hardlink、open前後のfile identity drift、最終passまでのTOCTOUを拒否する。外部trust rootsもopen前後と検証終了時にfile identityを再確認する。

launch bindingは、trusted runner validate-only argvと、detached driverを直接起動するresident driver argvを分けて固定する。後者はdetached driver absolute path/SHA、`--served-model-manifest` absolute path/SHA、device index 1、normative driver build commit、protocolを固定する。served manifestと全protocol linkに対するabsolute/no-parent-traversal契約がnormative driver sourceに存在することもblobから再検証する。

検証終了時にはbundle root directoryを再列挙し、exact names、member type、inode、nlinkとdirectory自体のidentityを初回snapshotと比較する。検証途中に追加・削除・置換されたentryを拒否する。

旧`0fd7993` bundleは`superseded_historical_prepared`、`execution_eligible=false`として履歴だけを残し、実行入力として使用しない。

`status=prepared_not_executed`、`promotion=false`、`service_touched=false`を固定し、actual runtime identity、power、VRAMはnullとする。synthetic readyの`model_loads=1`はprotocol形状検査用であり、実model loadの証拠として扱わない。

## 次の行動

実GPU smokeへ進む場合は、このbundle validationとは別の承認単位でdevice lockを取得し、actual runtime identity、power、VRAM、model-load evidenceを新しいrun artifactへ記録する。

## B v4 sidecar更新

### 前回の要点

791a20c input rootは19 membersをexactに持つ。root内のprepared dry-run/evidenceをfinal generic runner出力で置換すると、runner planが列挙する入力member SHAとplan自身のSHAが循環する。

### 今回の変更点

final generic runner trust rootをcommit `e9065925d7b5af0352cb8dfd454a7e106abd7172`、tree `9f2ff38d06d5ea5724a6e84af1c00d2b8147f241`、runner Git blob `9c097d1a97af3e15ca695c6da08b1e2928d08df7`、source SHA-256 `3140574c4f50f9b09aeb3780e400cbf8020ecf1c4ff69da685622858128f33cc`へ固定する。resident driverはe906592とnormative 319d618でGit blobが不変であり、clean binary SHA-256 `62f720835de60a61bad0a9aab5b80d778624d4d97ef5c8998e179418dab730f1`を継続する。

791a20c rootはbyte、member、mode、SHA256SUMSを変更しない。Bはroot外の`resident-one-case-smoke-binding-v4` sidecarとして、R input rootのpath/device/inodeと全19-member inventory、final runner source、checked-in validator source、実runner subprocess argv/exit/stdout/stderr、runner plan、mandatory validator reportを固定する。validatorはsymlink componentを持たないabsolute checked-in pathと事前指定SHAの一致をR自身が検証してからexactly once実行する。runnerもprepareからexactly once実行する。

sidecarのrunner plan、outer subprocess evidence、validator reportは相互のSHA-256を固定する。生成物はinput root外に置き、root内`dry-run.json`を置換しない。このためRのgeneric root schemaとBの実行証跡に自己参照cycleはない。旧791a20c artifactは`SUPERSEDED`、`execution_eligible=false`として記録する。ただしinput rootとしての履歴的検証対象は保持する。

### 次の行動

L immutable launcherがinput root fingerprint、B sidecar manifest SHA、R runner SHA、validator SHAを固定し、validatorからrunnerの順序を強制する必要がある。B単体は`launch_eligible=false`であり、GPU、model load、live service操作の許可にはならない。
