# Qwen3.5 AQ4 P2 resident smoke trust roots v0.1

## 前回の要点

初版のoffline bundle validatorは、bundle内のcase、identity、file hashを相互照合していた。この方法では、攻撃者が意味とhashを同時に再束縛した場合に、外部の正しい値との差を判定できない。

## 今回の変更点

v3 validatorの正しさはbundle内の値から導出しない。次の独立trust rootから全payloadを再構築し、JSONの全階層とexact bytesを比較する。

- normative Git commit `319d6187b29e877536aa5dfe80c02bde0c77ed7a`、tree、resident source blob、runner source blob、expander blob、fixture generator blob
- detached clean worktree buildで確定したresident binary SHA-256
- `/etc/ullm/served-models/active.json`、active worker、package manifest、1045-file package tree、required guard set
- official P2 case manifestと、trusted expanderが生成するofficial case

R9700 host bindingはofficial caseを上書きして隠さず、source device、bound `gfx1201` device index 1、visible device 1を独立したruntime-binding objectとして記録する。fixture、identity、synthetic preflight、policy、fake-ready、dry-runは、この明示的なbindingとtrust rootsから決定的に再生成する。

bundle memberはsingle-link regular file、exact mode、固定SHA-256、exact directory coverageを要求する。JSON duplicate、全階層のunknown/semantic drift、symlink component、hardlink、open前後のfile identity drift、最終passまでのTOCTOUを拒否する。外部trust rootsもopen前後と検証終了時にfile identityを再確認する。

launch commandはdetached driver absolute path/SHA、`--served-model-manifest` absolute path/SHA、device index 1、build commit、protocolをexact argvとして固定する。served manifestと全protocol linkに対するabsolute/no-parent-traversal契約がnormative driver sourceに存在することもblobから再検証する。

検証終了時にはbundle root directoryを再列挙し、exact names、member type、inode、nlinkとdirectory自体のidentityを初回snapshotと比較する。検証途中に追加・削除・置換されたentryを拒否する。

旧`0fd7993` bundleは`superseded_historical_prepared`、`execution_eligible=false`として履歴だけを残し、実行入力として使用しない。

`status=prepared_not_executed`、`promotion=false`、`service_touched=false`を固定し、actual runtime identity、power、VRAMはnullとする。synthetic readyの`model_loads=1`はprotocol形状検査用であり、実model loadの証拠として扱わない。

## 次の行動

実GPU smokeへ進む場合は、このbundle validationとは別の承認単位でdevice lockを取得し、actual runtime identity、power、VRAM、model-load evidenceを新しいrun artifactへ記録する。
