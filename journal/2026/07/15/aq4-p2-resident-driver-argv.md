# AQ4 P2 resident driver argv passthrough

## 前回の要点

resident batch runnerの`--driver-command nargs='+'`は、resident driverの`--served-model-manifest`などをrunner optionとして再解釈し、production argvを透過できなかった。

## 今回の変更点

- `--driver-command`を最後のoptionとし、以降を`argparse.REMAINDER`で保持する。
- driver argvを7要素のproduction schemaへ固定し、device indexとbuild commitをidentityへ照合する。
- one-case modeではbundleの`launch-command.json`とも完全一致させる。
- option風引数の透過正例と、末尾追加、順序変更、欠落、one-case差し替えの負例を追加した。
- 通常84-case fake live testはproduction形argvのまま84件を完走する。

## 次の行動

このrunner修正を独立コミットにし、その後にbenchmark worker registry file snapshotの強化へ戻る。
