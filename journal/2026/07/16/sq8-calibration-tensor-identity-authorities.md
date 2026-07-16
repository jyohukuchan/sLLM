# SQ8 calibration tensor identity authorities

- 前回の要点: 一回だけの calibration GPU capture は 24/24 rows を immutable target に publish 済みで、strict validator は通過した。offline metrics build は target と binding manifest の `tensor_names` の順序差で fail closed した。
- 今回の変更点: binding manifest は actual receipt の manifest SHA と正規 48 tensor 名の集合に、capture target は同じ 48 tensor 名の runtime 数値順に、それぞれ独立して固定した。target artifact は exact 5-field shape と文字列型も検証し、未知・欠落・型・順序・重複・未知名・binding SHA 改ざんの回帰を追加した。
- 検証: focused は 53 passed / 112 subtests、calibration 関連 Python 全回帰は exit 0、`git diff --check` は通過した。target の変更、GPU 再実行、holdout 実行はしていない。
- 次の行動: 修正を commit し、同一 immutable target で offline comparison、metrics、freeze、execution ledger を固定する。
