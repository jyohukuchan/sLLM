# Subagent model selection probe

- `terra_probe` と `luna_probe` の2つの subagent は正常に起動・完了した。
- 現在の subagent 起動APIには、モデル名や推論強度を指定する引数がない。
- タスク文として `GPT5.6-terra-xhigh` / `GPT5.6-luna-xhigh` の役割名は渡せる。
- 両 subagent とも、実際の基盤モデルIDや選択結果を示すメタデータは参照できなかった。
- 結論: subagent 自体は利用できるが、現インターフェースから terra / luna を明示選択できたとは検証できない。
