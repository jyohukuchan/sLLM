# 参照source採用計画

## 状態

**完了（2026-08-02）**

2026-08-02の候補調査結果から正式なlocal参照sourceを選び、取得済みcheckoutと文書正本を同期する。過去の取得計画・履歴は当時の事実として変更しない。

## 採用判断

- LMDeploy `v0.15.0` とKTransformers `v0.6.4` だけを正式採用し、40桁の完全commit SHAで固定する。
- MLC LLM、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIは今回未採用とし、cloneせず、採用予定に置かない。
- 参照sourceはsLLMの対応実績、正しさ、直接reuse許可を意味しない。llama.cpp以外はreader-onlyとする。

## 完了条件と結果

| 完了条件 | 結果 |
| --- | --- |
| source lockを現行7件へ更新する | [source-lock manifest](../../../../../references/source-lock.md) に両sourceのorigin、release、完全SHA、license、checkout、再現コマンドを追加した |
| KTransformers固有のsubmodule状態をfail-closedに扱う | 未初期化gitlink 4件、各statusの `-`、空worktreeを個別検査する手順を記録した |
| 参照範囲と採用判断を同期する | [推論engine参照](../../../../../references/inference-engines.md) を固定7件へ更新し、未採用6件から将来優先表現を除いた |
| 開発計画とCI再調査対象を同期する | main planとactive CI計画を固定7件へ更新した |
| 文書の整合性を検証する | whitespace、relative Markdown link、旧件数、採用済みsourceの将来候補扱い、KTransformersのsubmodule誤記を検査した |

[対応する履歴](../../../../../history/2026/08/1-10/reference-source-adoption.md)
