# SQ8 authorization lineage v2

## 前回の要点

- v1は実装GO、capture No-Go 2件、actual failure 2件、restore No-Go 1件のexact 6 entriesだった。
- actual failureが増えるたびに固定件数・relation順を変更する必要があり、継続的な認可履歴に適さなかった。

## 今回の変更点

- `ullm.sq8_authorization_lineage_input.v2`を追記専用のtyped manifestとして追加した。
- sequence順、許可relation、path/SHA/schema/status/request/source commitを全entryでlive検証し、current GO exact 1、capture No-Go 2以上、restore No-Go 1以上、actual failure 3以上を必須にした。
- predecessorのlive identityと旧entriesの完全prefix一致により、履歴の削除・置換・並べ替え・重複を拒否する。
- v2 referenceへentry countとcurrent implementation audit path/SHAを追加した。builderはactual authorization時に同auditのCLI path/SHAを必須とする。
- v1の検証互換は残したが、prepared diagnostic専用としてactual authorizationを拒否する。
- Python gatewayにもv1/v2 typed contractとv2 entry/predecessor live validationを追加した。Rust served-model parserはこの作業では変更していない。

## 検証

- lineage validator: 17 passed。
- builder: 11 passed。
- Python gateway served-model loader: 62 passed。
- runner/generatorを含む関連集合: 142 passed、既存deployment profileのworktree外絶対pathに依存する1件のみ対象外。
- GPU、service、sudo、actual retry: 0。

## 次の行動

1. current sourceのimplementation GO audit receiptをcreate-newで完成させ、path/SHAを確定する。
2. 実在する3件のactual failureを含む初版v2 manifestをcreate-newで生成する。
3. v2入力からfresh runtimeをmaterializeし、独立runtime audit後にだけ次のone-shot authorizationへ進む。
