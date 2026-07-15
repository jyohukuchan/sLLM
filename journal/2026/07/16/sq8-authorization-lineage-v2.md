# SQ8 authorization lineage v2

## 前回の要点

- v1は実装GO、capture No-Go 2件、actual failure 2件、restore No-Go 1件のexact 6 entriesだった。
- actual failureが増えるたびに固定件数・relation順を変更する必要があり、継続的な認可履歴に適さなかった。

## 今回の変更点

- `ullm.sq8_authorization_lineage_input.v2`を追記専用のtyped manifestとして追加した。
- sequence順、許可relation、path/SHA/schema/status/request/source commitを全entryでlive検証し、current GO exact 1、capture No-Go 2以上、restore No-Go 1以上、actual failure 3以上を必須にした。
- 初版v2 migrationはv1 exact6をlive typed validationし、旧GOをhistorical relationとして意味と順序を保ったsequence 0..5へ正規化する。
- 初版はv1 predecessor schema/path/SHAとmigrated prefix digest/countを固定し、v1 sourceのcommit/tree/archiveに対応する最新failureとv2 sourceに対応するcurrent GOを追加したexact8だけを許可する。
- 9件目以降はv2 predecessorのlive identityと旧entriesの完全prefix一致を必須にし、履歴の削除・置換・並べ替え・重複・second migrationを拒否する。
- v2 referenceへentry countとcurrent implementation audit path/SHAを追加した。builderはactual authorization時に同auditのCLI path/SHAを必須とする。
- v1の検証互換は残したが、prepared diagnostic専用としてactual authorizationを拒否する。
- Python gatewayにもv1/v2 typed contractとv2 entry/predecessor live validationを追加した。Rust served-model parserはこの作業では変更していない。

## 検証

- lineage validator: 25 passed。
- builder: 11 passed。
- Python gateway served-model loader: 62 passed。
- runner/generatorを含む関連集合: 142 passed、既存deployment profileのworktree外絶対pathに依存する1件のみ対象外。
- GPU、service、sudo、actual retry: 0。

## 次の行動

1. migration実装をcommitし、その新しいsource全体のimplementation GO audit receiptをcreate-newで完成させる。
2. v1 predecessorと実在する3件のactual failureを含む初版exact8 v2 manifestをcreate-newで生成する。
3. v2入力からfresh runtimeをmaterializeし、独立runtime audit後にだけ次のone-shot authorizationへ進む。
