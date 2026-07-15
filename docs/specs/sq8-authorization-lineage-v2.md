# SQ8 authorization lineage v2

`ullm.sq8_authorization_lineage_input.v2`は、SQ8 actual promotionの認可に使う
追記専用の証跡manifestである。v1の固定6件契約は診断用に限り受理し、actual
authorizationには使用できない。

## Manifest

manifestは`schema_version`、`disposition`、`source`、`predecessor`、`entries`の
exact objectである。`source`は認可対象のcommit、tree OID、source archive
SHA-256を固定する。`predecessor`は常に必須である。初版v2だけはv1 predecessorの
schema、canonical absolute path、manifest SHA-256、migrated prefix SHA-256、count 6を
指定する。以降は直前のv2 manifestについてschema、path、manifest SHA-256、canonical
entries SHA-256、entry countを指定する。

初版migrationではv1 exact 6 entriesをlive typed validationし、旧GOをhistorical
implementation auditとして保持した共通v2 entryへ、意味と順序を変えずsequence 0..5に
正規化する。sequence 6にはv1 source commitに対応する最新actual failure、sequence 7には
v2 source commitに対応するcurrent GOだけを追記できる。最新failureのsource provenanceは
v1 sourceのtreeとarchiveにも完全一致しなければならない。これにより初版はexact 8となる。
9件目以降にv1 migrationを再利用できず、直前v2 predecessorが必須となる。

通常のv2追記ではpredecessorをlive readし、旧entriesが同じ順序・同じ内容で完全な
prefixとして残り、少なくとも1件が末尾に追加された場合だけ受理する。v1/v2とも削除、
置換、並べ替え、relation rewrite、重複、source spoof、predecessor cycleを拒否する。

## Entry

各entryは次のexact fieldを持つ。

- `sequence`: 0から始まる連番。canonical orderを定義する。
- `relation`: 許可されたrelation enum。
- `path`、`sha256`: source receiptのlive immutable identity。
- `schema_version`、`status`、`request_id`、`source_commit`: receipt内容から検証する型付きidentity。

許可するrelationは`implementation_ready_current`、
`capture_implementation_no_go`、`restore_implementation_no_go`、
`actual_failure`、`historical_implementation_audit`、
`historical_runtime_audit`である。pathとSHA-256はmanifest内でそれぞれ一意でなければ
ならない。manifest、predecessor、全source receiptはcanonical absolute path、regular
file、mode 0444、link count 1、non-symlink、bounded read、読み取り中identity不変を
必須とする。

認可可能なv2は、manifest source commitに一致するcurrent implementation GOをexactly
1件、capture implementation No-Goを2件以上、restore implementation No-Goを1件以上、
`actual_failed`を3件以上含む。

## Reference and authorization

`ullm.sq8_authorization_lineage_ref.v2`はexternal input path、runtime copy path、manifest
SHA-256、canonical entries SHA-256、entry count、current implementation auditのpath/SHAを
固定する。同じreferenceをpromotion request identity、profile、prepared/actual receipt、
served-model、build receipt、Gate、runner snapshot、SHA256SUMSへ伝播する。

actual authorizationのbuilder呼び出しではv2 manifestに加え、
`--current-implementation-audit-receipt`と
`--current-implementation-audit-sha256`を必須にする。両値はcurrent GO entryとexact一致
しなければならない。これによりcurrent GO receiptが完成する前でもコードとテストを
確定でき、完成後にcreate-new manifestをmaterializeできる。
