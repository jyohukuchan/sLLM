# AQ4 profile maintenance v11 recascade

## 前回の要点

- immutable actual-v8 は、最初に device lock の `device` / `inode` 契約差で rc 1 となり、retry 0 の failure evidence として commit `4b651cd5c46212349b5a598b344da6ea11993d30` に封印された。
- producer は device lock と diagnostic live preflight の current evidence contractを commit `dac045244d7609c42c2db1ea0f91aa707ffb717b` で検証するようになった。

## 今回の変更点

- maintenance が launcher commit `b81066dbf86857afbeb0dc7d41493fdef680266d`、tree `ba44559a4778504eaef37dc2cf4d052076fab838`、blob `a9f0498d9dc51b276addda0410560b2d8e696859`、SHA-256 `bcd25ffa719e04d8535560ec179506f1bcae2ede417023d3d6303c864dadb5e3` を固定するように更新した。
- capture tool commit `a098ca53c1c3e5c16ec02a08013c55b82f18301c`、tree `ad8396a5ebf6b632d362042830fcaff3bf995c3c`、blob `dc1b475f006702d30aab3f777c4e055a823a0c1b`、SHA-256 `d0d7093e2fe8575c1105432cabf801c04b3deee8b6772d792382486116657527` を固定した。
- fresh namespace を profile-ready-v11、profile-ready-dry-run-v11、profile-maintenance-evidence-v8、profile-execute-v8、profile-execute-evidence-v8、rocprof-capture-v8 へ進めた。runtime/evidence paths は launcher authority から継承する。
- QA exact manifest を current committed blobsへ更新し、aggregate を 623 collected / 623 passed / 0 failedへ更新した。
- historical ready v7-v10 と actual-v8 の全 sealed roots を SHA256SUMS で再確認する回帰境界を追加した。既存 root は変更していない。
- 検証は maintenance trust chain 382件、resident driver 22件、ROCTx 5件、capture 58件、producer 103件、family exclusion 27件、selector 26件の合計623件がpassedした。GPU、service、actualは実行していない。

## 次の行動

- maintenance source commit後、その commit/blob/SHAを authority として profile-ready-v11 を生成する。
- profile-ready-v11 と dry-run-v11 を封印・commitし、operator v9 の ready authorityへ再連鎖する。
- actual-v8は過去のfailure authorityとして不変に保ち、新しいauthorizationでのみfresh v8 runtime/capture outputsを一回生成する。
