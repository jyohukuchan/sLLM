# P3 profile quiet-v19 / command-v14

## 前回の要点

- v14再認可source/testsはcommit `c69070e12c474c62f26c671dee5bf1c2ea72d570`、journal poststateは`297c64426ccdf8f2a1e970865bc01193f084a127`に封印された。
- command-v13は`authorized_not_invoked_preflight_blocked`、reason `external_owner_after_seal_before_invocation`、invocation 0/1、fresh9 absent、result/audit absentとして固定された。
- ready-v16 commit `09324284ab27d61642f126d8e052fa05c1cbb3cf`とoffline-reassembly-v11 commit `aa26f4e85dbdf2bc000c32a9869fc22b6597e888`は不変だった。

## 今回の変更点

### 開始前の安定poll

- 外部SQ8 family、GPU owners、service、fresh9を15秒以上の間隔で4回read-only監視した。
- poll時刻（Unix ns）は`1784154548698911284`、`1784154570573571085`、`1784154603474959227`、`1784154625366072409`。
- 4回ともcargo、ullm_engine、SQ8 workload、対象`/tmp` workerは不在だった。
- production serviceはMainPID `2356631`、active/running、NRestarts 0で不変だった。
- production workerはPID `2357251`で、AMD-SMI/KFD ownersは4回とも`[2357251]`だった。
- current v14 fresh outputsは4回とも9/9 absentだった。

### quiet-v19

- operator既定値を変更せず、interval 5秒、maximum 900秒、minimum span 130秒、27連続clean sampleで収集した。
- 結果はGO、27/27 clean、span `358.729450382`秒、reset 0、final confirmation passed。
- 全sampleとconfirmationでblocking identity、HEAD `297c64426ccdf8f2a1e970865bc01193f084a127`、tree `a09830599394e4b4888d99e733aad7b7e320fe89`、service epoch、worker、owners、fresh9 absenceが単一だった。
- 全sampleでprevious-v13 stateは`authorized_not_invoked_preflight_blocked`、invocation 0だった。
- quiet JSON SHA-256: `946b778aab81c5ab555ecd427a0c6548dc3326c7f2558244ff1a3affd447af1a`
- quiet `SHA256SUMS` SHA-256: `52dbf1058ca113932b1dcee57147235c26a539ae88a50bfa452af7bfe1ac1434`
- commit: `1a45447b1eaa76a645fff6cca31cc007f034b4ff`
- overall tree: `253f0fd6b080c4d152e493452bb7013189379c31`
- root tree: `a12d0d7e9b10a734bf9d8518ce5ba445c3351787`
- JSON blob: `2807596ef29c42f75ea4f4c7ccb0b7ba640d6b62`
- SUMS blob: `2c5c411485568c4a17ff44b976f54e9eb8e9d33c`
- archiveとGit objectのSHA-256一致を確認した。

### command-v14

- exact-one pending manifestとして1回生成し、selfhash、SUMS、semantic validatorを通した。
- manifest file SHA-256: `6a85c47818e7fe97fda348203f0721e883e6bbe31c18366c19e76a22ff0f72d3`
- manifest selfhash: `bf95bd0e4c2146abbb083d48db0effb744da09a98261d91382b98cd562cfc45e`
- command SHA-256: `5693d75b17f91187b6841566815ad717d001a91280d651860aa127dc20277079`
- command `SHA256SUMS` SHA-256: `e041014c7c77103a8c2237d5331508be7061bb7075d3deed24d9159eaafd8af0`
- commit: `ba7ab7d41c6de84a9165aa8e3592a9b18fcb0e6d`
- overall tree: `60e0e6d29f29ac2545e0cbdfe7dff6da44a38598`
- root tree: `6b0ad082ad999eb7e9269686949beb4868dca1a8`
- manifest blob: `6695e0af2d436cbfe998a6ffa7af41c06ba99aa1`
- SUMS blob: `1c93b0a6d1ed6620a828517a13c641a6a59020f9`
- archiveとGit objectのSHA-256一致を確認した。
- previous-v13のcommit/tree/root/raw/selfhash/command、invocation 0/1、fresh9 absent、result/audit absentをpinした。
- previous actual-v12のcommit `44617f7fd46c39f71f04502b248739cc116fe095`、tree `813c4ffc88fb58cf8764b91d3c80cea9ef351f0f`、35 files、invocation 1/1、retryなしをpinした。

## 安全境界

- result/audit-v13、result/audit-v14、maintenance evidence-v11、runtime/execute-evidence/capture-v10は未生成のまま維持した。
- actual execution、GPU workload、service stop/start、外部process操作は0回だった。

## 次の行動

- quiet-v19とcommand-v14を独立監査へ渡し、commit/tree/blob/raw/selfhashとprevious-v13 stateを再確認する。
- command-v14はexact-one pendingとして保持し、別の明示的actual指示まではmanifest argvを実行しない。
