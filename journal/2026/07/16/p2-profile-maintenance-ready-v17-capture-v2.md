# P2 profile maintenance / ready v17 capture v2 cascade

## 前回の要点

- launcher-v11 source authorityは`4cd57c1c0da182224df15c842e072dcc2c4a1de0`、execute-binding-v11 artifact authorityは`9111b2a6c9479ebccb61a55641b5be52f86d5dda`である。
- actual-v14はcommit `a2fe1ebac5d631919ca9082e17fda2126759a385`でfailure evidence 35 filesを封印済みで、raw capture-v10のkernel trace orderを理由に失敗した。
- ready-v16、offline reassembly-v11、actual-v14はhistorical artifactとして不変に保つ。

## 今回の変更点

- capture fix authorityをsource commit `418e507214b2a4c0352ac8867bf9689b81948ca4`、tree `dc0100092c6e0fa85d66a6082c134349544f5e83`、blob `95c4e156e3546aa7fe2ff29a3ff00f39b0932b22`、raw SHA-256 `afd3eec63e3621984f500f3f99457173081bed8e04a141a117daf8c1372941ef`へ固定した。test authorityはcommit `376b733b097db37701529014e4e698093976d689`、blob `6e8a76d30702bf3f2f42fb511fde91091dd1b60c`、raw SHA-256 `51e9d51a881f8e8044332078a493082391192db9469dfa9a08c7746774fab776`で、canonical driver環境の84/84 testsがpassedした。
- maintenanceはcapture schema v2のexact top-level keysを要求し、raw kernel/markerからmarker-bounded groupsを再構成する。raw kernel before/afterのidentityとSHA-256、全10 derived split、row/dispatch/correlation/duration conservation、normalization provenanceをpinned capture moduleで再計算する。
- actual future namespaceはprofile runtime-v11、execute-evidence-v11、capture-v11、maintenance-evidence-v12へ進めた。offline laneはcapture-offline-reassembly-v12とdistinct maintenance-offline-reassembly-evidence-v12へ進めた。
- offline-v12 generatorはactual-v14 raw capture-v10をsourceにし、旧generic memcpy adapterと手動sortingを除去した。canonical offline-v12 artifactsは生成せず、後続P3へhandoffする。
- historical ready-v16 loaderはembedded maintenance sourceとlauncher-v10 sourceを再構成する。historical offline-v11 loaderはGit root trees、SHA256SUMS、embedded generator/parser authority、actual-v12 derivationを検証する。
- maintenance direct test authorityはcommit `fb9ee7efab27359dd89348f79d7c7bed1fbd1a67`、tree `57ee6918bf8415a3831de54a7245098a88b84ec3`、blob `6abb7d6eadd6e8a17b5faf656d84279a43c053c7`で、172/172 testsがpassedした。
- maintenance source/QA authorityはcommit `9ff2b8861f6d91935679db3bdf1b4af37bc6a543`、tree `ac63aa5dadb213ba4f1f43ef6ff3b2a9c8157e2a`、blob `1d7c99815a4ee3cbc5f8ef4f5ab438d752338dcd`、raw SHA-256 `fb2fc515570b1889c21bd170434159845048971339a084a85027707f79665345`である。
- QA manifestは13 distinct files、resident trust 416、driver 22、ROCTx 5、capture 84、selection producer 105、family exclusion 39、selector 26の合計697/697 testsをjobs 1でpassedした。QA attestation SHA-256は`b0984765a29fadd23aa58d2936b8daeb91d9a85237df88017e814174389fb91b`である。
- ready-v17とdry-v17を各1回生成した。ready binding、harness trust、QA attestation、ready SUMSのSHA-256は順に`0bcf74c243c0744bdfa66af44d641fc230dbf1819e2f18443948aa2406f1cc9a`、`d9c6c3002b1d430e5ba4f82236a48c7dbb8659e4838529b3105e54dc51b89ead`、`b0984765a29fadd23aa58d2936b8daeb91d9a85237df88017e814174389fb91b`、`20b724180f313e9f662af8905193ba4636c7e91b176840df7a98fde496e83d9f`である。
- dry evidence/SUMSのSHA-256は`6ad059d952b3832793b0d6820938eb6556ea90c247ebf5e2e2c6db85025edc06`／`77a0ede95ee41d223035e619e411e4c75e14e19795b9f78c5227d136a9253979`で、全process count 0、service/GPU/model flagsはfalseである。
- ready artifact authorityはcommit `19f7d390b97b1e8f0daa72e1007267c27ab4061b`、repository tree `42b1298def7eb20aa1e1b307160400463bce26cb`、ready root tree `452edaf85efd34c77b8fecf02a2f00ba8b76a592`、dry root tree `b27e322e1ed6bd901fc84e468fef961f193f77a4`である。両rootは`0555`、6 filesは`0444`かつnlink 1である。
- ready-v16/dry-v16、offline-v11、actual-v14は各artifact commitとの差分がない。actual future 4 pathsとoffline future 2 pathsは全てabsentである。GPU、service、model load、actualは実行していない。

## 次の行動

- P3はoffline capture-v12とdistinct offline maintenance evidence-v12を、今回のmaintenance source authorityから生成・検証する。
- actual実行を行う場合はready-v17を単一authorityとし、runtime-v11／execute-evidence-v11／capture-v11／maintenance-evidence-v12を各1回だけ使用する。
