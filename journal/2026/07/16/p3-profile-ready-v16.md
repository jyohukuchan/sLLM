# P3 profile ready v16

## 前回の要点

- ready-v15／dry-run-v15はcommit `b39e21822db40e7fd5060da66db885b3a9ff0b8a`で封印済みで、生成時source／embedded QAを使うhistorical artifactとして保持する。
- offline再構成はcanonical actual outputと分離し、actual capture-v10／maintenance-evidence-v11をabsentに保つ方針になった。

## 今回の変更点

- historical loaderを生成時source commit/tree/blob/rawとembedded QAから再構成する方式へ変更し、後続HEADでもready-v15を検証可能にした。trust、QA、source authorityを再封印して改変する負例はすべて拒否した。
- current sourceをcommit `c4fe279e6c0bf9a8899c2cd36642f45bf145fe8f`、tree `49685f2b9194d6128d8e92ad04d52c01540eed38`、blob `53ad6ab6eeec43eb77478397ad0fcd8c09caa45b`、raw SHA-256 `4330469041c664454165844e2f1de452f207ddd27814876d4f35caf9775698c4`へ固定した。
- current ready-v16／dry-run-v16を各1回生成した。readyは`ready_for_one_case`、`actual_eligible=true`、profile diagnostic、maximum invocation 1、canonical capture-v10のoutput no reuseを維持した。
- ready binding、harness trust、QA attestation、ready SUMS SHA-256は順に`54c218a203a19643eae8983bfb2ac84b8132341dc04d42e7dc30f080ea02e42d`、`ef614c668a5d59e76849fa83b7cbd2afc5e005875402c568002f762f0ea9afb8`、`1b9d6704f55b898be36f2ac237a69ede91472cdfe535ea96dcfd309a356a3738`、`76fc710b78d384c890f3a3c7c21dad7ed912299a21e20e431085e0d319e78686`である。
- dry-run JSON／SUMS SHA-256は`82d7a17c5d71c9e0e4019280a1f14a75c569ad8faa6bb2b00702088dd0d93f17`／`d6b11306091f1132ef1485bf01ee67005b5335ccb32060b6f25987fd8a7c7fc8`である。
- current loaderとhistorical v15 loader、両rootの`SHA256SUMS`がpassedした。dry-runは全process count 0で、service、GPU、model、capture、actualを操作していない。
- QA manifestは13 test files、690/690 passedで、maintenance 170/170、正規driver環境付きcapture 80/80を含む。

## 次の行動

- ready-v16／dry-run-v16のartifact authorityを下流へ渡す。
- offline再構成はcapture-offline-reassembly-v11とmaintenance-offline-reassembly-evidence-v11だけへ生成し、actual capture-v10／maintenance-evidence-v11はabsentに保つ。
