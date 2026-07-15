# P3 profile operator v15 authority binding

## 前回の要点

- operator-v15はquiet-v20、command/result/audit-v15の骨格まで用意されていたが、current actual pathとauthorityは未確定だった。
- path-bound preworkではmaintenance-evidence-v12、profile-execute-v11、profile-execute-evidence-v11、capture-v11を候補として割り当て、execute-binding-v11を通常execute namespace専用の非実行authorityとして追加していた。
- previous operator-v13はinvocation 0のblocked state、previous actual-v14は1/1実行後のsealed failureかつrestore passedとして変更不能に維持する必要があった。

## 今回の変更点

- current readyをprofile-ready-v17へ、offline readbackをoffline-reassembly-v12へ固定し、`CURRENT_V15_AUTHORITY_BOUND = True`とした。
- ready-v17はartifact commit `19f7d390b97b1e8f0daa72e1007267c27ab4061b`、repo tree `42b1298def7eb20aa1e1b307160400463bce26cb`、ready root tree `452edaf85efd34c77b8fecf02a2f00ba8b76a592`、dry root tree `b27e322e1ed6bd901fc84e468fef961f193f77a4`へ固定した。ready binding raw SHA-256は`0bcf74c243c0744bdfa66af44d641fc230dbf1819e2f18443948aa2406f1cc9a`、ready SUMSは`20b724180f313e9f662af8905193ba4636c7e91b176840df7a98fde496e83d9f`である。
- ready-v17はmaintenance loaderで形式的に再読込し、schema v1、status `ready_for_one_case`、actual eligible true、run id profile-diagnostic-v11、current 4 output path、restore timeout 120秒を確認した。dry-run-v17はGPU/service false、全process count 0である。
- maintenance sourceはcommit `9ff2b8861f6d91935679db3bdf1b4af37bc6a543`、tree `ac63aa5dadb213ba4f1f43ef6ff3b2a9c8157e2a`、blob `1d7c99815a4ee3cbc5f8ef4f5ab438d752338dcd`、raw SHA-256 `fb2fc515570b1889c21bd170434159845048971339a084a85027707f79665345`へ固定した。maintenance testはcommit `fb9ee7efab27359dd89348f79d7c7bed1fbd1a67`、172/172 passedである。
- capture sourceはcommit `418e507214b2a4c0352ac8867bf9689b81948ca4`、tree `dc0100092c6e0fa85d66a6082c134349544f5e83`、blob `95c4e156e3546aa7fe2ff29a3ff00f39b0932b22`、raw SHA-256 `afd3eec63e3621984f500f3f99457173081bed8e04a141a117daf8c1372941ef`へ固定した。capture testはcommit `376b733b097db37701529014e4e698093976d689`、84/84 passedである。
- offline-v12はartifact commit `fcd0bbdae7e27e137e2b149f701298146e50e878`、repo tree `f913d49d8e19f8f4e4abde577bea35044ad205bd`、capture root tree `96e35590e749d833bd82083540ac363ec8c90a6f`、evidence root tree `7e7aa5190fac27c5a987ffccb945c4ccb4084184`へ固定した。2 root合計は42 filesである。
- offline capture SUMSは`555b8a1711b02cb51dbb5d1b3bc5ee1f2c4a28feabe858e7f13d78b85a99ce73`、artifact raw/selfは`f07b520a46b1b0e641e10c3a22179aa4c499a337a3328505e49955f9be333f4a` / `d7d55ce12bfa3f91b0945922ed73aff900dff436d76a5646f9f78bd481ec8fcc`、evidence raw/selfは`731a8b81c9553280412f8ae3d45028bf71f5c64cc5439d5296461fcd2a99d991` / `0a4b1f421f807cb9e2fe857d05e9ef198c4a957608966c0df9bb26f1bde9a657`である。
- maintenance loaderでoffline-v12を形式的に再読込し、evidence schema v12、capture schema v2、parser/generator provenance、kernel normalization、member count 40、measurement/promotion false、workload/rocprof/GPU/service/operator/actual/model load count 0を検証した。
- execute-binding-v11はcommit `9111b2a6c9479ebccb61a55641b5be52f86d5dda`、root tree `f76c878764aff5d4290bc48967928c0d1e1f6bac`、launcher source commit `4cd57c1c0da182224df15c842e072dcc2c4a1de0`へ固定した。これは通常execute-v11 path namespaceの証明だけであり、profile namespace authorityとexecution authorityはfalseである。
- current actualは6 root、fresh setは正確に9 pathで、全て不在である。quiet-v20とcommand-v15も不在であり、operator result-v15、actual audit-v15も不在である。previous v13 blocked stateとprevious v14 sealed failure/restore passedは不変である。
- current-v15 authority bindingの判定は`GO`である。このGOはread-only authority統合だけを示し、quiet収集、command生成、artifact生成、GPU処理、service操作、actual実行を認可または実施したことを意味しない。
- source/tests commitは`f280dd306e2aed5a4c2560c0d956a47623d9511f`、treeは`6152a682703d1593da09144de6a951e9bf57212c`である。source blob/rawは`18b260d1be44dac2c0702f5b6dba398e639770ef` / `5ffa7c86183e57c554bf73fea3292634c6a47e81012dd46a3bdae7192d2c660d`、test blob/rawは`6e5f2289acbe16f09087caa8e3d538a96e49e5f0` / `aa3a7c2a353bbc77eac1f27092ac98a6384f815c77893001f6453a88244d1e39`である。
- targeted authority testsは10/10、source commit前の全体は61/61、source commit後の全体は62/62 passedだった。py_compileとdiff checkも通過した。

## 次の行動

- 独立監査ではsource/tests commit、ready-v17、offline-v12、launcher-v11、fresh 9/9 absence、previous v13/v14不変性を別経路で再確認する。
- quiet-v20の収集、command-v15の生成、actual実行は、それぞれの明示的な後続作業として扱う。
