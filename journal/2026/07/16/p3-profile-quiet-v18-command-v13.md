# P3 profile quiet-v18 / command-v13

## 前回の要点

- operator verifierのmode契約を修正し、実`audit-current`をcleanに戻した。
- source/testsは`dd725b6db382c757530459f857c94288ed1d2035`、poststate journalは`a8afd2fab34e2154472e58c9d587b7c6e82f18f7`で封印した。
- quiet-v18、command-v13、result-v13、actual-audit-v13は作業開始時点で未生成だった。

## 今回の変更点

### 外部SQ8 family監視

- 厳密pollを15秒以上の間隔で4回実施した。
- poll時刻（Unix ns）は`1784151898426969134`、`1784151920263545258`、`1784151940280379346`、`1784151962137904578`。
- cargo、ullm_engine、SQ8 workload、`/tmp/ullm-sq8-main-integration`配下の実行workerは4回とも不在だった。
- production serviceはMainPID `2356631`、`active/running`、`NRestarts=0`で不変だった。
- production workerはPID `2357251`で、AMD-SMI/KFD ownerは4回とも`[2357251]`だった。
- 別のread-only監視シェルがコマンド文字列中の`sq8`で一度誤検出されたため、実行ファイル・作業ディレクトリ・プロセス種別に基づく厳密pollへ切り替えた。その監視シェルはGPU ownerではなく、外部操作は行っていない。

### quiet-v18

- operator既定値を変更せず収集した。interval 5秒、最大900秒、最低span 130秒、連続clean sample 27件。
- 結果は`GO`、27/27 clean、span `355.922490087`秒、reset 0、最終confirmation passed。
- 全sampleとconfirmationでblocking identity、HEAD、tree、service epoch、worker、GPU owners、trusted sources、fresh9 absenceが単一だった。
- quiet JSON SHA-256: `0fb7e3346e7f38d0b9d844d3bac2815b533945eb7d25b3981ac3d5542eb36e00`
- quiet `SHA256SUMS` SHA-256: `081e220fd195c3576eeced4d59464c309be4d1304bb5cfbc771cbe197c59608b`
- commit: `cb774ac0090380d4fff5b613a942fad9b3d106c8`
- overall tree: `add160bacc5f372cd21bbaa6840ebcb1735c94f4`
- quiet root tree: `18c7e4c0c83142bab61be025022e77696c259ea7`

### command-v13

- exact-one pending manifestとして生成し、selfhash、SUMS、semantic validatorを通した。
- manifest file SHA-256: `78168089ff34e2eb8560bcaa85c94f49c0f3ae23ee4a614f0d0fc7e077a0d4f0`
- manifest selfhash: `42c8498adc6c8f97382ef17421d3145a14d50126a549a66d0693f114f8cad313`
- command SHA-256: `5693d75b17f91187b6841566815ad717d001a91280d651860aa127dc20277079`
- command `SHA256SUMS` SHA-256: `1c157f9d864b4e75d62e2acc7b5b5189b1765e3795b3109ef4e815df26b87fd6`
- commit: `764045355ee06c3b5c53f296d4bcbe47e1495ece`
- overall tree: `cb73e9c7c34c884eac567510f6d89da238b57a49`
- command root tree: `d187b2902aa9f83503c17d6c0c8665210744f2e0`
- previous command-v12: commit `2185ac90f7188402c60280e87b8eded3cbfc65e8`、state `authorized_sealed`、maximum invocation 1。
- previous actual-v12: commit `44617f7fd46c39f71f04502b248739cc116fe095`、tree `813c4ffc88fb58cf8764b91d3c80cea9ef351f0f`、35 files、state `executed_sealed`、invocation 1/1、retryなし。
- result-v13、actual-audit-v13、maintenance evidence-v11、profile runtime/evidence/capture-v10は未生成を維持した。
- actual execution、GPU workload、service stop/startは0回。

## 次の行動

- command-v13は監査待ちのpending authorityとして保持する。
- explicitなactual実行指示が出るまでは、manifest argvを実行せず、result/audit/actual outputsを未生成のまま維持する。
