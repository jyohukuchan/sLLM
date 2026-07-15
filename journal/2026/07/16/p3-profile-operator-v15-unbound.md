# P3 profile operator v15 unbound skeleton

## 前回の要点

- command-v14はcommit `ba7ab7d41c6de84a9165aa8e3592a9b18fcb0e6d`で認可され、actual-v14は1/1回実行後のimmutable failureとしてcommit `a2fe1ebac5d631919ca9082e17fda2126759a385`、35 filesへ封印された。
- actual-v14はreturn code 1、retry false、restore passedであり、失敗理由は`kernel trace row 83 interval/order is invalid`だった。
- operator-v13はcommit `764045355ee06c3b5c53f296d4bcbe47e1495ece`のinvocation 0 blocked stateとしてhistorical専用に維持する必要があった。

## 今回の変更点

- current namespaceをquiet-v20、command/result/audit-v15へ更新した。
- 未確定のmaintenance evidence、runtime、execute evidence、captureは`None`のUNBOUND状態とし、4 rootがすべてPath、absolute、unique、previous-v14非重複になるまでfail-closedにした。
- ready、maintenance、capture、offline reassemblyの新authorityは追加せず、current-v15 authority flagもfalseで固定した。したがって、authority統合前にはquiet収集、current audit、operator生成を開始できない。
- operator-v13 validatorはlive worktree absenceを参照せず、commit `7640453`のGit treeでabsenceを検証するhistorical poststate-independent境界へ変更した。
- quiet-v19、command-v14、actual-v14の専用validatorを追加した。actual-v14については6 rootのtree、全35 files、SUMS、result/audit schema、return code 1、invocation 1/1、retry false、restore passed、owner/cleanup/failure reason、journal parentageを検証する。
- current-v15の後続ファイルを生成してもv14 actualとv13 blocked stateが変化しないpoststate independence testを追加した。
- v15 finalizer fixtureはresult/audit schema v15と、previous-v14に重ならない一時actual namespaceを使用する。
- source/tests commitは`b5df6192874314d47e55628176c2b4b937b46d35`、treeは`ec3f9243c43381a59f17f18403955d03f105a605`である。
- source blobは`f035cb9e80c895704701582b9b5c6759253c92db`、raw SHA-256は`0e92d5388f1a131c5fc8f8f86a5ef51b0a42f009302270deefb29f18c671e27a`である。
- test blobは`bd3b14ff1b9586fd0dc738a0b2a46f2f0fcbcfe3`、raw SHA-256は`d9f262d1cfb3182acfcb3c7c26e712160dcf42817e95c579e4418cc2a06ca023`である。
- `tests/test_prepare_aq4_p3_profile_operator.py`は58 passed、py_compileとdiff checkも通過した。後続commitでHEADが進んだpoststateでもsource path last-change authorityを含めて58 passedだった。
- GPU、service、benchmark artifact、actual executionには触れていない。

## 次の行動

- maintenance/runtime/execute-evidence/captureのversion planと、ready/maintenance/capture/offline authority commitが確定した後にだけ、4つのUNBOUND pathとcurrent authorityを統合する。
- 統合時にはprevious-v14 6 rootとの非重複、current fresh9 absence、ready launcher binding、exact argvを再検証する。
- 新しいsource/tests authorityをcommitした後、quiet-v20をread-onlyで収集し、command-v15は別の明示的認可があるまで生成・実行しない。
