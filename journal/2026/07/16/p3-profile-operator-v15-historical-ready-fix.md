# P3 profile operator v15 historical ready validation fix

## 前回の要点

- operator-v15 authorityはcommit `f280dd306e2aed5a4c2560c0d956a47623d9511f`でcurrent ready-v17とoffline-v12へbind済みだった。
- quiet-v20開始前のread-only pollは、historical ready-v15 validatorがcurrent maintenance 9ff2 loaderへready-v15を渡し、対応path外として拒否されたため、成果物生成前にfail-closedで停止した。

## 今回の変更点

- `historical_ready_v15_authority()`からcurrent maintenance loaderへの依存を除去し、b39 artifact専用の自己完結validatorへ変更した。
- ready-v15 commit `b39e21822db40e7fd5060da66db885b3a9ff0b8a`、tree `4daa8f0cafe93274aeddd902bea58727633b3080`、ready root tree `8045019bc2346efccc3c37781fc8bd6280e95dac`、dry root tree `b375ac9a0e55b738715dd637d38b864ccf6a2204`を固定検証する。
- ready binding、SUMS、trust、QAのraw SHA-256、4-member inventory、Git archive byte一致を検証する。
- embedded maintenance sourceはcommit `2167c33fe56c0efcbd3745055e6de8604aafd456`、tree `b76cdd6937d3f5f63565049596d8192ed6f87cd2`、blob `cf4fedca1912cc6cbe54ffbd63456c3ff1dbba53`、raw SHA-256 `f86f5be10968eab00f1fabae7827cd557514437098545049ac82def2ddbf2f0c`をGit objectから再読込して検証する。
- embedded QAは13 distinct files、685/685 passedを再集計し、各source commit/pathからGit blobを再解決する。historical v10 output semantics、actual eligible、measurement/promotion false、restore timeout 120秒、dry-runのGPU/service falseと全process count 0も検証する。
- current ready-v17/v16とcurrent maintenanceの定数を参照せず、後続poststateから独立している。root inventory、source bytes、QA aggregate、trust source pathのtamperを拒否するテストを追加した。
- 実際のread-only `audit_current()`を通すintegration testを追加した。
- source/tests commitは`7a11d5b204cff70a5b1dfd55b7fe7be3879a5e41`、tree `870e0ef206983735b333ec31aea6863a9a140f8d`である。source blob/rawは`d203782344946c9046e4e9ee7b98807836c4561b` / `abe26f141030a2ba48bee57050d1d42880221a18658f5ce8d4a75a27cdad801a`、test blob/rawは`6e7da7a11e75b65fb2e3d2f19574b9a9ee8f27fd` / `3713b9eb1f2de2219b8653604bc15ab416e63d9338ad709a8765d18c2b7906dc`である。
- targeted 5/5、commit前full 66/66、commit後full 68/68 passed。py_compileとdiff checkも通過した。
- quiet、command、actual artifactは生成しておらず、GPU workloadとservice操作は行っていない。

## 次の行動

- Lunaの独立read-only QA後に、開始前4 stable pollsを最初からやり直す。
- 4 pollsがcleanかつ同一のときだけ、quiet-v20の既定27 samples、130秒以上、reset 0、final confirmationへ進む。
