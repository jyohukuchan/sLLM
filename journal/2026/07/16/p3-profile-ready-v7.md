# P3 profile ready v7

## 前回の要点

- launcher/runtime v7 authorityはlauncher commit `60461d796ba64a7f0ba28353cb4f263d08d18dab`、execute-binding artifact commit `c5d17d7524608ea4438ba82a1fb51637b0876de6`で確定した。
- capture parser authorityはcommit `e86cf512183574340ddfc6564477395766262092`、SDK ROCTx SHA-256は`1a5831a3817eac29f63d1442dc348ba31b417202b7ce15f3aed9c09a8f4773c9`。

## 今回の変更点

- maintenance final authority `3fc2b8cd6f6910fbebd3ff4728855d55bf2cbbd2`、tree `9b3cb93d921d4829505f6243ac6adf7e43cfabbd`、blob `9b5566a0f6d1381732342c0ee26f9778c54f852b`、raw SHA-256 `6a964e0dc93c889a31e28e89ccbc25ba5e0db095aad3d7c2ca427230c36428b0`をpinした。
- fresh profile-ready-v7を生成し、status `ready_for_one_case`、execution mode `profile_diagnostic`、actual eligible trueをformal loaderで確認した。
- fresh profile-ready-dry-run-v7を1回実行し、sudo、systemctl、launcher、rocprof、capture、Docker、health probeを含む全process countが0であることを確認した。
- dry-runではservice、GPU command、model loadを実行していない。
- ready binding JSON SHA-256は`cb3df439fc0fcf4ed403d27048b87311dbee4ed12c733cb2e6250af3ba977977`、ready SUMS SHA-256は`fdffda0011b407c037d3c54664b280a6f8b70a5a52f5231f77f2124d72c7dc57`。
- dry-run evidence JSON SHA-256は`9cac57a7883699c796f124f5b0ccdb22e8664b5b5d89a23b68f2823c596c3879`、dry-run SUMS SHA-256は`1be77044ddb9b3cda271b3530200f3ec0c2b05185d34b70f17d7200e5b35c816`。
- 両rootは`0555`、全filesは`0444`かつnlink 1。SUMS、artifact専用CPU tests 2件がpassedした。
- maintenance authorityの全CPU QAは543/543 passed。old profile-ready-v6、dry-run-v6、profile actual/capture failure evidenceは不変。
- GPU、production service、actualは実行していない。

## 次の行動

- profile-ready-v7 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- 独立QAと新しいquiet/operator chainが確定するまではactualを実行しない。
