# P3 profile operator v13

## Scope

- Prepared the offline `quiet-window-v18`, `operator-command-v13`,
  `operator-result-v13`, and `actual-audit-v13` cascade.
- Bound the new profile namespace to `profile-ready-v16`, profile output v10,
  maintenance evidence v11, and capture v10 without generating any actual
  artifact or touching the GPU or service.

## Historical authority

- Validated sealed operator command v12 at commit
  `2185ac90f7188402c60280e87b8eded3cbfc65e8`.
- Validated sealed actual v12 at commit
  `44617f7fd46c39f71f04502b248739cc116fe095`, tree
  `813c4ffc88fb58cf8764b91d3c80cea9ef351f0f`, with exactly 35 committed
  files, one failed invocation, no retry, preserved capture-failure streams,
  and passed restoration.
- Kept historical paths separate from current v13 paths so later ready,
  quiet, command, maintenance, execute, capture, result, audit, and dry-run
  artifacts do not change the sealed v12 classification.

## New input authority

- Bound execute-binding v10 at commit
  `2b477ed0dd1344d368e684e413cb756706af22f3` and its launcher trust to source
  commit `fc4559ee4fb8c7c1e62353fb3978a1a1e0a7d86d`.
- Kept the execute binding's ordinary execute outputs distinct from the
  profile diagnostic v10 outputs.
- Preserved profile-ready-v15 at commit
  `b39e21822db40e7fd5060da66db885b3a9ff0b8a` as historical authority and
  validated its original maintenance source and QA manifest from Git objects.
- Bound current profile-ready-v16 and its process-zero dry run at commit
  `09324284ab27d61642f126d8e052fa05c1cbb3cf`, tree
  `984136dfc469d15394f00bba8e1adfca742ad30f`, with exact per-root trees,
  ready-binding hash, dry-run evidence hash, and both `SHA256SUMS` hashes.
- Confirmed the canonical actual capture-v10 output remains absent and the
  ready artifact remains `actual_eligible: true` after the distinct offline
  reassembly poststate exists.
- Bound the distinct offline reassembly-v11 capture and sidecar at commit
  `aa26f4e85dbdf2bc000c32a9869fc22b6597e888`, tree
  `79446c68a4e0c4b4782a37d6a48646f8583d92f3`. Formal readback validates 42
  sealed files under the two artifact roots, zero GPU/service/operator work,
  and the unchanged 35-file actual-v12 source seal.
- Bound current maintenance source `c4fe279e6c0bf9a8899c2cd36642f45bf145fe8f`,
  its 170-test authority `6af8dfa47968fed55b1f198bb03409f496bfb6c1`,
  and capture parser `eb00cbd83b90d6fd8d519f6662ddea16d5f4438c`.

## Verification

- `python3 -m pytest -q tests/test_prepare_aq4_p3_profile_operator.py`
  - 37 passed.
- `python3 -m py_compile tools/prepare-aq4-p3-profile-operator.py tests/test_prepare_aq4_p3_profile_operator.py`
- `git diff --check -- tools/prepare-aq4-p3-profile-operator.py tests/test_prepare_aq4_p3_profile_operator.py`

- Source and tests were committed as
  `c6562fd6` before the final trusted-source readback.
- Current ready, historical ready, offline reassembly, and 20 trusted sources
  all passed exact readback; all nine future actual outputs remained absent.
