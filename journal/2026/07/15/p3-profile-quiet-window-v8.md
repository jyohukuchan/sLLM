# P3 profile quiet window v8

## Scope

- Read-only pre-actual rolling observation for the fresh profile v4 boundary.
- Sampling policy: maximum 900 seconds, nominal interval 5 seconds, at least 27 consecutive clean samples, and at least 130 seconds from the first through last clean sample.
- PTS inventory changes were diagnostic-only. Targeted external `systemctl`, maintenance, capture, `rocprof`, GPU diagnostic, or probe commands remained blocking.
- Actual execution, GPU workload commands, model loading, sudo, and service operations were forbidden and were not performed.

## Binding and blocking gates

- Final HEAD: `4cadba39e1310f28552abcbe55c88425149d6614`.
- Final index tree: `93588f9642eb6bafaafed497c641c4844f5557f0`.
- The relevant set contained 50 regular files across the prepared, binding-v4, execute-binding, ready, and profile-ready roots plus the relevant source/test files.
- Relevant byte aggregate: `98dc3cc51efdfdd8e719a3e67c06a34c2c18fb88864c410a603e5211e930d8f9`.
- Relevant no-follow identity aggregate: `013a6612c2b4e30adeba59ce2fe6e2c16570a61bf867c3feb556695d19dd7563`.
- Service identity, unique worker PID, lock inode/holder, formal container-namespace health, and exclusive AMD/KFD ownership stayed fixed.
- The fresh profile v4 execute output, execute evidence, maintenance evidence, and diagnostic capture paths remained absent through confirmation.

## Rolling result

- Decision: `GO`.
- Final streak: 29 consecutive clean samples over `137.175538188` seconds.
- Monitoring elapsed time: `138.959114225` seconds.
- Final-run reset count: 0.
- PTS diagnostic identity change count: 0.
- Confirmation sample passed after final verification.

Three preliminary collector checks were intentionally stopped before any evidence root was created. They identified and removed three diagnostic-only values from blocking identity comparison: host-direct route timeout behavior, the whole `/proc/locks` digest, and per-probe capture timestamps. The formal endpoint results, exact lock inode/holder, and every other blocking gate remained enforced. These preliminary checks did not execute actual work, issue a GPU workload command, or touch the service.

## Final verification

- Strict QA provenance resolved exactly 12 of 12 source commit/path/blob bindings.
- `SHA256SUMS` passed for all five bound roots.
- The four canonical targeted maintenance/readback/dry-run tests passed (`4 passed in 0.31s`).
- Start and end formal health captures passed with identical blocking identity.
- Quiet-window evidence SHA-256: `c48ed2885b7b53e8e9ee62fcd2f67274ee54329b209e56e577e603afbf49d85b`.
- Evidence `SHA256SUMS` SHA-256: `580c0eb28ac32382276c5571fd9eec99380985f4b9e61653597947e77900a9b4`.
- Collector source SHA-256 recorded in evidence: `ac0de260f4e1563add4645a1dd88a89608b7b2ecf6392a4ef690c227f3945365`.

## Safety result

- `read_only=true`
- `actual_executed=false`
- `gpu_command_executed=false`
- `service_touched=false`
- `secret_material_recorded=false`
