# AQ4_0 P3 pre-promotion coordination record

## Immutable inputs verified

- Expected active manifest SHA-256:
  `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`
- Observed immediately before the pending promotion window: the same SHA-256.
- Candidate manifest SHA-256:
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`
- Candidate validation and generic promotion dry preflight both returned `ready: true`.

## First attempted window and containment

At 21:18:37 JST, a baseline gateway start was attempted while BK's GPU singleton-lock owner was
still active.  The active manifest was not changed and `tools/promote-served-model.py --yes` was
not invoked.  The gateway's first start exited with `WorkerBusy: another process owns the GPU
singleton lock`; systemd made one automatic restart and the old active worker then became ready.
The gateway was stopped at 21:19:52 JST to release the R9700 for BK.

Consequences:

- `NRestarts=1` after containment.
- No candidate binary executed through the gateway in that attempt.
- No active-manifest bytes were changed.
- No rollback action was needed.

The service has `StartLimitBurst=3` and `StartLimitIntervalSec=900` (reported by systemd as
15 minutes).  The failed start was 21:18:41 JST; no subsequent start will be attempted before
21:33:51 JST, and only after all listed GPU/promotion processes are absent and the active SHA is
rechecked.

## GPU and service boundaries

The direct P3 timings used the R9700 (`gfx1201`) only, with the gateway stopped.  The prohibited
`llama-qwen35-udq4.service` was confirmed `inactive` and `disabled`; it was never started.

An initially completed BK Python controller later spawned/continued its actual
`ullm-sq8-r9700-prefill-tail-fix-profile` workload.  Promotion remains paused until both the
controller and its worker have exited.  A process merely displaying a `promote-served-model.py
--help` command is not treated as an active promotion; an actual `--yes` process would block this
window.

## Attempt 1 and subsequent BH ownership

After BK's wrapper restored the old worker, the generic lightweight promotion tool passed its
input preflight and baseline readiness probe but stopped with
`baseline_failed_before_mutation`.  The active code-generation request had started at 21:45:58
JST.  Global journal records show that a different session issued `systemctl stop
ullm-openai.service` at 21:45:59.160 JST while that request was active; the following EOF is a
teardown effect, not a demonstrated active-worker failure.  Candidate bytes were never made
active.  The complete immutable tool output was copied to
`../lightweight-promotion-attempt-1/`.

BH began an explicit service stop and acquired `/run/ullm/r9700.lock` at 21:46:18 JST for its
decode-attention measurement.  A later old-service start at 21:47:11 JST therefore failed with
`WorkerBusy`; that was a deliberate ownership conflict, not a candidate start.  No additional
start/restart attempt is made while this lock is held.  The manifest remained
`c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`.

At 21:57:48 JST, an unattributed other session issued another `systemctl start` while BH still
held the same lock.  Its systemd retries at 21:58:01 and 21:58:15 also failed with `WorkerBusy`;
the retry scheduled at 21:58:28 reached `StartLimitBurst=3` and was rejected.  This external
collision left `NRestarts=3` / `start-limit-hit` behavior in the service history.  It did not
change the active manifest or execute the candidate.  For safety, no new service operation will
be attempted before **22:13:29 JST** (15 minutes after the last attempted start), and then only
after BH has released the lock and the active SHA is rechecked.

## Final safe window and activation

BH released the lock and restored the old gateway before the second promotion attempt.  Immediately
before that attempt, no promotion/rollback/measurement process was present, the active SHA was
still `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`, and
`NRestarts=0`.  The fresh generic run used a new evidence directory, performed exactly one
successful `systemctl restart`, required no StartLimit recovery, and activated manifest
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` at 22:26:36 JST.

Post-activation service state was `active/running`, `Result=success`, and `NRestarts=0`.  The
rollback tool's no-`--yes` preflight returned `ready: true`, binding the saved old manifest
`c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` to the active candidate.
`llama-qwen35-udq4.service` remained `inactive` and `disabled` throughout.

BJ then opened a separate SQ8_0 handwritten-projection measurement window at 22:28:42 JST and
explicitly stopped the candidate gateway while holding `/run/ullm/r9700.lock`.  This happened
after the promotion suite and activation outcome had completed.  The candidate manifest did not
drift.  BJ's trap restored the gateway; the final independent check observed `/readyz` HTTP 200,
the running P3 worker executable SHA-256
`ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257ed74e7723636ace9dfca69b`,
and `active/running`, `Result=success`, `NRestarts=0`.

## Handoff-only concurrent window

At 22:35:50 JST, after the preceding BJ window had restored the gateway and after the P3
activation had already completed, BJ began a second `--speed-first` SQ8_0 window.  It stopped the
gateway at 22:35:52 JST and held `/run/ullm/r9700.lock`.  At the 22:41 JST handoff observation,
the manifest was still `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`, while
the gateway was intentionally `inactive/dead` with `Result=success` and `NRestarts=0`.  No AQ4
deployment service command was issued during this lock-held window; its trap owns the eventual
gateway restore.  The historical post-activation response/ready check remains valid and this
later stop is not a rollback or a P3 promotion failure.
