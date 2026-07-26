# AQ4_0 `matvec_add` optimization — 2026-07-27

## Scope

This directory contains the evidence for a gfx1201-only investigation of
`ullm_aq4_matvec_add_f32_kernel` in the production AQ4_0 Qwen3.5-9B decode
path.  The intended comparison is an in-process, identical-worker A/B between
the retained pre-specialization shuffle body and the group-size-specialized
candidate.  Profiler traces are used only for launch and diagnostic evidence;
unprofiled full-model runs are the throughput authority.

## Phase 0 finding

[`phase0-static-analysis.md`](phase0-static-analysis.md) records the static
ISA, resource, geometry, payload, and residual-traffic audit.  In brief:

- The prior add and the successful SiLU-mul body both already reduce with
  wave32 shuffles, without an LDS tree or spills.  Reduction replacement is
  therefore not a remaining add optimization.
- Add's generic g8/g16 traversal contains invalid/predicated slots and dynamic
  packed-byte selection.  SiLU-mul has explicit group-size traversal and
  shares each input pair between two weight streams.
- At C=1339, 72.97% of add payload is g16 MLP-down, so both g8 and g16 need
  to be preserved but the g16 path is the dominant one.
- Residual read plus output write are at most 2,097,152 bytes/token for 64
  launches, only 0.1689% of the 1,241,513,984-byte weight payload.  They do
  not explain the 52.4594% versus 83.2007% payload-bandwidth difference.

The static candidate reduces the whole-code-object count from 1,434 to 820
instructions and SALU from 922 to 395, with no spill or LDS allocation.  It
raises VGPRs from 30 to 49, so the result is intentionally not accepted until
the full-model R9700 A/B is complete.

## Candidate and validation design

The candidate keeps AQ4_0 low-nibble-first decoding, g8/g16 scale addressing,
group scaling, row scaling, f32 output, and residual-add order.  The exact
pre-specialization shuffle source remains selectable only by
`ULLM_AQ4_MATVEC_ADD_USE_SHUFFLE_REFERENCE=1`, allowing the staged worker to
run a source-identical rollback reference in a separate clean process.

The first main-based staging build was rejected before measurement: current
main does not contain the 4:1 grouped-split implementation that the active
artifact uses, so an A/B from that build would not be a production comparison.
`candidate-grouped-build-provenance.json` will instead bind a worker rebuilt
from `9d864350`, whose `part_01.inc` and HIPRTC-source blobs are identical to
BZ's `c8074928` grouped artifact, plus only this candidate's two source-file
changes.  `run-candidate-window.sh` is fail-closed: it requires the BZ
protected `/opt/ullm/` active artifact, checks its grouped-decode contract,
uses `/run/ullm/r9700.lock`, verifies greedy tokens before timing, confirms
the 292-module/64-add launch invariant for both bodies, and restores the
service once after the window.

## Measurement status

The locked R9700 window completed.  The retained shuffle reference versus the
group-specialized candidate was 74.591159 versus 78.284628 full-model grouped
decode tok/s (1.049516×); cold p=2048/M=128 prefill was 974.984645 versus
977.087601 tok/s.  The candidate passed GPU/CPU differential tests, runtime
greedy equivalence, and the 292-module/64-add launch invariant.  See
[`post-window-results.md`](post-window-results.md) for the static ISA,
counter limitation, thermal, trace diagnostic, release, and service record.

The lightweight promotion transaction completed with ten actual prompt-suite
responses, no blocking finding, and `activated` outcome.  A later active
manifest change returned the service to the pre-existing BZ SHA; per the
no-unexpected-overwrite rule this task did not re-activate the candidate.
The validated root-owned candidate artifact remains under `/opt/ullm/`.
