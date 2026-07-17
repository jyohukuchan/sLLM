# SQ8_0 implementation plan

2026-07-09

- User clarified that `FQ8_0` was a typo and should be `SQ8_0`.
- Added `uLLM-project/docs/plans/sq8-implementation-plan-v0.1.md`.
- The plan reframes old SQ FP8 candidate work as formal `SQ8_0` implementation work:
  - `SQ8_0` is the public format ID.
  - `sq-fp8-w8a16-r9700-v0` is implementation lineage / legacy alias.
  - FP8 quality acceptance is not the blocker; implementation correctness and regression diagnostics are.
- The plan defines implementation scope, compatibility, artifact boundary, Rust loader boundary, runtime kernel boundary, backend dispatch integration, C++ organization, milestones, regression categories, completion criteria, and immediate work queue.
- Updated `uLLM-project/docs/words.txt` so `FQ8_0` is explicitly documented as a typo and `SQ8_0 implementation plan` is defined.
