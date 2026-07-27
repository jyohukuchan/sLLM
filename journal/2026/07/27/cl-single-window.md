# CL: SQ8_0 tile-128 quality then Qwen3.5-35B-A3B AQ4_0 MoE

## 前回の要点

- CK recorded tile-128 speed but failed before numeric/quality capture because
  the capture output contract was not respected.
- CH fixed the MoE mRoPE validator and the current release generator was
  rebuilt before this window.

## 今回の変更点

- Corrected `run-sq8-grouped-tile-sweep-window.sh`: `sq8_ck_serving` creates
  the supplied capture directory itself, so each route now supplies a fresh
  `numeric/<route>` directory rather than a pre-created directory or an
  `oracle` child.
- Took exactly one R9700 service window.  The first attempted correction was
  incomplete: it removed the capture target *and* its parent, producing
  `No such file or directory` before GPU runtime initialization.  The runner
  fail-fast path restored production, hence tile-128 quality and MoE execution
  were not reached.  No second window was taken.
- Manifest SHA-256 was unchanged before/after:
  `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`.
  The service is active with `NRestarts=0`; an OpenWebUI bridge completion
  returned HTTP 200 / `restored`.  The disabled llama service remains inactive.
- MoE release generator SHA-256 verified as
  `6ee827e43fa4e4a5e54fd66c1b20eb444e05632245f66349e10cfe409b9e39cd`.
  It was not launched.

## 次の行動

- A new explicit GPU-window authorization is required to run the now-corrected
  tile capture and then the MoE physical check in a fresh single window.
- Do not infer tile-128 quality, split-vs-direct error, MoE VRAM admission,
  generation, routing, or speed from this setup failure.
