# Lightweight promotion route self-test v0.1

This record validates the generic route against the running AQ4_0 Qwen3.5-9B
service. The candidate is a machine-created semantic self-test: it has the
same JSON meaning as the starting active manifest but intentionally different
raw bytes. It therefore exercises the strict `active_snapshot.raw !=
rollback_snapshot.raw` path without introducing a new model binary.

| Attempt | Result | Evidence |
| --- | --- | --- |
| Direct host probe | Stopped before mutation | `lightweight-promotion-selftest-v0.1-promotion/` |
| First bridge-container run | Activation passed; rollback restored bytes but systemd rejected its start at `start-limit-hit` | `lightweight-promotion-selftest-v0.1-container-promotion/`, `lightweight-promotion-selftest-v0.1-container-rollback/` |
| Final bridge-container rerun | Promotion and generic rollback both passed | `lightweight-promotion-selftest-v0.1-rerun-promotion/`, `lightweight-promotion-selftest-v0.1-rerun-rollback/` |

The direct host probe did not swap `active.json`: this deployment binds the
gateway on the Docker bridge, which is intentionally not reachable from the
host network namespace. The generic `--gateway-container` transport was added
instead of creating a candidate-specific route.

The first bridge run is retained as factual evidence. Its rollback copied the
saved bytes back exactly (`bytes_equal_rollback: true`), but the subsequent
`systemctl restart` hit the unit's rate limit and did not prove service
availability. The service was repaired with `reset-failed` followed by a
single start. The generic tool was then changed to perform that bounded,
recorded recovery only after it has positively observed `start-limit-hit`.

The final rerun is the conclusive validation:

- active input and final rollback SHA-256:
  `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`;
- semantic self-test candidate SHA-256:
  `159f4d743b65977bc3602bc613216693bcd7f50812fc3d6338fa97e3cdd73b1c`;
- active and candidate generation records: 10 each; the readable side-by-side
  text is in `rerun-promotion/comparison.md`;
- automated blocking findings: none; exact-match rate 1.0 (diagnostic only for
  this identical-semantics test);
- promotion performed one successful `systemctl restart`; rollback performed
  one successful `systemctl restart`, took eight bounded readiness probes, and
  then received HTTP 200 with a non-blocking generated response;
- the final `validate-served-model.py` result retained worker SHA-256
  `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b`.

The root-owned transaction and append-only ledger for the successful rerun are
under `/var/lib/ullm/lightweight-promotions/20260726T111155944069Z-818283c23a84c672/`.
They are intentionally not copied into the repository because they are live
machine state; their SHA-256 values are recorded in the outcomes.
