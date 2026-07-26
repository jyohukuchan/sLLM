# AQ4_0 runtime-hardening activation execute r3

Plan `0e12fe09ad4d00578ee74f1bcc730a6b401e63a6fc91bb1d237346251e8f81f8` was activated once on 2026-07-26 JST after the explicitly authorized, current-time preflight. The locked r3 control route returned `status: activated`; it did not enter rollback.

The authoritative root-owned immutable records remain outside this repository:

- intent: `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/activation-intent.json` (`268ceb1b4fa78d80ffd2b9c7e191cea6fff5aa4e898a5c04b079b5183d5c1de1`)
- outcome: `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/outcome.json` (`b022f91aa6118f379a79e59a6d35e30ba90b348511bdc789cfdd1c8c97f2d340`)
- candidate live proof: `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/proofs/candidate-live-proof.json` (`a5f623e238e55f7829818bea96c861a77b2305d6e2f58d3948f2bf910da7fbed`)
- reusable isolated-worker receipt: `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/proofs/candidate-isolated-preflight.json` (`65c4cd4d595e83e1c9bdeef34e14fab3c7f2dcddc85cca28cbf28a25c7f1973f`)

`preflight.json` records the eight required admission checks. `activation.json` binds the successful immutable route records and live-proof result. `postflight.json` records the final manifest, worker, guard, service, rollback-copy, and GPU observations. `inference-smoke.json` contains only structural metadata from one successful live chat completion; neither its prompt response body nor credentials were stored.

The isolated-preflight command was invoked again through the locked wrapper. For an already sealed plan it validates the reusable immutable receipt rather than launching a second candidate worker; the receipt's actual worker observation reports `ready_after_milliseconds: 3195` and deliberate SIGTERM cleanup. This is the control route's no-replace/idempotent behavior, not a fresh worker process at activation time.
