P2 tests create their bounded synthetic artifacts in a temporary directory.

No model weights, prompts, logits, GPU traces, or live-request data belong in
this fixture directory.  The production path is tested only for fail-closed
behaviour when its real binary/package/trace prerequisites are absent.
