# CPU strict-F32 SMT scaling v1

`summary.json` is the review entry point. Each configuration directory holds
the immutable `measurement.json` and one small `run.json` receipt per worker.
The runs used `--no-capture`, so they do not contain or overwrite corpus
payloads.

The actual corpus baseline is separately recorded in `summary.json` from a
189-second in-progress capture interval. The selected configuration remains
eight 8-thread workers on CPU 0--63. The best SMT result was only 1.0033x that
physical-core no-capture steady rate; all other SMT layouts were slower.

The launcher now requires explicit `--allow-smt` for logical CPUs 64--127.
Compatible resume keeps the corpus identity immutable (frozen gate, binary,
artifact/package, seed, 8-thread worker plan, and jobs) while recording a
separate immutable scheduling invocation. A 16-thread resume of this existing
8-thread corpus fails before worker launch.
