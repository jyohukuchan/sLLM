# AQ4 chain 0--11 resource estimate

## Scope and safety boundary

- This run reuses the committed Phase 2 hybrid fixture: 3 contexts and 9 output records.
- It invokes `ullm-aq4-layer0-family-isolation` only through its CPU chain mode and compares against the BF16 source on CPU.  It does not invoke a GPU, a resident service, a systemd unit, or an active manifest.
- The requested range is `0:11`: 12 layers, with linear-attention layers `0--2`, `4--6`, and `8--10`, and self-attention layers `3`, `7`, and `11`.

## Compute estimate

The committed 0:3 measurement contains exactly one `[linear, linear, linear, self]` block.  Range 0:11 contains exactly three such blocks with the same fixture and 9 records, so its layer-local work is estimated at **3.0 times** the prior measurement.  The Phase 2 commits bracket the original implementation/evidence work by nine minutes, rather than exposing a clean command duration; therefore this run uses a conservative 45-minute wall-clock cap and records `/usr/bin/time -v` output.

## Memory estimate

The relevant AQ4 package files for one layer total approximately 118 MiB for a linear-attention layer and 114 MiB for a self-attention layer.  The comparator loads BF16 tensors one at a time; its largest single MLP tensor is `12288 * 4096 * 2 = 96 MiB`.  The largest linear recurrent state is `32 * 128 * 128 * 4 = 2 MiB` per active sequence.  Fixture residuals and each streamed output frame are at most tens of KiB; all 12 layers' frames would be about 1.7 MiB even if retained, but they are streamed and not retained.

The source identity hash reads the three already-used safetensors shards (about 13.2 GiB in the file cache), which is reclaimable page cache rather than an all-layer hidden/state collection.  Before the run the host reported about 69 GiB available RAM.  Thus the expected process RSS is far below available RAM and does not scale threefold with range length.  The run records actual maximum RSS and is not retried automatically if it reaches the timeout.

## Stop condition

The command is capped at 45 minutes.  A timeout, nonzero exit, or a materially unexpected memory result ends the 0:11 attempt; only then would a smaller range be considered, with the reason recorded.
