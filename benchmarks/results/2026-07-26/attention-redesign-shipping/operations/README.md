# R9700 execution record

No R9700 work has been started by this evidence directory yet.  The following
is the bounded run order prepared for the first free lock window:

1. Record the required `fuser`, `pgrep`, and `systemctl show` preflight.  If
   `/run/ullm/r9700.lock` is held, do not run or wait on the lock.
2. Run one isolated loopback request through the unchanged active `AQ4_0` P3
   manifest.  This is a regression smoke of the old manifest contract, not an
   `AQ4_0`/`SQ8_0` output comparison.
3. Acquire the regular lock with non-blocking `flock` and profile the pinned
   P3-compatible C=1339 decode driver with `rocprofv3`.  Release it as soon as
   the profile exits.
4. Run the fixed ten-prompt suite through isolated direct and grouped
   `SQ8_0` gateways sequentially.  Each gateway owns the normal lock through
   its worker supervisor and is terminated before the next one starts.
5. Record the required postflight and re-hash the active manifest.

The sequence neither calls `systemctl start/stop/restart` nor invokes
`promote-served-model.py`.  It uses only `HIP_VISIBLE_DEVICES=1` / R9700
(`gfx1201`), leaves `llama-qwen35-udq4.service` untouched, and must leave the
active manifest unchanged.
