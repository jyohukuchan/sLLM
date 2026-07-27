# CL single-window MoE status — not run

CL reserved one R9700 service window for tile-128 quality followed by this
MoE check.  The tile runner failed before GPU initialisation because its
numeric-capture directory parent was absent.  Its fail-fast cleanup released
the lock and restored production, so the MoE binary was never launched.

The release binary was nevertheless rebuilt/verified before the window:

```
6ee827e43fa4e4a5e54fd66c1b20eb444e05632245f66349e10cfe409b9e39cd
target/release/ullm-qwen35-moe-aq4-generate
```

Therefore the 262,144-token F16-KV ledger (`30,858,010,436 B`) was not
allocated or sampled; no OOM/overage number exists.  Generation text,
all-40-layer raw-BF16 route verification, and prefill/decode speed are also
**not run**.  A second window was deliberately not taken.
