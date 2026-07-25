# AQ4_0 runtime hardening Phase 4

Phase 4 produced fresh protected-path promotion evidence, a receipt, and a frozen candidate manifest.  It also sealed the already-reviewed activation control source into a concrete immutable plan and re-ran its default read-only preflight.

No activation action was performed: `/etc/ullm/served-models/active.json` remained SHA-256 `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`; no promotion campaign ran and no campaign authorization was consumed.

## Result

- The mechanically derived candidate profile has exactly 30 unique guard flags, in the live-manifest order.  Its P3-only-key intersection is empty.
- Fresh resident-versus-legacy evidence passed on R9700 / `gfx1201` / HIP GPU index `1`.  Both raw comparisons had exact token matches, and both workers shut down cleanly.
- The frozen candidate differs from live only at `/tokenizer/root`, `/worker/binary`, `/product/root`, `/promotion/receipt`, and `/promotion/receipt_sha256`; it contains no `/home/` reference.
- The sealed plan-bound preflight is `ready: true`, with `blockers: []`.  It is still not execution authority; the Phase 6 human approval gate remains mandatory.

## Main protected artifacts

| Artifact | SHA-256 |
| --- | --- |
| candidate profile | `ee3d9d4374b79f03e402027a48c6e32601912f79429013893a023083a497439e` |
| promotion evidence | `4a604453abb6c7a672731d2b17d3333e471d6c5239b4fed1f6b338fe19a19adb` |
| promotion receipt | `99ead62f6d5d6062690d78431dbb888949e100bf8951c55f9ff16c71545f1f24` |
| protected-path binding | `e1b6158cddfab37b84afc2b85351a109d4530af7c4668adb932e5b94532ebe2b` |
| frozen candidate manifest | `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` |
| rollback manifest | `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` |
| reviewed operations | `b635793815792d2c75b95ac7b9824343f200a2025c4ad96eb90a717e5d779cef` |
| activation plan | `72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194` |

The telemetry record distinguishes one successful evidence window from one earlier stopped-and-restored preflight abort.  The empty telemetry files left by aborted probes were retained under protected staging and were not used as evidence.

See `candidate-profile-verification.json`, `fresh-evidence-verification.json`, `service-window-record.json`, `activation-readiness.json`, and `read-only-preflight.json` for machine-readable details.
