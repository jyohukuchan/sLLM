# Registered served-candidate inventory

Read-only inventory performed on 2026-07-26.  These are all `AQ4_0` served
candidates; none is the `SQ8_0` handwritten projection or `SQ8_1` W8A8
candidate evaluated by this task.  They are therefore recorded for the
requested registry audit but not measured, modified, or promoted here.

| Manifest | Public model | Format / device | SHA-256 | Static validator |
| --- | --- | --- | --- | --- |
| `qwen35-9b-aq4-4be10d0.json` | `ullm-qwen3.5-9b-aq4` | `AQ4_0` / gfx1201 | `7589b9db7734d176bef21130b31e1ba679d1e0599e9a3c0d8af6699f86eded80` | pass |
| `qwen35-9b-aq4-reasoning-fidelity-f1a3cf4c.json` | `ullm-qwen3.5-9b-aq4` | `AQ4_0` / gfx1201 | `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` | pass |
| `qwen35-9b-aq4-reasoning.json` | `ullm-qwen3.5-9b-aq4` | `AQ4_0` / gfx1201 | `e6f749654e85a5f69f2d077bd55d4e27aff869d71803809386c5d36865183e72` | pass |
| `qwen35-9b-aq4.json` | `ullm-qwen3.5-9b-aq4` | `AQ4_0` / gfx1201 | `c2ce3265f2e21fcf8ef3e11ff720c860a43988df764090aee450107282edd61b` | fail (`served-model validation failed`) |
| `temp-test-prefill-wmma-v4-ef62dc48.json` | `ullm-qwen3.5-9b-aq4` | `AQ4_0` / gfx1201 | `74e32476322132fc91d93ee31793d5dbe23ff3bf6bdabb10a107bfe908f4908c` | pass |

The check used `python3 tools/validate-served-model.py --manifest <manifest>`.
This confirms the reported four-pass/one-fail split without changing any
manifest or invoking promotion.
