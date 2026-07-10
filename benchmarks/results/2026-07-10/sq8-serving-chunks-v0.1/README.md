# SQ8 serving chunks initial evidence

## 前回の要点

P8-Bのall-M1 sessionは4096 contextで正しく動作したが、prompt 4095 / G=1に369.55秒かかり、製品TTFT用のM=8 prefill chunkが必要だった。

## 今回の変更点

- 実行runner commit: `e76c33118e60200b1e26892b060fa0eb251b97f9`
- release binary SHA-256: `4c80ce814053a210298877b2d18ea5a1ada4771d84a9b1c0eaae6548b32ab1bd`
- artifact SHA-256: `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`
- package manifest SHA-256: `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`
- GPU: isolated Radeon AI PRO R9700 / `gfx1201`、driver `6.16.13`
- workload: ascending raw-token prompt 8/9/16/17、G=1、active1/waiting0、batchなし
- M=8 chunkとall-M1は別processで同じbinary/artifact/packageを使用し、それぞれ同一model load内で4 promptを連続実行した。
- 独立validatorはproducerの`passed`を使わず、raw payload、hash、unit trace、40層cache、数値gateを再計算した。
- chunk対all-M1は4 promptでtop-1全一致、最悪relL2 `0.047343390`、最低cosine `0.998961757`、top-10 overlap最低9で合格した。
- prompt 8のchunk対vLLM sourceはhidden relL2 `0.046073592` / cosine `0.998980942`、logits relL2 `0.042203942` / cosine `0.999110258`、top-1一致、top-10 overlap 9で合格した。

Files:

- `runtime-chunk-p8-p17.json`: M=8 chunk producer result and per-unit cache trace
- `runtime-all-m1-p8-p17.json`: all-M1 comparison result
- `runtime-*-captures/`: final prompt hidden/logits raw F32 payloads
- `runtime-p8-p17-validation.json`: independent validation result

Clean build-identity evidence for prompt 32/128/512 is under `p32-p512-clean-28cd88e/`:

- runner commit: `28cd88eef728c35a492c3c50e22a9b036eeb83c1`
- binary SHA-256: `74310b7b576cfc4c38e78587553a01fc161fd4a6732dad6df498170c965c43db`
- both chunk/all-M1 producers recorded `runner_worktree_clean=true`
- chunk request time at prompt 32/128/512: `0.313870 / 0.613021 / 3.054293` seconds
- all-M1 request time at prompt 32/128/512: `1.131851 / 3.988003 / 18.741727` seconds
- chunk vs all-M1 worst relative L2: `0.055494862`; minimum cosine: `0.998492050`; top-1 exact for all three prompts
- chunk vs vLLM source: top-1 exact and top-10 overlap at least 9 for all three prompts

Clean build-identity evidence for prompt 4095 is under `p4095-clean-55562d9/`:

- runner commit: `55562d901d4f8e356b1d1a097903b84515b570cb`
- binary SHA-256: `74310b7b576cfc4c38e78587553a01fc161fd4a6732dad6df498170c965c43db`
- both chunk/all-M1 producers recorded `runner_worktree_clean=true`
- chunk/all-M1 request time: `78.043268 / 369.181784` seconds
- both paths emitted token `291`, reached position 4094 with all 40 cache lengths at 4095, and returned to the Ready/reset baseline
- chunk vs all-M1 hidden relative L2/cosine: `0.011411250 / 0.999950021`; logits relative L2/cosine: `0.008940925 / 0.999987181`
- chunk vs vLLM source hidden relative L2/cosine: `0.019835477 / 0.999888264`; logits relative L2/cosine: `0.020959889 / 0.999974552`
- both comparisons have exact top-1 and top-10 overlap 10

The exact 4096-token deep-boundary run is under `deep-boundary-p3584-g512-clean-5084396/`:

- runner commit: `5084396b35e6d74e7fce8fa298bd58580b2d7e7b`
- binary SHA-256: `58d1af401459cd321798a6d8b50292da53c49d43a1d7828ced34bbc256ff6f13`
- prompt 3584 plus 512 actual generated tokens ran with the explicit test-only ignore-EOS policy
- execution used 448 fixed M=8 prefill chunks and 511 M=1 decode writes, for 959 synchronized execution calls
- every generated index 0 through 511 records all 40 cache lengths, scheduler active1/waiting0, and the exact cache write position
- final cache length/position/block are `4095 / 4094 / 255`; reset returned all 40 caches and the allocator to zero
- resident request time was `136.763141` seconds, reset time was `0.003174` seconds, and model load was `24.088511` seconds
- the independent validator passed after recomputing all prefill, decode, terminal, reset, and external build-identity constraints

The first formal performance run is under `performance-clean-08bdcec/`:

- runner commit: `08bdcecdbfad78827131b8b2d390122e4e19457a`
- binary SHA-256: `ee1090689062f9d604513de142d784aa39afc97c5d6cbfef1e85623d6166c71b`
- raw result SHA-256: `71a896684a361cbc050bdbdd37d188e41dbc7b0f0ad90ec4d467ac810f22b03f`
- one resident model load took `24.379317` seconds; every prompt length used two warmups and five measured samples
- TTFT p50/p95 at prompt 32, 128, 512, and 2048 was `0.144360/0.144457`, `0.602628/0.603478`, `3.035701/3.037958`, and `23.481711/23.503208` seconds; all four gates passed
- prompt 3584 TTFT was `61.023836/61.025951` seconds and failed the fixed `50/60` second gate
- prompt 32 / generation 64 passed at p50 `27.779928` token/s and p95 inter-token latency `0.036897` seconds
- all 42 requests completed their required cancel or length terminal path and returned active0/waiting0, all 40 cache lengths, and the allocator to the Ready baseline
- all 44 AMD SMI/KFD VRAM captures agreed exactly and found only the worker PID; initial/final resident VRAM was `18,183,073,792 / 18,183,774,208` bytes
- the independent validator accepted the complete raw structure and recorded only the two prompt-3584 threshold failures; producer output contains no self-reported pass flag

The clean correctness evidence for the selected M=128 serving path is under
`m128-p32-p4095-clean-72008b9/`:

- runner commit: `72008b91d3e2ada892208803b1891a5af466c5f2`
- binary SHA-256: `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`
- both M=128 and all-M1 producers recorded `runner_worktree_clean=true` and the same commit/binary identity
- workload: raw-token prompt 32/128/512/4095, G=1, active1/waiting0, no request batching
- M=128 request time at prompt 32/128/512/4095 was `1.131062 / 0.176792 / 1.005426 / 56.753855` seconds; all-M1 was `1.160059 / 3.979503 / 18.786001 / 369.124277` seconds
- prompt 32 used 32 M=1 calls; prompt 128 used one M=128 call; prompt 512 used four M=128 calls; prompt 4095 used 31 M=128 calls plus 127 M=1 tail calls
- every request emitted the same top-1 token on both paths, reached the expected 40-layer KV length/position/block, and returned to active0/waiting0 with zero allocated blocks and zero cache lengths
- M=128 versus all-M1 had worst relative L2 `0.055494862`, minimum cosine `0.998492050`, exact top-1 for all prompts, and top-10 overlap 10; prompt 32 was bitwise equal, while the other prompts passed the numeric gates rather than bitwise equality
- M=128 versus the frozen vLLM source had exact top-1 for all prompts, top-10 overlap at least 9, worst relative L2 `0.065402638`, and minimum cosine `0.997865524`
- `chunk.json` and `m1.json` hold producer results; `m128-captures/` and `m1-captures/` hold distinct regular-file F32 payloads; `validation.json` is the independent comparison result regenerated from the repository-relative paths

The selected M=128 exact 4096-token deep-boundary run is under
`deep-boundary-p3584-g512-m128-clean-3bb1ef2/`:

- runner commit: `3bb1ef206e05aafc47bde82f105eea0bd8278443`
- binary SHA-256: `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`
- raw result SHA-256: `885bbd1a84fdd18c81829bc87f0e558d46f1267180263c5adf865a55cb07235e`
- prompt 3584 plus 512 actual generated tokens ran under the isolated test-only ignore-EOS contract and reserved exactly 4096 context tokens
- execution used 28 M=128 prefill calls and 511 M=1 decode calls, for 539 synchronized calls; all 512 generated steps were independently checked
- final KV length/position/block were `4095 / 4094 / 255`, with scheduler active1/waiting0 before release
- model load took `23.497561` seconds, resident request time was `107.083953` seconds, and reset took `0.003267` seconds
- reset returned the session to Ready with active0/waiting0, zero allocated blocks, and all 40 cache lengths zero
- the independent validator recomputed the full prefill/decode/terminal/reset structure and accepted the external clean commit and binary anchors

The selected M=128 formal performance run is under
`performance-m128-clean-c271e01/`:

- runner commit: `c271e010f18e6683dc53834188c45287434a34ef`
- binary SHA-256: `2ed172ab192f5d3d775959fb060910e290d893f23b74552cb77f190aaa416204`
- raw result SHA-256: `cb6119c9d6be9cbc8c7f55dcf2968be0b543c2e50bff602c046fb908201577e3`; validation SHA-256: `388f97dbc3702d182eb0dfe739143a5c0c2fce973bf8b3f43ad85715830d0bd7`
- one resident model load took `27.693288` seconds; every prompt used two warmups and five measured samples
- TTFT p50/p95 at prompt 32, 128, 512, 2048, and 3584 was `0.958687/0.960489`, `0.150361/0.150400`, `0.995855/1.216792`, `10.817689/10.825768`, and `31.286809/31.291056` seconds; every fixed gate passed
- prompt 32 remains an all-M1 tail in the selected fixed-M128 mode and is slower than the old M=8 result, but it stays below the unchanged `2.5/3.0` second product limits; no optional hybrid-tail optimization was added
- prompt 32 / generation 64 passed at p50 `27.757735` token/s and p95 inter-token latency `0.036882` seconds
- TTFT prompt call counts were `32/1/4/16/28`; the decode case used 32 prompt calls, 31 prompt progress events, and 95 total calls
- all 42 requests completed their required cancellation or length terminal path and reset to active0/waiting0, zero allocator use, and all 40 cache lengths zero
- all 44 AMD SMI/KFD VRAM captures agreed and found only the worker PID; initial/final resident VRAM was `18,275,348,480 / 18,276,048,896` bytes
- the independent validator reported `passed=true` with no gate errors after validating raw v2 structure, timing, sampling, terminal/reset, isolation, VRAM, and clean build identity

## 次の行動

M=128のclean correctness、3584+512 deep boundary、formal TTFT/decodeはすべて固定gateに合格し、P8-B2は完了した。次はP8-Cでdeterministic sampling、cross-thread cancellation、resident worker protocolを実装する。request batchingとwaiting queueは対象外のままとする。
