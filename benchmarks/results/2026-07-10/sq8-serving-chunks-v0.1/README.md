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

## 次の行動

prompt 8/32/128/512/4095のcorrectness oracleは完了した。P8-B2の残件は3584+512 deep boundaryと、warmup 2 / repeat 5のTTFT/decode性能gateである。
