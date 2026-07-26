# Throughput status and comparison

## Same-accounting control

The only completed full-model rate available at this source boundary is the
BR M=128 control below. It used the conditions and five-repeat accounting in
`../../2026-07-26/r9700-prefill-comparison/`; values are repeated here only as
the control, not presented as a new measurement.

| prompt tokens | SQ8_0 M=128 tok/s | llama.cpp Q8_0 F32-KV tok/s | llama/uLLM |
| ---: | ---: | ---: | ---: |
| 128 | 883.021 | 1,165.756 | 1.320x |
| 512 | 561.905 | 1,195.722 | 2.128x |
| 1024 | 358.745 | 1,145.351 | 3.193x |
| 2048 | 196.585 | 1,058.379 | 5.384x |
| 4095 | 105.040 | 1,008.683 | 9.603x |

## Wider-M status

| M | 128 | 512 | 1024 | 2048 | 4095 | reason no rate is recorded |
| ---: | --- | --- | --- | --- | --- |
| 256 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | lower CK/layer/stack admission rejects M=256 before model allocation |
| 512 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | same lower admission blocker |
| 1024 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | same lower admission blocker |
| 2048 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | same lower admission blocker |
| 4096 | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | same blocker; at N=4095 the no-padding scheduler correctly cannot use M=4096 |

No extrapolated tok/s value is reported. The scheduler proves that, once the
lower execution contract admits each width, N=4095 would have 640/320/160/80
planned attention calls for M=256/512/1024/2048, respectively, versus the
observed M=128 baseline's 1,280. That count reduction is not a substitute for
a trace or full-model timing result.
