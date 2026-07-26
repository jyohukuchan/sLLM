25-options:
26-  -h, --help
27-  --numa <distribute|isolate|numactl>       numa mode (default: disabled)
28:  -r, --repetitions <n>                     number of times to repeat each test (default: 5)
29-  --prio <-1|0|1|2|3>                       process/thread priority (default: 0)
30-  --delay <0...N> (seconds)                 delay between each test (default: 0)
31-  -o, --output <csv|json|jsonl|md|sql>      output format printed to stdout (default: md)
--
81-'first-last' or 'first-last+step' or 'first-last*mult'.
82-```
83-
84:llama-bench can perform three types of tests:
85-
86-- Prompt processing (pp): processing a prompt in batches (`-p`)
87-- Text generation (tg): generating a sequence of tokens (`-n`)
--
91-
92-Each test is repeated the number of times given by `-r`, and the results are averaged. The results are given in average tokens per second (t/s) and standard deviation. Some output formats (e.g. json) also include the individual results of each repetition.
93-
94:Using the `-d <n>` option, each test can be run at a specified context depth, prefilling the KV cache with `<n>` tokens.
95-
96-For a description of the other options, see the [completion example](../completion/README.md).
97-
98-> [!NOTE]
99:> The measurements with `llama-bench` do not include the times for tokenization and for sampling.
100-
101-## Examples
102-
