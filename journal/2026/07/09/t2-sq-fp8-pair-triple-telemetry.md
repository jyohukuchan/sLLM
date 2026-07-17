# T2 SQ FP8 pair/triple telemetry

## 前回の要点

- `q/k` pair候補と `q/k/v` triple候補はfull mixed B=1/4/8でAQ4 final top1と一致した。
- ただしstdout/JSONLでsingle/pair/triple境界を区別できていなかった。

## 今回の変更点

- SQ FP8 direct kernelのsingle/batch/pair/triple呼び出しcountをengineに追加した。
- layer load後にresetし、測定区間だけを数える。
- parserとunit testを更新し、JSONLの `workload` にtelemetryを保存した。
- R9700 B=4でpair/tripleを再測定した。

## 結果

| candidate | boundary | single | batch | pair | triple | top1 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| qk-layer3 | pair | 0 | 0 | 12 | 0 | match |
| qkv-layer3 | triple | 0 | 0 | 0 | 12 | match |

## 次の行動

- `q/k/v` layer3 triple候補をprompt bundleまたは長めのprefill gridへ広げる。
- layer7以降の `q/k/v` を安全に足せるかを見る。
