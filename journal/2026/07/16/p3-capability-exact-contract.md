# P3 capability exact contract

## 前回の要点

- P3 raw は型付き P2 qualification に結合され、diagnostic raw は選考から分離された。

## 今回の変更点

- base capability を family-exclusive timing、D2H count/bytes/time、stream sync count/time のexact boolean fieldsにした。
- Candidate A capability を direct sequence output、D2D bytes/copy count、launch count、component/full-model latency、workspace、peak VRAM、fallback、alias/size/admission safety、fidelity binding のexact boolean fieldsにした。
- 現producerにはbyte-counted memory-copy trace referenceがないため、`d2h_bytes=false`を明示し、API call countからbyte数を推定しない。
- 全capability fieldについてmissingとbool-as-intを拒否し、unknown fieldも拒否するmatrix testを追加した。

検証結果: selector/producer/diagnostic capture 140 tests passed。

## 次の行動

- verified selection artifactだけがproduction direct routeを有効化できるactivation contractを実装する。
