# Qwen3.5 AQ4 register BM8 prefill evidence

## 前回の要点

full-native AQ4 prefillの7ケースとM1比較、物理・token-equivalent audit、self-attention chunk ID coverageを既存証跡として固定していた。

## 今回の変更点

- experimental rawのcrossoverでwidth8をfirst-candidate coldとして分離し、width16..128の定常比率を未丸めraw値から算出した。tokens_matchは全行trueで、manual/repeated観測はrawへ追加していない。
- promoted raw HEAD `67740a0`（環境変数なし）を7ケース比較し、token/progress/reset/clean/child checksをfull-nativeとM1へ機械比較した。physical/token-equivalent式とnew self chunk ID coverageも全件で成立した。
- canonical profileのBM8 guard、raw/source SHA256、JSON parse/assert、read-only rocprof DB selected rowsをsummaryへ記録した。DBはbaseline/promotedとも単一p128 profileで、rocprof overheadを明記した。

## 次の行動

既存rawと`.rocprofv3/`を変更せず、親エージェント統合後に必要なら新source commitで再測定する。
