# AQ4 register BM8 experimental evidence

## 前回の要点

`e1c9877`では、BM8 register GEMMを環境変数で明示的に選ぶ実験経路として測定した。

## 今回の変更点

raw resident/crossoverは改変せず、token一致とcrossoverを機械検証した。width 8の最初の候補呼び出しはcompile/setupを含むcold値（0.7781x）なので、定常比較から分離した。width 16/32/64/96/127/128のBM8対Legacy比は、それぞれ1.2052x、1.1819x、1.1618x、1.1561x、1.1599x、1.1576xだった。

全crossover rowでtokenは一致した。単発rawに含まれない反復値や手動観測は、このsummaryへ測定値として追加していない。

## 次の行動

実験環境変数ではなく、typed registryとforced ABIで適格形状だけをBM8へ昇格したpromoted evidenceを正本とする。
