# 射影 / format 却下候補の lightweight 再評価

## 前回の要点

- 旧来の strict top-1 / logits gate は、自己回帰における 1 ULP 級の軌道分岐を
  候補の破損と誤認し得た。2026-07-26 の lightweight promotion policy は、実際の
  生成文の品質を判定対象に変更した。
- `SQ8_0` handwritten WMMA projection は component 4/4 を通過していたが、
  full-model trace には layer 3 の 2 / 5,120 要素、最大 abs `6.1035156e-5` の
  差が残っていた。旧運用では速度未測定だった。
- `SQ8_1` W8A8 は旧 logits / top-1 基準で落ちたが、最大 logits 差 `7.889` は
  1 ULP の話ではないため、実生成文を必ず確認する必要があった。

## 今回の変更点

- `SQ8_0` WMMA を R9700/gfx1201 の isolated full-model decode で、CK と
  interleaved に 5 組測定した。1,028-token prompt、16 generated tokens のうち
  feedback decode index 1--15 のみを使い、CK は **14.662647430 tok/s**、WMMA は
  **9.624875977 tok/s** だった。WMMA は 34.357857% 遅く、速度-first で不採用に
  した。プロファイラ range は throughput の根拠にしていない。
- 有効 window の各開始サンプルは edge 42 C / hotspot 43 C、13--17 W、
  `UNTHROTTLED` を満たした。window は 22:35:50--22:59:33 JST、service は
  `active/running` に復旧した。初回 pre-measurement abort と、WMMA invocation の
  decode-oracle path 欠落による不完全 pair は結果に使っていない。
- `SQ8_1` W8A8 の CPU full-model fixed suite を source-reference と並置し、
  `javascript_debug` で `NaN is truthy` / `Boolean(NaN) is true` という実際の
  誤説明を確認した。Node の確認では `Boolean(NaN) == false`。空答・文字化け・
  loop はなかったが、コード説明の事実性に具体的な欠陥があるため不採用にした。
- 現在 active の `AQ4_0` P3 からも同一 10 prompt の出力を read-only capture
  した。この active 出力にも同じ NaN truthiness の誤説明が含まれたため、現行を
  この一点の clean semantic control としては扱わない。同一 source model の
  source-reference は正しいため、`SQ8_1` candidate にも実害があることは確認できる。
- `SQ8_1` は gfx1201 dispatch / served selector / manifest / worker がなく、R9700
  full-model tok/s は **未測定ではなく測定不能** と記録した。V620 historical timing
  は使っていない。
- registry の 5 manifest を static validate し、4 pass / 1 fail を確認した。全件
  `AQ4_0` であり、本 task の `SQ8_0` / `SQ8_1` candidate ではない。
- active manifest は待機中に他 task により
  `c57a2b6...1d05fca4` から `a98910dc...f274fe49` へ変わった。BJ は active manifest
  を書かず、valid window の stop 前・restore 後で後者 SHA-256 が同一であることを
  確認した。

## 次の行動

- WMMA projection は full-model で大幅に遅いため、現 revision の昇格や追加の
  lightweight output capture は行わない。再検討するなら、まず CK を上回る別の
  projection 実装を用意し、同じ speed-first protocol から開始する。
- `SQ8_1` W8A8 は gfx1201 実行路・generic served candidate が存在しないうえ、
  実生成文の factual defect がある。format と kernel を新たに設計する作業は本 task
  の範囲外であり、現 candidate の昇格は行わない。
- current `AQ4_0` output に見つかった NaN truthiness の説明欠陥は、別途 active
  quality issue として追跡可能だが、本 task では manifest を上書きせず記録のみに
  留める。
