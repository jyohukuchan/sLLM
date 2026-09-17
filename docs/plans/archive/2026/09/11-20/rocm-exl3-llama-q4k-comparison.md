# rocm_exl3とllama.cpp Q4_Kの追加比較

2026-09-17ユーザー依頼。前回のQwen3.5-2B比較表にllama.cpp Q4_Kの実測値を加える。

## 受入範囲

- 同一source model revisionをBF16 GGUFへ変換し、llama.cppのQ4_K aliasが選ぶQ4_K_Mへ量子化する。
- 保存済みclean llama.cpp `bc52a12b38941b0a690ade65fbc5749715224e30` とgfx1201 HIP binaryをidentity照合して使用。
- R9700単独、MTPなし、FP16 KV、入力128/512、decode入力128＋出力64、warmup1＋measured2で比較する。
- API生成・GPU確認・終了を検証。測定器、sampling、prompt/decode時間の定義差を記録し、同等品質とは主張しない。
- 前回表と機械可読記録へ追加。models/raw logsはcheckout外、production変更・commit/pushなし。

## 状態

完了。同一sourceのQ4_K_Mを生成し、R9700でQ4_K演算子6/6、9requestのprotocol、計算smoke、
server exit0を確認した。既存表へ最終の4057.8/9553.3/164.5 tok/sと測定器の定義差を追加した。
初回の語彙範囲推定誤りを訂正し、元のEXL3関数で全9token列を照合。照合用runtime停止後の一式を採用した。
既存reference/binaryの再使用であり、追加の外部source copyはない。model/raw logsはcheckout外、commit/pushなし。

履歴: [調査結果](../../../../../history/2026/09/11-20/rocm-exl3-investigation.md)
