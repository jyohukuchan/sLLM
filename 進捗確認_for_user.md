# Phase 3d CPU-only 進捗

- 既存chainをlayer 0--31、final norm、LM head固定row sampleまで拡張した。
- 0:11実測を基に、0:31は約7分・RSS 512 MiB未満と見積もり、20分上限の単発CPU測定を準備した。
- 初回測定は6:45・RSS 331188 KiB・swap 0でdecoder後まで到達したが、LM headがAQ4 tensorであるのにpassthrough readerを選んだためterminal前に無効終了した。失敗ログを保存し、全語彙を保持しないAQ4固定34行readerへ修正して別出力先で再測定する。
- 修正後の2回目はAQ4 terminal frameを出力できたが、比較器がproducerのper-timestep順を誤って検証したため6:52・RSS 334748 KiB・swap 0で無効終了した。stream順テスト（17 passed）を追加し、保存済みattemptを残して最終CPU測定を別出力先で実行する。
- 最終attempt-3は6:48・RSS 330744 KiB・swap 0でvalid完走した。layer 31のrelative L2 0.127881からfinal RMSNormで0.501033へ3.92倍急増し、dominant boundaryはfinal normである。sourceの`1 + weight`とAQ4のraw weight handling差をpayload一致確認付きで記録した。fix実装やGPU/service操作はしていない。
- GPU、service、systemd、active manifest、P3 harness、service-stop関連には触れていない。
