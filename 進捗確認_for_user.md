# Phase 3d CPU-only 進捗

- 既存chainをlayer 0--31、final norm、LM head固定row sampleまで拡張した。
- 0:11実測を基に、0:31は約7分・RSS 512 MiB未満と見積もり、20分上限の単発CPU測定を準備した。
- 初回測定は6:45・RSS 331188 KiB・swap 0でdecoder後まで到達したが、LM headがAQ4 tensorであるのにpassthrough readerを選んだためterminal前に無効終了した。失敗ログを保存し、全語彙を保持しないAQ4固定34行readerへ修正して別出力先で再測定する。
- GPU、service、systemd、active manifest、P3 harness、service-stop関連には触れていない。
