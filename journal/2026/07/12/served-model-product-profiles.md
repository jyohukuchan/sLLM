# Served-model product profiles

- SQ8/AQ4の現在のproduct、tokenizer、workerの実パスを既存systemd環境から確認した。
- SQ8には既存の`promotion.json`がある。AQ4 compatibility productにはpromotion receiptがない。
- worker build後に実ファイルをstreaming SHA-256で固定する生成toolと、WRX80用profileを追加した。
- SQ8 manifestは生成後にstrict validatorを通過した。AQ4はreceipt不足で意図どおりfail-closedになった。
- workerが`--served-model-manifest`に対応するまでは、生成済みmanifestをactivationに使わない。
