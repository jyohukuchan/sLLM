# AQ4 register BM8 promoted evidence

## 前回の要点

experimental BM8は定常幅でLegacyを上回ったため、`67740a0`でtyped registryとforced ABIによるproduction選択へ昇格した。

## 今回の変更点

no-env rawは`experimental_env=false`で、7ケースのtoken、progress、outcome、reset、clean shutdown、67件の子プロセス検査がfull-native baselineとM1に一致した。p127/128/129/255/256は136.7504/135.7696/134.9581/135.1377/134.7264 tok/sで、full-native比は1.164〜1.173x、M1比は1.714〜1.765xだった。

operation auditは全ケースで`physical = 64*C + 192`、`token-equivalent = 64*M + 192`を満たし、paged KV writerとsigmoid-gated readerのID coverageはfull-nativeから変わらない。canonical profileには`ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL`があり、experimental環境変数は含まれない。

rocprofの単一p128 profileでは、baseline AQ4 batchは249 calls、1,010,487.923 us、91.7448%。promotedはregister 201 calls、761,549.001 us、79.3997%、Legacy 49 calls、117,133.115 us、12.2124%、recurrent 26 calls、58,268.813 us、6.0751%、paged GQA 10 calls、5,676.021 us、0.5918%だった。DB SHA256は`summary.json`に固定した。

profileにはrocprof overheadが含まれ、各DBは単一p128 requestなので、反復スループット分布ではない。

## 次の行動

production guardを維持し、より広いprompt長・反復・同時実行でregister/Legacy境界とdecode側の比率を再測定する。
