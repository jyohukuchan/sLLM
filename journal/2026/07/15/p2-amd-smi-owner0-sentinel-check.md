# P2 AMD-SMI owner-zero sentinel check

## 前回の要点

- production activeのR9700 GPU 2ではexact per-GPU AMD-SMI process JSONが新observer parserの期待形と一致した。
- stop直後のv2 failure rawは保存されておらず、owner zero時のsentinel形状が通常の空listか別形状かは未確認だった。

## 今回の変更点

- read-onlyで`amd-smi list --json`を1回、全GPU対象の`amd-smi process --general --json`を1回実行した。
- GPU inventoryはindex 0、1、2の3台。process inventoryは全GPUでroot keys `gpu/process_list`、`process_list` type `list`、process count `1/1/1`だった。
- R9700 GPU 2を除くGPU 0と1にもownerがあり、owner zero候補は0台だった。このため指定された`amd-smi process --gpu N --general --json`は実行していない。選択GPU、owner-zero sentinel raw SHA/shape、exact process_info文字列/keysはいずれも該当なし。
- service/GPU/HTTP/actualには変更を加えていない。R9700 serviceはactive/running、`NRestarts=0`のままである。

## 次の行動

- owner zeroの別GPUが現れた時だけ、決定的に最小indexを選び、指定per-GPU commandを1回実行してsentinel rawをimmutable保存する。
- production serviceや別GPU ownerを停止してsentinelを作ることはしない。
