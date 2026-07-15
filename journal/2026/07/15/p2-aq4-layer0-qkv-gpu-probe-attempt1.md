# P2 AQ4 layer-0 QKV standalone GPU probe attempt1

## 前回の要点

固定commit由来のprobe binaryと診断専用gateを準備済みだった。GPU/service実行は未実施で、`promotion_eligible=false`、`unclassified`、holdout未観測の契約を固定していた。

## 今回の変更点

- main HEAD `eb7bf4513a5bdcc8ea44f111ef42e7fa735a7edf`（gate commit `1cb69e23ba08bd35a953488bae19e4f6e244a464`）で、sudo資格確認と同じPTYからattempt1を一度だけ実行した。
- service stop、RuntimeDirectory/lock作成、ROCm observer、`HIP_VISIBLE_DEVICES=1`、global device1、AQ4 matvec guard、standalone probeまで到達した。
- runtime package loadで `unsupported backend operation Aq4MatvecBatch for phase ColdPrefill` が発生し、report/outputは生成されなかった。期待outputは3×8192×4=98304 bytes、観測値は0 bytesである。数値Go/No-Go、holdout、promotionは実施していない。
- gateのobserver background子へEXIT trapが継承され、失敗時cleanupで`observer_pid: unbound variable`が発生した。このため手動でsudo-v→service startを実行して復旧した。
- serviceはactive/running、MainPID=1553870、NRestarts=0、lock owner=MainPID、RuntimeDirectory mode750、lock mode600を確認した。active/package/worker SHAは固定値と一致した。
- raw run/monitor/markers、開始前・終了後dirty snapshot、SHA256SUMSをattempt archiveへ保持した。observer rawは物理card2を含み、card2のGPU使用率は3%/6%、card0/1は0%/0%だった。

## 次の行動

attempt1の再実行は禁止する。unsupported operationの診断は別担当に委ねる。observer trap非継承の最小修正と固定テストを通常follow-up commitとして追加し、GPU/serviceを再実行せずにarchive/journalとともに限定commitする。
