# Phase 3c GPU window 進捗

- `RuntimeDirectoryPreserve=yes` の専用 drop-in を追加し、daemon-reload 後も service は active/running のまま。worker の lock 再作成は既存 regular file に対して冪等であることを確認済み。
- 新しい service-stop window は未実行。stop 後の lock 存続を同じ window で初めて確認し、成功時だけ R9700-only guard、telemetry、trace、比較へ進む。
- 07/16 停止中 P3 harness と既存 evidence には触れていない。
