# AQ4 P2 worker benchmark wire phase1

## 前回の要点

現行AQ4 resident workerは通常generation専用で、request-scoped M、生成0のprefill-only、reset後のstdout auditを持たなかった。

## 今回の変更点

- 通常`ullm.worker.v1/v2`と分離したAQ4 benchmark protocol/runtimeを追加した。
- case/hash/run/M/fixture/inputをexact検証し、M gridとall-M1を受理する。
- resident sessionへrequest-scoped Mとprefill-only resetを追加した。
- reset完了後のterminal evidenceをreleasedより先に同期flushする。
- sanitized audit、lifecycle/reset、actual width、fallback、operation/resource linkを出力する。
- cancel/reset failure/reuse禁止、unknown/duplicate/hash drift、M grid、event orderingをCPU test化した。
- stderrの通常audit logと通常worker wireは維持した。
- independent QA followupでcommandへcanonical full case objectを追加し、case ID/SHAをobjectから再計算してouter commandと意味的に結合した。
- status okのactual M/token width/request width/operation audit SHAとlifecycle/resetをsanitized auditからexact再構築する。failedは常にreuse禁止とし、failure code/reset整合、後続case/process fail-closeを検証する。
- case ID swap、width=999、operation audit non-SHA、failed reuse allowedの負例をCPU testへ追加した。

## 次の行動

gatewayのadmin benchmark routeとTTL evidence storeからこのopt-in wireを使用する。GPU/live実行は別途許可されたone-case smokeまで行わない。
