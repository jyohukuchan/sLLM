# AQ4 P2 worker benchmark wire v0.1

## 前回の要点

通常workerは`ullm.worker.v1/v2`のgenerate/cancel/shutdownとstrict event schemaを提供する。通常wireはOpenAI gatewayの製品契約なので、benchmark fieldを追加しない。

## 今回の変更点

AQ4 manifest workerを`--benchmark-wire`付きで起動した場合だけ、`ullm.aq4_p2.worker_benchmark.v1`へ切り替える。readyはcommand/evidence schema、prefill-only、configurable M、M grid、input hash algorithmをcapabilityとして明示する。通常modeのready、generate、released schemaは変えない。

`benchmark_prefill`はrequest/case ID、case SHA-256、warmup/measured、run index、requested/resolved M、`generated_tokens=0`、fixture SHA-256、input SHA-256、token IDsをexactに受理する。input hashはtoken IDをunsigned 64-bit little-endianで連結したSHA-256である。resolved Mはrequested Mまたはall-M1の1だけを許可する。

resident sessionはmodelとoperation planを再loadせず、request単位のresolved Mでprefillする。最後のprefill progress後に生成を行わず同期resetする。stdout順序はready、started、progress、terminal_evidence、releasedである。terminal evidenceはsanitized request audit、actual width、fallback、lifecycle/reset、operation-audit hash、resource observation keyを保持する。resource sample本体はworkerへ埋め込まない。

cancelはreset後にcancelled evidenceを返す。OOM、HIP、reset、publish、その他execution failureはfailure codeと`reuse=forbidden`を持つ。terminal evidenceのflush後にだけreleasedを出す。stdout publish failure時はreleasedを出せず、process全体を非再利用として失敗させる。

## 次の行動

gateway側は専用admin benchmark routeからだけこのwireを使用し、通常OpenAI requestを接続しない。resource observerとraw-v2/P1 hash chainはgateway phaseでterminal evidenceのresource observation keyへ結合する。
