# Backend operation registry M1

## 前回の要点

文字列中心の `backend_dispatch` は互換matcherとして維持し、実行可能なtyped registryを別境界として追加する方針だった。

## 今回の変更点

- 実runtimeのbackend/device/ABIと、値が厳密に`1`のproduction guardからだけcapabilityを構築する。
- manifest shapeをresident upload前に検査し、linear QKV prepare、recurrent scan、paged GQA readを3 phase分解決する。
- served profileの`gfx1201`をmodel loadへ渡し、context情報を得た直後かつstream/probe/weight allocation前にexact一致を検査する。
- architectureはHIP device propertyの`gcnArchName`だけを使う。R0600/R0000 symbolを動的解決し、header・symbol・値のどれかが欠ければ名称やcompute capabilityから推定せず拒否する。
- conv/recurrentはそれぞれ正直なin-place state effect、paged GQAはKV read-onlyとして分離した。
- gated/ungated attentionを別executableにし、geometryをplanから導出してABI引数の差し替えを防いだ。
- plain/fused KV writerもregistryに登録し、Qwen3.5-9B実geometry（q=16、kv=4、head/value=256）でload-timeに解決する。layer hot pathとprewarmには直接ABIを残さない。
- production guardに加えて独立scratch ABI probeを実行し、同期成功した機能だけをadmitする。
- probe cache keyはbackend、architecture、device、ABI、policy bitsetを含み、QKV、recurrent、plain/gated paged read、plain/fused writer、syncの各故障段階ではcacheを公開しない。
- R9700隔離実行で各probe故障段階のcache非公開、無故障再実行、以後のcache hitを確認した。
- package resident bytesと全layer stateのchecked和、保持activation上限、headroomをモデル単位で先にadmitする。registryのoperation workspaceは同じstate/IOの監査値なので二重加算しない。
- fallbackは未開始attemptを消費し、started executionも一回のABI呼び出しで消費する。
- ColdPrefill、CachedPrefixPrefill、Decodeをsessionからlayerまで明示的に伝播する。
- 成功した各layer stepには、linearならprepare/recurrent、selfならwriter/readの2件だけを固定長execution recordとして残す。
- sessionは32 layer×2 recordをload-time traceのphase別順序と照合し、リクエスト中はphase counters、6 implementation counters、固定長SHA-256 stateだけを累積する。正常終端ではprompt/generation期待step数、総record数、coverageを検査する。
- terminal auditはstdoutの`released` flush/ack後に既存stderr `request_released` logへ入れる。stdout token protocolは変更しない。
- cancel、execution failure、reset failureではcoverage=falseのpartial auditを残し、失敗stepのphase/layer/operationと、それ以前に成功したoperation数を保持する。InPlace ABI失敗後のlayer stateは同期resetまで再実行できない。
- prepare、recurrent、writerは不正sizeの実runtime bufferでABI failureを起こし、各planが一回だけsys callして消費されることを検証した。geometry mismatchはsys call前に拒否する。
- workspace byte計画は実manifest parserとloader共通のf32 allocation formulaを使う。容量はtotalGlobalMemに512 MiB headroomを引く静的な保守判定で、free VRAMの予約ではない。

## 正式な検証

- `cargo test -p ullm-engine --lib backend_operation_registry -- --nocapture`: registry契約、全3 phaseのCPU direct-ABI差分、state/output一致、architecture fail-closed、probe故障注入。
- `cargo test -p ullm-engine --bin ullm-aq4-worker -- --nocapture`: served profileからresident modelへのexact architecture bindとworker audit JSON。
- `qwen35_aq4_session::tests::*operation_audit*`: Cold/Cached/Decodeのexact counts/digest、32-layer pattern、2-request reset、empty/missing/order不正の拒否。
- `cargo test -p ullm-engine --lib`: engine全体の回帰。
- `ROCR_VISIBLE_DEVICES=1 ... cargo test ... isolated_hip_probe_faults... -- --ignored`: R9700実scratch probeの段階故障/cache検証。
- `cargo fmt --all -- --check` と `cargo check -p ullm-engine --lib`: formatとproduction library build。

## 次の行動

M>1/chunk implementationは今回追加していない。将来の実装は同じregistry key、workspace admission、state effect、typestateを満たすdescriptorとして追加する。
