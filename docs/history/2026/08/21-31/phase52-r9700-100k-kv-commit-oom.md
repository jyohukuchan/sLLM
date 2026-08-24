# Phase 52履歴: R9700 100k KV物理コミットOOM

## 2026-08-24: 計画作成

- ユーザー指示により、R9700 exact `gfx1201`のQwen3.5-4B BF16／FP16 KV `100,000/2` OOM解消をPhase 52へ割り当てた。
  Phase 51のMI300X移植とは独立に進行でき、どちらも相手の開始条件にしない。
- Phase 50の旧selectorではlayer 31 KV commit、HBM peak `26,414,587,904` bytesでOOMとなった。
  自動prefill capacity tier修正後のcommit `159bc526cb26d180161f2cd7abcc22abb7e67e84`では、32 GiBのcandidate列を
  `[2048, 512]`へ制限してHBM peakを`13,160,554,496` bytesへ約50.18%下げたが、約`152.867`秒後に
  `layer.23.kv_append`、HIP status 260、`grow virtual KV physical commitment`で再びOOMとなった。
- 失敗rowは自動指定を`prefill_chunk_tokens: null`とだけ記録し、実効2K／512を保存していない。また、K／V／scale planeを
  順にgrowする現行VMM経路は途中失敗時に新規physical pageを完全rollbackしたことを証明できない。この二点をPhase 52の
  最初の観測・host failure injection対象とした。
- Phase 52は実効chunkとper-plane commitmentを観測し、VMM transactional rollback、extent／handle削減、provider-aware
  preflight、gfx1201長context限定`contiguous-resident`を根因に応じて個別比較する。明示512、silent retry、CPU fallbackを
  完了解決として扱わない。
- 最終条件は同一R9700 tuple、自動prefill、`100,000/2`の1 warmup＋3 measured完走、生成token `[23066, 23066]`、
  HIP-only、fallback 0、cleanup 0、process消滅、HBM／GTT復帰である。llama.cpp性能同等は報告項目でありgateにしない。
- 既存Phase 50 summary／schemaとraw evidenceを変更せず、Phase 52専用summary／schemaへsource、selector、provider、
  per-plane commit、resource、rollback、各反復を固定する方針とした。この時点では計画だけであり、実装・GPU PASSは主張しない。

## 2026-08-24: 実装候補とhost検証

- ユーザー指示によりPhase 51を一時保留し、Phase 52を先に進める状態へ変更した。
- exact `gfx1030`/`gfx1201`のlogical capacity 65,536以上だけを`contiguous-resident`へ固定し、65,535以下、unknown target、
  exact `gfx942`の既存選択を変えないcandidateを実装した。実行中OOM後のfallbackは追加していない。
- virtual-contiguous providerの全plane growとshared-tail COWをappend transaction化した。create/map/access途中失敗時は
  今回追加したpage/handleを戻し、旧shared handle/read-only accessと物理accountingを復元する。rollback失敗はcontextをpoisonする。
- fake HIPへVMM failure injectionを追加し、create first/middle/last、map/access、COW first/cross-plane、retry、query、
  release、live resource baselineをhost testでPASSした。Rust core 283件、HIP 112件、CLI checkもPASSした。
- profiled executionのabortは一つのtotal timeout内でpending completionを全てdrainし、元のbackend/OOMをcleanup errorで
  上書きしない`CleanupFailure`へ変更した。実効prefill candidate、棄却見積り、選択chunk、KV physical auditをdirect結果へ追加した。
- この時点ではR9700実機を未実行であり、provider candidateの最終採用、GPU PASS、Phase 52完了は主張しない。

## 2026-08-24: R9700実機PASSと完了

- 実装をcommit `3ed002c476b49417cc702119e37c2389cefb96bc`へ固定し、ROCm 7.14.0、HIP `7.14.60850`、
  LLVM 23、Code Object V6、wave32のexact `gfx1201` release binaryをfresh buildした。binary SHA256は
  `79b0099f0c8981c46d1629debaf2aacfe551107adb13ec00465f4ebce11c8f81`である。
- 自動prefillは両行とも候補`[2048,512]`から2,048を選択した。`10,001/2`はcapacity 10,003の
  `virtual-contiguous`で3 warmup＋10 measured、`100,000/2`はcapacity 131,072の
  `contiguous-resident`で1 warmup＋3 measuredを全てPASSした。明示512、silent retry、fallbackは使っていない。
- 両行の全requestで生成tokenは`[23066,23066]`、HIP-only、fallback 0、timeoutなし、cleanup failure 0だった。
  process groupは消滅し、全GPU合計sysfs HBM/GTTはbaseline `98,664,448/100,134,912` bytesへexactに復帰した。
- 100kの8 KV layerはlogical/mapped capacity 131,072、observed 100,001、K/V合計commit
  `4,294,967,296` bytesだった。E2E measuredは`325.439180387/326.827859973/325.593963905`秒、中央値
  `325.593963905`秒、TTFT中央値`325.526989625`秒、sysfs HBM/GTT peakは
  `15,388,794,880/106,524,672` bytesだった。旧VMM failureのHBM peak `13,160,554,496` bytesを越えて完走し、
  Phase 50旧selectorの`26,414,587,904` bytesより41.74%低い。
- 10,001行のE2E中央値は`4.096388783`秒、TTFT中央値`4.0516689515`秒、HBM peak
  `11,429,343,232` bytesだった。KVは8 layer、observed 10,002、K/V合計commit `335,544,320` bytesである。
- 原因は総HBM不足ではなく、R9700長capacityのVMM page/handle commit経路に局所化した。driver内部の個別上限値までは
  推測せず、同じcontiguous pointer契約でpreflightに収まるresident providerをexact gfx1201長capacityだけへ採用した。
  extent集約は不要、明示512は原因分離不要、runtime retryは意味を変えるため不採用とした。
- raw rowはrepository外の
  `/home/homelab1/.local/share/sllm-evidence/phase52/r9700/final-3ed002c476b49417cc702119e37c2389cefb96bc/sllm/raw/`
  に置き、10,001/100k row SHA256は`b2d73f7fc1a1900b224b40b6a1ee452bcab0d9ecae9d4893963a31533cb71dfe`／
  `ac367c6320de15a581148828f67a22563c0bd4302004ab478d3ea0d63a0817b0`である。全反復値とidentityは
  [`phase52-r9700-kv-commit-summary-v1.json`](../../../../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)へ固定した。
- Phase 52を完了し、詳細計画をarchiveへ移した。Phase 51はユーザー指示どおり一時保留のままで、自動再開しない。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)
