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

[対応する計画](../../../../plans/active/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)
