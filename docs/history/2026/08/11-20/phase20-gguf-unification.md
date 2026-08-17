# Phase 20 GGUF unification history

## 2026-08-17: P20-A0 source/format lock and handoff inventory

- Phase 20をGGUF converter、reader/runtime、standard/extension format、derived model lock、移行・互換性closeoutへ限定し、
  request batching、chunked prefill、永続KV、追加model/KV形式、multi-GPUを非対象とするactive planを作成した。
- base sourceをlocal source-lock済みllama.cpp `b10453` commit
  `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`へ固定し、GGUF header/parser/writer/constants/quant blockと
  Qwen3.5/Gemma 4 converterの8 fileをSHA-256付きでinventory化した。A0はinspection-onlyでありcode importはない。
- 公開containerをGGUF v3、little-endian、32-byte alignment、single-fileに固定した。standard architectureは
  `qwen35`、`qwen35moe`、`gemma4`、standard tensor typeはBF16=30、MXFP4=39、NVFP4=40とした。
- NVFP4/MXFP4はsource value/scaleを再量子化せずstandard blockへlossless repackする。pinned GGUFに標準FP8 tensor typeが
  ないため、FP8のdequantized BF16/F16/Q8_0代用とA0でのprivate numeric type ID割当を拒否し、A1でversioned extensionを固定する。
- Phase 17/16F/19からQwen dense 738 tensor + MTP/vision、Gemma BF16 677/666 tensor、Gemma mixed
  1,389 physical/677 logical tensor、Qwen MoE 62,053 text tensor/493-entry planをcontainer-neutral handoffとして固定した。
- derived lockはsource fingerprints、converter identity/config/environment、GGUF size/hash、metadata/catalog digestを含み、
  semantic identityをcontainer間で維持し、verified file descriptorをpathから再openしない契約とした。
- local Qwen3.8 subagentへA0全体のread-only design auditを委譲したが、32,081-token prefill後に複数turnを継続し、
  wrapperの600秒上限でfinal reportなしに終了した。serverは正常で、観測decodeは概ね22〜31 token/sだった。A0を止めず、
  後続のQwen試用を単一test fileの実装へ縮小した。
- Qwenへ`ci/tests/test_phase20_gguf_a0.py`だけの実装を委譲すると205.51秒で完了し、13 testをPASSした。main agentは
  testを全行reviewし、source identity setとhandoff core inventoryの2 testを追加した。focused A0 + model-lock回帰は
  43 test/71 subtest PASS、local pinned sourceのcommit/8 file hash/GGUF constants照合、Markdown link、diff checkもPASSした。
- machine-readable schema/manifest、format/model-lock/provenance/main plan/historyを同期し、A0の全受入条件を満たした。
  converter、reader、GPU、full-model evidenceは取得しておらず、計画どおりP20-A1以降の範囲である。

[対応する計画](../../../../plans/active/2026/08/11-20/phase20-gguf-unification.md)
