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

## 2026-08-17: P20-A1〜A6 implementation and closeout

- GGUF v3 little-endian/alignment 32のbounded reader、same-FD payload read、deterministic writer、
  `derived-gguf-lock-v1`を実装した。duplicate、overflow、overlap、truncation、unknown type/extension、recipe driftをallocation前に拒否する。
- BF16=30、MXFP4=39、NVFP4=40を実装し、MXFP4/NVFP4のnibble/scale repackを独立decoderへexact照合した。
  FP8 E4M3FNはstandard I8 carrierとversioned `sllm.fp8.*` bindingで値・scaleを保持し、dtype代用品へ変換しない。
- Qwen3.5 dense BF16/FP8、Gemma 4 mixed NVFP4、Qwen3.5 MoE MXFP4 converterとruntime loweringを実装した。
  final outputのsize/SHA-256は9,343,583,840 / `50582d6c...9ca3`、5,779,142,624 / `1a9db28b...74b5`、
  9,337,229,760 / `4e0410c6...2fb5`、24,617,123,424 / `44022302...1fce`。converter commitは
  `ded2264035b8138da581773e42f37d11e3693fe1`で、BF16独立2回生成はbyte-identicalだった。
- 公開CLI/serverを`--gguf PATH --derived-lock PATH`へ統一し、source `--lock`/`--cache`、sidecar/provider引数、direct benchmark laneを
  公開parserとhelpから削除した。derived lockのsource fingerprintはbuild内reviewed lockへ解決し、GGUF failure時のsource fallbackはない。
  safetensors/sidecar readerはconverter・開発adapterに残した。
- R9700 `gfx1201`ではQwen BF16/FP8がtoken 11、Gemma mixedがtoken 236770、MoEがtoken 11となりsource経路と一致した。
  V620 `gfx1030`でもQwen BF16、Gemma mixed、MoEが同じtokenへ一致した。全caseはHIP-only、fallbackなし、cleanup 0。
- R9700 MoE serverはmodel-ready 22,009,574,016 byte、`/v1/models`、1-token chat completion、774 kernel dispatch、
  graceful shutdownのfinal tracked bytes/retryable/durableが全0でPASSした。
- 最終回帰はcore 173、GGUF contract 11、CLI 24、server 27 test、workspace check、Markdown link、diff checkをPASSした。
  4 final artifactの公開`verify-model`も旧引数なしでPASSした。生成GGUF、model、binary、raw traceはGitへ含めていない。
- integration reviewでは標準NVFP4/MXFP4 resident readbackのinverse nibble orderを修正し、exact-byte回帰を追加した。
  verified descriptor、graph schema、built-in source identity、公開parserを再確認し、未解決のcorrectness/security findingはない。
  llama.cppはA0のinspection sourceだけで、code import/adaptationはない。

## 2026-08-17: P20-A6 compatibility re-audit and final closeout

- closeout後の監査で、Qwen vision patch weightのrank 5がpinned llama.cppの`GGML_MAX_DIMS=4`を超えること、GGUF公開経路が
  vision request graphとMTP planを接続していないこと、A5のload/resident/peak/TTFT/TPOT値が未記録なことをfindingとして再開した。
  さらに公開helpに削除済み`--fp8-provider`の1行が残っていた。
- converter/writer/readerをmax rank 4へ固定し、rank 5以上はelement countを保つrank-4物理shapeとversion-1
  `sllm.tensor_recipe.logical_shapes`へ変換した。Qwen vision patch `[1024,3,2,16,16]`はGGUF dimension
  `[16,16,6,1024]`として格納し、runtime planで元shapeへ復元する。missing/extra/ambiguous、rank 4以下への不要override、
  element-count driftをallocation前に拒否する。
- 4 artifactをconverter commit `1189a3e22a135a9bc547372fdebf3b22e0ce6641`で再生成した。BF16は
  9,343,583,936 byte / `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、FP8は
  5,779,142,720 byte / `cf143f6c138f0e4a6372959bf348568159278202eca6081ce29346fdef1cfe0d`、Gemma NVFP4は
  9,337,229,760 byte / `4e0410c6afa45daef0a723c5adc7ab89c410c1f106d199b1c3c023c15e902fb5`、MoE MXFP4は
  24,617,123,520 byte / `0fddb97b41868e72efa4aa9aaa690bf53599f785927975c4eacbfa32cebc9620`となった。
- pinned llama.cpp `3cb7ffb1...c77ad934a70`のPython `GGUFReader`と`gguf_dump --no-tensors --json`は4 fileを
  version 3、extension 1、max rank 4としてparseした。C++ semantic loaderはFP8 fileをallocation前の固定tensor-name
  limitで明示的に拒否し、未知extensionを無視した実行や破損読出しへ進まないことも確認した。
- GGUF plan schemaをMTP graphのidentity/digest検査へ追加し、CLI/serverからGGUF MTP residentを接続した。R9700 text greedyは
  token 90700、492 kernel dispatch、HIP-only、fallbackなし、cleanup 0でPASSした。OpenAI serverはmodel-ready
  9,924,199,936 byte、`/v1/models`、1-token chat、shutdown後のcurrent/request/workspace/retryable/durableが全0だった。
- GGUF vision request graphとGGUF vision resident/manifest/prompt assemblyを接続した。実画像1枚の233-token prefillはR9700で
  token 760、493 kernel dispatch、10.599 s、process max RSS 885,180 KiB、V620でも同token/dispatch、13.157 s、
  max RSS 600,516 KiBとなり、両方HIP-only、fallbackなし、cleanup 0だった。途中で露見したmRoPE上限の
  `UINT32_MAX`→`int32_t`縮小比較は、I32 ABIで表現可能な全非負値を受理する境界へ修正しhost contract testを追加した。
- 再生成FP8はR9700でtoken 90700、740 kernel dispatchをPASSした。再生成MoEはR9700/V620の双方でtoken 90700、
  774 kernel dispatchをPASSした。Gemma NVFP4はbyte-identical SHAのため既存の両target source/GGUF照合を再利用した。
- A5固定laneはQwen BF16、prompt 11、output 2、3 warmup + 10 measuredで実施した。R9700はload
  10,653,616,324 ns、median TTFT 46,653,136 ns、median TPOT 26,689,093 ns、V620はload
  10,331,200,235 ns、median TTFT 184,142,667 ns、median TPOT 29,684,885 ns。両targetのresidentは
  8,411,592,192 byte、peakは8,512,933,508 byteで、1 loadを13 sampleとuntimed correctness controlで再利用し、
  HIP-only、fallbackなし、全request cleanupとmodel drop後allocation 0を確認した。
- `bc09604018da08434fc6e42f94d7397e21c22fc8`でGGUF MTP/vision、mRoPE境界、旧help表記を固定した。最終host回帰は
  core 174、GGUF contract 11、CLI 24、server 27 testをPASSした。push前監査でsmall-row matmulの公開dispatch ID
  12/13に名称定数がなく、host testが旧hipBLAS ID 4を期待していたstale contractを検出した。ABI値を変えずC/Rust定数、
  layout probe、serial-row symbol期待値を同期し、native public-runtime host suiteも新mRoPE caseを含めてPASSした。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase20-gguf-unification.md)
