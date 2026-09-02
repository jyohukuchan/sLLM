# Phase 71: Qwen3.5-27B MXFP6 compatibility and bounded-VRAM benchmark

状態: `完了（2026-09-02）`

## ユーザー指示と目的

2026-09-02のユーザー指示により、公式Qwen3.5-27B BF16をreviewed modelとして追加し、Phase 70で確立した
model非依存MXFP6経路へ接続する。canonical V620 exact `gfx1030`とRadeon AI PRO R9700 exact `gfx1201`の
各32 GiB VRAMへ収まる入力長で、実model prefillを測定する。

27B専用matmul kernelやselectorは追加しない。27Bの24 query heads、4 KV heads、GQA比6を既存の汎用
attention preprocess／causal attention fallbackへ通し、既存の8／16／32 headsとGQA比2／4／8／16を退行させない。

## 固定identity

- source: `Qwen/Qwen3.5-27B`
- revision: `fc05daec18b0a78c049392ed2e771dde82bdf654`
- model lock fingerprint:
  `sha256:a4a0a6192babfdb7b1fc3ac75cc340e96df87fe2b0e629cc1510085bfeced97f`
- architecture: hidden 5,120、intermediate 17,408、64 layers、24 query heads、4 KV heads、head dim 256、
  48 linear-attention layers＋16 full-attention layers、untied output projection。
- target model format: OCP MXFP6 E3M2 W6A6、block 32、E8M0 scale、FP32 accumulation、BF16 output。
- KV: 明示FP16。KV default、model recipe、sampling、public ABIは変更しない。

## 作業単位

1. 11 shardとmodel card／tokenizer類をimmutable revisionへ固定したmodel lockを追加し、全file hash、tensor catalog、
   required weight load plan、非整列境界を含むgraph buildを検証する。
2. `--model-size 27B`とreviewed direct benchmark identityを追加する。
3. 24-head sigmoid gate、attention preprocess、causal attentionとGQA比6を汎用経路へ追加する。既存の
   gfx1030／gfx1201 optimized GQA4 selectorはscopeを広げず、27Bは対応済みbaselineを使う。
4. reviewed BF16 sourceからMXFP6 bundleを生成し、manifest、GGUF tensor count、payload size、SHA-256を固定する。
5. 同じsourceからtarget別release binaryをbuildし、direct pretokenized input 512、最大4 output、greedy、ignore EOS、
   3 warmup＋10 measuredを両GPUで実行する。512が安全に収まった場合だけ2,048も試す。
6. 実dispatch、HIP-only、fallback 0、cleanup 0、resident／peak VRAM、prefill中央値とMADを記録し、計画、互換性、
   履歴、追跡summaryを同期する。

## 受入条件

- model cache、load plan、1／3／17／255／256／257 token graphがfail-closed検証を通る。
- MXFP6 conversionが851 required language weightsを欠落なく格納し、derived artifact identityを検証できる。
- exact `gfx1030`とexact `gfx1201`で少なくとも512-token行がVRAM内に収まり、HIP-only、fallbackなし、正常cleanupで完了する。
- OOM、timeout、crash、CPU fallbackはGPU PASSに数えない。2,048-tokenが収まらない場合はfail-closed結果とpeakを記録し、
  512-token成功を否定しない。
- 既存の2B／4B／9B model lock、head構成、MXFP6／MXFP8 selectorは変更しない。

## 対象外

- 27B固有kernel最適化、FP32 attention/KV保存、KV量子化変更、multi-GPU／tensor parallel、vision入力、
  batch throughput、MXFP8／NVFP4の27B artifact生成は対象外とする。
- draft local artifactはrelease provenanceを主張しない。公開時はclean candidateとimmutable final identityを別途固定する。

## 完了結果

- 公式revisionの23 fileを`model-lock-v1` fingerprint
  `sha256:a4a0a6192babfdb7b1fc3ac75cc340e96df87fe2b0e629cc1510085bfeced97f`へ固定した。全hash、1,199 tensor catalog、
  851 loadable text weight、348 known-unconsumed tensor、1／3／17／255／256／257-token graphを検証した。v1 schemaの
  `generation_config`を明示的なabsent／locked-path組として一般化し、27Bの`generation_config.json`をlock内fileへ結合した。
- `--model-size 27B`、24 query heads、GQA比6、linear-attention value heads 48、hidden 5,120、intermediate 17,408を
  reviewed shapeとして追加した。既存target専用selectorは広げず、汎用attention／linear-attention経路へ接続した。
- repository外のMXFP6 GGUFは25,909,762,816 byte、SHA-256
  `3b7151e5c601f3efee524e4998e403b800699fbf6e9097918f983e3c72876ddd`、1,695 tensor、derived fingerprint
  `sha256:d1142468252af487d52ebf72a29a4bb62487a635c174e709bebd73b0c337a82c`として検証した。
- 512入力、chunk 512、3 warmup＋10 measuredのprefill中央値／MADは、gfx1030が
  `34.298907／0.157267 tok/s`、gfx1201が`81.746517／0.065546 tok/s`だった。両行ともresident
  `24,115,002,880` byte、peak `24,777,018,880` byteで、13/13 request、HIP-only、fallback 0、cleanup 0だった。
- 2,048入力、chunk 1,024、1 warmup＋3 measuredはgfx1030 `33.448016／0.148480 tok/s`、gfx1201
  `77.409011／0.018784 tok/s`、peak `25,351,937,536` byteでPASSした。gfx1030の外部VRAM snapshotは
  `26,990,432,256 / 34,342,961,152` byteで、終了後baselineへ復帰した。
- gfx1201の2,048入力／単一2,048 chunkはplacement見積り後、layer 56 MLP downのlow-precision matmul workspace
  `hipMalloc`でOOMとなった。この試行はPASSに数えず、2,048入力で安全に確認した上限はchunk 1,024とする。
- 27B専用matmul kernel、persistent展開、KV default変更は追加していない。MXFP6は既存Phase 70のmodel非依存経路を
  そのまま再利用しており、今回の速度は新しい27B最適化値ではなく互換性・capacity evidenceである。

[全体計画](../../../../main-plan.md) /
[Phase 70保存済み計画](../../../../archive/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md) /
[Phase 71履歴](../../../../../history/2026/09/1-10/phase71-qwen35-27b-mxfp6-compatibility.md) /
[Phase 71追跡要約](../../../../../../ci/matrix/phase71-qwen35-27b-mxfp6-compatibility-v1.json)
