# Phase 56: Gemma 4 12B MTP assistant production path

> 状態: 完了
> 作成日: 2026-08-31
> 完了日: 2026-08-31

## 目的

計画済みのGemma 4 MTPを、既存Gemma 4 12B Dense target、model-neutral speculative transaction、GGUF、CLI／API／WebUIへ
統合する。architecture文字列のacceptだけでは完了とせず、Google公式assistantを単一32 GiB AMD GPUでtarget-onlyと数値的に
同じvisible token列を生成するproduction pathまで実装する。

## 固定対象

- target: `google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`、既存
  `model-lock-v2` fingerprint `sha256:381c94bcb48a26d8ef83d1c3d7c5a3513ef8fac4a638752731b85c119385f09d`。
- assistant: `google/gemma-4-12B-it-assistant` revision
  `46d4c6f13f0ac0ad827b915669b8df9b81c64c51`、BF16、845,719,296-byte safetensors、48 tensor。
- topology: backbone hidden 3,840、assistant hidden 1,024、4 layer、sliding 3／full 1、Q-only attention、
  pre projection 7,680→1,024、post projection 1,024→3,840、assistant vocab head 262,144×1,024。
- KV mapping: assistant sliding layer 0..2はtarget sliding layer 46、assistant full layer 3はtarget full layer 47のrequest-local KVを読む。
- scope: text-only、既存reviewed mixed NVFP4 W4A4／FP8 W8A8 targetとBF16 assistant、single GPU、single active request、
  greedy、draft width 1、CLI／OpenAI-compatible API／dynamic model lifecycle／WebUI。primary targetはR9700 exact `gfx1201`とする。
- reader結果は[Gemma 4 MTP reader記録](../../../../../references/gemma4-mtp-reader.md)を正本とし、外部engine codeを流用しない。

## 受入条件

1. assistantの完全revision、license、7 fileのsize/hash、config、generation config、tokenizer、safetensors header／catalog／全48 tensorを
   fixed lockへ記録する。target tokenizerをwire正本とし、両者のvocabと共通generation IDを検査しつつ、targetだけが名前付きで持つ
   `<|video|>` ID 258,884は固定pairのdocumented差として許可する。backbone hidden、KV layer type対応もallocation前に検証する。
2. target embedding＋hidden連結、pre/post projection、4層Q-only attention、4種norm、GELU-tanh MLP、full-vocab argmaxを既存semantic opへ
   lowerし、K/V projection捏造、CPU numerical fallback、assistantによるtarget KV追記を行わない。
3. target executionはprefill／decodeの必要なnormalized hidden rowと、draft verification用の連続target rowsを同じrequest ownerから返す。
   assistantは同一末尾positionとtarget KV snapshotを読み、proposal中は公開target stateを更新しない。
4. model-neutral speculative transactionでdraft width 1を逐次target検証し、target-only greedyとvisible token列、finish reason、stop除外、usageを
   一致させる。reject、length、stop、cancelで未消費rowを確実にrollback／破棄する。
5. canonical GGUF architecture／metadata／tensor mappingとderived lockを追加し、source assistantとGGUF assistantのdescriptor、tensor bytes、
   proposal／visible outputを照合する。safetensorsは変換入力、GGUFは公開runtime artifactとする。
6. 通常CLI、Chat Completions非stream／SSE、raw Completions、benchmark、model folder選択、load／unload、metrics、WebUIを追加の隠しflagなしで
   利用できるようにする。未指定時の既存target-only経路は維持し、MTPを有効にしたaliasだけassistantを要求する。
7. context 2,048を初期actual scopeとし、exact `gfx1201`でtarget-only／MTPのfixed・Unicode・code・stop、prefill、decode、reject、
   cancel／recovery、連続要求、unload／shutdownを
   HIP-only、fallback 0、nonfinite 0、cleanup 0でPASSする。token完全一致とdraft proposed／accepted／rejected計測を記録する。
8. focused host/GPU checks、1回のintegration review、runtime、model lock、GGUF、provenance、compatibility、main plan、historyを同期する。

性能向上率は完了条件にしない。draft width 1がtarget-onlyより遅い場合も正しさと実測を記録し、幅拡張や融合はprofile後の別作業とする。

## 実装順序

1. assistant lock、reader記録、target互換性、48 tensor load planを実装する。
2. assistant graph／resident／requestと、target KV read-only view／hidden outputを接続する。
3. Gemma MTP generation executorを既存model-neutral speculative adapterへ接続する。
4. assistant GGUF converter／reader／derived lockを追加する。
5. CLI、server、dynamic model library、metrics、WebUIへ統合する。
6. actual source／GGUF、exact GPU、service smoke、計画・履歴同期、integration reviewを行う。

## 非対象

- sampled MTP、draft width 2以上、centroid masked embedding、vision／audio MTP。
- Gemma 4 MoE assistant、DeepSeek v4 DFlash、MiniMax M3、multi-GPU／tensor parallel。
- CUDA／Triton実装、vLLM／SGLang／Transformers sourceのcopy。
- INT GGUF量子化、assistant low-bit化、性能最適化の先行実装。

[対応する履歴](../../../../../history/2026/08/21-31/phase56-gemma4-mtp.md)
