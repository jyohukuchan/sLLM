# Gemma 4 MTP reader記録

> 調査日: 2026-08-31
> 実装へのコード流用: なし

## 目的と参照境界

計画済みのGemma 4 MTPを実装する前に、公式checkpointと固定済み推論engineから、assistant固有のtensor topology、
targetとの状態共有、投機decode上の不変条件だけを抽出する。vLLMはreader-onlyであり、source codeのcopy、adapt、portを行わない。
llama.cppも本調査ではGGUF metadata候補とtensor分類の確認だけに使い、コードを流用しない。

参照した固定sourceは次のとおりである。

- 公式model: `google/gemma-4-12B-it-assistant` revision
  `46d4c6f13f0ac0ad827b915669b8df9b81c64c51`、Apache-2.0。
- target model: 既存lock済み`google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`。
- vLLM `v0.26.0` / `568afb3a13806beb53bb2e6bd518269357b237c0`。
- llama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`。

engineの固定条件とreader-only方針は[参照元固定](source-lock.md)と[推論engine参照方針](inference-engines.md)を正本とする。

## 公式artifactから固定した事実

- top-level architectureは`Gemma4UnifiedAssistantForCausalLM`、model typeは`gemma4_unified_assistant`。
- assistant hiddenは1,024、target backbone hiddenは3,840、4層、intermediate 8,192、vocab 262,144。
- layer typeはsliding 3層、full 1層。sliding head dim 256、full head dim 512、16 Q headである。
- attention weightはQ/Oだけを持ち、K/V projectionを持たない。全assistant層はtargetが既に保持するKVを読む。
- `pre_projection=[1024,7680]`はtarget token embedding 3,840とtarget hidden 3,840の連結をassistant hiddenへ写像する。
  `post_projection=[3840,1024]`はassistant出力を次draft stepのbackbone hiddenへ戻す。
- assistant自身の`embed_tokens=[262144,1024]`はdraft logits headとして使う。targetの3,840-wide embeddingとは同一tensorではない。
- `use_ordered_embeddings=false`で、centroid／token ordering tensorはartifactに存在しない。初期sLLM経路は完全vocab argmaxを使う。
- safetensorsは845,719,296 bytes、file SHA-256は
  `3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df`。headerは5,360 bytes、header SHA-256は
  `d0f1537ec1254122003a892254cefcf44c538f2cc42ba612b5791f4c6c5fdcb4`。
- 48 tensorはすべてBF16で、catalog SHA-256は
  `fd87240fd7fe1beac3b7f39ff3d4ae93e4c5a3fb4192fc556a8a2f28d892cc3d`。

## 実装へ渡す意味契約

1. assistantのsliding層はtargetの最後のsliding層46、full層は最後のfull層47のKVを読む。近似layerやattention typeを
   暗黙選択しない。
2. 一つのdraft列の各stepは同じtarget末尾position／KV lengthを使い、assistantがtarget KVへ追記しない。draft feedbackだけを
   `post_projection`出力で更新する。
3. target hiddenとtoken embeddingをBF16入力として連結し、pre projection、4層、final norm、assistant logits headの順に実行する。
4. 初期draft widthは1とし、既存のmodel-neutral speculative adapterがproposalをtargetで逐次検証する。visible token、stop、usage、
   RNGの正本はtarget-only生成であり、assistant proposalを無検証で公開しない。
5. target tokenizerをwire上の正本とする。両artifactはvocab 262,144とgenerationに必要な共通IDを一致させる一方、固定targetだけが
   `<|video|>` ID 258,884を名前付きtokenとして持つ差はpair contractへ明記して許可する。tokenizer fileのbyte一致へ一般化しない。
6. targetとassistantのrepo、revision、documented pair、vocab幅、共通special token、hidden幅、layer mapping、tensor catalogのいずれかが
   不一致ならresident allocation前にfail-closedとする。

この記録は外部engineのperformance、CUDA実装、scheduler構造をsLLMへ移植する許可ではない。実装は既存sLLM semantic op、
Gemma resident state、speculative transactionから独立に構成する。
