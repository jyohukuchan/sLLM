# Qwen3.5 Phase 3 reader記録

## 範囲とprovenance境界

この記録はPhase 3のmodel lock、RMSNorm、G2実装へ渡すコード表現を含まない技術要点である。vLLMのcodeをcopy、adapt、portしない。llama.cppからも今回はcodeを直接reuseせず、直接reuseへ変更する場合は[provenance方針](../provenance/README.md)のnoticeとimport記録を先に完了する。

| source | local path | 固定commit SHA |
| --- | --- | --- |
| llama.cpp b10227 | `reference/llama.cpp` | `f5919bf458ef190468b5c329bb293f8a54a1e69c` |
| vLLM v0.26.0 | `reference/vLLM` | `568afb3a13806beb53bb2e6bd518269357b237c0` |

固定identityは[source-lock manifest](source-lock.md)を正とする。

## configとlayer schedule

- Qwen3.5 text stackは`linear_attention`と`full_attention`のhybridで、明示`layer_types`を正とする。
- `layer_types`の長さは`num_hidden_layers`と完全一致し、許可値は上記2種類だけとする。`full_attention_interval`と明示listが同時にある場合は整合を検証し、暗黙の既定値へfallbackしない。
- 採用revisionのtext configはhidden size 2560、32 main layers、4層ごとのfull attention、`rms_norm_eps=1e-6`、BF16、dense FFNである。
- MTP layerはmain stackへ数えず別componentとする。visionとMTPはPhase 3 runtimeで使わないが、未知tensorとして黙って無視せずlock済み既知未消費componentへ分類する。
- top-level multimodal configから`text_config`を明示的に選び、vision対応済みとは表記しない。

## safetensors mapping

- indexの`weight_map`を正として必要shardを解決し、index、全参照shard、実shard内tensor名を相互照合する。
- shard size、実bytes SHA-256、Git blob、LFS identityをmodel lockへ照合する。indexまたはshard metadataだけをcontent verificationの代用にしない。
- tensorごとにdtype、shape、header/data offsetとfile boundsを検証する。missing、duplicate、overlap、out-of-range、index外tensor、未参照shardを拒否する。
- vLLMの一般的なbias/optional skip規則をsLLMのglobal ignoreとして採用しない。必須、lock済み既知未消費、config条件付き、常時拒否の4分類をmodel contractへ列挙する。
- llama.cpp runtimeはGGUF loaderであり、direct safetensors runtime loaderの根拠にはしない。

## RMSNorm semantics

- Qwen3.5はGemma系のoffset-one RMSNorm variantで、HF checkpointのraw weightに対する実効scaleは`1 + raw_weight`である。
- raw BF16 weightを通常scaleとして直接乗算せず、disk上で事前変換しない。semantic descriptor、NumPy oracle、HIP kernelでscale modeを明示する。
- epsilonはlocked configの`text_config.rms_norm_eps`だけを使い、既定値へfallbackしない。
- 二乗和、平均、epsilon適用、逆平方根、offset-one scale適用はFP32で行い、出力をBF16へ丸めるbaseline contractとする。
- linear attention内部のL2 normalizationはinput/post-attention RMSNormと別opであり、Phase 3 RMSNormへ混ぜない。
- 固定sourceはstride、alignment、alias、公開ABIを規定しないため、sLLMでは最終次元連続、明示shape/stride/alignment、in-place非対応、input/output alias拒否を独自contractとして固定する。

## Phase 3 test分割

1. config、index、shard metadata、hash、tensor集合のhost validation。
2. synthetic tiny fixtureによるRMSNorm semantic contractと独立NumPy oracle。
3. private diagnostic G1とは別のsemantic RMSNorm G1。
4. verified read-only model cacheからの実weight slice抽出。
5. 独立生成したBF16 activationとNumPy FP32 oracleによるG2。
6. canonical `gfx1030`/`gfx1201`の同一candidate evidenceと短いP0 smoke。

full model、generation、vision、MTP、CPU fallback、GPU emulationはこのG2から除外する。raw model/sliceはGitまたはreportへ保存せず、source fingerprint、tensor、byte range、recipe、size、SHA-256だけを記録する。

## 採否

- 採用: fail-closed config/tensor validation、明示layer schedule、offset-one RMSNorm semantics、component単位のtensor分類、host/synthetic/real-slice/GPU test分割。
- 不採用: 広いtensor ignore、converter後のGGUF表現をHF raw weightと同一視すること、固定sourceの既定値fallback、他engine出力を数値oracleにすること。
- 固定sourceに存在しないためsLLMで独自に実装する: model lock/JCS、direct safetensors byte validation、public C ABI、stride/alignment/alias contract、semantic G1/G2/P0 evidence。

[対応するPhase 3 Stage A計画](../plans/archive/2026/08/1-10/phase3-model-lock-rmsnorm-g2.md)
