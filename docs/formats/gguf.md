# sLLM GGUF format contract

## Status and scope

This document is the implemented Phase 20 format authority for the public model
container. The bounded reader, deterministic writer, converters, derived lock,
and GGUF-only public runtime were completed on 2026-08-17.

The public artifact is one GGUF file containing the model metadata, tensor
payloads, tokenizer, vocabulary, special-token data, and chat template needed by
the supported runtime path. Safetensors, quantization sidecars, and external
frontend files remain conversion inputs only.

## Base container

- GGUF magic bytes: `GGUF`.
- GGUF version: `3`.
- Byte order: little-endian for the initial Linux runtime.
- Default and required Phase 20 tensor-data alignment: 32 bytes.
- Tensor-table rank: at most 4, matching pinned `GGML_MAX_DIMS`. A source
  tensor above rank 4 is flattened without changing its element count, and its
  original shape is carried by the versioned recipe described below.
- Distribution shape: one file. Split GGUF is outside the initial contract.
- Implemented production architecture values are `qwen35`, `qwen35moe`,
  `gemma4`, and `gemma4moe`. `gemma4mtp` is a target-bound companion.
  `deepseek4` is parser-recognized for the Phase 57 foundation catalog only
  and `minimax-m3` is parser-recognized for the Phase 58 foundation catalog
  only. `diffusion-gemma` is an sLLM Phase 59 write-disabled foundation key;
  the fixed llama.cpp revision has no merged DiffusionGemma architecture.
  None of these three foundation keys is production-loadable.
- Standard metadata keys are used wherever the pinned source defines the same
  semantic value, including `general.architecture`, `general.alignment`,
  tokenizer fields, token IDs, and `tokenizer.chat_template`.

The inspected base is the MIT-licensed llama.cpp `b10453`, full commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. Exact inspected file hashes are in
`ci/matrix/phase20-gguf-a0-v1.json`. Inspection does not by itself create a code
import. A later direct reuse must add the provenance event and notices required
by `docs/provenance/README.md` before release.

## Tensor encoding boundary

| sLLM semantic encoding | GGUF representation fixed by A0 | Conversion |
| --- | --- | --- |
| unquantized BF16 | standard `GGML_TYPE_BF16` (`30`), block 1 / 2 bytes | preserve BF16 bits |
| OCP MXFP4 E2M1, block 32, E8M0 scale | standard `GGML_TYPE_MXFP4` (`39`), block 32 / 17 bytes | lossless interleave of scale byte and packed codes |
| NVFP4 E2M1, block 16, E4M3 scale, outer scale | standard `GGML_TYPE_NVFP4` (`40`), super-block 64 / 36 bytes plus named scale tensors | lossless grouping of four source blocks; preserve scale/code bits |
| FP8 E4M3FN plus channel BF16 scale | no standard type in the pinned source | versioned sLLM extension required in A1; no dequantized substitution |

`MXFP4` and `NVFP4` are not interchangeable. Their block size, scale encoding,
packing, and outer-scale rules remain distinct. The converter may move bits into
the standard block layout but must not run a new quantizer. Source and output
logical values are checked by independent decode in a later work unit.

The pinned enum has no FP8 tensor type. sLLM therefore stores FP8 value planes as
standard I8 carrier tensors and binds them to scale tensors with versioned
`sllm.fp8.*` metadata. Readers unaware of the extension can still inspect the
GGUF structure; sLLM rejects missing, unknown, or ambiguous extension bindings.
No dequantized BF16, F16, or Q8_0 substitute is produced.

The same version-1 `sllm.tensor_recipe` contains a `logical_shapes` table for
source tensors whose rank exceeds the standard four-dimensional tensor table.
Each entry names one physical tensor and its original logical shape. The reader
requires rank greater than 4, exact element-count preservation, one-to-one
names, a matching recipe digest, and no unused override. The Qwen vision patch
weight `[1024,3,2,16,16]` is therefore stored physically as
`[16,16,6,1024]` in GGUF dimension order and restored to the source logical
shape before graph planning.

## Container-neutral lowering

The GGUF reader produces the same internal contracts as the reviewed source
importers. Container-specific offsets and names do not enter execution planning.

- Qwen3.5 dense: reviewed model identity, 738-tensor catalog, text plan, MTP 15
  tensors, vision 297 tensors, tokenizer/chat metadata.
- Gemma 4 BF16: reviewed identity, 677 physical tensors, 666 loadable text
  tensors, architecture/frontend metadata.
- Gemma 4 NVFP4 mixed: 1,389 physical to 677 logical tensors; 144 NVFP4 MLP,
  184 FP8 attention, 48-layer static FP8 KV recipe, BF16/ignored remainder.
- Qwen3.5 MoE MXFP4: reviewed text inventory of 62,053 tensors, 493-entry load
  plan, expert-axis recipe, tokenizer/chat metadata. Vision 333 and MTP 785
  source tensors stay known-unconsumed until their existing execution scope is
  intentionally enabled.
- Gemma 4 26B-A4B MoE NVFP4: canonical architecture `gemma4moe`, 35,513
  tensors, 597-entry resident plan, 30 layer-packed routed-expert blobs,
  direct BF16/F32 planes, tokenizer/chat metadata, and an implicit-unit static
  E4M3 KV recipe. The 356 vision tensors remain known-unconsumed in the
  text-only scope.

The machine-readable manifest contains the exact fingerprints, revisions,
counts, and recipe digests. A converter or loader must reject a mismatch rather
than infer a nearby layout.

## Derived identity

A GGUF output is a derived artifact of one or more reviewed source locks. Its
model lock records:

- every source lock fingerprint;
- converter repository and full commit;
- complete arguments and effective configuration, including defaults;
- relevant environment and dependency identity;
- output path, size, and SHA-256;
- GGUF metadata and tensor-catalog digests;
- semantic model identity and the encoding/recipe digest lowered to runtime.

The semantic model identity is preserved across source and GGUF containers. The
container digest changes, while aliases move only through an explicit lock
change. Runtime verification binds the opened GGUF descriptor before metadata
or tensor reads and does not reopen the path after verification.

## Fail-closed parser requirements

Before allocation or GPU work, the reader rejects unsupported version or byte
order, duplicate metadata or tensor names, unknown architecture/type/extension,
integer overflow, invalid dimension/block multiple, range overlap, range beyond
EOF, bad alignment, rank above 4, truncated string/array/table, and incomplete
or ambiguous recipe or logical-shape bindings. The writer applies the same rank
limit. A GGUF failure never falls back to an unverified safetensors or sidecar
path.

## Implemented boundary

The public CLI and server accept exactly one GGUF plus its derived lock. Source
safetensors and sidecars remain converter/development inputs only. Runtime
verification hashes the GGUF, validates metadata and tensor ranges, and retains
the verified open descriptor for payload reads; it never falls back to a source
container.

### Gemma 4 MTP companion

`gemma4mtp`は単独生成modelではなく、reviewed dense `gemma4` targetへ結合するcompanion architectureである。metadataはassistant model-lock
fingerprint、target fingerprint、pair semantic identity、Q-only topologyを固定し、tensor catalogは48 BF16 tensorをcanonical name／shape／rangeで
保持する。derived lockはsource revisionとGGUF digestへ結合し、source safetensorsとGGUFの全tensor byteをlosslessに照合する。

model libraryは`gemma4mtp`をsupported architectureとして認識するが、standalone aliasへ登録しない。canonical lockが隣接し、exact target pair、
exact `gfx1201`、合計resident capacityを満たすときだけtargetのcompanion pathへ付加する。target GGUFのmixed low-bit encodingとassistant BF16を
一つのflattened weight planへ偽装せず、それぞれのartifact identityとresident accountingを保持する。

### DeepSeek V4 foundation catalog

`deepseek4`はPhase 57でcontainer parserとmodel libraryが認識するが、GGUF production loader、CLI／API generation、または
書込み可能なconversion planへ接続しない。固定sourceは`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`で、公式indexは72,317 source tensor、advertised payload
166,878,536,440 bytesを持つ。reader／writerのbounded tensor-table上限は100,000であり、このcatalogを表現できる一方、
100,001以上は引き続き拒否する。

foundation dry-runはconfigとindex identity、target main 43 layer、checkpoint DSpark 3 stage、source family、canonical target
metadata、mapping digestだけを検証する。70,656 routed-expert value／scale sourceを46 block×3 projectionへnumeric expert順で
stackすると138 MXFP4 tensorになり、direct 1,661 tensorと合わせた将来候補は1,799 physical tensorである。ただしsafetensors
headerだけは48 shard全てをbounded-readし、72,317 tensorのshape、dtype、payload rangeをindexへ照合済みである。tensor payload bytesと
全local shard hashは未読で、現行GGUF type／recipeもblock-128×128 E4M3＋UE8M0とsource I32をlosslessに表せないため、
`GgufWritePlan`、output bytes、file type、quantization versionを生成しない。`num_nextn_predict_layers=1`とcheckpointの3-stage DSparkを
一つのblock countへ潰さず、DSparkのpublic artifact／
canonical tensor nameも後段決定までfreezeしない。

model libraryはこのarchitectureを灰色行で表示し、少なくとも166,878,536,440 resident bytesがKV／workspace前に必要であることと、
production loading未対応を理由として返す。reviewed architectureの認識はload可能性の主張ではなく、dynamic registration callbackを
呼び出さない。

### MiniMax M3 foundation catalog

`minimax-m3`はPhase 58でcontainer parserとmodel libraryが認識するが、GGUF production loader、CLI／API generation、または
書込み可能なconversion planへ接続しない。固定sourceは`MiniMaxAI/MiniMax-M3` revision
`f0e1c1e04d40177e4673a22097036854f536e9c0`で、公式indexは23,416 source tensorと
`metadata.total_size=869,157,697,024`を持つ。一方、59 shardのfile size合計は854,176,398,808 bytes、
header由来payloadは854,172,958,720 bytesであり、manifestは整合しない。admissionは最大値へfail-closeし、差を消去しない。

foundation dry-runはtext 22,893 tensorとvision／projector 523 tensorをtyped familyへ分類する。57 layer×128 expert×3 projectionの
21,888 routed-expert sourceをnumeric expert順で171 expert-axis output候補へstackし、direct source 1,528と合わせて
1,699 physical candidateとなる。mapping serializationのSHA-256は
`93ad9f5467bb9a7ba3b77c96db5aa0641e5d9e9801f99dc49bf46a8a4a18dd3f`である。

59 shardのheader prefix 3,440,088 bytesだけはbounded-readし、23,416 tensorのBF16／F32 dtype、shape、range、shard coverageを
照合したが、weight payloadと全local shard hashは未読である。Gemma-style normの`+1` bake、vision patchのtemporal slice、
dtype／quantization変換をmetadata catalogが実行したとは扱わないため、`GgufWritePlan`、output bytes、file typeを生成しない。
configのMTP 7 moduleに対応するtensorもreleased indexにないため、MTP weightやcanonical output名を推測しない。

model libraryはMiniMax Community License、manifest不整合、少なくとも869,157,697,024 resident bytesがKV／workspace前に必要であること、
production loading未対応を含む灰色行を返す。reviewed architectureの認識はload可能性の主張ではなく、dynamic registration callbackを
呼び出さない。

### DiffusionGemma foundation catalog

`diffusion-gemma`はPhase 59でcontainer parserとmodel libraryが認識するsLLM固有のwrite-disabled foundation keyであり、
固定llama.cpp revisionにはmerged architecture定義がない。公開artifactは`google/diffusiongemma-26B-A4B-it` revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`、Apache-2.0、11 shard／1,047 source tensorである。index payloadは
51,647,562,456 bytes、header prefix込みshard file合計は51,647,701,024 bytesなので、admissionは大きい後者を使う。

foundation dry-runは固定config／index／header catalog、decoder text、encoder layer scalar、vision、self-conditioningのsource family、
将来のcontainer-neutral mappingだけを検証する。公式shard payloadはlocal full hash未検証であり、現段階ではnormalization bake、
tensor transform、payload write、GGUF output hash、production load、generationのいずれも主張しない。openな第三者proposalに同じ
spellingが存在しても、merge済みstandardまたはcanonical upstream approvalとして扱わない。

model libraryはApache-2.0、少なくとも51,647,701,024 resident bytesがKV／workspace前に必要であること、production loader未対応を
灰色行へ表示する。dynamic registration callback、CLI／API／WebUI generationへは渡さない。

### Direct official `mistral3` GGUF

Phase 60のproduction候補は公式`Ministral-3-3B-Instruct-2512-BF16.gguf`を直接読む。tensor catalogは236件で、
183 BF16 matrixと53 F32 normalization tensorから成る。全BF16 payloadの下位16 bitが0であることと、F32 normalizationを
損失なく保持することを確認する。text graphは`mistral3` text tensorだけを受理し、vision／projector tensor、未知のtensor、
固定metadataまたはshapeのdriftをallocation前に拒否する。

この公式GGUFの`tokenizer.ggml.scores`はI32 zero arrayである。sLLMは公式型をそのまま受理する。固定llama.cppとの品質比較では、
比較器を起動するためだけにreflink copy上のarray element typeをF32へ読み替え、zero payloadは変更しなかった。この一時的なoracle
copyはproduction artifact、derived lock、またはsLLM parser contractではない。
