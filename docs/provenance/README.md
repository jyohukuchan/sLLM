# Third-party provenance and code reuse

This document defines the minimum provenance record for external source material
used by sLLM. It is an engineering policy, not legal advice. License obligations
must be checked for the exact upstream revision and files being used.

## Reuse policy

### llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) may be used as an
implementation reference. Direct reuse is allowed and should be considered
before writing a clean implementation. Draft development has no precondition of
an independent human review or a complete import record; the record below is
required for release or distribution.

Direct reuse must not be represented only by a few general sentences in the
project `LICENSE`. For every exact copy, adaptation, or source-to-source port
intended for release or distribution:

1. Verify the license and copyright notices in the exact upstream revision and
   source path before release or distribution.
2. Add an entry to the repository-root `THIRD_PARTY_NOTICES.md`. Create that file
   when the first actual third-party import is made; an empty placeholder is not
   required.
3. Put a concise provenance header in every copied-to local source file. The
   header must identify the notice entry and must survive later refactoring.
4. Record the upstream repository URL, full commit SHA, source path, upstream Git
   blob ID, local destination, imported SHA-256, copyright, license, reuse mode,
   modifications, and the sLLM import commit.
5. Preserve all upstream license and notice material required for redistribution.

`exact`, `adapted`, and `ported` are all direct reuse modes:

- `exact`: source text is copied without substantive changes.
- `adapted`: source text or structure is copied and modified.
- `ported`: an upstream implementation is translated to another language or API
  while retaining protectable implementation expression or close structure.

A clean implementation based only on separately documented technical facts is
not a direct import and does not use one of these reuse modes; its design notes
must still identify the references consulted. It must not contain copied source
text or a close translation. When uncertain, treat the work as direct reuse and
record it as direct reuse.

Suggested source-file header:

```text
// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#<stable-entry-id>
// Upstream: <repository URL> @ <full commit SHA>, <source path>
// SPDX-License-Identifier: <license identifier for this file>
```

The SPDX line describes the licensing conclusion for that local file; it does
not replace required copyright or notice text.

### Other inference engines

Do not directly copy, adapt, or port source from vLLM or other inference engines.
Use them only to identify technical facts, algorithms, constraints, and evaluation
ideas. Keep inspection notes separate from implementation: the inspection output
documents technical points without source expression, and implementation is based
on that document rather than the upstream code. Separate agents are optional.

Public papers, standards, and original project documentation should be preferred
as the implementation basis when available. The same separation rule applies to
generated code and snippets whose provenance is unclear.

### Vendored dependencies and assets

Before adding anything under `vendor/`, `third_party/`, generated source trees, or
bundled assets, check the license of the exact artifact and revision. Record its
origin, integrity hash, copyright, license files, notices, modifications, and
redistribution requirements independently. A package-level license assumption is
not enough when individual files carry different terms.

## Required record

Use one stable ID per imported upstream source unit. A record may cover several
local files only when they share the same upstream revision, licensing conclusion,
and reuse mode. Keep the record machine-readable inside the corresponding
`THIRD_PARTY_NOTICES.md` entry or in a linked manifest tracked by Git.

```yaml
schema_version: 1
id: llama-cpp-<short-purpose-name>-001
component: <sLLM component name>
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: <40-character full commit SHA>
  sources:
    - path: <upstream/source/path.cpp>
      git_blob: <upstream Git blob object ID>
      url: https://github.com/ggml-org/llama.cpp/blob/<full-commit>/<path>
local:
  files:
    - path: <local/destination/path.cpp>
      imported_sha256: <SHA-256 of this file in the import commit>
copyright:
  - <verbatim applicable copyright notice>
license:
  spdx: MIT
  file: <path to retained upstream license text>
reuse:
  mode: adapted # exact | adapted | ported
  modifications:
    - <specific semantic or structural change>
    - <specific rename, API adaptation, or bug fix>
import:
  commit: <full sLLM commit SHA that first introduced the material>
  reviewed_by: <optional reviewer or approval reference>
  reviewed_at: <optional YYYY-MM-DD>
```

`imported_sha256` is fixed to the bytes introduced by `import.commit`; it is not
updated after ordinary maintenance edits. Subsequent content and provenance are
tracked by Git history. A later import of additional upstream expression, or an
update to a newer upstream revision, receives a new provenance event rather than
overwriting the original import hash.

For an import commit that is not known until the commit is created, use a clearly
marked pending value in the working tree and replace it with the full commit SHA
before release. A provenance-only follow-up commit is not required at each
development checkpoint. Do not leave the pending value in a release.

## AI-assisted code

AI output is not accepted into sLLM merely because it was generated rather than
copied manually. AI origin alone does not add an independent review requirement,
hard gate, or broader verification. At integration, a similarity or provenance
check may be used; if identifiable third-party expression was reproduced,
closely adapted, or ported, apply this policy for release or distribution.
Prompts, model output, or an agent's claim of originality are not provenance
evidence. Any human review is the ordinary review for the active work lane.

## Phase 10 FP8 implementation record

Phase 10のOCP E4M3FN converter、sidecar loader、HIP quantization、hipBLASLt integration、RDNA2 emulationは
AMD/ROCmの公開datatype・API contractと独立数値oracleを基に実装した。llama.cpp、vLLM、その他の推論engineから
source expressionをcopy、adapt、portしておらず、`THIRD_PARTY_NOTICES.md`へ追加する直接importはない。
性能比較では既存の固定llama.cpp binary/resultだけをpeer baselineとして参照した。

## Phase 11 CDNA3 implementation record

Phase 11のE4M3FNUZ codec/converter、hipBLASLt FNUZ integration、wave64 BF16 provider、contiguous-resident KV、
MI300X candidate runnerはAMD/ROCmの公開datatype/API contractとsLLM既存実装を基に作成した。新たな第三者
source expressionのcopy、adapt、portはなく、Phase 9で既に記録したllama.cpp由来MMVF organizationの
wave64化は同じ既存provenance範囲に含まれる。`THIRD_PARTY_NOTICES.md`への新規import追加はない。

## Phase 15 NVFP4 implementation record

Phase 15はNVIDIA Transformer Engine v2.18、annotated tag `62f366a50b8e5a96fac7f123a554ab4db928b2a9`、peeled commit
`27486e03cfc1fa41f6932dcecdc47c71c47eac3e`（BSD-3-Clause）の公開format documentationとrecipe contractを、
E2M1 code point、OCP E4M3FN block scale、FP32 tensor scale、zero/underflow挙動を固定する参照sourceとして使用した。
sLLMのRust converter/oracleとHIP packed-dequant kernelは独立実装であり、Transformer EngineのCUDA kernel、training recipe
source expression、swizzle/RHT実装をcopy、adapt、portしていない。したがって`THIRD_PARTY_NOTICES.md`へ直接importを
追加せず、format-source identityとlicenseをsidecar manifest、量子化contract、Phase historyへ記録する。

## Phase 16F provider artifact implementation record

Phase 16FのUnsloth/NVIDIA/Kimi repositoryはprovider artifactのschema、revision、mixed recipe、model capabilityを固定する
data sourceとして使用し、runtime source expressionをcopy、adapt、portしていない。MXFP4/MXFP8 decoderはOCP Microscaling
Formats v1.0の公開format contractから独立実装し、W4A4/static-FP8 HIP providerはsLLM既存NVFP4/FP8 codeと独立oracleを基に
作成した。vLLM/SGLang sourceは参照・移植しておらず、llama.cppからの新規直接reuseもない。このPhaseによる
`THIRD_PARTY_NOTICES.md`へのcode import追加はない。model artifact自体はGitへ含めず、配布時のmodel license/noticeは
runtime codeのMIT licenseと別に扱う。

## Phase 17 MTP and vision implementation record

Phase 17は固定Qwen3.5 model config/tensor catalog、公開Qwen multimodal processor contract、OpenAIの公開image-content wire
documentationをdata/semantic sourceとして使用した。MTP/vision graph、processor、speculative verifier、HIP executionはsLLM既存の
model-neutral contractと独立oracleから実装した。llama.cppはMTP tensor mappingとvision tensor organizationの技術参照に限り、
新しいsource expressionのcopy/adapt/importはない。vLLM/SGLang sourceは参照・移植していない。このPhaseによる
`THIRD_PARTY_NOTICES.md`へのcode import追加はない。model/image payloadはrepositoryへ含めない。

## Phase 18 exact MTP implementation record

Phase 18のserial-equivalent small-M Matmul、target block transaction、KV/linear-state rewind、generation adapter、performance runnerは
sLLM既存のM=1 kernel、owned execution contract、Phase 17 MTP graphから独立実装した。llama.cpp issue #25618は量子化targetの
出力分岐というdefect classにだけ使用し、llama.cppのMTP source/control flowをcopy、adapt、portしていない。vLLM/SGLang sourceも
参照・移植していない。このPhaseによる新規第三者code importと`THIRD_PARTY_NOTICES.md`追加はない。model、raw logits、KV dump、
profile traceはrepositoryへ含めない。

## Phase 19 Qwen3.5 MoE implementation record

Phase 19は`amd/Qwen3.5-35B-A3B-MXFP4`の公開artifactをtensor schema、mixed-precision recipe、OCP MXFP4 dataの
sourceとして、`Qwen/Qwen3.5-35B-A3B-FP8`をarchitecture/lineage controlとして使用した。router/top-8/shared-expert semantics、
artifact loader、HIP route/expert kernel、full-model integrationはsLLM既存execution contract、OCP Microscaling Formats v1.0、
独立NumPy actual-weight oracleから実装した。llama.cppからの新規直接reuseはなく、vLLM/SGLang sourceを参照・移植していない。
vLLM containerによる同一artifactのblack-box起動はhealth到達前に停止しておりcorrectness evidenceには使用せず、source expressionの
入力にもしていない。このPhaseによる新規第三者code importと`THIRD_PARTY_NOTICES.md`追加はない。model shard、生成token trace、
profile artifactはrepositoryへ含めず、model artifactのlicense/noticeはsLLM runtimeのMIT licenseと別に扱う。

## Phase 55 Gemma 4 MoE implementation record

Phase 55は`google/gemma-4-26B-A4B-it`と`nvidia/Gemma-4-26B-A4B-NVFP4`の固定公開artifactを、architecture、
tensor catalog、chat/template identity、NVFP4 expert recipe、implicit-unit static FP8 KV contractのdata/semantic sourceとして
使用した。公開Transformers実装はrouterの演算順を確認するsemantic referenceに限定し、source expressionやcontrol flowをcopy、
adapt、portしていない。Gemma固有graph/router、HIP expert/attention/KV、GGUF mapping、state/prefix/checkpoint、CLI/server統合は
sLLM既存のmodel-neutral execution contractと独立oracleから実装した。vLLMはartifact提供元が示すreference runtimeだが、local
AMD環境では同一artifactを実行しておらず、vLLM/CUDA/FlashInfer sourceのcopyまたは移植はない。このPhaseによる新規第三者code
importと`THIRD_PARTY_NOTICES.md`追加はない。model shard、生成GGUF、raw trace、profile artifactはrepositoryへ含めず、model
artifactのlicense/noticeはsLLM runtimeのMIT licenseと別に扱う。

## Phase 56 Gemma 4 MTP implementation record

Phase 56はGoogle公式`google/gemma-4-12B-it-assistant`の固定revision、config、tokenizer、safetensors metadata／tensor bytesを
semantic/data sourceとして使用した。target KV mappingとQ-only assistant semanticsは固定reader記録へ分離し、実装はsLLM既存Gemma graph、
model-neutral speculative transaction、owned HIP execution contractから行った。llama.cpp、vLLM、SGLang、Transformers engineのMTP
control flowやkernel sourceをcopy、adapt、portしていない。

公開runtime artifactは公式assistant BF16 tensorをlosslessに格納するcanonical `gemma4mtp` GGUFであり、source safetensorsは変換入力である。
既存targetはreviewed mixed NVFP4 W4A4／FP8 W8A8 GGUFを再利用し、BF16 targetへ変換したとは主張しない。pair lock、reader記録、全48 tensorの
source/GGUF byte照合、exact GPU target-only比較をrelease provenance evidenceとする。このPhaseによる新規第三者code importと
`THIRD_PARTY_NOTICES.md`追加はない。

## Phase 57 DeepSeek V4 foundation implementation record

Phase 57はDeepSeek公式`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`のMIT-licensed config、artifact index／header、tokenizer、generation metadata、
encoding documentation／fixtureをsemantic/data sourceとして使用する。公式Python inference／encoding sourceはidentityと
concept確認のreader-onlyであり、source expression、control flow、kernelをcopy、adapt、portしない。固定llama.cpp
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`もGGUF naming、shape、layer schedule、state境界の概念cross-checkに限り、
本Phaseではcodeを直接reuseしない。

typed config／catalog／capacity、mHC、CSA／HCA、Lightning Indexer、hash／score MoE、mixed FP8／MXFP4、GGUF contractは
sLLM既存のmodel-neutral execution contractと独立oracleから実装する。公式48 shardのLFS identityは固定するが、全166.9 GBを
local取得していない段階でfull-byte SHA-256 verified、full-model resident、generation correctnessとは主張しない。
DSparkをDFlashへ読み替えず、model-free／tiny-random／verified slice証拠はそのscopeを明記する。このPhaseによる新規第三者code
importと`THIRD_PARTY_NOTICES.md`追加はない。model shard、GGUF、raw trace、profile artifactはrepositoryへ含めない。

## Phase 58 MiniMax M3 foundation implementation record

Phase 58はMiniMax公式`MiniMaxAI/MiniMax-M3` revision
`f0e1c1e04d40177e4673a22097036854f536e9c0`のCommunity-Licensed config、artifact index／header、tokenizer、
generation／processor metadataと、公式MSA paper／repositoryをsemantic/data sourceとして使用する。公式MSA実装は
NVIDIA SM100／CUDA向けoperator境界のconcept確認に限り、source expression、control flow、kernelをcopy、adapt、portしない。
固定llama.cpp `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`もcanonical GGUF naming、shape、layer schedule、
released MTP tensor absenceのcross-checkに限定し、本Phaseではcodeを直接reuseしない。

typed config／catalog／capacity、manifest mismatch、MSA block selection／causal attention、sigmoid MoE、GGUF contractは
sLLM既存のmodel-neutral execution contractと独立oracleから実装する。公式59 shardのLFS identityとbounded headerを固定するが、
全854 GB payloadをlocal取得していない段階でfull-byte SHA-256 verified、full-model resident、generation correctnessとは主張しない。
configが示す7 MTP moduleに対応するtensorがreleased indexにないため、未公開weightを推測で補わない。model-free／tiny-random／
verified slice証拠はそのscopeを明記する。このPhaseによる新規第三者code importと`THIRD_PARTY_NOTICES.md`追加はない。
model artifactのMiniMax Community LicenseはsLLM runtimeのMIT licenseと別に扱い、model shard、GGUF、raw trace、profile artifactを
repositoryへ含めない。

## Phase 59 DiffusionGemma foundation implementation record

Phase 59はGoogle公式`google/diffusiongemma-26B-A4B-it` revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`のApache-2.0 config、index／bounded header、tokenizer、processor、generation、
scheduler metadataをartifact sourceとして使用した。意味論はGoogle DeepMind Gemma repository commit
`7b785991bd78626c73b317eb43fdbb6c292f7b9c`、Transformers commit
`42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`、Diffusers commit
`c1bf18c92c6285334adcaac7e75ef8946a227f49`をreader-onlyで照合した。source expression、control flow、test、kernelは
copy、adapt、portしていない。

fixed llama.cpp `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`にはDiffusionGemmaのmerged GGUF architectureがない。

Phase 60 Ministral 3はMistral公式BF16 source revision
`b6d637bef2393152b3da2b2fde72eecdee30557e`と公式GGUF revision
`eb599d408350ea2bb60452cb86be7c7b2fc28227`をartifact identityとして直接固定する。Transformers commit
`3e9d3e50e71442a3173bdf01cd45ba5833533efe`はYaRN／Q-only position scale／stage orderのsemantic照合、fixed
llama.cpp `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`はcanonical `mistral3` GGUF metadata／tensor naming／QK layoutの
相互運用cross-checkだけに使い、source expressionをcopy／portしない。詳細は
[Phase 60 reader](../references/ministral3-phase60-reader.md)を正とする。
`diffusion-gemma`はwrite-disabled sLLM foundation keyとして分離し、open proposalのspellingをapproved upstream standardへ
読み替えない。typed config／catalog／capacity、causal encoder、read-only encoder KVを参照するbidirectional decoder、
self-conditioning、entropy-bound sampler、adaptive stop、GGUF dry-runは独立oracleとして実装した。

11 shardのHub LFS identityとbounded headerだけを固定し、51.6 GB payload全体をlocal取得していないためfull-byte SHA-256 verified、
derived GGUF、single-GPU resident、generation correctnessとは主張しない。このPhaseによる第三者code importと
`THIRD_PARTY_NOTICES.md`追加はなく、model shard、生成GGUF、raw trace、profile artifactをrepositoryへ含めない。

## Phase X llama.cpp Qwen3.8 performance investigation record

Phase Xはllama.cpp build 901 commit `4df29be4f4c3673f428170fda944a5b19f743bb8`を外部local-subagent runtimeと
技術比較対象にし、Qwen3.8-27B/Qwen3.5 architectureのHIP性能低下をprofileした。採用変更はllama.cppの既存CMake option
`GGML_CUDA_FA_ALL_QUANTS=ON`によるQ5_1 Flash Attention build coverageであり、sLLM sourceへllama.cppのcode expression、
control flow、kernelをcopy、adapt、portしていない。sLLMの`linear_attention.gdn.v1`も変更していないため、既存
`llama-cpp-phase9-gdn-layout-001`を更新せず、新しいimport eventまたは`THIRD_PARTY_NOTICES.md` entryを追加しない。
local llama.cpp exact-shape test patchはupstream未投稿でsLLM repositoryへ含めず、model、raw trace、生成全文、binaryも含めない。

## Phase 20 A0 GGUF source inspection record

P20-A0はsource-lock済みllama.cpp `b10453` commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`の
`gguf.h`、`gguf.cpp`、Python reader/writer/constants/quants、Qwen3.5/Gemma 4 converterをGGUF v3 format、標準metadata、
tensor block、converter mappingの技術参照として読んだ。8 fileのSHA-256と用途は
[P20-A0 manifest](../../ci/matrix/phase20-gguf-a0-v1.json)へ固定した。本作業でllama.cppのcode expression、control flow、
test、generated artifactをsLLM sourceへcopy、adapt、portしていないため、import eventと`THIRD_PARTY_NOTICES.md` entryは
追加しない。後続Phaseで直接reuseする場合は、実際のimport前に本書の通常手順で別recordを追加する。

P20-A1〜A6もこの境界を維持し、reader、writer、converter、block repack、testsをsLLM内で実装した。
llama.cppからのcode expression、control flow、test vectorのimport/adaptationはなく、Phase 20 closeout時点でも
import eventと`THIRD_PARTY_NOTICES.md` entryは不要である。

## Phase 23 performance discovery record

Phase 23はllama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`の既存immutable Phase 5 resultを
system-level performance peerとして再利用した。vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0`とSGLang commit
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`はscheduler構造のfacts-only inspectionに限定した。
[technical-difference note](../references/phase23-inference-engine-performance-differential.md)をimplementation案から分離し、
source expression、control flow、testをcopy、adapt、portしていない。Phase 23はproduction sourceを変更せず、新規第三者code
importと`THIRD_PARTY_NOTICES.md` entryはない。model、binary、raw trace、生成全文はrepositoryへ含めない。

## Phase 24 terminal-row implementation record

Phase 24のlast-row view、row-policy contract、physical terminal allocation、distinctive-row GPU oracle、host tests、dual-GPU計測は
sLLMの既存Qwen graph/runtime contractとPhase 23の内部trace observationから独立に作成した。外部engineのsource expression、
control flow、kernel、testをcopy、adapt、portしていない。改訂後の採用条件を満たしたshared candidateはproduction sourceへ保持したが、
新規第三者code import、import event、`THIRD_PARTY_NOTICES.md` entryはない。model、candidate binary、raw result、rocprof trace、
生成全文はrepositoryへ含めず、追跡するbounded summaryにはaggregateとSHA-256だけを記録する。

## Phase 27 llama.cpp weight-stream comparison record

Phase 27はllama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`（tree
`e9b6173953477054a4068884aa5fc9aeef6475e8`）の`ggml/src/ggml-cuda/mmvf.cu`、`common.cuh`、
`ggml-cuda.cu`と`src/models/qwen35.cpp`を、BF16 matvec provider、gate/up/GLU fusion、kernel dispatchのfacts-only性能比較に使用した。
sLLM sourceへcode expression、control flow、kernel、testをcopy、adapt、portせず、Phase 27はproduction sourceを変更していない。
したがって新規import eventと`THIRD_PARTY_NOTICES.md` entryはない。llama.cpp build、model、raw trace DB、生成全文はrepositoryへ
含めず、tracked summaryにはrevision、aggregate、digestとE1比較限界だけを記録する。

## Phase 30 attention provider comparison record

Phase 30はrepository内のllama.cpp `ggml/src/ggml-cuda/fattn-mma-f16.cuh`と`fattn-tile.cuh`を、RDNA向け
FlashAttention tiling、mask、online-softmax、matrix-provider dependency surfaceの技術比較として読んだ。rocWMMA installed
headersもprovider feasibilityの確認に使用した。既存sLLMのopaque KV layoutとN0/N1範囲へ直接取り込めるbounded kernelではないと判断し、
llama.cpp/rocWMMAのsource expression、control flow、kernel、testをcopy、adapt、portしていない。採用したnative FP8 readと
wave32 reductionはsLLM既存kernel、公開AMD compiler builtin、独立256-code oracleから実装した。したがって新規import eventと
`THIRD_PARTY_NOTICES.md` entryはない。model、binary、raw trace、生成全文はrepositoryへ含めない。

## Phase 31 chunked prefill implementation record

Phase 31はSGLangの公開運用で見られる階層的chunk-sizeという概念だけをproduct policyの参考にし、512/2K/4K/8K/16Kの
resource-based selector、Qwen chunk orchestration、completion-boundary liveness planner、CLI/server KV encoding設定を
sLLM既存execution/KV contractから独立実装した。SGLang、vLLM、llama.cppその他のsource expression、control flow、kernel、testを
copy、adapt、portしていない。従って新規import eventと`THIRD_PARTY_NOTICES.md` entryはない。model、binary、raw trace、
生成全文はrepositoryへ含めず、bounded aggregateとidentityだけを追跡する。

## Phase 35 Full Attention and GDN implementation record

Phase 35のFull Attention Q_TILE=4 providerはsLLM既存Phase 33 kernel、opaque KV contract、独立scalar oracleを基に実装し、
新しい外部source expressionをcopy、adapt、portしていない。GDN column-state providerはllama.cpp commit
`f5919bf458ef190468b5c329bb293f8a54a1e69c`の`ggml/src/ggml-cuda/gated_delta_net.cu`にあるcolumn ownership、
register state shard、wave reductionの近接構造をbounded adaptationしたため、新規notice
`llama-cpp-phase35-gdn-column-state-001`へ記録した。既存`llama-cpp-phase9-gdn-layout-001`のlayout-only eventは上書きしない。
sLLMのBF16/FP32 round stage、exact-target state mapping、transaction、runtime、short/decode complementへ合わせて変更し、
ggml tensor/runtimeやgeneric CUDA dispatchは移植していない。import commitは
`bca482251bd21b144d950956af39a769c4211417`、導入時local file SHA-256は
`cf8e8aafa5e7e64c8fe5bc082912b5b8a328d0a9ed407965d6782cad72b3bc4a`へ固定した。model、binary、raw trace、生成全文はrepositoryへ含めない。

## Phase 66 reusable low-precision provider record

Phase 66はPhase 65で固定したllama.cpp、SGLang、vLLM、LMDeploy、KTransformers、TensorRT-LLM比較から、
target／format／shape別provider、consumer向けactivation layout、複数tile familyというfacts-only設計点だけを使用した。
Q8/Q8_1の式、packed layout、tile table、kernel／dispatch source、symbolは移植していない。MXFP8 ID37、prepared provider、
typed attention、MXFP6／NVFP4／MXFP4 routingはsLLM既存実装、OCP format contract、独立oracleから実装し、
NVFP4／MXFP4 W4A4は既存sLLM device kernelをfrozen providerへ接続した。

第三者source expressionのcopy、adapt、portはなく、新規import eventと`THIRD_PARTY_NOTICES.md` entryはない。詳細なreference
identityと境界は[Phase 65/66 inference-engine comparison](phase65-inference-engine-comparison.md)を正とする。

`Co-Authored-By` and similar commit trailers record development participation.
They are not evidence of copyright ownership, license compatibility, assignment,
or authority to grant legal rights.

## Updating or removing imported material

An upstream update is a new provenance event: inspect the new revision and add its
imported hashes, modifications, notices, and import commit without rewriting the
earlier event. Ordinary downstream maintenance is represented by Git history.
When imported code is removed, retain enough history in the notice entry to
explain past distributed versions; mark the entry as removed and identify the
removal commit instead of erasing the record.
