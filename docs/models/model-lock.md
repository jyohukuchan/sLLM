# Model artifact lock format

Model identifiers such as a Hugging Face repository name or branch are mutable.
sLLM therefore resolves every model input to an immutable revision and verifies
every downloaded byte before loading it. The lock file, not a cache directory or
floating alias, is the record of what was used.

Official references:

- [Hugging Face Hub repositories](https://huggingface.co/docs/huggingface_hub/guides/repository)
- [`hf_hub_download`](https://huggingface.co/docs/huggingface_hub/guides/download#download-a-single-file)
- [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository)
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)
- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [Safetensors format](https://huggingface.co/docs/safetensors/index)

## Required lock contents

Each lock records:

- `repo_id` and `repo_type`;
- the user-supplied `requested_revision` and the resolved 40-character Git commit
  SHA;
- every file consumed by loading, tokenization, prompting, or inference, with its
  path, byte size, SHA-256, repository Git blob ID, source page, immutable download
  locator, and Git LFS object ID when stored through LFS;
- evidence files used for the model card, license, and base-model declarations,
  with the same size, hash, blob, download, and LFS metadata as runtime files;
- the license identifier or exact upstream statement and the evidence path;
- declared base models and their revision information when available;
- complete derivation metadata for converted, quantized, merged, or generated
  artifacts;
- a lock schema version; and
- a deterministic lock fingerprint used by runtime model aliases.

“Every file” includes, as applicable, model configuration, generation
configuration, tokenizer model/vocabulary/configuration, added and special token
maps, chat templates, custom modeling/configuration code, safetensors index files,
and all referenced weight shards. Files discovered indirectly through an index or
configuration are still inputs and must be locked. Do not use unrecorded files
from an existing local cache.

README files, model cards, license files, and metadata used to reach a licensing
or base-model conclusion belong in `evidence_files`; a URL alone is insufficient.
If an evidence file is also a runtime input, record it once in `files` and refer
to its path from the evidence field rather than duplicating it.

The lock records provenance and integrity; it does not grant rights to download
or redistribute a model. License terms, gated access conditions, and model-card
restrictions must be reviewed separately.

## Source pages and downloads

A Hugging Face `/blob/<revision>/<path>` URL is a human-readable source page, not
a download locator. Record it as `source_page_url`. Obtain bytes with
`hf_hub_download` using `revision: <resolved full SHA>`, or with the corresponding
immutable `/resolve/<resolved-full-SHA>/<path>` URL recorded as `download_url`.
Never download locked bytes using the requested branch or tag after resolution.

For an LFS file, `git_blob` identifies the repository's pointer blob and `lfs_oid`
identifies the LFS content object. Neither replaces the SHA-256 computed over the
actual bytes sLLM consumes.

## Resolution procedure

1. Start with an explicit Hugging Face `repo_id`, `repo_type`, and requested
   revision. Never treat an omitted revision as immutable.
2. Resolve the requested revision through the Hub to one full commit SHA before
   selecting files.
3. Fetch repository metadata, the model card, and applicable license evidence at
   that resolved SHA. Record declarations without guessing when metadata is
   missing.
4. Determine the complete transitive file set: parse index and configuration
   files, include all referenced shards and tokenizer/chat-template inputs, and
   reject unexpected remote code unless it was explicitly reviewed and locked.
5. Download only from the resolved revision. Compute SHA-256 and byte size from
   the downloaded content and record the Git blob and LFS OID where applicable.
6. Record derivation metadata whenever locked outputs were produced from other
   locked artifacts.
7. Compute the fingerprint as defined below, then bind runtime aliases to it. On
   startup, reject a duplicate alias, fingerprint mismatch, missing file, or
   content mismatch.

## Verified cache and model slices

The lock is integrity metadata, not a storage container. Full model bytes belong
in a cache outside the repository checkout. Before any model-bound execution,
verify the cache path, complete resolved revision, every byte size and SHA-256,
and every LFS identity against the lock. Treat a cache miss, extra or missing
runtime input, writable trusted-cache mount, or content mismatch as a non-PASS
infrastructure or validation result; never download a replacement during an
offline test run.

A model slice is a temporary local artifact extracted at execution time from a
verified read-only cache. Do not commit or upload raw slices. A reproducible
slice record must contain the source lock fingerprint, exact tensor name, shard,
safetensors byte range and logical shape/dtype, extractor repository and
40-character commit SHA, script path and SHA-256, ordered arguments, relevant
execution environment, and output size/SHA-256. The expected numerical result is
computed from the verified extracted bytes and an independent input fixture; a
slice hash or cache path never substitutes for the source model lock.

Qwen3.5-2B/9BのPhase 4で固定したraw非保存のrange recipeとhashは
[Qwen3.5 Phase 4 real-weight slice identities](qwen3.5-phase4-slices.md)を正とする。

Phase 14の`model-lock-v2`はreviewed `google/gemma-4-12B` direct-safetensors sourceだけに限定する。単一
`model.safetensors`の8-byte header length、完全header hash、derived tensor catalog hash、全tensorの
name/shape/dtype/absolute rangeを固定し、index fileが存在するかのように補わない。base sourceにはchat templateがないため
`chat_template_path: null`と`prompt_mode: raw-text-only`を固定し、instruction-tuned siblingのtemplateを混在させない。
text 666 tensorだけをloadableとし、locked audio/vision 11 tensorはknown-unconsumedとして明示する。runtimeは全locked fileを
full hash検証した後でのみweight planを作り、23.8 GBをrepository、slice、生成artifactへ追加しない。

## Derived artifacts

For each converted, quantized, merged, or otherwise generated artifact, record:

- every source model-lock fingerprint;
- the tool repository URL and full commit SHA;
- the exact ordered arguments and complete effective configuration;
- the relevant execution environment, including platform, toolchain/runtime,
  dependency lock or versions, and accelerator details when they can affect the
  output; and
- every output path, byte size, and SHA-256.

Phase 10のQwen3.5 FP8 sidecarは、tracked model lockを変更せずに上記契約を実行時検査する
`sllm-fp8-sidecar-v1` manifestを使う。manifestはsource lock fingerprintと内容hash、converterのrepository
commit/script hash、ordered arguments、Python/host環境、完全artifact hash、各tensorのsource range/hash、
shape、OCP E4M3FN value range、per-output-row FP32 scale range/hashを含む。runtimeはmanifest、source lock、
artifact全体、全rangeをfail-closedに照合し、cache外の派生artifactをrepositoryへ追加しない。

このsidecarはtext-linear 248 tensorだけを量子化し、非linear tensorはverified BF16 source cacheを使用する。
model alias/fingerprintはsource lockとsidecar manifest identityの双方へ結び付ける。OCP E4M3FN byteをFNUZとして
再解釈してはならない。Phase 11のgfx942 pathは、同じverified sidecarをload時に数値変換したE4M3FNUZ resident
representationとして監査し、source model lockや派生artifact自体は変更しない。

Phase 15のweight NVFP4は`sllm-nvfp4-sidecar-v1`を使う。manifestはverified BF16 source lock、Transformer Engine
v2.18/commit `27486e03cfc1fa41f6932dcecdc47c71c47eac3e`、converter script hashとNumPy version、tensor選択、
artifact全体と各source/value/block-scale/tensor-scale hashを固定する。各recordはrank-2 logical shape、low-nibble-first
E2M1、K-axis block 16、OCP E4M3FN scale、FP32 tensor scaleを必須とし、missing/extra/range/shape/hash/provider不一致を
load前に拒否する。Qwen3.5-2Bのrepository外sidecarは186 tensor、772,236,184 byteで二回のartifact SHA-256が一致した。

Phase 15OのFP8/NVFP4最適化は実行providerだけを変更し、上記sidecar bytes、manifest、source lock fingerprint、
scale/packing規則を変更しない。decode/prefill provider IDとkernel symbolはdispatch auditへ含めるが、派生artifact identityを
置き換える情報ではない。runtimeは同じverified sidecarをloadし、requestごとの全weight展開を行わない。

Phase 15Qは`google/gemma-4-12B-it` revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`を
`model-lock-v2` fingerprint `sha256:381c94bcb48a26d8ef83d1c3d7c5a3513ef8fac4a638752731b85c119385f09d`で追加した。
7 file、model artifact SHA-256 `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`、complete
safetensors header SHA-256 `e432b3ee11ff7f7d179ccbf3827af9669c03a0a28e603000d89c6e1b6c9d4bb7`、tokenizer SHA-256、
chat template、stop IDsをexactに検証する。base model lockとのrepo/revision/fingerprintのORだけを許し、未知のGemma identityへ
一般化しない。

Unsloth import manifestも同じ`sllm-nvfp4-sidecar-v1`を使い、source BF16 lockに加えてquantized repository/revision、完全artifact
SHA-256、importer script hash、`weight_global_scale`のreciprocal変換、`input_global_scale`未適用を固定する。full primary artifactは
MLP 144 tensorを必須とする。1〜144 tensorのsubsetはlayer感度診断だけに使え、manifestの`gemma-mlp-subset`選択とsidecar
fingerprintが一致しないrequestへの再bindを拒否する。

Phase 17では既存Qwen3.5-4B lockを変更せず、locked catalogからMTP 15 tensorとvision 297 tensorをcomponent-enabled
manifestへ昇格する。manifestはexact repo/revision/model fingerprint、全tensor name/shape/dtype/source range、shared embedding、
special token、processor geometryを検査する。text-only weight planは従来どおりこれらをloadせず、MTP/visionを選択した経路だけが
required componentとして消費する。画像bytes、decoded raster、patch列、projected embeddingはderived model artifactではなく
request-local dataであり、lockやrepositoryへ保存しない。

Phase 19のprimary lockは`amd/Qwen3.5-35B-A3B-MXFP4` revision
`2e19c6576db91e5d5a93455415619262218bf8a1`であり、architecture/lineage controlは
`Qwen/Qwen3.5-35B-A3B-FP8` revision `9d1823d2dee688a6b25e77009dc727688c44936e`である。
text-only inventoryは62,053 tensor、22,009,481,856 source byte、model fingerprintは
`sha256:5bca203f6ec8ab9cab4e340a6c337fff7387f9ca2fa12526c48ce999748e83b0`とする。loaderはconfig、index、全shard、
9個のsupport file、license、tensor name/shape/dtype/source range、expert ID 0..255、projection/value/scale planeをexactに検査し、
missing/extra/duplicate/range/hash不一致をload前に拒否する。lowered execution planは493 entry、digest
`sha256:f96a3389cfaca4ab947fe060ccd6f048d078946e704464277d87019a13fb7ae4`である。
検証済みshardはhash確認後も同じopen file descriptorへ固定し、upload時のpositional readでpathを再openしない。
config、index、support fileも同じdescriptorから上限付きで読み、読み込み前後のdevice/inode/size/mtime/ctimeとpath bindingが
変化した場合は拒否する。

Phase 55のGemma 4 MoE semantic sourceは`google/gemma-4-26B-A4B-it` revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`、primary quantized artifactは
`nvidia/Gemma-4-26B-A4B-NVFP4` revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`である。
loaderは2 shard、11 support identity、全47,033 source tensor、128 expert × 30 layer、NVFP4 value/block-scale/
outer-scale/input-scale plane、implicit-unit static E4M3 KV recipeをallocation前に検証する。canonical GGUF semantic identityは
`gemma4moe:<reviewed fingerprint>`であり、35,513 tensorと17,636,771,900-byte resident planを保持する。
source snapshot、GGUF container、resident plan、state image/checkpointは別digestで結び、近似architecture名、未知source revision、
欠落scale、保存されていないper-layer KV scaleを受け入れない。

このlockはsource container固有の検査結果と、container-neutralなMoE config、expert-axis inventory、mixed recipe、verified load plan、
tokenizer/chat metadataを分離する。Phase 20のGGUFは後者と同じsemantic identityを保持し、GGUF化を理由に別modelとして扱ったり、
source safetensorsと量子化sidecarを最終ユーザー形式として残したりしない。model shardや生成GGUF自体はrepositoryへ含めない。

Phase 20では`derived-gguf-lock-v1`を実装した。各outputは全source lock fingerprint、converter repository/full commit、
完全な引数とeffective config、environment、GGUFのsize/SHA-256、metadata digest、tensor catalog digestを含める。
container digestは変わるがsemantic identityは維持し、runtimeは検証済みGGUFを同じopen file descriptorから読む。
GGUF tensor tableはstandard readerに合わせてrank 4以下とし、rank 5以上のsource logical shapeはdigest対象のversioned
`sllm.tensor_recipe.logical_shapes`へ一対一で固定する。物理shapeとlogical shapeのelement count不一致はload前に拒否する。
format、encoding、handoffの正本は[GGUF format contract](../formats/gguf.md)と
[P20-A0 manifest](../../ci/matrix/phase20-gguf-a0-v1.json)とする。公開runtimeはderived lockのsource fingerprintをbuild内の
reviewed lockへ解決するため、変換元lockを別のユーザー入力として要求しない。

Phase 46の変換toolはこのderived lockを`model.gguf`と同じatomic directory transactionで公開し、さらに
`sllm-phase46-tool-run-v1`の`run-manifest.json`を同梱する。共通manifestはsource lock、使用file、converterの
40-hex commit、実行binary SHA-256、完全な引数、recipe、GGUFとderived lockのsize/SHA-256を結合する。
`run-manifest.json`自身は循環digestを避けるためoutputsへ含めず、bundle全体の存在をpublication boundaryとする。
非dry-run変換は`--output-bundle`だけを許可する。別々の`--output`/`--derived-lock`はGGUFとlockを
一つのatomic transactionとしてpublishできないため拒否する。
split/merge、LoRA、repack、quantize、imatrixも同じ共通identityを持つが、変換成功だけでruntime supportや
品質defaultを宣言しない。詳細は[Phase 46 tools](../development/phase46-tools.md)を正本とする。

`model-lock-v1` only represents original upstream snapshots and therefore
requires `derivation: null`. The requirements below define the information that
a future derived-artifact schema must preserve; they are not accepted fields in
`model-lock-v1`. A later schema must explicitly define nullable repository
identities for unpublished local outputs and bind every derivation output to its
corresponding file record. Do not encode a derived artifact by weakening or
overloading `model-lock-v1`.

```yaml
derivation:
  source_lock_fingerprints:
    - sha256:<source model-lock fingerprint>
  tool:
    repository: https://example.org/owner/conversion-tool
    commit: <40-character full commit SHA>
  arguments:
    - --input
    - <source path>
    - --output
    - <output path>
  config:
    quantization: <complete effective configuration, including defaults>
  environment:
    platform: <OS and architecture>
    runtime: <compiler or interpreter and version>
    dependencies: <lock-file hash or complete relevant versions>
    accelerator: <hardware and driver/runtime versions, or null>
  outputs:
    - path: <output path also present in files>
      size_bytes: <integer>
      sha256: <same SHA-256 as the corresponding file record>
```

## Fingerprint

The fingerprint input object is exactly the JSON object containing the root
`schema_version` member and the complete root `model` member. It therefore
includes the schema version, all runtime and evidence file records, license and
base-model evidence, and derivation metadata. It excludes `fingerprint`,
`aliases`, `generated_at`, and any other root-level bookkeeping metadata.

Serialize that input object with RFC 8785 JSON Canonicalization Scheme (JCS), hash
the resulting UTF-8 bytes with SHA-256, and encode it as
`sha256:<64-lowercase-hex>`. Do not hash YAML text, ordinary JSON serialization,
or an implementation-dependent map order.

Changing any member of the fingerprint input changes the fingerprint. An alias
may move to a new fingerprint only through an explicit lock-file change; aliases
must never resolve a floating Hub branch at runtime.

## YAML authoring excerpt

YAML is shown only as a non-validating authoring excerpt. The normative v1 shape
is `ci/schema/model-lock-v1.schema.json`, and the checked-in Qwen lock is the
complete example. The fingerprint is computed from the JCS canonical JSON
representation of `{ "schema_version": ..., "model": ... }` defined above.

```yaml
schema_version: model-lock-v1
model:
  repo_id: Qwen/Qwen3.5-4B
  repo_type: model
  requested_revision: <tag, branch, or commit supplied by the maintainer>
  resolved_revision: <40-character full commit SHA>
  license:
    id: <SPDX identifier or null>
    statement: <exact upstream statement when no SPDX identifier is available>
    evidence_paths:
      - README.md
      - LICENSE
  base_models:
    - repo_id: <namespace/base-model>
      revision: <full commit SHA when declared and resolved, otherwise null>
      evidence_path: README.md
  evidence_files:
    - LICENSE
    - README.md
  files:
    - path: LICENSE
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/LICENSE
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/LICENSE
      lfs_oid: null
    - path: README.md
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/README.md
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/README.md
      lfs_oid: null
    - path: config.json
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/config.json
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/config.json
      lfs_oid: null
    - path: model.safetensors.index.json
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/model.safetensors.index.json
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/model.safetensors.index.json
      lfs_oid: null
    - path: model.safetensors-00001-of-00002.safetensors
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID for the LFS pointer>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/model.safetensors-00001-of-00002.safetensors
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/model.safetensors-00001-of-00002.safetensors
      lfs_oid: sha256:<64 lowercase hexadecimal characters>
    - path: tokenizer.json
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/tokenizer.json
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/tokenizer.json
      lfs_oid: null
    - path: tokenizer_config.json
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/tokenizer_config.json
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/tokenizer_config.json
      lfs_oid: null
  derivation: null
fingerprint: sha256:<RFC-8785-JCS hash of schema_version and model>
aliases:
  - qwen3.5-4b-bf16
generated_at: <RFC-3339 timestamp; excluded from fingerprint>
```

The excerpt omits required architecture, tensor, slice, tokenizer, excluded-file,
and other v1 fields. The actual lock must validate against the schema and enumerate
all files used by the selected revision, including every shard named by its index.
Derived artifacts require a future schema version; do not replace `derivation:
null` inside `model-lock-v1`.

## Provider low-bit artifact lock

Phase 16Fの提供元low-bit checkpointは、architecture/tokenizer contractを既存reviewed lockから継承しつつ、provider artifact
固有のrepository、full revision、全使用file size/SHA-256、container header、tensor inventory、mixed recipe digest、topology
plan digestを追加lockで固定する。primary lockは
[`unsloth-gemma4-12b-it-nvfp4.json`](locks/unsloth-gemma4-12b-it-nvfp4.json)である。

このlockはBF16から生成したsidecarのderivationではなく、提供元が公開した独立artifact identityである。runtime importerは
load前に全identityとselectorを検証し、architecture lockとlogical topologyが一致する場合だけcontainer-neutral descriptorへ
lowerする。NVIDIA 31BとKimi K3のmetadata lockはbounded schema/referenceまたはarchitecture handoff用であり、full payloadを
検証していない状態をfull-model supportへ昇格しない。

## Phase 41 derived state identity

Prefix entryとsession checkpointはpath、alias、cache directoryをmodel identityとして使用しない。canonical identityは少なくとも
reviewed model-lock fingerprint、derived artifact/recipeまたはweight-plan digest、adapter identity、renderer/template digest、
tokenizer fingerprint、exact token digest、KV encodingとdescriptor/layout digest、target semantics、context-policy digestを含む。
いずれかが異なるstateはmissまたは明示rejectとし、別model/adapter/template/encoding/targetへsilent reuseしない。

Qwen/Gemma checkpointは保存された全state planeのdescriptor metadataとpayload checksumを検証してからfresh request ownerへimportする。
同じmodel familyでもcapacity、static FP8 scale、full/sliding topology、linear/GDN layoutが異なる場合は同一identityではない。
checkpoint filenameとdirectoryはlookup locatorにすぎず、identity比較を代替しない。Phase 41 productionのstateless prompt checkpointは
token historyのnonempty suffix continuationだけを許し、mid-generation session identityやwire session IDをmodel lockへ暗黙追加しない。

## Phase 42 frontend and inference capability identity

Phase 42のpublic endpointは、model aliasだけでfrontend capabilityを推測
しない。lockまたはderived verified manifestは、少なくとも次のcapability
identityを提供し、欠落・不一致をload/request前にfail closedする。

The profile reference identity is OpenAI OpenAPI `2.3.0`, commit
`117ce5680e4269f6656a4fd70d28f9755630d938`, and the technical llama.cpp
reference is `b10453`, commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. These are API/adapter pins, not
model bytes and not evidence that a model supports every referenced endpoint.

- tokenizer identity: tokenizer filesのfingerprint、special-token IDs、byte
  fallback policy、normalization、utility version、最大input bytes/tokens;
- renderer identity: verified template kind/version、template digest/size、
  supported roles、assistant-prefill policy。任意Jinjaやcustom kwargsは
  capabilityにならない;
- embedding identity: final-hidden-row source、pooling `arithmetic-mean`、
  L2 normalization、F64 accumulator/F32 output、model-lock hidden dimension;
- rerank identity: cosine embedding profile、query/document dimension、
  higher-score ordering、stable original-index tie break、bounded `top_n`;
- FIM identity: prefix/suffix/middle token IDs、verified template digest、
  context limit、extra-context policy、production status。未検証FIMは
  `unsupported`であり、generic completion fallbackを許可しない;
- target/provider identity: exact target semantics、weight/KV encoding、
  provider and plan digest. MI300X `gfx942` capability is deferred until a
  fresh exact-runtime evidence row exists; compile-only evidence cannot be
  recorded as a model capability.

The Phase 42 fixture and schema pin these semantics in
[`phase42_profiles_v1.json`](../../tests/fixtures/phase42_profiles_v1.json) and
[`phase42-profile-v1.schema.json`](../../ci/schema/phase42-profile-v1.schema.json).
The fixture is an API boundary/identity artifact, not a model lock and not a
claim that a model has passed production inference. A lock may advertise a
Phase 42 capability only after the corresponding verified tokenizer/template,
model execution path, numerical oracle, and exact target evidence are all
bound to the same immutable identity.

## Phase 44 template・reasoning・interactive identity

Phase 44 generic templates are request-level verified providers, not floating model-lock defaults. A custom source is admitted only after its
regular-file bytes, UTF-8 validity, bounded size, and lowercase `sha256:<64 hex>` digest are checked. The generic renderer profile is
`generic-jinja-v1` with exact MiniJinja `2.24.0`; its identity includes profile version, template/source digest and size, canonical kwargs digest,
and rendered bytes digest/size. A reviewed Qwen renderer remains the model-lock default, while current Gemma raw-text capability is not silently
replaced by generic messages.

Prefix/cache/checkpoint identity must include the exact renderer/template identity and tokenizer fingerprint in addition to the Phase 41 model,
plan, target, KV/layout, and token identity. Generic and reviewed-template state therefore never share a cache entry merely because the model alias
matches. Reports expose only bounded identity values; local template paths, source prompt, kwargs payload, and checkpoint locations are not identity
or error fields.

Reasoning mode and budget are frontend generation semantics, not a model-family claim. The checked controller uses existing selector/grammar/stop/
cancellation ownership, admits 1..=4,096 reasoning tokens including any forced close sequence, and preserves token history/usage accounting. A lock may
advertise a reasoning capability only when its reviewed template/markers and closing-token identity are verified; absent capability, Gemma/raw-text and
unsupported backends reject before tokenizer/GPU work. The Phase 44 `chat` CLI stores typed conversation through the Phase 41 opaque checkpoint
boundary; it does not add a new model-lock state plane or a mid-generation session identity. MI300X `gfx942` runtime evidence remains deferred.

## Phase 45 adapter・dynamic lifecycle identity

The `sllm-model-manifest-v1` is an offline, strict input to model admission. Manifest paths must resolve to absolute regular files opened with
no-symlink semantics; bounded size, file digest, model-lock fingerprint, and derived-artifact source/recipe/output digest are checked before any
backend or GPU allocation is published. Network URLs, downloads, unverified cache entries, and request-supplied paths or credentials are rejected.
The manifest's alias is a routing label, not part of model identity.

The derived lock for a Phase 45 adapter/control binding records the immutable base model fingerprint, derived plan digest, artifact digest, target
tensor catalog and shape, dtype, LoRA rank/orientation or control-vector half-open layer range, and canonical ordered scales. The production catalog
identity is checked both before backend open and against the loaded resident owner; a path or cache-directory change cannot create a new identity or
permit silent cross-identity reuse. Disabled adapters use `adapter:none-v1` and retain base prefix/checkpoint/logit/token identity.

Prefix and checkpoint keys bind the ordered adapter/control artifact IDs and scales together with target semantics, renderer, and tokenizer identity.
Admin lifecycle actions are alias-only (`load`, `preload`, `unload`, `clear-quarantine`, `evict-idle`) and do not alter the lock. The profile/schema
fixture records the host contract and rejection matrix. Exact RDNA `gfx1030`/`gfx1201` release-build full-model smoke and BroadcastAdd standalone
evidence pass with bitwise two-run repeatability, HIP-only dispatch, fallback false, resident/request-workspace baseline restoration, and zero
pre/final allocations; the compact [GPU summary](../../ci/matrix/phase45-adapter-lifecycle-gpu-summary-v1.json) records bounded identity prefixes and
dispatch counts without tracking raw artifacts. `gfx942`/MI300X runtime remains deferred.

## Phase 53 KV encoding selection identity

Phase 53のKV default判定はsource model lockやderived artifact lockを変更しない。固定Qwen3.5-4B BF16 model lock fingerprint
`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、model lock SHA-256
`4071e1b36901e523a3c5c65559f2cecda7c9cc258185770f049886f52d1fe678`、derived lock fingerprint
`d553db4d10df5655b681b067ac0e8359defe85ab384e805c97f8a296854b4c12`、derived lock SHA-256
`821e43dc1c568f4c5b0fdea8d831a15177a6c652e9f5c0390b5aba0b99b47547`を品質evidenceへ結合する。

runtime state identityはrequested／resolved KV encoding、canonical descriptor、descriptor version、block size、E8M0 scale recipe、
physical OCP／FNUZ／software variant、exact target semantics、policy version/digest、selection sourceを含む。block16 v1/v2 identityは
履歴payloadを誤認しないため残すが、2026-08-30以降は新規state生成を拒否する。current default identityは
`kv-mxfp8-e4-v1`、OCP E4M3FN、block 32、E8M0、`mxfp8-e4-default`を結合する。
model lock alias、path、request length、空きHBMはselection identityを代替しない。

descriptor v1/v2のgfx1201／gfx1030 correctness／品質と空mappingはPhase 53/54の履歴である。2026-08-30の明示決定により、
reviewed Qwen3.5-4B BF16 dense text scopeではexact `gfx1030`、`gfx1201`、`gfx942:sramecc+:xnack-`をstandard OCP
MXFP8 E4M3 defaultへmappingする。このpolicy変更はmodel weight lockの新revisionではない。明示`fp16`はrollbackである。

## Phase 56 Gemma 4 MTP pair identity

Gemma 4 MTP targetは既存`google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`、fingerprint
`sha256:381c94bcb48a26d8ef83d1c3d7c5a3513ef8fac4a638752731b85c119385f09d`を変更しない。assistantは
[`gemma4-12b-it-assistant-bf16.json`](locks/gemma4-12b-it-assistant-bf16.json)へrepo、完全revision、license、7 file、config、
generation、tokenizer、safetensors header／catalog／全48 tensorを固定する。pair semantic identityはtarget fingerprintとassistant fingerprintの
両方を含み、片方の差替えを拒否する。

wire tokenizerはtargetを正本とする。vocab幅と共通generation IDを一致させ、targetだけが名前付きで持つ`<|video|>` ID 258,884はこの固定pairの
documented差としてのみ許可する。assistant GGUF、derived lock、target lock、pair identity、Q-only topology、target KV layer mappingのいずれかが
一致しない場合はresident allocation前に拒否する。

## Phase 57 DeepSeek V4 foundation identity

Phase 57は`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`、MIT license、公式support file、48 shardのHub LFS identity、
5,602,871-byte index SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`、
72,317 tensor、advertised payload 166,878,536,440 bytesを
[`deepseek-v4-flash-0731-foundation.json`](locks/deepseek-v4-flash-0731-foundation.json)へ固定する。
Hub LFS OIDはlocal shard payloadを実読した証拠ではなく、foundation lockは`local_full_payload_sha256_verified=false`を保持する。
一方、48 shardの先頭8-byte length fieldと宣言headerだけはbounded rangeで取得し、合計7,998,896 bytes、各prefix SHA-256、
全72,317 tensorのdtype／shape／relative offset／absolute byte rangeを検証した。header catalog SHA-256は
`6d90aa665f26217f4488809b1fdf87a1459702aa4ec46c8b02b44ce66bd4afcc`であり、これはweight payload hashではない。

typed configは43 main layerの先頭3 layerだけをhash routingとし、これを46 main layerへ数えない。root configの
`num_nextn_predict_layers=1`とindexに存在する`mtp.0..2`のcheckpoint DSpark 3 stageも別fieldで照合する。DSparkと要件上の
DFlashは別identityであり、名前、sampling contract、evidenceを相互流用しない。

このlockはidentity／config／header catalog／index／mapping foundationであり、tensor payload bytes、全shard local hash、derived GGUF、
single-またはmulti-GPU resident、generationを証明しない。model libraryが`deepseek4`を認識してもproduction aliasへ登録せず、
full-model対応はmulti-GPUまたは別のreviewed fitting artifactをscopeに含む後段計画と、そのartifactへ結合した新しいruntime evidenceを要求する。

## Phase 58 MiniMax M3 foundation identity

Phase 58は`MiniMaxAI/MiniMax-M3` revision
`f0e1c1e04d40177e4673a22097036854f536e9c0`、MiniMax Community License、18 support file、59 shardのHub LFS identity、
2,706,437-byte index SHA-256 `54dbde502126d07f6999077437a06b5df1f71e317518956d0aad1c8197df524e`、
23,416 tensorを[`minimax-m3-foundation.json`](locks/minimax-m3-foundation.json)へ固定する。
Hub LFS OIDはlocal shard payloadを実読した証拠ではなく、foundation lockは`local_full_payload_sha256_verified=false`を保持する。

公式index `metadata.total_size=869,157,697,024`に対し、59 shard file合計は854,176,398,808 bytes、header由来payloadは
854,172,958,720 bytesで整合しない。lockは三値と14,981,298,216-byteのindex／file差を別fieldで保持し、capacity admissionを
最大の869,157,697,024 bytesへfail-closeする。59 header prefix 3,440,088 bytes、全tensorのBF16／F32 dtype、shape、rangeを
検証したheader catalog SHA-256は`341285506267abca7bf50507d4bd39adf3eb430d1454d3f4dbfe74eb84b35982`であり、
weight payload hashではない。

typed configはdense layer 0..2、MSA／MoE layer 3..59、block 128／top-16／index 4×128／local block 1、
128 routed expert／top-4／shared expert 1／routed scale 2.0、vision topologyを固定する。configの7 MTP module／
`num_nextn_predict_layers=1`に対しreleased indexのMTP tensorは0件であるため、未公開weightやspeculative control flowを補わない。

canonical `minimax-m3` mappingはsource text 22,893、vision／projector 523、routed-expert source 21,888、
stacked expert候補171、combined physical候補1,699とdigest
`93ad9f5467bb9a7ba3b77c96db5aa0641e5d9e9801f99dc49bf46a8a4a18dd3f`を保持する。ただしpayload transform、GGUF書込み、
full-model resident／generation、multimodal／MTP productionを証明しない。model libraryはCommunity License、manifest不整合、
capacity、production loader未対応を灰色行に表示し、production aliasへ登録しない。

## Phase 59 DiffusionGemma foundation identity

Phase 59は`google/diffusiongemma-26B-A4B-it` revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`、Apache-2.0、10 runtime／evidence support file、11 shardのHub LFS identity、
104,650-byte index SHA-256 `6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a`、
1,047 tensorを[`diffusion-gemma-26b-a4b-it-foundation.json`](locks/diffusion-gemma-26b-a4b-it-foundation.json)へ固定する。
Hub LFS OIDはlocal shard payloadを実読した証拠ではなく、foundation lockは`local_full_payload_sha256_verified=false`を保持する。

11 shard file合計51,647,701,024 bytesからbounded header prefix 138,568 bytesを除いたpayloadは、index宣言
51,647,562,456 bytesと一致する。header catalog SHA-256は
`fd2cdedb367cd6c9aa52af6463e73baff3df52477b9cc3d61b9c6c4213cdc86f`であり、weight payload hashではない。
capacity admissionはKV／workspaceを追加する前からfile合計51,647,701,024 bytesを使う。

typed configはcanvas 256、text 30 layer、hidden 2,816、5 sliding＋1 full schedule、128 expert／top-8、vision 27 layer、
context 262,144を固定する。decoder text weightはencoder language-model weightへtieされ、indexに独立して存在するencoder tensorを
勝手に補わない。causal encoder KV、read-only KVを参照するbidirectional decoder、processed-logit self-conditioning、
entropy-bound sampler、adaptive stopはcontainer-neutral contractとして分離する。

`diffusion-gemma` GGUF foundation mappingはwrite-disabledである。fixed llama.cpp revisionにmerged architectureがなく、
full shard payload、derived GGUF、single-GPU resident、multimodal／generationを証明しない。model libraryはApache-2.0、capacity、
production loader未対応を灰色行に表示し、production aliasへ登録しない。

## Phase 60 Ministral 3 official GGUF identity

Phase 60は`mistralai/Ministral-3-3B-Instruct-2512-GGUF` revision
`eb599d408350ea2bb60452cb86be7c7b2fc28227`の`Ministral-3-3B-Instruct-2512-BF16.gguf`を
[`ministral3-3b-instruct-2512-official-bf16-gguf.json`](locks/ministral3-3b-instruct-2512-official-bf16-gguf.json)へ固定する。
file sizeは6,866,745,504 bytes、SHA-256は
`17ef932bea952e007f9dad63151da5699132ec513d1033d618df7382e24aa3ee`である。これはsource safetensorsからsLLMが
導出したartifactではなく、公式公開GGUFを直接固定したlockである。

lockは236 text tensor、losslessなF32 normalization、Tekken tokenizer、chat template、YaRN metadata、text／vision分離を
検証する。model identityとcontainer integrityの一致はruntime wiringを許可する条件だが、参照生成とのtoken一致やproduction品質を
証明しない。Phase 60の品質差が解消するまで、このlockの検証成功をmodel対応完了へ読み替えない。
