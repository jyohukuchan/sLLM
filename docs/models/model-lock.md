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
