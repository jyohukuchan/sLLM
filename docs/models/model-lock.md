# Model artifact lock format

Model identifiers such as a Hugging Face repository name or branch are mutable.
uLLM therefore resolves every model input to an immutable revision and verifies
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
actual bytes uLLM consumes.

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

## Derived artifacts

For each converted, quantized, merged, or otherwise generated artifact, record:

- every source model-lock fingerprint;
- the tool repository URL and full commit SHA;
- the exact ordered arguments and complete effective configuration;
- the relevant execution environment, including platform, toolchain/runtime,
  dependency lock or versions, and accelerator details when they can affect the
  output; and
- every output path, byte size, and SHA-256.

Original upstream snapshots use `derivation: null`. A derived output must also
appear in `files`; the output hash in `derivation.outputs` must match its file
record. Derived local outputs use `source_page_url`, `download_url`, `git_blob`,
and `lfs_oid` as `null` unless they are subsequently published in a repository
that supplies those identities.

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

## YAML example

YAML is shown only as an authoring format. The fingerprint is computed from the
JCS canonical JSON representation of `{ "schema_version": ..., "model": ... }`
defined above.

```yaml
schema_version: 1
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
    - path: README.md
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/README.md
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/README.md
      lfs_oid: null
    - path: LICENSE
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/LICENSE
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/LICENSE
      lfs_oid: null
  files:
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
    - path: model-00001-of-00002.safetensors
      size_bytes: <integer>
      sha256: <64 lowercase hexadecimal characters>
      git_blob: <repository Git blob object ID for the LFS pointer>
      source_page_url: https://huggingface.co/Qwen/Qwen3.5-4B/blob/<full-commit>/model-00001-of-00002.safetensors
      download_url: https://huggingface.co/Qwen/Qwen3.5-4B/resolve/<full-commit>/model-00001-of-00002.safetensors
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

The example file list is illustrative, not complete. The actual lock must enumerate
all files used by the selected revision, including every shard named by its index.
Replace `derivation: null` with the complete derivation record described above for
generated artifacts.
