# Served-model manifest v0.1

## 1. Purpose

`ullm.served_model.v1` is the immutable contract for one model exposed by an
uLLM serving slot. It contains public API metadata, generation limits,
tokenizer identity, backend selection, worker launch identity, product
identity, and the promotion receipt. The gateway, worker, activation tooling,
and OpenWebUI reconciliation consume the same document.

Bind address, port, API key, GPU lock, GPU visibility, and the active manifest
path are slot operations settings and are not model fields.

## 2. File and JSON boundary

The manifest MUST be a non-symlink regular file, MUST NOT be world-writable,
and MUST be at most 1 MiB. It is strict UTF-8 JSON with a top-level object.
Duplicate keys, non-finite numbers, invalid UTF-8, trailing data, unknown keys,
missing keys, wrong JSON types, structures deeper than 16 levels, more than
16,384 JSON nodes, or a string larger than 65,536 UTF-8 bytes are rejected.

Every object described below has an exact field set. Optional values are
represented by an explicitly permitted `null`; omission is not permitted.
Integers do not accept JSON booleans.

## 3. Exact document shape

```json
{
  "schema_version": "ullm.served_model.v1",
  "public": {
    "id": "ullm-qwen3.5-9b-aq4",
    "name": "uLLM Qwen3.5 9B AQ4",
    "description": "Qwen3.5 9B served locally by uLLM AQ4_0.",
    "upstream_id": "Qwen/Qwen3.5-9B",
    "revision": "aq4-cli-compat-v0.1",
    "context_length": 4096
  },
  "generation": {
    "max_completion_tokens": 512,
    "vocab_size": 248320,
    "eos_token_ids": [248044, 248046],
    "sampling": {
      "top_k": 1,
      "temperature": false,
      "top_p": false
    }
  },
  "format": {
    "format_id": "AQ4_0",
    "implementation_id": "qwen35_aq4_rdna4_v1"
  },
  "tokenizer": {
    "root": "/srv/ullm/tokenizer",
    "transformers_version": "5.12.1",
    "class": "Qwen2Tokenizer",
    "chat_template_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "files": {
      "tokenizer.json": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "template_options": {
      "add_generation_prompt": true,
      "enable_thinking": false
    }
  },
  "worker": {
    "protocol": "ullm.worker.v1",
    "binary": "/opt/ullm/bin/ullm-aq4-worker",
    "binary_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "arguments": ["--served-model-manifest", "{manifest}"],
    "required_environment": ["ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL"],
    "identity": {
      "device": "gfx1201",
      "execution_profile": "rdna4_aq4_resident"
    }
  },
  "product": {
    "root": "/srv/ullm/product",
    "artifact": null,
    "package": {
      "manifest_path": "package/manifest.json",
      "manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  },
  "promotion": {
    "source_commit": "fa69c10",
    "receipt": "/srv/ullm/product/promotion.json",
    "receipt_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
}
```

For a separate quantization artifact, `product.artifact` is an object with
exactly `manifest_path`, `manifest_sha256`, and `content_sha256`. Otherwise it
is `null`. `product.package` is always required.

## 4. Field rules

- All text fields are nonempty, bounded, and contain no C0 control characters.
- SHA-256 values are exactly 64 lowercase hexadecimal characters.
- `public.context_length`, `generation.max_completion_tokens`,
  `generation.vocab_size`, and `generation.sampling.top_k` are positive.
- EOS is a nonempty list of unique nonnegative token IDs. Every EOS ID is less
  than `vocab_size`.
- `max_completion_tokens` is at most `context_length`; `top_k` is at most
  `vocab_size`.
- If either temperature or top-p sampling is unsupported, `top_k` MUST be one.
- `worker.arguments` has at most 128 entries and contains the exact string
  `{manifest}` once. The launcher replaces it with the validated manifest path.
- `worker.required_environment` has at most 128 unique POSIX-style uppercase
  environment names. The list names required boolean guards; GPU visibility is
  deliberately excluded as a slot setting.
- `tokenizer.files` is nonempty and contains at most 128 entries.

## 5. Paths and identity

`tokenizer.root`, `worker.binary`, `product.root`, and `promotion.receipt` may
be absolute. A relative value is resolved against the directory containing the
manifest and MUST NOT contain `.` or `..` components.

Tokenizer file paths and product manifest paths are relative POSIX paths. They
MUST NOT be absolute or contain empty, `.` or `..` components. Their resolved
targets MUST remain within the declared root.

Every declared root must be a non-symlink directory. Every manifest,
tokenizer file, worker binary, product manifest, and receipt must be a
non-symlink regular file and must not be world-writable. No path component may
be a symlink. The worker binary must be executable. The loader streams and
verifies every declared file SHA-256 before returning a model contract.

`product.artifact.content_sha256` binds the promoted artifact content identity;
the other product hashes bind the actual manifest files. Worker-ready
validation derives model revision and product identity from this single
document rather than repeating them in another configuration source.

## 6. Failure and compatibility

Any validation failure prevents worker launch and model readiness. There is no
best-effort parsing, implicit field default, unknown-field preservation, or
fallback to a different model profile.

During migration, manifest mode and legacy model environment mode may both be
implemented, but selecting both for one process is an error. Wire protocol
`ullm.worker.v1` remains permitted until workers emit the validated manifest
digest in a later protocol revision.
