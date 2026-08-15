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
