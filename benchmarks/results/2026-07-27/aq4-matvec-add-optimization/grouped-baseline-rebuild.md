# Grouped-baseline rebuild decision

The first staging worker was built from current main (`16d101e2` at build
time).  It is unsuitable for a production grouped-decode comparison because
main does not contain the 4:1 grouped-split change.  Passing the grouped
environment variable to that binary would not establish that it runs the same
implementation as the active grouped artifact.

The approved build baseline is `9d8643506a36659ecec3fc2d931deba26d29f574`
(`bq-aq4-grouped-integration`).  The following blobs are identical between
that commit and the active artifact's recorded source commit `c8074928`:

| file | blob SHA-1 |
| --- | --- |
| `runtime/src/ullm_runtime_hiprtc_sources.inc` | `16b9793914417c14497e976d22de1550a258d245` |
| `runtime/src/ullm_runtime_parts/part_01.inc` | `56d80284971148275564be09a4b7987fbe0c3936` |

An isolated detached worktree at
`/tmp/ullm-aq4-add-grouped-source-20260727T0416` contains that baseline plus
only the group-specialized add source and its differential-test update.  It
will be rebuilt into a separate staging target before the GPU window.  This
separation prevents unrelated dirty worktree changes, and specifically avoids
using the main-only build, in a worker that could be promoted.

The extracted standalone HIP source for the candidate is byte-identical between
the original static-ISA input and this grouped-baseline worktree:
`ba542a6a5f65bc578688adc1fb9d19607bf518d275b4a537168dfe3fd4397bb3`.
Consequently the existing candidate ISA object remains applicable; only the
surrounding runtime/grouped dispatch baseline changed for the full-model A/B.

The non-evidence source difference from `c8074928` to `9d864350` is limited to
SQ8 serving examples and `sq8_serving_runtime.rs`; no AQ4 engine/runtime source
file differs outside the two already-identical runtime blobs.  The isolated
AQ4 worker is therefore an appropriate grouped-split comparison baseline,
rather than merely an environment-variable approximation.
