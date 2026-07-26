# Candidate source isolation

Concurrent work was modifying the shared attention runtime sources.  To avoid
attributing any such change to this scheduler result, the measured candidate
was built in a detached temporary worktree from clean AY base commit:

```text
base commit:       0216b131cf5377d90125abd9c1c49c5a8a210511
candidate patch:   d8270d1cf4c2a30369f673eaa26932836246fbe66dc5815edda689aa7331194b
```

The candidate patch contains only:

```text
crates/ullm-engine/src/decoder.rs
crates/ullm-engine/src/sq8_stack_runtime.rs
crates/ullm-engine/src/sq8_serving_runtime.rs
crates/ullm-engine/examples/sq8_ck_serving.rs
```

It has 214 insertions and 65 deletions.  It does not include
`runtime/src/ullm_runtime_parts/part_01.inc`,
`runtime/src/ullm_runtime_hiprtc_sources.inc`, loader changes, or AQ4_0
production code.

The timing driver is AY’s driver source linked to this candidate worktree.
Its resulting binary hashes are:

```text
prefill driver: 84e4cb70e2ff3359c39788f4aef1527d46532700072687b5e5d50cb4a94eac26
oracle binary:  467484c8bbfe2770c8e3a63f0938a7c21762f4f21cd4763b7a479af60655eed6
```

For the old-path oracle, the clean AY-base `sq8_ck_serving` binary hash is:

```text
da792beeb36e9025f27775e6c0344f03d53babf87a9af0c14fe93c00c1ad7a0a
```

The generated candidate Cargo manifest is kept under
[`environment/candidate-prefill-driver/`](environment/candidate-prefill-driver/)
to make the link target explicit.
