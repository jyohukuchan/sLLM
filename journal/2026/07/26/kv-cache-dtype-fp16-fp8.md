# KV cache FP16 / FP8 dtype work

## 前回の要点

- paged KV は F32 payload で、block size 16 / cache blocks 256 の physical page
  mapping を用いていた。
- BH の GQA 協調 decode redesign 後の full-model F32 reference は 27.378731
  tok/s。旧 direct-path の 20.002232 tok/s 見積もりを流用してはいけない。
- BR は `runtime/src/ullm_runtime_parts/part_01.inc` と
  `runtime/src/ullm_runtime_hiprtc_sources.inc` を編集しているため、typed HIP
  kernel の同時編集は禁止だった。

## 今回の変更点

- `KvCacheDtype` / `KvCacheDtypes` / `KvCacheLayout` を追加した。F32 が既定、
  `ULLM_KV_CACHE_DTYPE` と `ULLM_KV_CACHE_TYPE_K` / `_V` で F16 または
  FP8 E4M3FN を K/V 別に選べる。`Q8_0` は明示拒否した。
- FP16 payload と、FP8 E4M3FN + `(physical_token, kv_head, plane)` FP16 scale
  の exact allocation/ABI を実装した。Qwen3.5 指定 geometry では layer あたり
  F32 32 MiB、F16 16 MiB、FP8 8.0625 MiB である。
- generic `PagedDecodeState` に typed allocation, writer, direct reader,
  scale-aware readback, typed causal prefill fallback を接続した。F32/F32 は従来の
  F32 writer/reader API をそのまま選ぶ。
- CPU host reference と同一 ABI の HIP staging fallback を追加した。これは
  correctness fallback であり、native HIP performance path ではない。
- CPU targeted tests は decoder 36 件と layout 3 件が成功した。FP8 scale は
  all-zero reset row の zero だけを許し、負値・NaN・infinity は readback と
  attention の双方で破損として拒否する。
- `ed641675 feat(runtime): add typed paged KV cache ABI` を作成した。
- `1c7cc3f3 fix(kv): reject invalid FP8 cache scales` を作成した。
- 実行後に、broad `ullm-runtime-sys --lib` に opportunistic HIP tests が含まれる
  ことを発見した。R9700 lock preflight 前の実行だったため、GPU evidence としては
  採用しない。以後 GPU を使う全テスト/計測は lock/service preflight 後だけにする。

## 次の行動

- BR 完了を `pgrep -af 'codex exec' | grep -c '依頼BR'` で確認するまで
  `part_01.inc` / HIPRTC source を編集しない。
- 完了後に native typed paged writer/decode reader、fused Qwen writer、paged
  causal GQA / cached-prefix typed readers を追加する。GQA cooperative mapping と
  online softmax order を保つ。
- AQ4_0 resident production path の typed allocation/operation registry 接続は
  今回の禁止範囲であり、別途許可された変更として扱う。
- R9700 lock が空き、native path と production integration が揃ってからのみ、
  F32/F16/FP8 の長文脈 full-model decode/prefill と生成文並置を保存する。
