# Phase 15 Weight NVFP4履歴

## 2026-08-15: format、artifact、runtime

- NVIDIA Transformer Engine v2.18（tag object `62f366a50b8e5a96fac7f123a554ab4db928b2a9`、commit
  `27486e03cfc1fa41f6932dcecdc47c71c47eac3e`、BSD-3-Clause）をformat sourceとして固定した。
  weight-only v1はlow-nibble-first E2M1、K-axis block 16 OCP E4M3FN scale、tensor FP32 scaleである。
- Rust codec/oracle、U8+NVFP4 descriptor、deterministic NumPy converter、versioned safetensors sidecar、loader verifierを
  実装した。全E2M1 code point、nearest-even tie、zero/underflow、NaN/Inf、15/16/17 tailをhost testへ通した。
- Qwen3.5-2B full sidecarは186 tensor、772,236,184 byte、artifact SHA-256
  `c99efefd0e209976f50e5e55f6fd4e265ab5f5242788e1ceb7becf49e8a0a36d`、manifest fingerprint
  `4f78d2f6eb271db3d79eedffd5e0a4638d39331d7896324d0ec11c08e21e1aca`である。二回のartifactはbyte-identicalだった。
- public C ABI matmul v3とkernel id 8を追加した。packed value、block scale、aligned tensor scaleをmodel load時に一度uploadし、
  request間でresident reuseする。exact target/provider不一致とruntime errorはfallbackしない。

## operator、model accuracy、memory

- canonical V620 `gfx1030`とR9700 `gfx1201`でM=1/M>1、K/N `15/16/17`、`31/32/33`を含む7 caseを
  `matmul.nvfp4.block16.packed_dequant.v1`へ通した。両targetともkernel id 8、fallbackなし、cleanup 0、独立FP32
  dequant oracle内でPASSし、最大relative errorは約`0.00364`だった。
- Qwen3.5-2Bは両targetで3 input set、1,056 submission、1,110 kernel dispatchをBF16と比較した。top-1は3/3一致、
  最大KLDは両targetとも`0.2637522997`で、既定budget `0.05`を超えた。thresholdは緩めなかった。
- V620最終再実行ではBF16 resident/high-water `3,763,686,080/3,798,400,520` byte、NVFP4
  `1,790,406,056/1,825,120,496` byteで、52.43%/51.95%削減した。loadは4,672 ms対2,343 ms、3 case requestは
  266 ms対199 msだった。これは短いaccuracy setでありdefault性能claimには用いない。
- official Gemma 4-12B layer 0 gate `[15360,3840]`を33,178,084-byte sidecarへ変換した。artifact SHA-256は
  `22bc5c2285f72a273602db6c2505a79a47f3249e44c4381ce333450273ef3ccc`、weight relative L2 `0.09262`、
  cosine `0.99593`、3 activationの最大KLD `0.002202`、top-1 2/3だった。

## CLI、OpenAI service、採用判断

- R9700でQwen NVFP4 CLIのfixed、Unicode、stop generationを実行し、全dispatch HIP、fallbackなし、cleanup 0を確認した。
- 同一resident serverでnon-stream 2件、SSE、stop、disconnect後のrecovery、graceful shutdownを通した。5 completedと
  1 cancelled requestを監査し、model drop/session shutdown後のcurrent/request/workspaceとretryable/durable cleanupは0だった。
- `gfx1201`/`gfx1030`ともproviderは`packed-dequant`でありnative FP4ではない。Qwen KLD budget超過とGemma slice
  top-1不一致により、両targetを`correctness-only opt-in`としdefaultへ昇格しない。`gfx942`はcompile-onlyである。
- closeout再計測時、R9700はPhase中にPASSした既存binaryを含めkernel imageをdriverから拒否した。この試行はPASSへ
  読み替えず、先行R9700実機証拠とV620最終再実行を分けて扱う。

## integrationとcloseout

- workspace testで新しい厳密NVFP4 descriptorより前に失敗する既存negative fixture 6件を検出し、正規NVFP4 encodingで
  本来の後段contractを検査するよう修正した。correctness/security blockerの残件は0件である。
- Rust dependency inventory、H3 public runtime/RMSNorm immutable source inventory、runtime/model lock/GPU/software互換性、
  provenance、main plan、Phase 12 forward queueを同期した。large sidecar、model、binary、raw traceは追跡していない。
- integration review 1回はcorrectness/security blocker 0件だった。最終host evidenceはH0 `513/513`、H1
  `425/458 selected`、H2 `36/44 selected` PASSで、workspace all-target test、clippy、format、matrix/schema/link検査もPASSした。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase15-weight-nvfp4.md)
