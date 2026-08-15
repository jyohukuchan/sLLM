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

## 2026-08-15: VRAM内訳とdirect-engine性能のfollow-up

- Qwen3.5-2B sidecarの186 tensorは`1,372,717,056` elementで、BF16では`2,745,434,112` byteである。
  NVFP4 device payloadはpacked value `686,358,528`、block scale `85,794,816`、tensor scale `744` byte、合計
  `772,154,088` byteで、比は`0.281250271`だった。理論`4.5/16 = 0.28125`との差はtensorごとのFP32 scaleだけである。
  residentの残り`1,018,251,968` byteはBF16/NVFP4で同一だったため、全residentの52.43%削減は正常である。
- direct benchmarkへNVFP4 sidecar verification、graph、resident reuse、provider identityを接続した。Qwen3.5-2B、batch 1、
  greedy、correctness control 1回、warmup 3回、measured 10回、同一resident model、17/17 `short-odd`と32/32を
  canonical V620 `gfx1030` / R9700 `gfx1201`で実行した。中央値は次の通りである。

| GPU | case | encoding | resident bytes | peak bytes | TTFT ms | prefill tok/s | TPOT ms | decode tok/s | E2E ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 | short-odd | BF16 | 3,763,686,080 | 3,826,398,220 | 160.464 | 108.231 | 19.102 | 52.466 | 471.211 |
| V620 | short-odd | NVFP4 | 1,790,406,056 | 1,853,118,196 | 197.714 | 87.919 | 23.958 | 41.544 | 587.612 |
| V620 | 32/32 | BF16 | 3,763,686,080 | 3,864,493,120 | 168.693 | 194.069 | 18.780 | 53.340 | 759.716 |
| V620 | 32/32 | NVFP4 | 1,790,406,056 | 1,891,213,096 | 333.858 | 97.136 | 24.158 | 41.384 | 1,088.965 |
| R9700 | short-odd | BF16 | 3,763,686,080 | 3,826,398,220 | 33.750 | 558.516 | 15.628 | 63.943 | 288.181 |
| R9700 | short-odd | NVFP4 | 1,790,406,056 | 1,853,118,196 | 201.155 | 86.353 | 19.672 | 51.165 | 519.482 |
| R9700 | 32/32 | BF16 | 3,763,686,080 | 3,864,493,120 | 46.008 | 755.676 | 15.168 | 65.892 | 521.087 |
| R9700 | 32/32 | NVFP4 | 1,790,406,056 | 1,891,213,096 | 372.232 | 87.013 | 19.372 | 51.333 | 982.361 |

- short-odd NVFP4はBF16比で、V620がTTFT `+23.21%`、prefill `-18.77%`、decode `-20.82%`、E2E
  `+24.70%`、R9700がTTFT `+496.02%`、prefill `-84.54%`、decode `-19.98%`、E2E `+80.26%`だった。
  32/32でもdecodeはV620 `-22.42%`、R9700 `-22.10%`で、R9700のprefill/TTFT退行が大きい。
  両targetともnative FP4でなく同じpacked-dequant providerなので、RDNA4のBF16 prefill優位を利用できない結果と整合する。
- short-oddの生成token digestはBF16/NVFP4で一致した。32/32は各encoding内の14 requestでexact一致したが、encoding間では
  異なった。これは既知のKLD budget超過と整合し、性能結果によって`correctness-only opt-in`判断を変更しない。
- model loadは各row一回だけの観測で、V620 BF16/NVFP4がshort-odd `5,299/5,932 ms`、32/32
  `5,317/5,869 ms`、R9700が`5,431/6,992 ms`、`5,294/5,795 ms`だった。sidecar全体のhash verificationと
  filesystem cacheを含むため、この単発値からload性能の優劣は主張しない。
- 8/8 row、80/80 measured sampleはdirect v1 schema、event算術、反復token/stop/dispatch signature、HIP-only、fallbackなし、
  request/model/session cleanup 0をPASSした。R9700は今回target別binaryを受理し、先行closeoutのkernel image拒否は再現しなかったが、
  過去の失敗試行を遡ってPASSへ変更しない。binary SHA-256はV620
  `305b556c4fc22ed314ab37b375c670c7820e70049bccd4a14e1776acc3d848e0`、R9700
  `60a5f4c5ad15a6037137afc8700e07698457a9b3d424ab7c3d33cf531c625e2b`である。
- raw JSON SHA-256はV620 BF16/NVFP4のshort-oddが`6f0e12bd7ce4fe268e5a69594f956dce3bc0f2dbc5ae50a4edca18bcf2f4369f` /
  `c278dd8df0a9850a1f4d1d1aa711a390b5278d1bb89803fee50a2aad362e11fa`、32/32が
  `5d33dce196d2d2b8cb11217bf2ea69843195bdc0e8fd273021a1ea33b9fd89ca` /
  `737b8f4d232f53cdaa11dcff1587e7a734b691bd3ce83f3aed266717e3453521`、R9700の同順が
  `69bbdcd957b950cae1052b3df428226db529520737d7aa030079a77ca00a1b79` /
  `2979ec994924528870160fc95507fe26ef79ba9839dfe348eda7f05e2e6bce3d`、
  `39bfb92a5cf04f4cf1dc274233f0d69dada96227567d324a1e74827c22ad444f` /
  `5f19d5fa71c389157a1883084deb9c9e566d21c375ff1d5db074fe473c079416`である。raw/binary/sidecarは追跡しない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase15-weight-nvfp4.md)
