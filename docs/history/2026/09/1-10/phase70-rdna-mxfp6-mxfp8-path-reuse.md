# Phase 70: 両RDNA MXFP6のMXFP8実行骨格再利用

## 結論

Phase 70は2026-09-02に完了した。OCP MXFP6 E3M2 W6A6のresident形式を維持したまま、format固有処理を
packed E3M2のunpackとexact E3M2→E4M3FN tile ingressへ限定し、既存MXFP8経路のschedule、decode、scale、
FP32 accumulation、BF16 outputを再利用した。

exact `gfx1201`ではID44を経て、追加P70-Fのpacked 4-value ingress ID45をmodel非依存shapeへ既定採用した。
exact `gfx1030`のID43は数値的には旧ID29と同一だったが、full-modelで遅いためbenchmark-onlyとした。
decode、KV default、sampling、GGUF recipe、MXFP8 selectorは変更していない。

## 実装

- 共通codecへ全64 E3M2 codeを実数値exactなE4M3FN bit patternへ写すhost/device primitiveを追加した。normalはsign、
  exponent bias、mantissa shiftだけで変換し、subnormalは`0x00/0x18/0x20/0x24`へ明示写像した。
- provider planへMXFP6 MMQ-via-E4／WMMA-via-E4を別identityで追加し、prepare後に環境を変えてもvariantが変化しない契約を固定した。
- ID43 `matmul.mxfp6.w6a6.gfx1030.mmq-col8.via-e4m3.v1`は既存col8 MMQ bodyへconverter policyを接続した。
- ID44 `matmul.mxfp6.w6a6.gfx1201.wmma128x64.via-e4m3.v1`はMXFP8 N64 WMMA bodyを共有し、activation／weightの
  K32 tile ingressだけをpacked E3M2からE4M3 byteへ変換した。
- P70-FのID45 `matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4.v2`は、各scalar laneが同じ3 byteを再読込するID44 ingressを
  4-value group loadと32-bit LDS storeへ置換した。ID46は同じingressをN128へinstantiationした。
- ID45の既定scopeはexact `gfx1201`、`M>=17`、`K>=2048`、`1024<=N<=16384`である。ID44 rollbackは
  `SLLM_MXFP6_PREFILL_FORCE_PHASE70=gfx1201-n64`、旧tiled16 rollbackは
  `SLLM_MXFP6_PREFILL_FORCE_TILED16=1`を用意し、既存のbaseline／MMQ／row8強制指定も既定採用より優先する。

## 正しさと互換性

- device codec oracleはexact gfx1030／gfx1201とも64 code×4 packed laneをbit exactにPASSした。
- gfx1030 ID43はQwen3.5-4B production operator 5 shapeでID29とBF16 digest一致し、最大相対誤差は`0.00294194`以下だった。
- gfx1201 ID44は独立FP32 oracle、非有限位置一致、repeat determinismをPASSした。WMMA reduction treeへの変更によりlarge-Mの
  BF16 digestは旧providerと異なるためN1分類としたが、固定full-modelの生成tokenはcontrol／candidate全反復で`23066`×4だった。
- P70-FのID45／46も同じ独立FP32 oracleをPASSし、全shapeで非有限不一致0、repeat digest一致、最大相対誤差
  `0.00387579`だった。ID45とID44は同じ算術treeで、最終full-model全13 sampleの生成tokenは`23066`×4で一致した。
- host selector境界、prepare freeze、rollback、gfx942／unknown非選択をPASSした。gfx942はROCm 7.14／LLVM 23でrelease
  compile-onlyもPASSしたが、実機GPU PASSとは扱わない。
- shared source変更後のMXFP8は、gfx1030 512／2,048-tokenが約`249／252 tok/s`、gfx1201が約`3872／3777 tok/s`で、
  ID41／ID37の既存selectorと既知速度水準を維持した。

## 性能採否

固定artifactはQwen3.5-4B MXFP6 GGUF
`sha256:d0ff2e1de9d87dddddcde8f85ef305bbf21a06d5f7586d077ba1178580a0264e`、FP16 KV、direct input 512／2,048、
最大4 output、greedy、ignore EOSである。

- gfx1030、1 warmup＋3 measured: 512-tokenはID25 `247.688 tok/s`に対してID43 `191.373 tok/s`、
  2,048-tokenは`244.236`に対して`191.375 tok/s`だった。約22.7%／21.6%の退行なのでbenchmark-onlyとした。
- gfx1201、3 warmup＋10 measured: 512-tokenはID25中央値`307.588 tok/s`からID44 `1302.342 tok/s`へ4.234倍、
  2,048-tokenは`299.929`から`1508.692 tok/s`へ5.030倍となった。prefill時間は約76.38%／80.12%、E2Eは
  約69.98%／78.19%短縮した。
- P70-F、同一最終binaryの3 warmup＋10 measured: 512-tokenはID44 `1276.494 tok/s`からID45 `2157.868 tok/s`へ
  1.690倍、2,048-tokenは`1506.933`から`2423.308 tok/s`へ1.608倍となった。prefill時間は40.84%／37.82%、
  E2Eは29.30%／33.68%短縮した。ID46 draftは512／2,048で`1931.884／2348.276 tok/s`となり、どちらもID45を下回った。
- gfx1201のcontrol／candidateは生成token、dispatch 3,008、resident `4,061,763,072` bytes、peak
  `4,400,391,680`／`5,261,350,400` bytes、fallback 0、cleanup 0が一致した。

## Resourceとprofile

- gfx1030 ID43: wave32、workgroup 256、LDS 8,704 bytes、SGPR 34、VGPR 45、spill/private 0。
- gfx1201 ID44: wave32、workgroup 256、LDS 6,912 bytes、SGPR 34、VGPR 103、spill/private 0、WMMA 8。
- P70-F最終code objectではID44／45／46がLDS `6,912／6,912／9,216` bytes、SGPR `34／38／34`、
  VGPR `114／115／167`、spill/private 0、static WMMA `8／8／16`だった。
- production operator M=`17/127/128/512/2048`、K=`2560`、N=`9216`で、ID45はID44比
  `1.535／1.838／1.838／1.768／1.740倍`だった。ID46はM=2,048だけID45比1.061倍で、それ以外は0.478〜0.623倍だった。
- rocprofv3の`VALUInsts`、`MemUnitBusy`、`LDSBankConflict`、`OccupancyPercent` derived metricは対象環境で0を返し、
  有効な比較値ではなかった。raw `SQ_WAVES=8`とcode object resource、static instruction分類を記録し、0値を採否根拠にしなかった。
- ID46 N128は実装・測定したが、VGPR 167とfull-model退行によりbenchmark-onlyとした。

## Identity

- 開始時Git HEAD: `8e7ca87e2127da610cd765c9e29559a745448c45`（dirty working sourceのためsemantic identityではない）。
- gfx1030／gfx1201 CLI SHA-256: `9f511b3354a47bc5aae58fa14e7d103d26a402728148d953416e98c20ec8ed2e` /
  `6a5de12e02f4e328785105a27b701af6b2a9e585ce691285dccd35602582bfce`。
- gfx1030／gfx1201 matmul code object SHA-256: `fd665233fba20d5138242a94f51e7e44612e5dafdc5ba079257bf2f38d1d8f8e` /
  `2acacc40208fa1917f4d9da0977018a823722fc58254120f8bdbde61dacae654`。
- P70-F最終gfx1201 CLI／operator runner／matmul code object SHA-256:
  `8234358a64db92d7932481ae1f892360842b5b83e58ecaa525c6200927c3dd99` /
  `0752705a58b1cdc30d2c91d6c4102ef25938f196863dad7f8d1aff6501f8fd46` /
  `2a773926e1d5a014c7e89cbc093736be98fbc042cf423522b1e572e92292eb8c`。
- compiler: ROCm 7.14.0、AMD clang 23、Code Object V6。

[保存済み計画](../../../../plans/archive/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md) /
[全体計画](../../../../plans/main-plan.md) /
[追跡要約](../../../../../ci/matrix/phase70-rdna-mxfp6-mxfp8-path-reuse-v1.json)
