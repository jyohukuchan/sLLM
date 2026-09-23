# KV memory方式: vAttentionからPaged Attentionへの移行方針

2026-09-24。vAttention（HIP VMMによるvirtual-contiguous KV）を続けるか、Paged Attentionへ移るかを検討した。
ユーザー決定により、vAttentionを完全に廃止してPaged Attentionへ完全移行する方針とした。
ただし、pagedを最適化しても性能低下が10%以上になる場合は再考する。
判断材料を取る試作は[Phase 87計画のWU-P1](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)で行う。

## 現状

[KV memory方式の決定](../../../../architecture/kv-memory.md)はvAttentionだが、実際の確保は既にGPUで分かれている。

- R9700: Phase 83で、VMMを伸ばした直後に別layerのKVが壊れる事象が出たため、容量によらず全量確保（resident）。
- V620: 容量65,536 token未満はVMM、それ以上はresident（Phase 52）。

## 検討の経過

1. **単一要求・1 GPUだけなら**、連続KVの方が有利で、vAttentionの続投でよい。
   Phase 6の同一演算proxyでは、表を引くpaged版が長いcaseでV620 +17%、R9700 +31%遅かった（素朴な実装で、調整すれば数%の見込み）。
   Qwen3.8のようなhybridモデルはKVを持つ層が少なく、pagingの利点が小さい。
2. **1M以上のcontext、8〜16並列、最大8 GPUを考えると**、pagedが有利になる。
   - AMDのVMMは、実行中のcommit数に上限や不具合があり、規模に耐えない（下記）。pagedは起動時に1つのpoolを確保して中で分けるだけで、実行中にVMMを呼ばない。
   - VMMは2 MiB単位でcommitするため、層・plane・要求の数だけ無駄が積み上がる。tokenあたりのKVが小さいモデルほど1 pageのtoken数が膨らみ、単位が粗くなる。
   - agent session間のprefix共有、KVの退避と復帰、GPU間の転送、blockを選んで読むsparse attentionは、いずれも固定サイズのblockと相性がよい。
3. **READMEの対象ハードウェア（RDNA2〜4、CDNA1〜5、gfx906、Volta／Turing、Ascend、Hygon、CPU）まで広げると**、
   pagedは普通のメモリ確保と表の参照しか必要としない。VMMはメーカーごとに機能の有無・成熟度・上限を検証する必要がある。
   CDNA、古いNVIDIA、Ascendの既存attention実装も多くがblock tableを前提にしている。
   llama.cppの各backendのattentionは、全系列で共有する連続bufferをtoken単位のセルとmaskで管理する方式であり、流用の際は別途比べる。

## 「10万token制約」の正体（Phase 52）

Phase 52で、R9700が10万token入力のprefill途中にVMMの物理commitで失敗した（`layer.23.kv_append`、HBMは13.2 GB）。
AMD公式のHIP VMM文書には、確保数・handle数・map数の上限の記述はない。
一方、[ROCm/legacy-rocm-build #6661](https://github.com/ROCm/legacy-rocm-build/issues/6661)では、
ROCm 7.14の`hipMemCreate`が確保1回ごとにdmabufのfdを1つ開き、fdのsoft limit（既定1024）に達すると
VRAMが空いていても`hipErrorOutOfMemory`になると報告されている（1,012個目で失敗。ROCm 7.1.1ではfd数は一定）。

sLLMは推奨granularity（7.14では2 MiB）でpageを作る。Phase 52のモデル（Qwen3.5-4B、FP16 KV、KV層8、K/Vの16 plane、1 page＝1,024 token）では、
約1,012 pageに達するのはKV約63〜64k token時点で、失敗の位置とよく合う。
2026-09-24時点で動いているCodexのプロセスはfdのsoft limitが1024（hardは1,048,576）であり、
Phase 52もCodexから起動していた可能性が高い。当時のfd上限は証拠に残っていないため、確定ではない。
これは入力の新しい部分だけでなく、要求のKV総量（cacheした過去の入力を含む）に対する制約である。

あわせて、gfx1201ではmap完了後もGPUにmappingが反映されておらず直後のkernelが0を読むという報告が未解決である
（[ROCm/rocm-systems #11693](https://github.com/ROCm/rocm-systems/issues/11693)。
HIP側の回避策[ROCm/clr #288](https://github.com/ROCm/clr/pull/288)は、正しい修正はHIPより下の層だとして取り下げられた）。
Phase 83の破損との関係は断定しない。
公式文書には、RDNAで同じ物理pageを複数の仮想アドレスから共有すると結果が正しくないことがある、という注意もある
（sLLMのVMMのpage共有と末尾COWに当たる）。

## 進め方

1. 段階9・4の完了後に、WU-P1としてpaged kernelを試作し、attention kernel単体で現行の連続KVと比べる（本番sourceは変更しない）。
2. decodeとprefillのそれぞれで増加が10%未満なら、本移行をPhase 88のバッチ処理より前の独立した作業として計画する。
3. 10%以上なら、原因を切り分けて報告し、方針を再考する。

移行するまでの間、Codexなどfdのsoft limitが1024の環境から起動するV620の長いcontextの実行は、VMM経路で同じ壁に当たりうる。
今のベンチマーク（8192入力）では当たらない。

計画: [Phase 87](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
