# SQ8_0 R9700 attention 経路と最適化余地

Date: 2026-07-26

## 前回の要点

- Phase 0 は R9700 の SQ8_0 decode で paged attention が 50.9968%、M=128 prefill で Flash2 attention が 74.8409% を占めることを確認した。
- paged decode の通常 source は wave-shuffle reduction だったが、計測時に共有メモリ fallback が有効だったかは環境記録不足で未確認だった。
- generic SQ8_0 matvec は選択されず、quantizer と RMSNorm の decode 上限は合計 2.6067% に留まった。

## 今回の変更点

- R9700 の単一 stop -> isolated test -> restore 窓で、SQ8_0 production-artifact driver の default decode、強制 shared-LDS fallback、decode PMC、Flash2 prefill trace/timing/PMC を取得した。全8ステップは exit 0 だった。
- GPU_DUMP_CODE_OBJECT=1 の runtime object と selected-region trace を同じ fresh process で取得した。default paged object は SHA-256 26fa813c...b6bb1 で ullm_paged_decode_attn_f32_kernel の ds_bpermute が10本、共有 fallback object は SHA-256 b8561368...e9fa で ds_bpermute が0本だった。従って、測定した SQ8_0 decode は wave-shuffle 経路である。
- fallback の選択は ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE の存在だけで決まり、値は解釈されない。=0 でも fallback になるため、是正は unset である。現状は default なので是正による現在の増分は0%。負の対照では default が forced fallback より unprofiled decode で 4.724077% 高速、attention trace で 8.914070% 短かった。
- decode は 40 workgroup/layer、320 wave32 だけで、R9700 の 64 CU x 32 wave/CU に対する wave 供給上限は15.625%だった。Flash2 は 5120 workgroup/layer launch を持つ一方、64-token full tile 当たり source-level CTA rendezvous が661回ある。
- KV logical metric は decode C=1036 で 55.157770 GB/s (640 GB/s 比 8.618402%)、prefill causal 1..1024 で 391.459814 GB/s (61.165596%) だった。FETCH_SIZE と VALUInsts は全 PMC sample が0であり、物理 HBM 効率と最終的な memory-bound/compute-bound 判定は未確認とした。
- Flash2 を最優先、decode split/tile を次点へ更新した。P3 の wave-shuffle、uint4、VGPR/LDS 監査はそれぞれ staged reduction、lane/tile redesign、candidate gate に適用するが、未試作の性能値は推測しない。
- ullm-openai.service は04:21:33 JSTに停止し、04:28:34 JSTに最初の start で active/running/enabled へ復旧した。reset-failed は不要だった。llama-qwen35-udq4.service は前後とも inactive/dead/disabled、gdm.service は inactive/dead/static を維持した。

## 次の行動

1. Flash2 を別 symbol で QK、max、sum の順に wave32 staged reduction 化し、short/tail/real prompt differential、metadata/ISA、unprofiled prefill を同じ窓で比較する。
2. existing split API を使い、source tile 128/256/512 の paged decode differential と M=1 end-to-end timing を取得する。direct legacy dispatch は変更しない。
3. 物理 byte/compute counter が0になる原因を isolated profiler setup で解消できるか確認してから、lane re-layout または uint4 の価値を判定する。
4. CDNA3 案 A では canonical SQ8_0 payload/scale と differential harness を共通化し、R9700 wave32 body と CDNA3 wave64/MFMA body は別実装にする。
