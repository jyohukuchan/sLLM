# SQ8_0 R9700 decode attention root-cause

## 前回の要点

- 依頼BAのPhase 0は、SQ8_0 generic matvec の要素ループに64-bitソフトウェア除算があるという仮説と、FP8変換のバイトセレクタでmask/shiftを削除できるという仮説を反証した。どちらも実コードでは成立せず、generic matvec 4種は本番decodeに起動していなかった。反証証跡はコミット`98422bbb`に内容を変えず保存した。
- 本番SQ8_0 decodeは `Rdna4W8a8BlockCk` の投影経路と `ullm_paged_decode_attn_f32_kernel` を使う。旧トレースではattentionが640呼出、492,371,584 ns、decode合計の51.05%だった。

## 今回の変更点

- R9700 (`gfx1201`) のdirect paged attentionは一層あたり40 Q head = 40 workgroup、256 thread/workgroup、320 wave32であることを、ランチャとrocprof dispatch `(10240,1,1)/(256,1,1)` で確認した。64 CU x 32 waveの供給目安では15.625%であり、llama.cpp vector FATTN mainの400 workgroup / 78.125%と並置すると並列度不足が明確である。これは実測achieved occupancyではなく、achieved occupancyは未確認である。
- C=1,036のunique KV常駐量8,486,912 Bに対する640 GB/s roofは13.2608 us、観測769.3306 usは58.0154倍であり、依頼元の概算は正しかった。GQAの5回再走査を含むsemantic load量は42,434,560 Bで、physical HBM byteではない。root rocprofのGL2 request counterは0、SQ_WAVESも不整合で、物理帯域・cache hit・achieved occupancyは未確認として残した。
- source-tile splitの最小R9700試験では、C=128/tile=128の一タイル退化はdirectとbit一致した。C=130の2タイルでmax abs 2.9802e-8 / 2,250 bit差、C=1,036の9タイルで1.08033e-7 / 4,934 bit差となった。初期化、tail/empty tile、ページ、merge scaleの明白なバグ経路はこの範囲で否定され、有限精度のreassociationとSQ8_0逐次量子化による増幅が最も支持される。full-modelのhard top-1 regression記録があるためmulti-tile default化は行わない。
- direct順序を保つ実験的 `ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX` を追加した。Vを持つ各waveのlane 0だけがmax/denominatorを更新するが、トークン順序は変えない。C=1,036でデフォルトと20,480 bytesがbit一致した。既定は無効のままである。
- full-modelのcandidate/control測定を試みたが、20:19:35 JSTに `ullm-openai.service` がactiveとなったため、候補側全区間とdirect末尾がAQ4_0 workerと競合した。journalはこのstartを特定の並行タスクへ帰属していない。14.685730 / 14.959300 tok/sは無効化し、性能改善とは報告しない。サービスはactive/runningへ復旧済みで、StartLimit窓を追加消費する再停止はしなかった。`llama-qwen35-udq4.service`はinactive/disabledのまま。V620 (`gfx1030`) はdevice infoで拒否した一度を除きGPU計算に使っていない。

## 次の行動

- 他セッションと調整した固定HEAD・isolated R9700窓で、既定directとwave-scalar candidateのfull-model decode tok/sを再計測する。サービス停止が必要なら一回のstop/isolate/restoreにまとめ、終了時に必ずactive/runningを確認する。
- gfx1201で信頼できるDRAM/GL2物理counter取得経路を特定するか、別の検証済み計測手段を使う。現時点では物理HBM帯域とachieved occupancyを推測で補わない。
- split-KVは、directと同等のSQ8_0 feedback出力品質を最小・full-modelの両方で示せるまで候補隔離を維持する。高速であることだけを昇格根拠にしない。
