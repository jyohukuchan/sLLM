# Phase84量子化順序の決定（2026-09-11）

ユーザーは、低bit draftでも採用率低下が小さい研究があることを踏まえ、まずMXFP8を実装し、採用率・速度で大きな問題がなければ同じPhaseでMXFP6まで進めるよう指示した。従来の「形式未決定」と先のAI提案「既存FP8を第一候補」は今回の決定で置き換える。

- MXFP8 E4M3 W8A8 → 条件付きMXFP6 E3M2 W6A6。同Phaseで通常CLI/APIへの統合と採否まで扱う。
- companion専用8行列を初期対象とし、norm・target重み・共有embedding/head・KVは維持する。必要なBF16残置は理由とrecipeを記録する。
- 共通化後BF16 companionの8192/128測定と言語/タスク12 prompt×3 seedを比較基準にする。GPU間で採用率が一貫してV620<R9700ではなかったため、量子化前後をGPUごとに比較する。
- 採用率の微減だけで停止せず、draft短縮と検証負担を含むdecode/E2E、prefill/TTFTを合わせて移行・採否を判断する。新しい固定倍率や一律percentageを完了条件にしない。
- MXFP6を試して不利ならMXFP8を維持できる。MXFP8に問題が残りMXFP6未実施なら、その理由・試行・残件を記録し、実施済みとは説明しない。
- MXFP4/NVFP4 MTP、汎用FP8 artifact、再学習へ範囲を広げない。研究のMXFP4 weight-only draftをsLLM MTP W8A8/W6A6の速度・品質保証には使わない。
- main-planとロードマップのPhase84/85を同期した。実装は未着手であり、この文書更新はPhase84完了ではない。

作業計画: [Phase84](../../../../plans/archive/2026/09/1-10/phase84-mtp-weight-quantization.md)。
参照: [8192/128基準](../1-10/phase83-common-qwen38-remeasurement.md)、[言語/タスク比較](mtp-language-task-acceptance.md)。
