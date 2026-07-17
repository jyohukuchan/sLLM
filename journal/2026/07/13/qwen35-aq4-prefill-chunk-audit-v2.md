# Qwen3.5 AQ4 prefill chunk と operation audit v2

- prompt prefill の論理実行幅を `min(remaining, 128)` とし、1/2/3/127/128/129/255/256 token の境界を固定した。
- chunk dispatch 後に同期し、取消を再確認してから prompt progress を確定する。同期後取消では実行済み state を reset し、prompt progress は確定しない。
- request audit を v2 に更新し、物理 operation 呼び出し数、M1 換算 coverage、実行済み/確定済み prompt token、chunk 幅 histogram を分離した。
- `cargo test -p ullm-engine --lib --no-fail-fast`: 638 passed、1 ignored。

次は resident linear-attention layer の M>1 native batch と、self-attention layer の chunk 内 M1 loop を実装する。

## Native prefill 接続

- model単位で最大128 tokenのlinear sequence workspace、`[128,H]` ping-pong 2本、self-attention splice用M1 rowを一度だけ確保した。workspaceはVRAM admissionに含める。
- linear-attention層はM=2..128を既存batch primitiveへ通し、QKV prepareとrecurrent scanはtyped registry StartedPlan経由で実行する。self-attention層はposition順のM1 row spliceとした。
- registryはM1とM2..128を幅で排他的に解決し、AQ4 matvec batch/QKV prepare batchのruntime feature、production guard、実ABI probeをfail-close接続した。
- auditはnative linearとself M1 fallbackを独立IDで数え、物理呼び出し`48+16M`とM1換算`64M`を区別する。native途中faultもlayer/widthを保持する。
- `cargo test -p ullm-engine --lib --no-fail-fast`: 645 passed、1 ignored。`cargo check -p ullm-engine`、format、diff checkも通過。
- R9700実機activationはまだ行わず、production guard/profileとunit contractまでを準備した。
- 配布用 `qwen35-9b-aq4.profile.json` にAQ4 matvec batchとlinear QKV prepare batchの必須guardを追加し、manifest生成fixtureとresident worker admissionの完全一致テストを通した。
- worker progress unitをモデル非依存の1..=128へ拡張し、128累積またはfinalだけをwire emissionとした。127、128+1、128+127のprompt tailを受理する。
- native dispatchの監査をstream sync前に記録し、sync失敗でもphysical/token-equivalentを保持する。commitはsync成功後だけで、失敗requestはabort/resetのみ許可する。
- AQ4 batch projectionをshape-exact typed registryとStartedPlanへ移し、package runtimeからdirect ABI callを除去した。AQ4/QKV/recurrent sequence probeはM=128を実行し、recurrent sequence専用guardもprofileへ追加した。
- R9700 ready probeで発見したAQ4引数束縛を修正した。M=128はbatch countだけに使い、scale countはscale table実体どおり2としてhelperと回帰テストで固定した。

## Native self-attention chunk 接続

- M=2..128のself-attentionを1層1 writer/readerのnative sequenceへ置換し、全32層をchunk幅で実行する。M1/decode経路は維持する。
- model-wide self workspaceは1組だけで、Qwen9B・M=128では41,418,752 bytes（39.5 MiB）。外部ping/pongと所有を分離し、self-only構成でもVRAM admissionへ一度計上する。
- writer成功後だけ`written_len`をM進める。readerまたは後段失敗ではin-place KV更新済みとしてrequest stateをpoisonする。
- 1 chunkの監査はphysical 64、token-equivalent 64M。複数chunkはphysical 64Cで、失敗時はlayer/width/operationを保持する。
- primitiveのblock table検証にはwriter/read各呼び出しでD2H+stream syncが入り得るため、実機性能測定の対象として残す。
- `cargo test -p ullm-engine --lib --no-fail-fast`: 662 passed、1 ignored。`cargo test -p ullm-runtime-sys`: 145 passed。
- forced-M1 differential hookは追加していない。同一モデル差分は旧binaryとの実機比較で確認する。
