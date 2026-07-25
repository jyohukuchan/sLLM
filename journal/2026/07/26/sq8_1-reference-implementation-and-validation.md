# SQ8_1 reference implementation and validation

## 前回の要点

- `SQ8_1` は K=32、signed symmetric I8、FP16 upward scale、payload/scale 分離の設計まで確定していた。
- W8A8 の activation-only と sampled linear evidence は reference implementation 着手を許したが、全モデル logits gate は未通過だった。このため W8A16 が必須 fallback、W8A8 は明示選択だけにする必要があった。
- `SQ8_0` / `AQ4_0` の artifact、candidate、release、campaign、authorization、active manifest は変更対象外だった。

## 今回の変更点

- canonical `SQ8_0` v0.2 を検証して row-major F32 に再構成する、独立した `SQ8_1` packer を追加した。出力は `sq8_1_manifest.json`、16-byte aligned I8 payload plane、little-endian F16 scale plane であり、`SQ8_0` payload decoder を共有しない。
  - K=32、`[-127,127]`、zero-point なし、RNE/ties-to-even、`ceil_fp16`、zero block の scale=1.0、tail padding zero を検証する。
  - row compensation は `quantizer-row-compensation-plan-v0.1.md` と整合し、payload に焼き込まず format-external のままにした。
- Rust reader/reference と runtime C ABI を追加した。W8A16 (`ullm_runtime_sq8_1_matvec_w8a16_f32`) が既定経路で、W8A8 は `ullm_runtime_sq8_1_matvec_w8a8_explicit_f32` の明示 API だけである。いずれも payload stride、plane length、tail、有限正 scale を fail-closed で検証する。
- HIPRTC reference kernel は K=32 full block を二つの aligned `uint4` load で読む。W8A8 は block ごとに I32 dot と `s_w*s_a` を適用する。gfx1030/CDNA は `sdot4`、RDNA3/RDNA4 は signed controls 付き `sudot4` を使い、後者は disassembly で `v_dot4_i32_iu8 neg_lo:[1,1,0]` と現れる。
- offline compiler audit は runtime の HIPRTC raw source 自身を抽出して、whitelist 全五 target（gfx1030/gfx1100/gfx1201/gfx942/gfx950）で実行した。W8A8 は gfx1030/gfx1100/gfx1201 とも VGPR 53、SGPR 59、LDS 1024 B、private/spill 0。opcode は gfx1030 が `v_dot4c_i32_i8`、gfx1100/gfx1201 が `v_dot4_i32_iu8`、gfx942/gfx950 が `v_dot4c_i32_i8_e32` だった。W8A16 は同じ source で dot なし、private/spill 0 を確認した。
- V620 GPU differential は BDF `0000:03:00.0` を HIP API で選び、同一 BDF の `card0` / junction `temp2_input` を読んだ。8回ずつの W8A16/W8A8 launch の全測定は 41–42 °C、85 °C guard を超えなかった。CPU reference との差は W8A16 relative L2 `6.076546605e-08`、W8A8 relative L2 `4.333164297e-08`、max abs は両方 `0.0078125` だった。
- verified `SQ8_0` source の Qwen3-14B `model.layers.0.self_attn.k_proj.weight`（5,242,880 values）を pack/read した。source 再構成値に対する weight relative L2 は `0.005592543546739809`、max abs `0.0017452239990234375`、post-storage clipping 0。これは single-tensor evidence であり、BF16/full-model logits evidence の代替ではない。

## 検証と非干渉

- Python packer tests 5/5、Rust SQ8_1 reader/reference tests 4/4、CPU runtime API tests 2/2、canonical SQ8_0 reader tests 14/14、format-ID/SQ8 policy Python tests 13/13 が通過した。
- Python が生成した実 artifact を Rust reader が checksum/shape/stride/plane を含めて検証した。
- `SQ8_1` の manifest、reader、runtime ABI、format-ID は sibling namespace であり、legacy `sq` / `sq-fp8` は引き続き `SQ8_0`。`AQ4_0` の artifact/ABI/dispatch は変更していない。
- `/opt/ullm` は読み取りのみ、`/etc/ullm/served-models/active.json`、service unit、candidate/release/campaign/authorization、remote repository は変更していない。

## 次の行動

1. BF16 基準の full weight-plus-activation logits gate を、事前に閾値を固定した held-out corpus で実行する。それまで W8A8 は runtime/artifact/release selection に採用しない。
2. W8A16 を default のまま維持し、architecture-specific WMMA/MFMA/private optimization は同じ public ABI の後ろに追加する場合だけ別途検証する。
3. gfx1201/gfx942 の device differential と matched performance は未確認である。hardware 固有の dispatch/promotion は、その numerical/performance evidence が揃うまで行わない。
