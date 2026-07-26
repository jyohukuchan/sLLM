# SQ8_0 手書き projection の累積契約診断

## 結論

private gfx1201 hand-written WMMA projection は **NO-GO 継続** とした。
default の CK dispatch、active manifest、campaign、authorization、release は
変更していない。candidate timing は実施していない。

今回の wave32 route については「CK の契約を守りながら速くできるか」への
回答は **現時点では no** である。契約を守れていないため速度候補ではない。
全ての手書き実装に余地がないことまでは **未確認** であり、CK の exact
fragment/load/issue mapping を再現できる別実装の可否と速度は未測定である。

## 発散箇所

- frozen gate と同じ raw-p0512、512 token、M8 chunk prefill の後、first
  feedback M=1 decode を40 layer全段階で採取した。両 route の prefill token
  は 66、decode 入力 token は 66、position は 512 だった。
- layer 0--2 は全 stage bit一致だった。最初の差は layer 3 の
  down_projected: 2 / 5,120、first index 1,954、max abs 6.1035156e-5。
  layer output も同じ2要素だけが異なる。
- その actual activation / artifact を使った down projection
  (M=1, N=5,120, K=17,408) の direct replay は各 route の layer trace と
  一致し、CK 対 handwritten は同じ2要素差だった。
- 以前の component gate 4/4 は synthetic one-projection と BF16 boundary
  だけを見ていた。actual layer-3 activation の136 K128 block、serving
  stage sequence、feedback quantization を検証していなかった。token一致が
  gate として不十分であることを再確認した。

## 累積契約

- cumulative K128 prefix は単調ではない。prefix 1--5 は一致、prefix 6 が
  最初の差、prefix 8 は再び一致、full prefix は不一致だった。この
  cancellation の理由は **未確認**。
- K128単体にすると block 1 (K=128--255) ですでに 1 / 5,120、
  output 1,986、max abs 9.536743e-7 の差が出た。したがって K128 間の
  scale accumulation order だけが原因ではない。
- block 1 内の K16 prefix は 1--7 がbit一致し、8番目
  (offset 112--127) を加えた時点で初めて同じ1要素差が出た。
- one-hot lane probe は K lane 0--15 と first output tile の16/16で一致。
  gross transpose/lane error はこの狭いprobeでは否定できるが、opaque
  fragment mapping 全体を証明するものではない。
- CK source は K128 raw accumulator を clear → XDL/WMMA → scale掛けで
  FP32 Cに加算する。手書きも同じ高水準のscale timingである。一方 CKは
  256-thread 16x128x256 tile、手書きは32-thread N=16 waveであり、CK object
  ではWMMAとFP32 FMACのregister issueがinterleaveしている。

よって確定したのは「K128 内（8番目 K16 を含めた時点）に契約差がある」こと。
それが final K16 operand/fragment mapping か WMMA reduction/issue
association か、または両方かは **未確認**。この状態で一致化を推測実装する
ことはしなかった。

## service window / 非干渉

- stop/isolate/restore は3回。attempt-1 は valid だが layer-0 限定で
  inconclusive。attempt-2 は 09:12:48 に service がstartした後も
  09:13:06--08 に diagnostic artifact が書かれたため invalid とし、
  結論から除外した。
- valid authority は attempt-3 のみ。09:19:29 stop、09:20:45 diagnostic
  complete / service single-start、09:20:46 active/running、NRestarts=0。
- R9700 AMD SMI GPU 2、gfx1201、BDF 0000:47:00.0 のみ使用し、
  HIP_VISIBLE_DEVICES=1 を指定した。V620 は使用していない。
- llama-qwen35-udq4.service は inactive/disabled、gdm3 は inactive を
  preflight/finalで確認した。
- telemetry endpoint は edge/hotspot/memory 38/38/36 C → 46/47/46 C、
  gfx 2,833 → 49 MHz、socket power 16 → 14 W。post-stopにTHROTTLEDを
  記録したが、物理原因は **未確認**。このため timing主張はしていない。
- active manifest、campaign、authorization、systemd unit、power cap、
  /opt/ullm、remote は変更していない。

証跡は benchmarks/results/2026-07-26/sq8_0-projection-contract/ に保存した。
