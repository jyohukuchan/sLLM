# R9700 idle clock pinning root cause

Date: 2026-07-26

## 前回の要点

- R9700（gfx1201、PCI `0000:47:00.0`、DRM `card2`）はアイドル時にも
  `gpu_busy_percent=100%`、コアクロック`3414 MHz`、約`98 W`、junction
  `85°C`となることがあった。
- 原因は比較用baselineの`llama-qwen35-udq4.service`（llama.cpp
  `llama-server`、Qwen3.5-9B UD-Q4_K_XL）である。`--no-warmup`指定かつ
  llama.cppログが`all slots are idle`でも、モデルロード完了の`6–9`秒後に
  busy `100%`、コアクロック`3443 MHz`、約`82 W`へ戻った。
- 双方向の切り分けでは、停止`10`秒後にbusy `100% → 3%`、sclk
  `3414 MHz → 41 MHz`、power `96 W → 16 W`、junction `85°C → 66°C`、fan
  `1059 → 893 rpm`となった。
- これは実workloadではない。`amdgpu_fence_info`の全ringで`Last signaled`と
  `Last emitted`が一致し、`mem_busy_percent=0`、memory clockは下限`96 MHz`
  （deep sleep有効）、llama.cppとuLLMの両processのCPU消費は`0`、`300 W`枠の
  cardが最大boost時にも約`98 W`しか引いていない。
- `ullm-aq4-worker`だけならKFD queue `3`本とVRAM `6.6 GB`を保持してもbusyは
  `3%`のままだった。単なる常駐KFD queueではなく、llama.cppのHIP backend固有の
  挙動である。ヘッドレスGNOME（gdm3停止済み、DP connector `4`本とも
  disconnected）、`lactd`、fence/ring滞留は除外済みである。

## 今回の変更点

- ユーザー決定に従い、`llama-qwen35-udq4.service`を常時起動する比較対象から、
  必要時だけ手動起動するbaselineへ変更した。`systemctl stop`の後、boot自動起動を
  `systemctl disable`した。
- 実行後`10`秒の今回の確認値は以下のとおり。`systemctl is-active`は`inactive`、
  `systemctl is-enabled`は`disabled`だった。

| sysfs項目 | 今回の読み取り値 |
| --- | --- |
| `gpu_busy_percent` | `3%` |
| `pp_dpm_sclk`の選択state | `1: 41 MHz` |
| `power1_average` | `17000000 µW`（`17 W`） |
| `temp2_input`（junction） | `66000 m°C`（`66°C`） |
| `fan1_input` | `892 rpm` |

- RDNA3/4 SMUのfan制御は正常である。`gpu_od/fan_ctrl/fan_target_temperature`は
  `85°C`、acoustic targetは`2100 rpm`、junction criticalは`110°C`であり、
  junctionが`85°C`のとき約`1050 rpm` / PWM `31%`なのは目標温度を維持する設定どおり
  の挙動である。
- P3のGPU windowは従来`ullm-openai.service`だけを停止していた。そのためP3の
  prefill `982 tok/s` / decode `56.6%`はすべて、R9700がjunction `85°C`、
  コアクロック最大固定、llama-serverがVRAM `5.3 GB`を占有した条件で得られた。
  固定クロックによりラン間のばらつきは小さく相対比較の一貫性は保たれる一方、
  絶対値の再現性にはこの熱条件の注記が必要である。
- `/etc/lact/config.yaml`のGPU key
  `1002:7551-1DA2:E499-0000:23:00.0`は現在のPCI address
  `0000:47:00.0`と不一致である。保存済み`power_cap 250 W`等は適用されず、
  実際のcapは`300 W`である。この既知事項は記録のみとし、今回は修正しない。
- `gdm3`はinactiveのまま、`lactd`はactiveのままとした。`/etc/ullm/served-models/active.json`、
  `ullm-openai.service`、systemd unit内容、SQ8_0/AQ4_0のcampaign・authorization・
  candidate・release、`/opt/ullm`には変更を加えていない。GPU benchmark、モデル
  load、計測の実行も行っていない。

## 次の行動

- llama.cppとの比較が必要なときだけ`sudo systemctl start
  llama-qwen35-udq4.service`を実行し、完了後に`sudo systemctl stop
  llama-qwen35-udq4.service`を実行する。
- すべてのR9700 GPU計測窓では、uLLM service windowやR9700 lockの前に
  `llama-qwen35-udq4.service`を停止し、`systemctl is-active`が`inactive`である
  ことを記録する。
- 次のGPU測定では、停止済みbaselineのtemperature、core clock、power、VRAM条件を
  evidenceへ残し、過去P3値との絶対比較にはこの熱条件差を添える。
