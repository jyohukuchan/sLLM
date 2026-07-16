SQ8 calibration は固定 v4 plan で 24/24 rows を一回だけ取得し、target を publish 済みです。
service は公式 poller で正常復旧し、target の SHA256SUMS と strict validator は通過しました。
GPU 再実行と holdout を禁止し、同じ target の offline metrics・freeze・ledger 固定を進めています。
