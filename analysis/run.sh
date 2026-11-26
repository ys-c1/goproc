#!/usr/bin/env bash
# 频谱验证全流程：Go 真实生成器导场 -> Python FFT/LS/绘图/反解 ->
# 用反解出的 g_req 再导验证场 -> 输出最终 summary 与图表。
# 产物：artifacts/ 下 report.md 之外的 summary.json/.csv 与三幅 PNG。
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GO="${GO:-go}"
PY="${PY:-python3}"

# 1) 三档增益 × 4/8 层 × 16 种子，1024²（约 385MB 原始 float32，build/ 已 gitignore）
"$GO" run ./cmd/dumpfields \
  -out build/fields1024 -size 1024 -seeds 16 \
  -gains 0.35,0.50,0.65 -octaves 4,8

# 2) 主分析：两幅频谱图、地形对照、summary，并写出反解增益
"$PY" analysis/spectrum_analysis.py \
  --main build/fields1024 --out artifacts --write-target build/target.json

# 3) 按给定粗糙度 rho*=0.15 反解出的增益生成独立验证场
GREQ="$("$PY" -c 'import json;print(json.load(open("build/target.json"))["g_required"])')"
echo "==> inverted gain g_req=${GREQ}"
"$GO" run ./cmd/dumpfields \
  -out build/verify -size 1024 -seeds 16 \
  -gains "${GREQ}" -octaves 8

# 4) 汇总验证结果并复写 artifacts/summary.{json,csv} 与图表
"$PY" analysis/spectrum_analysis.py \
  --main build/fields1024 --verify build/verify \
  --target build/target.json --out artifacts

echo "==> done: 见 artifacts/report.md、summary.json、summary.csv 与 spectrum_*.png / terrain_gains.png"
