# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

## 坡度分级与谱斜率预判

按坡度对地形分级着色，并在着色前用功率谱斜率预判每个色阶的面积占比
（面积占比 = P(a≤S<b) = F(b)−F(a)；谱 P(k)∝k^(−β) 给出梯度方差，
高斯场下坡度模服从 Rayleigh）。完整证明、区间划分与逐阶核对见
[`artifacts/slope_prediction_report.md`](artifacts/slope_prediction_report.md)。

```
make slope-demo
```

该目标生成 2048²、seed=7、6 层、frequency=0.01 的 float64 高度场，再运行
`analysis/slope_forecast.py`，产物写入 `artifacts/`：

- `slope_map_K5.png` / `slope_map_K10.png`：5 色阶与 10 色阶着色图（边界嵌套）
- `slope_table_K5.{md,csv}` / `slope_table_K10.{md,csv}`：两套预判 vs 实测占比表，`偏差>2pp` 的色阶显式标记
- `pred_vs_meas.png`：逐阶比对柱状图；`spectral_derivation_check.png`：谱拟合与 Rayleigh 检验
- `analysis_metrics.json`：全部中间数值

CLI 另支持 `--frequency` 与 `--format f64`（小端 float64，避免 8 位 PGM 量化污染坡度）。

