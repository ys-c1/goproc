# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化、PGM 导出与 P5 解码
- `internal/resample/` 两个归一化采样核（box / triangle）的低通下采样与最近邻放大
- `internal/fog/` 雾化度量：同尺寸对齐后的连续加权窗局部方差比 F=1−R
- `internal/calibration/` 五档标定、log-log 分段曲线、幂律拟合、在线预判与反解
- `cmd/goproc/` 命令行：`gen` 导出高度图，`calibrate` 标定，`predict` 预判/反解
- `scripts/plot_curves.py` 读取标定 JSON 绘制雾化曲线

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm

# 标定（五个缩放比 s=2^(-k/2)，两个采样核）
go run ./cmd/goproc calibrate --size 512 --seed 7 --octaves 4 --window 17 --outdir artifacts/calib

# 在线预判：只输入缩放比，不生成缩略图
go run ./cmd/goproc predict --calib artifacts/calib/calibration.json --kernel box --scale 0.55

# 曲线反解：给定雾化程度求缩放比
go run ./cmd/goproc predict --calib artifacts/calib/calibration.json --kernel triangle --fog 0.5

# 绘图
python3 scripts/plot_curves.py artifacts/calib/calibration.json artifacts/calib
```

标定方法、取点位置与逐档偏差对照见 `artifacts/calib/REPORT.md`。
