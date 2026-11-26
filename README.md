# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/stretch/` 带方向的剪切拉伸（`Warp{Theta, Ratio}`）
- `internal/hmap/` 归一化与 PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/genbatch/` 批量生成两组拉伸实验用 PGM
- `analysis/` 独立 Python 分析（梯度加权直方图、结构张量、交叉验证）

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm

# 带方向拉伸：theta 为拉伸方向角（度，正东为零逆时针），ratio 为轴比 r
go run ./cmd/goproc gen --size 512 --seed 20261127 --theta 35 --ratio 2 --out map.pgm
```

## 拉伸与主脊方向

拉伸在采样坐标上施加剪切 `M = R(θ)·[[1,r],[0,1]]·R(-θ)`，即沿拉伸方向
θ 的点不变、垂直方向的点被沿 θ 位移 r 倍。主脊（梯度主方向）的理论角为

```
θ_ridge = θ + atan(r)        （方向角，正东为零、逆时针，结果折回 [0,180°)）
```

推导、逐拉伸比的预判/实测比对与超 5° 偏差说明见 `analysis/REPORT.md`。
分析脚本以隔离解释器运行：

```
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install matplotlib
go run ./cmd/genbatch artifacts/pgm
( cd analysis && .venv/bin/python -I analyze.py )
```

