# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/edge/` 边缘过渡带：C¹ 连续且 |二阶导| ≤ M 约束下的最小宽度反解
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc gen --size 128 --edge-m 0.01 --out map.pgm  # 边缘过渡，宽度自动解算
```

边缘过渡曲线为对称抛物线（bang-bang 最优）：灰度变化量 Δ、二阶导上限 M 时
最小宽度 W = 2·√(Δ/M)，默认 M=0.01、Δ=1 得 20 px，与画幅无关。宽度再小则
任何 C¹ 曲线的 max|二阶导| ≥ 4Δ/W² > M，灰度二阶差分必然出现跳变。
