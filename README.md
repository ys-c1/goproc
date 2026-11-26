# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处；`Generate` 返回归一化到 [0,1] 的图，
  `Raw` 返回未归一化的场（倾角谱模型按原始尺度描述）
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/tilt/` 平均倾角的谱斜率预判与实测统计
- `internal/report/` 预判/实测逐档比对与分布图存档
- `cmd/goproc/` 命令行，子命令 `gen`、`overview`、`tilt`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm

# 只算概览，不生成地形
go run ./cmd/goproc overview --freq 0.03 --octaves 5 --gain 0.5

# 两组参数的预判/实测逐档比对，产物写入 artifacts/tilt/
go run ./cmd/goproc tilt --size 512 --report artifacts/tilt/report.md
```

平均倾角的推导、Nyquist 修正与两组参数的比对见 `docs/tilt-overview.md`。
