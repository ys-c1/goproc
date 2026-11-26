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

## 频谱验证与衰减反解

证明分形叠加功率谱的双对数斜率 `s* = 2 ln gain / ln lacunarity`、
斜率与视觉粗糙度的一一对应，并由给定粗糙度反解 gain；三档增益 × 4/8 层比对。

```
./analysis/run.sh
```

结论、推导、图表与单行宽表见 `artifacts/report.md`（图：`artifacts/spectrum_gains.png`、
`artifacts/spectrum_octaves.png`；数据：`artifacts/summary.{json,csv}`）。
`cmd/dumpfields` 负责把真实 `api.Generate` 高度场导出为原始 float32 供 Python 分析。
