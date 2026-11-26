# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/axis/` 相位锁定有限层、镜像相关、相位和轴预测与逐像素确认
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/axisdemo/` 两个画幅的相位预判/逐列相关逐图比对与存档
- `docs/axis-proof.md` 镜像相关极大性与相位和公式的证明

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/axisdemo --out artifacts/symmetry
python3 analysis/make_axis_charts.py
```

`axisdemo` 在 `artifacts/symmetry/` 产出两画幅轴号两栏简表 `axis_two_columns.md`、并排表 `axis_wide.csv`、
一图一行且带物理行号的 `axis_compare.csv`、各层相位表 `phase_layers.csv`、
每画幅一张相关曲线图 `correlation_A.png` / `correlation_B.png`，以及证明所需的
`report.txt` 与 `results.json`。轴列号均从零起算。
