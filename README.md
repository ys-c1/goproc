# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/shadow/` 地形阴影：逐格步进基准 + 包络一次扫掠（证明见 `docs/shadow-proof.md`）
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图，`shadow` 子命令比对两种阴影算法

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc shadow --size 128 --seed 7 --az1 45 --el1 30 --az2 300 --el2 15 --outdir artifacts
```

`shadow` 对两个光照角（各给方位角与高度角）分别输出：两法阴影格数、逐行
阴影/差异计数、不一致格坐标与两侧高度、误差率，并存档阴影图
（`*_march.pgm` / `*_sweep.pgm` / `*_diff.pgm`）。包络计算步长为一格。
