# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/bathy/` 水深量化与两种“水深→色阶”映射的逐点对照
- `internal/orient/` 结构张量主方向角与旋转角反解
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/analyze/` 色阶映射对照与旋转角分析，结论见 `artifacts/analysis.md`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/analyze artifacts
```
