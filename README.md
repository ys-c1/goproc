# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化、PGM 导出与点积光照（前向差分梯度、朗伯着色、偏度矩）
- `cmd/goproc/` 命令行，`gen` 导出高度图，`shade` 导出三档光向的梯度/光照存档
- `analysis/` 独立 Python 复核与出图；证明与数值结论见 `REPORT.md`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc shade --size 256 --seed 7 --outdir artifacts/shade
python3 -I analysis/shading_analysis.py .
```
