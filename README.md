# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出（八位与十六位）
- `internal/band/` 量化等高带统计（同灰度四连通段）
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/bandscan/` 带数扫描，逐位深比对预判与实测，见 `docs/banding-proof.md`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```
