# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出（八位与十六位）
- `internal/cliff/` 陡崖倒角双约束模型（几何限制与灰度量化限制）及过渡带实测
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/bevel/` 倒角宽度实验，输出交点、对照表与演示 PGM 到 `artifacts/`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/bevel artifacts
```
