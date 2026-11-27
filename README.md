# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/orient/` 结构张量、`½atan2` 主方向角、张量/像素场旋转与接缝条带
- `internal/field/` 注入已知主方向的平稳定向纹理
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/stitch/` 接缝主方向旋转对齐实验，方法与数据见 `artifacts/STITCH_ALIGNMENT.md`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/stitch --sweep          # 两组拼接方向的角差、像素数与误差界
```
