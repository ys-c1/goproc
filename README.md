# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化、分布拉平（经验累积分布单调映射）与 PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/histreport/` 分布对比报告，输出三档层数的分箱方差、映射曲线与直方图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/histreport
```
