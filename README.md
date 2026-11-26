# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/erode/` 热侵蚀、坡度场与中位数/方差统计
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/erodestudy/` 侵蚀实验，验证坡度中位数下降量 ∝ 轮数×方差并出报告与累计分布图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/erodestudy artifacts   # 报告与 SVG 写入 artifacts/
```
