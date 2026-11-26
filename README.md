# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/flow/` D8 八邻域流向判定、一次性预判与逐格追踪
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/flowdemo/` 两个画幅的流向预判/追踪逐格比对与存档
- `docs/flow-proof.md` 流向规则的数学证明、离散规则与验证方法

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/flowdemo --out artifacts/flow
```

`flowdemo` 产出（默认 `artifacts/flow/`）：

- `report.txt`：两画幅的出口数、出口/内陆汇逐行坐标、零坡度格统计、预判与逐格追踪的逐格比对结果（不一致会报坐标）
- `flow_*.pgm`：流向图，灰度映射 `dir*32+1`（0 为 Sink），规则写在 PGM 头注释，可逆解码
- `height_*.pgm`：八位高度图，仅供可视化
- `height_*.f64`：原始 float64 高度（小端行优先），供无损独立复算流向

