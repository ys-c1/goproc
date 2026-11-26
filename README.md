# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/erosion/` 热侵蚀：守恒版 `Step` 与定点量化对照版 `NaiveStep`
- `cmd/goproc/` 命令行，`gen` 导出高度图，`erode` 跑守恒实验并存档
- `docs/conservation.md` 守恒证明、舍入误差公式与实验数据

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc erode --size 128 --seed 1 --octaves 4 --steps 10,50,200 \
    --dir artifacts/erosion
```

侵蚀算子只搬运不增删物质：同一条边的通量从授格扣除、原样加到受格，
流入恒等于流出，三个迭代步数（10/50/200）下归一化总高差均低于千分之一。
逐格独立取整的定点实现会系统性丢物质，误差公式与对照实验见
`docs/conservation.md`。
