# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

地形曲率预算与逐块简化对照：

```
make budget
```

结果（证明、逐块/逐行列 CSV/Markdown 表、两档误差的简化结果图）写入
`artifacts/terrain_budget/`，从该目录的 `README.md` 开始查看。
