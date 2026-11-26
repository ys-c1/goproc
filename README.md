# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生，表长即格点周期
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/seam/` 平铺接缝度量
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/tilecheck/` 三档周期接缝校验，产出留档

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm

# 可平铺：第零层横跨 256 个格点，周期 256，其余层按 lacunarity 自动整除
go run ./cmd/goproc gen --size 512 --seed 7 --octaves 3 --cells 256 --period 256 \
  --repeat 2 --out tile_2x2.pgm

# 三档周期校验，写 artifacts/
bash artifacts/verify.sh
```

平铺的充要条件（每一层「频率 × 边长」都能被置换表周期整除）及其证明、相位差与
接缝跳变的推导见 `artifacts/seam_report.md`。
