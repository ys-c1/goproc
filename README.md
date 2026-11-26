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

## 置换表与重复山脊

默认已修复为**每层派生独立置换子表**（`noise.LayerSeed(seed, i)` + `fbm.EvalLayered`），
消除相邻层共用一张置换表导致的尺度同源重复山脊。需要复现旧缺陷时：

```
go run ./cmd/goproc gen ... --shared-perm --out shared.pgm
```

周期证明、重复片定位与三档层数（2/4/6）比对见
[`REPETITION_AUDIT.md`](REPETITION_AUDIT.md)；一键复现标记图、坐标与相关指标：

```
make audit
```

产物在 `artifacts/repetition/`（`marked_K2/K4/K6.png`、`COORDINATES.md`、
`repeat_coordinates.csv`、`repeat_metrics.json`）。

