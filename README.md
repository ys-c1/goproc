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

## 冲沟核心格预判与逐格核对

`internal/hydro` 做确定性 D8 流向/坡度/汇流累积；`internal/gully` 用两个阈值的合取判定冲沟核心格：

```text
Core(c) = [Acc(c) > 32] ∧ [Slope(c) > 0.030]
```

核心格的八邻域连通段由 BFS 计数。`cmd/gully` 在 seed=7、128×128 地形上跑弱（K=0.0015）、强（K=0.0045）两组侵蚀，把侵蚀前预判核心格与侵蚀后按前后高程差逐格统计的实测核心格全量比对，输出不一致格坐标、两张冲沟标注图与汇总表：

```
make gully   # 等价于 go run ./cmd/gully，产物在 artifacts/gully/
```

证明与适用边界见 `docs/gully-core-proof.md`，结论摘要见 `artifacts/gully_report.md`。

