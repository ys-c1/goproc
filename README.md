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

## 等高线开闭判定（无追踪预判）

`internal/contour/` 在四邻域网格上证明并验证：等值线闭合当且仅当它不触及边界；
开曲线条数 O=B/2（B 为边界边跨越次数，必为偶数），闭合曲线条数可用两种不追踪
曲线的独立方法预判——区域填充 L=(c−b)+holes 与逐格局部欧拉计数 L=2c−b−χ。
证明、约定（边界格/边界边、四邻域 vs 八邻域）与逐档比对见
`artifacts/contour/ANALYSIS.md`。

```
go run ./cmd/contour --size 128 --seed 7 --octaves 4 --outdir artifacts/contour
python3 scripts/cross_validate.py   # 与 scikit-image find_contours 逐项交叉核对
python3 scripts/plot_contours.py    # 两个等高距的等高线图分别存档
```
