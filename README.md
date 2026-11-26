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

## 洼地深度阈值预判（填洼复核）

四邻域严格局部极小计洼地，深度按灰度差（minimax 溢流高程 `S` 减坑底灰度 `G`）。
证明 `strictMin4(min(S,G+τ)) == {m∈strictMin4(G): Δ(m)>τ}` 且坐标不变，
故可只算一次溢流面就按 τ 预判剩余洼地数与位置，无需迭代填洼。规范地形
（128/seed7/o4）共 21 个极小：τ=20 剩 8 个、τ=40 剩 6 个，与填洼后逐格扫描
逐个比对全部 `d∞=0`，无偏差超一格者。

- 证明：[`docs/depression-proof.md`](docs/depression-proof.md)（副本 `artifacts/depression_proof.md`）
- 两档一行结论 / 明细：`artifacts/depression_rows.md`、`artifacts/depression_compare.csv`
- 标注图：`artifacts/depression_map_tau20.png`、`artifacts/depression_map_tau40.png`
- 复现：`make dep`（Python 分析需 `numpy`、`Pillow`；Go 侧 `internal/depress` 独立复核）

