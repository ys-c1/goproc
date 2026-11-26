# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/hydro/` 水位—连通分量分析：按高度的并查集扫描预测、独立 BFS 实测、临界点分类
- `internal/render/` 水陆 PNG 图渲染（四邻域水陆面 + 极小/鞍点/本档跳变标记 + 图例）
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/hydro/` 命令行，做水位分量预测、逐档 BFS 核对并导出 CSV/报告/水陆图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/hydro --size 128 --seed 7 --octaves 4 --bands 21 --out artifacts/hydro_run
```

## 水陆连通分量分析

约定（全项目统一）：水 \(W(w)=\{p:h(p)\le w\}\)；连通取**四邻域**；**网格外不算水域、不环绕、无外部海洋**，边界格点按普通格点处理。

核心结论与证明见 [`docs/theorem.md`](docs/theorem.md)：

1. 分量数是右连续阶梯函数，只在水位越过临界台地时变化——极小台地 \(K=0\) 使分量 \(+1\)，鞍点台地 \(K\ge2\) 使分量 \(-(K-1)\)，普通台地 \(K=1\) 不变；跳变与临界点一一对应。
2. 一次按高度的并查集扫描 \(O(n\log n)\) 产出事件表与分量数阶梯，之后任意水位 \(O(\log n)\) 查表，无需重复连通标记。
3. 每个水位都用独立 BFS 重新标记做核对（只核对、不参与预测），并校验恒等式 \(\#\text{极小}-\sum(K-1)=1\)。

`cmd/hydro` 输出：

- `levels.csv`：各展示水位的预测分量、BFS 实测分量、本档极小/鞍点数、净变化；
- `jumps.csv`：全部临界台地逐行（坐标、高度、类型、\(K\)、\(\Delta\)、四邻 pattern、几何鞍点/局部极大对照）；
- `report.md`：逐档对照表、跳变点逐行表、结构核对；
- `maps/level_XX.png`：每档一张水陆图，标出全部已跨过临界点并用红环标注本档刚跨过的跳变。

