# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/blend/` 两层混合：线性/smoothstep/数据相关最优权重，棱能量与理论下界
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/blendexp/` 接缝混合实验（三档宽度 × 三种权重），`render.py` 渲染对照图
- `docs/blend_theory.md` 权重泛函的推导、EL 方程与 Cauchy 下界

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/blendexp --size 256 --seed 7 --widths 8,16,32 --out artifacts/exp
python3 cmd/blendexp/render.py --exp artifacts/exp --out artifacts/exp/figs
```
