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

## 分层设色面积/宽度预判（analysis/）

出图前用等值线积分预判每条色带的面积与平均宽度，证明与逐带实测比对见
`analysis/REPORT.md`（coarea 恒等式 A′(h)=∮ds/|∇u|，宽度=面积/等值线总长）。

```
go run ./cmd/dumpheight --size 128 --seed 7 --octaves 4 --out artifacts/data/height_s7_n128.bin
python3 -I analysis/selftest.py        # 恒等式机器精度自检
python3 analysis/run_analysis.py       # 出 CSV、summary 与四张图
python3 -I analysis/make_tables.py     # 两栏逐带对比表
```

- `analysis/terrain_bands.py`：三角剖分 marching-triangles、闭式积分、四邻域连通（纯 NumPy）
- 面积按四邻域连通计；8 带无超差，16 带仅两个极值带偏差超一成

