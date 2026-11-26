# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处；`RectOptions`/`GenerateRect` 支持各向异性矩形网格
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/sampling/` 按拉伸后局部频率反解各向异性奈奎斯特步长
- `internal/hmap/` 归一化与 PGM 导出（含矩形）
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/exp/` 拉伸噪声采样实验，落盘 float64 原始场与两版 PGM
- `analysis/` 椭圆频谱证明（`theory.md`）与 FFT/保留度分析脚本

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

各向异性奈奎斯特采样实验（理论与实测解读见 `analysis/theory.md`）：

```
go run ./cmd/exp --out artifacts/exp
python3 -I analysis/analyze.py artifacts/exp
```
