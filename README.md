# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/perlin/` Perlin 梯度噪声（谱能不沿网格轴泄漏，各向同性场 HOG 无假峰）
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/aniso/` 多分量各向异性地形：相干 `Field` 与非相干能量合成 `FieldRMS`
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/anisogen/` 生成 A/B 两档各向异性地形与各向同性标定场
- `scripts/` 谱预判、HOG 双阈值实测与按档比对（Python，`-I` 隔离运行）

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm

# 各向异性地形：谱峰数预判 vs 梯度方向直方图实测
for p in iso A B; do go run ./cmd/anisogen --preset $p; done
python3 -I scripts/test_thresholds.py        # 双阈值（高度+间距）单元测试
python3 -I scripts/run_analysis.py           # 预判/实测比对 -> artifacts/results/report.json
python3 -I scripts/make_plots.py             # 每组一张 HOG
python3 -I scripts/make_spectrum_plots.py    # 谱支撑图
```

分析方法、两个命题的证明与按档比对见 `artifacts/results/ANALYSIS.md`。

