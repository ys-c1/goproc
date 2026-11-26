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

## 可通行坡度 × 谱斜率 × 面积幂律

完整证明与逐阈值核对见
[`artifacts/slope_powerlaw_report.md`](artifacts/slope_powerlaw_report.md)。
结论链：谱 `P(k)∝k^(−β)` 经求导谱 `k²P` 决定梯度方差 `σ_g²(β,K)`（离散八层和），
CLT 下坡度模服从 `Rayleigh(σ_g)`，故可通行面积
`A(θ)=1−exp(−θ²/2σ_g²)≈θ²/(2σ_g²)`——小坡度段是指数 **2** 的幂律；
有限窗口双对数最小二乘有效指数约 1.90–1.97（拟合区间 `u=θ/σ̂β∈[0.10,0.40]`）。
三组地形（G1 基准 / G2 改谱斜率 / G3 加层数）各出两档与四档阈值，
实测面积、LS 幂律预测、β 先验预测逐行核对；故意估错 β 会使阈值行成片超差。

```
make slope     # 生成三组 2048² f64 地形并产出报告、坡度图与逐行表
```

- 报告：`artifacts/slope_powerlaw_report.md`；逐行数据：`artifacts/slope/threshold_rows.csv`
- 三组分别存档的坡度图：`artifacts/slope/slope_map_G{1,2,3}.png`
- 两档/四档可通行掩膜：`artifacts/slope/walk_G{1,2,3}_{2tier,4tier}.png`
- CLI 另支持 `--frequency/--gain/--lacunarity`、`--format f64`（小端 float64）与 `--raw`（跳过归一化）。
