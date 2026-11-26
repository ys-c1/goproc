# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/normalmap/` 高度图→法线、b 位量化、角度误差界/反解、斜坡台阶计数
- `cmd/goproc/` 命令行：`gen` 导出高度图，`normalreport` 做法线位深实验并归档

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

## 法线贴图位深与斜坡台阶

6/8/10 位 unorm 法线的角度量化步长约为 `Δθ ≈ 2√(1+g²)/(2^b−1)`，
严格误差界 `ε ≤ 2√(1+g²)/(2^b−1)`（弧度，b≥3），按界反解最小位深
`b_min = ceil(log2(1 + 2√(1+g²)/ε_rad))`。

```
make normalreport     # = go run ./cmd/goproc normalreport
```

产物在 `artifacts/normal_quant/`：三档位深各自的法线贴图（PPM 原码 + PNG）、
着色台阶图、`normal_groups.json`（误差与台阶数同组）和带完整证明的
[`REPORT.md`](artifacts/normal_quant/REPORT.md)。`go test ./...` 会逐像素断言
三档位深的角度误差都在界内，并核验反解位深的最小性与归档 PNG 的台阶数。
