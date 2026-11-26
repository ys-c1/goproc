# goproc

程序化地形生成，确定性种子派生；并在着色前由功率谱预判法线朝向分布。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/dft/` 基二 FFT（标准库实现，含对定义式 DFT 的校验测试）
- `internal/normal/` 中心差分梯度、单位法线与方位角直方图
- `internal/spectrum/` FFT 谱矩、径向谱斜率 β、联合高斯闭式扇区概率
- `internal/viz/` 切线空间法线图与八方位着色图（PNG）
- `internal/analyze/` 谱预判→法线实测→逐箱 3pp 标注的完整流程
- `cmd/goproc/` 命令行

```
make test

go run ./cmd/goproc gen     --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc normal  --size 128 --seed 7 --octaves 4 --outdir artifacts
go run ./cmd/goproc analyze --size 128 --seed 7 --octaves 4 --outdir artifacts
```

`analyze` 在 `artifacts/` 写出两份占比表（8 箱 ×45°、16 箱 ×22.5°）、
两张可视化图与汇总 JSON。数学推导与验证结论见
[`artifacts/PROOF.md`](artifacts/PROOF.md)：法线方位角落在某区间的概率
等于梯度落在对应扇区的概率，而该扇区概率可由谱斜率与二维谱矩在着色前算出；
偏差超过 3 个百分点的区间在 CSV 与 JSON 中标出。
