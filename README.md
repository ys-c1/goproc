# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/fog/` Beer–Lambert/Koschmieder 雾衰减、高度分布预判与灰度渲染
- `cmd/goproc/` 命令行，`gen` 导出高度图，`fog` 跑雾可见比例实验

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
make fog   # 两档雾浓度，产物在 artifacts/fog/（含带雾图、REPORT.md、results.json）
```

## 雾实验要点

可见判据 `C=exp[-βR(1-h)] > 0.05`，截断高度 `h > 1 - (-ln0.05)/(βR)`，
因此可见比例由乘积 `βR` 与高度分布 CDF 共同决定，可在渲染前由高度分布直接预判。
详见 `artifacts/fog/REPORT.md`。
