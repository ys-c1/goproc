# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/comb/` 多层叠加梳齿（同相列）的受控反例、条件不等式与定量检测
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/combexp/` 梳齿实验，导出三组 PGM、`report.json` 与相位扫描
- `artifacts/comb/FINDINGS.md` 理论推导与实验结论

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/combexp --out artifacts/comb
python3 -I scripts/visualize.py
```
