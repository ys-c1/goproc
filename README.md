# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/nrm/` 高度转法线的差分核推导与反解
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图

```
make test
make norm
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

`internal/nrm` 把高度转法线的角误差写成差分核三阶矩的平方，
反解出使 O(d²) 残差消失的核；推导见该包文档。

