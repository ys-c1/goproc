# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处（`Kernel` 可选 `quintic`/`cubic`，默认五次核）
- `internal/noise/` 值噪声，置换表由 splitmix64 派生；插值核见 `Kernel` 接口
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/jumplab/` 插值核格点跳变对照实验（三档频率 × 两种核）
- `docs/kernel-smoothness.md` 三次核 C¹、五次核 C² 的证明与跳变消失条件

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/jumplab --size 256 --seed 7 --outdir artifacts/jumplab
```
