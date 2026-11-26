# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/diff/` 方格网有限差分（内部二阶中心差分，边界单侧二阶模板）
- `internal/ridge/` Hessian 特征分解、Mask 连通段、上升合并树、G=0 脊线提取与比对
- `docs/ridges.md` 脊线判据的证明、反例与实测
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/ridge/` 命令行，导出脊线与段数比对报告

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/ridge --seeds 7,42 --size 128 --out artifacts
```

注意：`EncodePGM` 的头部写成了 `"P5\\n..."` 字面量，输出里是反斜杠加 n 而不是
换行，标准 PGM 阅读器读不了；`cmd/ridge` 输出 PGM/PPM 时另写了正确的头。
