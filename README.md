# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/ridge/` 曲率脊线检测、连通段归类与像素总数预判
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/ridge/` 命令行，检测脊线、导出掩码并打印预判/实测对照
- `tools/verify_ridge.py` 独立复核脚本，不引用 Go 代码

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
make ridge        # 生成 artifacts/ridge/
make verify-ridge # 独立复核
```

脊线检测与总数预判的原理、结果与前提失效情况见 `docs/ridge.md`。
