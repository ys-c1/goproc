# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/fill/` 填洼与水面分析，数学依据见 `docs/fill-proof.md`
- `cmd/goproc/` 命令行，`gen` 导出高度图，`fill` 做填洼核验

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc fill --size 128 --seed 7 --octaves 4 --thresh 8,32 \
  --before before.pgm --after after.pgm
```

`fill` 子命令：单遍优先泛滥求各洼地溢流高度，按溢流高度预判水面像素数与
最大水深，再与迭代松弛填洼逐格比对（不一致格报坐标）。排水按八邻域，
水面像素按四邻域连通计，水深按灰度差给出。
